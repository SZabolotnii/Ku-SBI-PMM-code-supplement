"""
Phase 3 (real-code validation): plug a Kunchenko poly / PATP feature map into the
AUTHORS' OWN FSM machinery (approxml) and test the scale-parameter claim.

We import their core verbatim:
    approxml.scorematching.fit_linear_sm  (their Eq.5 closed form -- reproduction baseline)
    approxml.utils.gen_simulation_samples, grad_log_normal
    approxml.simulators.mvt_norm_simulator
and add ONE generalization, fit_feature_sm, which is fit_linear_sm with the single
line `sims_q_aug = concat([sims_q, 1])` replaced by `sims_q_aug = feature_fn(sims_q)`.
Everything downstream (Gram G_j, regularised inverse, W, the einsum prediction) is
untouched -- proving the closed form is agnostic to the feature map.

Result claim: their LINEAR surrogate recovers a MEAN but fails a VARIANCE (scale)
parameter; a degree-2 / PATP even feature fixes it, inside their own code path.
"""
import jax, jax.numpy as jnp, optax, numpy as np
from functools import partial

# ---- their code, imported verbatim --------------------------------------------------
from approxml.scorematching import fit_linear_sm
from approxml.utils import gen_simulation_samples, grad_log_normal
from approxml.simulators import mvt_norm_simulator

# ---- the ONLY addition: their fit_linear_sm, generalised to an arbitrary feature map -
def fit_feature_sm(key, theta_t, gen_sim_fn, grad_log_prop_fn, n_prop, n_sim_dst,
                   feature_fn, lamb=1e-3, thetas_q=None, sims_q=None):
    theta_dim = theta_t.shape[0]
    if thetas_q is None and sims_q is None and gen_sim_fn is not None:
        thetas_q, sims_q, _ = gen_sim_fn(key, theta_t)
    grad_log_q_1 = jax.vmap(grad_log_prop_fn, in_axes=(0, None))(thetas_q, theta_t)
    grad_log_q_2 = jnp.repeat(grad_log_q_1, n_sim_dst, axis=0).reshape(n_prop, n_sim_dst, theta_dim)
    sims_q_aug = feature_fn(sims_q)                       # <<< the single changed line
    G_j = jax.vmap(lambda x: x.T @ x, in_axes=0)(sims_q_aug).sum(0)
    reg_term = lamb * jnp.eye(G_j.shape[0], M=G_j.shape[1])
    W = - jnp.linalg.inv(G_j + reg_term) @ jax.vmap(
        jax.vmap(jnp.outer, in_axes=(0, 0)), in_axes=(0, 0))(sims_q_aug, grad_log_q_2).sum(0).sum(0)
    return W, sims_q, sims_q_aug, thetas_q

# ---- feature maps (operate on the last axis; broadcast over any leading dims) --------
def feat_linear(x): return jnp.concatenate([x, jnp.ones_like(x[..., :1])], axis=-1)          # = theirs
def feat_poly2(x):  return jnp.concatenate([x, x**2, jnp.ones_like(x[..., :1])], axis=-1)
def patp_p2(a):     return 1/2 + (4 - 2 - 3/2) * a + (2*2 - 4 + 2/2) * a**2                  # p_2(alpha): 1/2 -> 2
def feat_patp(a):
    p = patp_p2(a)
    return lambda x: jnp.concatenate([x, jnp.abs(x)**p, jnp.ones_like(x[..., :1])], axis=-1)

# ---- simulators in the authors' (key, theta, n_sim) -> (n_sim, d) convention ---------
def mean_sim(key, theta, n_sim):                          # theta = MEAN, fixed unit variance
    return (jax.random.normal(key, (n_sim,)) + theta[0]).reshape(n_sim, 1)
def var_sim(key, theta, n_sim):                           # theta = VARIANCE (scale param)
    v = jnp.maximum(theta[0], 1e-6)
    return (jax.random.normal(key, (n_sim,)) * jnp.sqrt(v)).reshape(n_sim, 1)
def var_sim_t(df):                                        # heavy-tailed scale, Var = theta
    def f(key, theta, n_sim):
        v = jnp.maximum(theta[0], 1e-6)
        W = jax.random.t(key, df, (n_sim,)) / jnp.sqrt(df / (df - 2.0))
        return (W * jnp.sqrt(v)).reshape(n_sim, 1)
    return f

# ---- FSM-MLE loop, mirroring their run_sgd (optax adam, negate score, Polyak avg) ----
def fsm_mle(key, theta_init, obs, feature_fn, simulator_fn, prop_sigma,
            n_prop, n_sim_dst, n_iter, lr, lamb, positive=False):
    obs_aug = feature_fn(obs)
    pdim = theta_init.shape[0]
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(theta_init)
    theta = theta_init
    traj = []
    for _ in range(n_iter):
        key, subkey = jax.random.split(key)
        prop_cov = prop_sigma * jnp.eye(pdim)
        gen = partial(gen_simulation_samples, simulator_fn=simulator_fn,
                      prop_sim_fn=partial(mvt_norm_simulator, cov=prop_cov),
                      n_prop=n_prop, n_sim_dst=n_sim_dst)
        W, _, _, _ = fit_feature_sm(subkey, theta, gen_sim_fn=gen,
                                    grad_log_prop_fn=partial(grad_log_normal, cov=prop_cov),
                                    n_prop=n_prop, n_sim_dst=n_sim_dst,
                                    feature_fn=feature_fn, lamb=lamb)
        grads = -jnp.einsum('mk,ik->im', W.T, obs_aug).sum(0)     # negate score -> ascend
        updates, opt_state = optimizer.update(grads, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        if positive:
            theta = theta.at[0].set(jnp.maximum(theta[0], 1e-3))
        traj.append(theta)
    return jnp.mean(jnp.stack(traj)[n_iter // 2:], axis=0)        # Polyak-Ruppert

# ---- analytic TRUE scale-scores (grading only; mu=0 since var_sim centres at 0) -----
def sv_gauss(x, v):
    return -1.0 / (2 * v) + x**2 / (2 * v**2)
def sv_t(df):
    def f(x, v):
        c2 = (df - 2.0) / df
        u2 = x**2 / (c2 * v)
        return -1.0 / (2 * v) + (df + 1.0) * u2 / (2 * v * (df + u2))
    return f

def score_recovery_sweep(simulator_fn, sv_true_fn, theta_star, alphas, prop_sigma,
                         m=3000, n=60, lamb=1e-3):
    """Fit W via THEIR fit_feature_sm at theta_t=theta*, grade score vs analytic truth."""
    tt = jnp.array([float(theta_star)])
    xg = np.linspace(-6 * np.sqrt(theta_star), 6 * np.sqrt(theta_star), 400)
    xcol = jnp.array(xg).reshape(-1, 1)
    sv_true = sv_true_fn(xg, theta_star)
    out = {}
    for a in alphas:
        feat = feat_patp(a)
        prop_cov = prop_sigma * jnp.eye(1)
        gen = partial(gen_simulation_samples, simulator_fn=simulator_fn,
                      prop_sim_fn=partial(mvt_norm_simulator, cov=prop_cov),
                      n_prop=m, n_sim_dst=n)
        W, _, _, _ = fit_feature_sm(jax.random.fold_in(KEY, 500 + int(a * 100)), tt,
                                    gen_sim_fn=gen,
                                    grad_log_prop_fn=partial(grad_log_normal, cov=prop_cov),
                                    n_prop=m, n_sim_dst=n, feature_fn=feat, lamb=lamb)
        s_est = np.array((feat(xcol) @ W)[:, 0])
        out[float(a)] = float(np.mean((s_est - sv_true) ** 2))
    return out

# =====================================================================================
KEY = jax.random.PRNGKey(2026)
NP, ND, NOBS, NIT, LR, LAMB = 120, 25, 1000, 200, 0.05, 1.0
RESULTS = {}

def run_block(name, simulator_fn, theta_star, theta0, prop_sigma, positive, bases):
    obs = simulator_fn(jax.random.PRNGKey(0), jnp.array([theta_star]), NOBS)
    print(f"\n  {name}: theta* = {theta_star}, init = {theta0}")
    rows = {}
    for label, feat in bases:
        ests = np.array([float(fsm_mle(jax.random.fold_in(KEY, r), jnp.array([float(theta0)]),
                         obs, feat, simulator_fn, prop_sigma, NP, ND, NIT, LR, LAMB, positive)[0])
                         for r in range(6)])
        err = abs(ests.mean() - theta_star)
        flag = "  <-- FAILS" if err > 0.25 * abs(theta_star) else ""
        rows[label] = {"mean": float(ests.mean()), "sd": float(ests.std()), "err": float(err)}
        print(f"    {label:24s} est = {ests.mean():6.3f} +/- {ests.std():4.2f}   "
              f"|err| = {err:5.3f}{flag}")
    return rows

if __name__ == "__main__":
    import json, os
    print("=" * 78)
    print("PHASE 3  -  Kunchenko poly/PATP feature map inside the AUTHORS' FSM core")
    print("           (approxml.fit_* imported verbatim; one generalised line added)")
    print("=" * 78)

    print("\n[A] SANITY: estimate a MEAN with their linear feature (must work)")
    RESULTS["A_mean"] = run_block("mean N(theta,1)", mean_sim, 3.0, 0.0, 0.3, False,
              [("linear [x,1] (theirs)", feat_linear)])

    print("\n[B] THE TEST: estimate a VARIANCE (scale) -- linear should FAIL")
    print("    (their lamb=1.0 + smoothing -> a known downward bias; the CONTRAST is the point)")
    RESULTS["B_variance"] = run_block("variance N(0,theta)", var_sim, 4.0, 1.0, 0.15, True,
              [("linear [x,1] (theirs)", feat_linear),
               ("poly2  [x,x^2,1]", feat_poly2),
               ("patp@0.7 [x,|x|^p,1]", feat_patp(0.7))])

    print("\n[B-t] HEAVY-TAIL scale: estimate a Student-t(6) variance end-to-end")
    print("     (integer x^2 needs E[x^4]; on heavy tails a fractional PATP term is better-conditioned)")
    RESULTS["Bt_variance_t6"] = run_block("variance t6 N(0,theta)", var_sim_t(6.0), 4.0, 1.0, 0.15, True,
              [("linear [x,1] (theirs)", feat_linear),
               ("poly2  [x,x^2,1]", feat_poly2),
               ("patp@0.6 [x,|x|^p,1]", feat_patp(0.6))])

    print("\n[C] PATP value: scale-SCORE recovery sweep (sensitive metric, via their fit)")
    alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0]
    g = score_recovery_sweep(var_sim, sv_gauss, 4.0, alphas, 0.15)
    t = score_recovery_sweep(var_sim_t(4.0), sv_t(4.0), 4.0, alphas, 0.15)
    a_g = min(g, key=g.get); a_t = min(t, key=t.get)
    gain = (t[1.0] - t[a_t]) / t[1.0]
    RESULTS["C_sweep"] = {"gaussian": g, "student_t4": t,
                          "alpha_star_gauss": a_g, "alpha_star_t": a_t, "t_gain_vs_x2": gain}
    print("    alpha :", " ".join(f"{a:5.1f}" for a in alphas))
    print("    MSE Gauss:", " ".join(f"{g[a]:5.2f}" for a in alphas))
    print("    MSE t(4) :", " ".join(f"{t[a]:5.2f}" for a in alphas))
    print(f"    -> Gauss alpha*={a_g} (x^2 best); t(4) alpha*={a_t}, "
          f"gain over pure x^2 = {100*gain:.0f}%")

    os.makedirs("results", exist_ok=True)
    with open("results/phase3_results.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    print("\n[written] results/phase3_results.json")
