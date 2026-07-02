"""
Ku-SBI-PMM  --  prototype: a Kunchenko stochastic-polynomial / PATP score surrogate
for simulation-based Fisher-score matching (Khoo et al., arXiv:2506.06542).

Thesis under test (see ../fsm_fisher_score_kunchenko_analysis.md):
  FSM with a LINEAR surrogate S_W(x)=W^T x = truncation order m=1 of the Kunchenko
  likelihood series.  For a SCALE parameter the Fisher score is EVEN in x (quadratic),
  which a linear (and the published odd sign-parity PATP) surrogate cannot represent.
  An EVEN PATP power |x|^{p2(alpha)} (continuously |x|^{1/2} -> x^2) fixes it and keeps
  the closed-form least-squares solution (paper Eq. 4-5).

This driver is moment-free: it never evaluates a density, only SIMULATES from the model
and uses the integration-by-parts (Stein) identity, exactly as the FSM paper does.

Run:  python run.py      (writes results/results.json, prints a summary)
Gate: python verify.py   (asserts the workflow's verification gates)
"""
import json, os
import numpy as np

# ----------------------------------------------------------------------------------
# PATP power map  p_i(alpha) = 1/i + (4-i-3/i) alpha + (2i-4+2/i) alpha^2
#   p_i(0)=1/i (fractional),  p_i(1)=i (integer)
# ----------------------------------------------------------------------------------
def patp_power(i, alpha):
    A = 1.0 / i
    B = 4 - i - 3.0 / i
    C = 2 * i - 4 + 2.0 / i
    return A + B * alpha + C * alpha**2

# ----------------------------------------------------------------------------------
# Feature maps phi(x) -> (..., d).  All include a constant (absorbs centering).
#   'linear' : [1, x]                       <- the paper's surrogate (m=1)
#   'poly2'  : [1, x, x^2]                  <- minimal mixed-parity Kunchenko
#   'poly3'  : [1, x, x^2, x^3]
#   ('patp', a): [1, x, |x|^{p2(a)}]        <- const, odd-location, EVEN-scale PATP term
#                                              patp@1 == poly2 ; patp@0 == [1,x,|x|^0.5]
# ----------------------------------------------------------------------------------
def get_basis(spec):
    if spec == "linear":
        return lambda x: np.stack([np.ones_like(x), x], axis=-1)
    if spec == "poly2":
        return lambda x: np.stack([np.ones_like(x), x, x**2], axis=-1)
    if spec == "poly3":
        return lambda x: np.stack([np.ones_like(x), x, x**2, x**3], axis=-1)
    if isinstance(spec, tuple) and spec[0] == "patp":
        p2 = patp_power(2, spec[1])
        return lambda x: np.stack([np.ones_like(x), x, np.abs(x)**p2], axis=-1)
    raise ValueError(spec)

# ----------------------------------------------------------------------------------
# DGPs: theta = (mu, v) with v = VARIANCE.  Vectorised: thetas is (m,2) -> X is (m,n).
# ----------------------------------------------------------------------------------
def sample_gaussian(thetas, n, rng):
    mu, v = thetas[:, 0], np.maximum(thetas[:, 1], 1e-9)
    return mu[:, None] + np.sqrt(v)[:, None] * rng.standard_normal((len(thetas), n))

def sample_t(thetas, n, rng, df=4.0):
    """x = mu + sqrt(v)*W, W ~ standardised t_df (Var W = 1, df>2) -> Var(x)=v."""
    mu, v = thetas[:, 0], np.maximum(thetas[:, 1], 1e-9)
    W = rng.standard_t(df, size=(len(thetas), n)) / np.sqrt(df / (df - 2.0))
    return mu[:, None] + np.sqrt(v)[:, None] * W

# Analytic TRUE Fisher scores (for GRADING ONLY -- the estimator never sees them).
def score_gaussian(x, mu, v):
    s_mu = (x - mu) / v
    s_v = -1.0 / (2 * v) + (x - mu) ** 2 / (2 * v**2)
    return s_mu, s_v

def score_t_v(x, mu, v, df=4.0):
    """d/dv log p for x = mu + sqrt(v)*standardised-t_df.  Bounded (redescending) in x."""
    c2 = (df - 2.0) / df                      # s^2 = c2 * v  (t scale)
    u2 = (x - mu) ** 2 / (c2 * v)
    return -1.0 / (2 * v) + (df + 1.0) * u2 / (2 * v * (df + u2))

# ----------------------------------------------------------------------------------
# Core: local FSM Fisher-score fit (paper Eq. 3-5), d_theta-dimensional, any basis.
# ----------------------------------------------------------------------------------
def fsm_fit(theta_t, sampler, basis, prop_scales, m, n, rng):
    """Return (score_fn: x->(...,2),  A: Gram matrix). Closed-form, moment-free."""
    theta_t = np.asarray(theta_t, float)
    d_theta = theta_t.size
    s = np.asarray(prop_scales, float)
    thetas = theta_t + s * rng.standard_normal((m, d_theta))
    thetas[:, 1] = np.abs(thetas[:, 1]) + 1e-6                      # variance > 0
    g = -(thetas - theta_t) / s**2                                  # grad_theta log q  (m, d_theta)
    X = sampler(thetas, n, rng)                                     # (m, n)
    Phi = basis(X).reshape(m * n, -1)                              # (m*n, d)
    G = np.repeat(g, n, axis=0)                                     # (m*n, d_theta)
    A = Phi.T @ Phi + 1e-8 * np.eye(Phi.shape[1])                  # sum_j G_j  (Eq 4 LHS)
    W = -np.linalg.solve(A, Phi.T @ G)                            # (d, d_theta)  (Eq 5)
    return (lambda x: basis(x) @ W), A

def fsm_mle(theta0, theta_star, sampler, basis, prop_scales, m, n, T, eta, rng):
    """Algorithm 1 + Polyak-Ruppert averaging. theta = (mu, v)."""
    D = sampler(np.array([theta_star]), N_OBS, rng).ravel()
    theta = np.asarray(theta0, float).copy()
    eta = np.asarray(eta, float)
    traj = []
    for _ in range(T):
        score_fn, _ = fsm_fit(theta, sampler, basis, prop_scales, m, n, rng)
        theta = theta + eta * np.mean(score_fn(D), axis=0)
        theta[1] = max(theta[1], 1e-3)
        traj.append(theta.copy())
    return np.mean(traj[T // 2:], axis=0)

def cond_centered(theta_t, sampler, basis, prop_scales, m, n, rng):
    """Conditioning of the standardised non-constant features (isolates parity/scale)."""
    thetas = np.asarray(theta_t, float) + np.asarray(prop_scales) * rng.standard_normal((m, 2))
    thetas[:, 1] = np.abs(thetas[:, 1]) + 1e-6
    X = sampler(thetas, n, rng)
    Phi = basis(X).reshape(m * n, -1)[:, 1:]                       # drop constant
    Z = (Phi - Phi.mean(0)) / (Phi.std(0) + 1e-12)
    return float(np.linalg.cond(Z.T @ Z / Z.shape[0]))

# ----------------------------------------------------------------------------------
# Globals
# ----------------------------------------------------------------------------------
N_OBS = 1500
PROP = (0.3, 0.4)
SEED = 2026

# ==================================================================================
def exp_score_recovery():
    """Exp 1: recover (s_mu, s_v) at theta_t=(2,4); known analytic Gaussian score."""
    rng = np.random.default_rng(SEED)
    tt = np.array([2.0, 4.0]); mu, v = tt
    xg = np.linspace(mu - 5 * np.sqrt(v), mu + 5 * np.sqrt(v), 400)
    s_mu_true, s_v_true = score_gaussian(xg, mu, v)
    out = {}
    for spec, name in [("linear", "linear [1,x]"), ("poly2", "poly2 [1,x,x^2]"),
                       (("patp", 1.0), "patp@1.0"), (("patp", 0.5), "patp@0.5")]:
        fn, _ = fsm_fit(tt, sample_gaussian, get_basis(spec), PROP, 2000, 120, rng)
        S = fn(xg)
        out[name] = {"mse_mu": float(np.mean((S[:, 0] - s_mu_true) ** 2)),
                     "mse_v": float(np.mean((S[:, 1] - s_v_true) ** 2))}
    return out

def exp_joint_mle():
    """Exp 2: estimate (mu, v) jointly from Gaussian data; linear vs poly2 vs patp."""
    theta_star = np.array([2.0, 4.0]); theta0 = np.array([0.0, 1.0])
    eta = (0.4, 0.6); res = {}
    for spec, name in [("linear", "linear [1,x]"), ("poly2", "poly2 [1,x,x^2]"),
                       (("patp", 0.5), "patp@0.5")]:
        ests = []
        for r in range(8):
            rng = np.random.default_rng(SEED + 100 * r)
            ests.append(fsm_mle(theta0, theta_star, sample_gaussian, get_basis(spec),
                                PROP, 500, 70, 300, eta, rng))
        ests = np.array(ests)
        res[name] = {"mu": float(ests[:, 0].mean()), "mu_sd": float(ests[:, 0].std()),
                     "v": float(ests[:, 1].mean()), "v_sd": float(ests[:, 1].std())}
    res["_truth"] = {"mu": 2.0, "v": 4.0}
    return res

def exp_patp_sweep():
    """Exp 3 (pattern N): sweep alpha; score-recovery of the SCALE score + cond(F),
    for Gaussian (expect alpha*~1, x^2 exact) vs Student-t (expect alpha*<1)."""
    alphas = np.round(np.linspace(0, 1, 11), 2)
    tt = np.array([2.0, 4.0]); mu, v = tt
    xg = np.linspace(mu - 6 * np.sqrt(v), mu + 6 * np.sqrt(v), 500)
    sweep = {}
    for dgp, sampler, sv_true in [
            ("gaussian", sample_gaussian, score_gaussian(xg, mu, v)[1]),
            ("student_t_df4", lambda th, n, rng: sample_t(th, n, rng, 4.0),
             score_t_v(xg, mu, v, 4.0))]:
        mse, cond = [], []
        for a in alphas:
            rng = np.random.default_rng(SEED + 7)
            fn, _ = fsm_fit(tt, sampler, get_basis(("patp", a)), PROP, 3000, 150, rng)
            mse.append(float(np.mean((fn(xg)[:, 1] - sv_true) ** 2)))
            cond.append(cond_centered(tt, sampler, get_basis(("patp", a)), PROP, 1500, 80,
                                      np.random.default_rng(SEED + 8)))
        a_star = float(alphas[int(np.argmin(mse))])
        sweep[dgp] = {"alphas": alphas.tolist(), "mse_v": mse, "cond": cond,
                      "alpha_star": a_star, "mse_at_star": float(min(mse))}
    return sweep

# ==================================================================================
if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "results"), exist_ok=True)

    print("=" * 78)
    print("EXP 1  score recovery at theta_t=(mu=2, v=4)   [Gaussian; lower MSE better]")
    print("=" * 78)
    e1 = exp_score_recovery()
    print(f"  {'basis':18s} {'MSE(s_mu) odd/loc':>20s} {'MSE(s_v) even/scale':>22s}")
    for k, d in e1.items():
        print(f"  {k:18s} {d['mse_mu']:20.5f} {d['mse_v']:22.5f}")
    print("  -> linear cannot represent the EVEN scale score (large MSE(s_v)).")

    print("\n" + "=" * 78)
    print("EXP 2  joint FSM-MLE of (mu, v) from Gaussian data   [8 runs, mean+/-sd]")
    print("=" * 78)
    e2 = exp_joint_mle()
    print(f"  truth: mu=2.0, v=4.0")
    for k, d in e2.items():
        if k.startswith("_"):
            continue
        print(f"  {k:18s} mu={d['mu']:6.3f}+/-{d['mu_sd']:4.2f}   "
              f"v={d['v']:6.3f}+/-{d['v_sd']:4.2f}")
    print("  -> all recover mu; only x^2-bearing bases recover v.")

    print("\n" + "=" * 78)
    print("EXP 3  PATP alpha-sweep: scale-score recovery + conditioning")
    print("=" * 78)
    e3 = exp_patp_sweep()
    for dgp, d in e3.items():
        print(f"  {dgp:16s} alpha* = {d['alpha_star']:.2f}  "
              f"(MSE@star={d['mse_at_star']:.5f}, cond range "
              f"{min(d['cond']):.1f}..{max(d['cond']):.1f})")
    print("  -> Gaussian favours alpha~1 (x^2 exact); heavy-tail favours alpha<1.")

    with open(os.path.join(here, "results", "results.json"), "w") as f:
        json.dump({"exp1_score_recovery": e1, "exp2_joint_mle": e2,
                   "exp3_patp_sweep": e3, "seed": SEED, "N_OBS": N_OBS,
                   "prop_scales": PROP}, f, indent=2)
    print("\n[written] results/results.json")
