"""
COSMO gate -- sigma_8 weak-lensing experiment (LSST Y10 lognormal, sbi_lens setup),
linear vs poly2/PATP score surrogate INSIDE the FSM authors' own pipeline.

Portable driver for an Apple M3 Max (48 GB) CPU box.  Everything but the feature
map is the authors' camera-ready code (../../camera_ready_nips_2025_direct_fsm),
imported verbatim; the generalization is fit_feature_sm = their fit_linear_sm with
ONE line changed (sims_q_aug = feature_fn(sims_q)) -- see phase3_real/cosmo_patch.md.

Design (pre-registered; see README_UA.md -- do NOT change hyperparameters or
seeds between arms, the comparison is PAIRED at equal simulation budget):
  - theta = [log Om_c, log Om_b, log sigma_8, h, n_s, w]; sigma_8 is component 2,
    an AMPLITUDE (scale-type) parameter of non-Gaussian lognormal maps -- the
    parity theory predicts the linear surrogate under-performs there.
  - authors' hyperparameters, kept identical for every arm:
      N_ITER=100, N_PROP=25, N_SIM_DST=4, sigma=1e-3, lr=1e-2, lamb=1.0,
      Polyak average over the last 50 iterates, adam + clip_by_global_norm(0.1).
  - arms: linear (their fit_linear_sm), poly2 (feature alpha=1.0),
          patp05 / patp07 (fractional even exponents).
  - pairing: run r of EVERY arm uses master key PRNGKey(2026000 + r) for the
    observed dataset and the SGD chain, and the same fixed theta_init
    (sample_lensing_prior(PRNGKey(0)), as in the authors' script).
  - per run we record theta_hat (Polyak), the authors' aggregate unbound-space
    mse, and per-component NATURAL-space absolute errors (exp() on the three
    log components), sigma_8 being the one the gate reads.
Checkpointing: one pickle per finished run in --results-dir; re-running the same
command skips finished runs, so the experiment survives interruptions.

Usage:
  python cosmo_gate_experiment.py --arm linear  --runs 20
  python cosmo_gate_experiment.py --arm poly2   --runs 20
  python cosmo_gate_experiment.py --arm patp05  --runs 20      # optional arm
  python cosmo_gate_experiment.py --smoke                      # 5-min env+timing probe
Then: python analyze_results.py
"""
import argparse
import importlib
import json
import os
import pickle
import sys
import time
import types
from functools import partial

HERE = os.path.dirname(os.path.abspath(__file__))
CR_REPO = os.environ.get(
    "FSM_REPO",
    os.path.abspath(os.path.join(HERE, "..", "..", "camera_ready_nips_2025_direct_fsm")))
if not os.path.isdir(os.path.join(CR_REPO, "approxml")):
    sys.exit(
        f"[FATAL] FSM authors' code not found at {CR_REPO}\n"
        "It is gitignored (third-party), so a fresh clone does not include it. Fix:\n"
        f"  git clone https://github.com/Shermjj/Direct_FSM '{CR_REPO}'\n"
        "or point FSM_REPO=<path-to-Direct_FSM-clone> (must contain approxml/).")
sys.path.insert(0, CR_REPO)

import numpy as np
import jax
import jax.numpy as jnp
import optax

# --- tensorflow_probability JAX-substrate shim (verbatim from the authors' script) ---
import tensorflow_probability as tfp
_jax_backend = None
for _candidate in ("tensorflow_probability.substrates.jax",
                   "tensorflow_probability.experimental.substrates.jax"):
    try:
        _jax_backend = importlib.import_module(_candidate)
        break
    except ModuleNotFoundError:
        pass
if _jax_backend is None:
    raise ImportError("Couldn't locate the JAX substrate inside tensorflow_probability.")
if not hasattr(tfp, "substrates"):
    tfp.substrates = types.SimpleNamespace()
tfp.substrates.jax = _jax_backend
if not hasattr(tfp, "experimental"):
    tfp.experimental = types.SimpleNamespace()
if not hasattr(tfp.experimental, "substrates"):
    tfp.experimental.substrates = types.SimpleNamespace()
tfp.experimental.substrates.jax = _jax_backend
# --------------------------------------------------------------------------------------

from approxml.utils import gen_simulation_samples, grad_log_normal          # noqa: E402
from approxml.scorematching import fit_linear_sm                            # noqa: E402
from approxml.simulators import mvt_norm_simulator                          # noqa: E402
from approxml.cosmo import lensing_simulator, sample_lensing_prior          # noqa: E402

# ---------------------------------------------------------------------------------
# Authors' hyperparameters -- IDENTICAL for every arm (do not touch)
# ---------------------------------------------------------------------------------
SEED_BASE = 2026000
N_ITER = 100
N_PROP = 25
N_SIM_DST = 4
N_PARAM_DIM = 6
SIGMA_PROP = 1e-3
LR = 1e-2
LAMB = 1.0
POLYAK_TAIL = 50

# Fiducial truth (authors' script), unbound space (log on the first three):
PARAMS_NATURAL = np.array([0.2664, 0.0492, 0.831, 0.6727, 0.9645, -1.0])
PARAM_NAMES = ["omega_c", "omega_b", "sigma_8", "h_0", "n_s", "w_0"]

ARM_ALPHA = {"linear": None, "poly2": 1.0, "patp05": 0.5, "patp07": 0.7}


def _p2(a):
    """PATP even exponent p_2(alpha) = 1/2 + alpha/2 + alpha^2  (1/2 -> 2)."""
    return 0.5 + 0.5 * a + 1.0 * a ** 2


def make_feature_fn(alpha):
    p = _p2(alpha)

    def feature_fn(x):                       # x: (..., 6) -> (..., 13)
        return jnp.concatenate([x, jnp.abs(x) ** p, jnp.ones_like(x[..., :1])], axis=-1)

    return feature_fn


def fit_feature_sm(key, theta_t, gen_sim_fn, grad_log_prop_fn, n_prop, n_sim_dst,
                   feature_fn, lamb=1e-3, thetas_q=None, sims_q=None):
    """The authors' fit_linear_sm with ONE changed line (the feature augmentation)."""
    theta_dim = theta_t.shape[0]
    if thetas_q is None and sims_q is None and gen_sim_fn is not None:
        thetas_q, sims_q, _ = gen_sim_fn(key, theta_t)
    grad_log_q_1 = jax.vmap(grad_log_prop_fn, in_axes=(0, None))(thetas_q, theta_t)
    grad_log_q_2 = jnp.repeat(grad_log_q_1, n_sim_dst, axis=0).reshape(n_prop, n_sim_dst, theta_dim)
    sims_q_aug = feature_fn(sims_q)          # <<< the only change vs fit_linear_sm
    G_j = jax.vmap(lambda x: x.T @ x, in_axes=0)(sims_q_aug).sum(0)
    reg_term = lamb * jnp.eye(G_j.shape[0], M=G_j.shape[1])
    W = - jnp.linalg.inv(G_j + reg_term) @ jax.vmap(
        jax.vmap(jnp.outer, in_axes=(0, 0)), in_axes=(0, 0))(sims_q_aug, grad_log_q_2).sum(0).sum(0)
    return W, sims_q, sims_q_aug, thetas_q


def load_data(data_dir):
    paths = {
        "shifts": os.path.join(data_dir, "lognormal_shifts_LSSTY10_om_s8_w_bin.npy"),
        "opt_state": os.path.join(data_dir, "opt_state_resnet_vmim.pkl"),
        "compressor": os.path.join(data_dir, "params_nd_compressor_vmim.pkl"),
    }
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        sys.exit(f"[FATAL] missing data files:\n  " + "\n  ".join(missing) +
                 f"\nRun fetch_data.sh or point --data-dir at them (see README_UA.md).")
    shifts = np.load(paths["shifts"])
    with open(paths["opt_state"], "rb") as f:
        opt_state_resnet = pickle.load(f)
    with open(paths["compressor"], "rb") as f:
        parameters_compressor = pickle.load(f)
    return shifts, opt_state_resnet, parameters_compressor


def build(data_dir):
    """Return (sim_fn, params_unbound, theta_init) exactly as in the authors' script."""
    shifts, opt_state_resnet, parameters_compressor = load_data(data_dir)

    @partial(jax.jit, static_argnames=("n_sim", "compress"))
    def sim_fn(key, unbound_params, n_sim, compress=True):
        p0, p1, p2, p3, p4, p5 = unbound_params
        params_bound = jnp.array([jnp.exp(p0), jnp.exp(p1), jnp.exp(p2), p3, p4, p5])
        return lensing_simulator(key, params_bound, n_sim, compress=compress,
                                 opt_state_resnet=opt_state_resnet,
                                 parameters_compressor=parameters_compressor,
                                 lognormal_shifts_params=shifts)

    params = jnp.array(PARAMS_NATURAL)
    for i in (0, 1, 2):
        params = params.at[i].set(jnp.log(params[i]))
    theta_init = sample_lensing_prior(jax.random.PRNGKey(0))   # fixed, as in their script
    return sim_fn, params, theta_init


def run_sgd(key, theta_init, obs, sim_fn, arm, n_iter):
    """The authors' run_sgd with the feature-map switch; budget identical per arm."""
    alpha = ARM_ALPHA[arm]
    if alpha is None:
        obs_aug = jnp.concatenate([obs, jnp.ones_like(obs[..., :1])], axis=-1)
    else:
        feature_fn = make_feature_fn(alpha)
        obs_aug = feature_fn(obs)

    optimizer = optax.chain(optax.clip_by_global_norm(1e-1), optax.adam(LR))
    opt_state = optimizer.init(theta_init)
    prop_cov = SIGMA_PROP * jnp.eye(N_PARAM_DIM)

    gen_sim_fn = partial(gen_simulation_samples, simulator_fn=sim_fn,
                         prop_sim_fn=partial(mvt_norm_simulator, cov=prop_cov),
                         n_prop=N_PROP, n_sim_dst=N_SIM_DST)
    if alpha is None:
        grad_fit = partial(fit_linear_sm, gen_sim_fn=gen_sim_fn,
                           grad_log_prop_fn=partial(grad_log_normal, cov=prop_cov),
                           n_sim_dst=N_SIM_DST, n_prop=N_PROP, lamb=LAMB)
    else:
        grad_fit = partial(fit_feature_sm, gen_sim_fn=gen_sim_fn,
                           grad_log_prop_fn=partial(grad_log_normal, cov=prop_cov),
                           n_sim_dst=N_SIM_DST, n_prop=N_PROP,
                           feature_fn=feature_fn, lamb=LAMB)

    @jax.jit
    def update(params_, opt_state_, key_):
        key_, subkey = jax.random.split(key_)
        W, _, _, _ = grad_fit(subkey, params_)
        grads = jnp.einsum("mk,ik->im", W.T, obs_aug).sum(0)
        grads = -grads
        updates, opt_state_ = optimizer.update(grads, opt_state_, params_)
        params_ = optax.apply_updates(params_, updates)
        return params_, opt_state_, key_

    theta = theta_init.copy()
    thetas = []
    for _ in range(n_iter):
        theta, opt_state, key = update(theta, opt_state, key)
        thetas.append(np.asarray(theta))
    thetas = np.stack(thetas)
    tail = min(POLYAK_TAIL, max(1, n_iter // 2))
    return thetas[-tail:].mean(0), thetas


def natural_errors(theta_hat_unbound):
    th = np.asarray(theta_hat_unbound, dtype=float).copy()
    th[:3] = np.exp(th[:3])
    return np.abs(th - PARAMS_NATURAL), th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARM_ALPHA), default="poly2")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--n-obs", type=int, default=250)
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--smoke", action="store_true",
                    help="1 run, 5 SGD iters, n_obs=50: environment + timing probe only")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    n_iter = 5 if args.smoke else N_ITER
    n_runs = 1 if args.smoke else args.runs
    n_obs = 50 if args.smoke else args.n_obs

    print(f"[cosmo-gate] arm={args.arm}  runs={n_runs}  n_obs={n_obs}  n_iter={n_iter}")
    print(f"[cosmo-gate] jax {jax.__version__}  backend={jax.default_backend()}  "
          f"devices={jax.devices()}")
    if jax.default_backend() != "cpu":
        print("[WARN] non-CPU backend detected. The pre-registered protocol is CPU "
              "(jax-metal is experimental; do NOT use it for the final runs).")

    sim_fn, params, theta_init = build(args.data_dir)

    t_sim0 = time.time()
    _ = np.asarray(sim_fn(jax.random.PRNGKey(1), params, 4))
    t_first_sim = time.time() - t_sim0
    print(f"[timing] first (compile+run) simulator call, n_sim=4: {t_first_sim:.1f}s")

    for r in range(n_runs):
        tag = "smoke" if args.smoke else f"run{r:03d}"
        out_path = os.path.join(args.results_dir, f"{args.arm}_{tag}.pkl")
        if os.path.exists(out_path):
            print(f"[skip] {out_path} exists (resume mode)")
            continue
        master = jax.random.PRNGKey(SEED_BASE + r)
        k_obs, k_sgd = jax.random.split(master)
        t0 = time.time()
        obs = sim_fn(k_obs, params, n_obs)
        theta_hat, traj = run_sgd(k_sgd, theta_init, obs, sim_fn, args.arm, n_iter)
        dt = time.time() - t0
        errs, th_nat = natural_errors(theta_hat)
        mse_unbound = float(np.linalg.norm(np.asarray(theta_hat) - np.asarray(params)))
        rec = {
            "arm": args.arm, "run": r, "n_obs": n_obs, "n_iter": n_iter,
            "seed": SEED_BASE + r, "theta_hat_unbound": np.asarray(theta_hat).tolist(),
            "theta_hat_natural": th_nat.tolist(),
            "abs_err_natural": errs.tolist(), "param_names": PARAM_NAMES,
            "sigma8_abs_err": float(errs[2]), "mse_unbound": mse_unbound,
            "wall_seconds": dt,
            "hyper": {"N_PROP": N_PROP, "N_SIM_DST": N_SIM_DST, "sigma": SIGMA_PROP,
                      "lr": LR, "lamb": LAMB, "polyak_tail": POLYAK_TAIL},
        }
        with open(out_path, "wb") as f:
            pickle.dump({**rec, "trajectory": traj}, f)
        with open(out_path.replace(".pkl", ".json"), "w") as f:
            json.dump(rec, f, indent=2)
        print(f"[done] {args.arm} {tag}: sigma8_err={errs[2]:.4f}  "
              f"mse={mse_unbound:.4f}  ({dt/60:.1f} min)")
        if args.smoke:
            per_run_full = dt / n_iter * N_ITER
            print(f"[smoke] extrapolation: ~{per_run_full/60:.0f} min per full run "
                  f"-> {per_run_full*20/3600:.1f} h per 20-run arm "
                  f"(x2 arms = {per_run_full*40/3600:.1f} h; x3 = {per_run_full*60/3600:.1f} h)")

    print("[cosmo-gate] arm finished. Next: run the other arm(s), then analyze_results.py")


if __name__ == "__main__":
    main()
