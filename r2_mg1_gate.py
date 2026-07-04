"""
R2 -- FSM-PATP estimation on the M/G/1 queue (Papamakarios-Murray-style SBI benchmark).

Thesis under test (see run.py and paper Sec. 3): the closed-form
FSM least-squares surrogate (Eq. 4-5) generalises from raw i.i.d. observations to a
SUMMARY-STATISTIC feature map when only compressed data are observable -- the canonical
simulation-based-inference setting, here with a non-smooth (recursive, non-differentiable)
simulator and no tractable likelihood. Only inter-departure times are observed.

M/G/1 queue (Papamakarios & Murray 2016 benchmark):
    service times  s_i ~ U[theta1, theta2]
    arrivals       Poisson(lam), lam known = 0.15  (inter-arrival ~ Exp(lam))
    departures     D_0 = 0,  D_i = s_i + max(D_{i-1}, V_i),  y_i = D_i - D_{i-1}
      (V_i = cumulative arrival times)
theta = (theta1, theta2), truth (1.0, 5.0), constraint theta1 < theta2.
Observed summary z(y) (6-dim): the 0.1/0.3/0.5/0.7/0.9 empirical quantiles of the M=50
inter-departure times, plus min(y).

FSM fit mirrors fsm_fit/fsm_mle in run.py, here applied to SUMMARIES instead of raw x:
    proposals theta_j ~ N(theta_t, sigma^2 I) (per coordinate), n simulated datasets each,
    features Phi = basis(z), targets u_j = -(theta_j - theta_t)/sigma^2 (per proposal,
    repeated over the n sims), closed-form ridge fit (Phi^T Phi + lambda I) W = -Phi^T U.
    Robbins-Monro ascent theta <- theta + eta * score_fn(z_obs), Polyak-Ruppert average
    over the second half of the T-step trajectory.
Basis: 'linear' = [1, z]; 'patp' = [1, z, |z_c|^p2(0.6)] (z_c = z centred at the
per-iteration proposal-batch mean -- the "median-anchor"; p2 = patp_power(2, 0.6) from
run.py).

R2a/R2b (gated): with the PATP-augmented basis, mean_over_reps(|theta_hat - truth|/truth)
< 0.25 on both coordinates (R=40 replications, FSM-MLE started at theta0=(2.0, 7.0)).
[info] (ungated): the same relative errors for the plain linear-summary basis, reported
for the manuscript's contrast table.

Run:  python r2_mg1_gate.py   (writes results/r2_mg1.json; prints [PASS]/[FAIL]; exit 1 on FAIL)
"""
import json
import os
import sys
import time

import numpy as np

from run import patp_power

# ----------------------------------------------------------------------------------
# Globals (fixed by the experiment design)
# ----------------------------------------------------------------------------------
SEED = 2026
LAM = 0.15                      # known arrival rate
M_OBS = 50                      # observed inter-departure times
THETA_STAR = np.array([1.0, 5.0])
THETA0 = np.array([2.0, 7.0])
R_REPS = 40
ALPHA_PATP = 0.6
REL_ERR_TOL = 0.25

# budget knobs (tuned only for runtime/convergence, within the spec's allowed ranges)
SIGMA = 0.25
J_PROP = 40
N_SIM = 3
T_STEPS = 150
ETA = np.array([0.05, 0.15])
RIDGE = 1e-6


# ----------------------------------------------------------------------------------
# M/G/1 simulator (vectorised over m*n independent queue realisations).
# ----------------------------------------------------------------------------------
def simulate_mg1(thetas, n, rng):
    """thetas: (m,2) -> summaries z: (m,n,6), n simulated datasets of M_OBS points each."""
    theta1, theta2 = thetas[:, 0], thetas[:, 1]
    m = thetas.shape[0]
    lo, hi = np.repeat(theta1, n), np.repeat(theta2, n)          # (m*n,)
    rows = m * n
    s = lo[:, None] + (hi - lo)[:, None] * rng.random((rows, M_OBS))       # service U[t1,t2]
    inter = rng.exponential(1.0 / LAM, size=(rows, M_OBS))                 # inter-arrivals
    V = np.cumsum(inter, axis=1)                                          # arrival times
    D_prev = np.zeros(rows)
    y = np.empty((rows, M_OBS))
    for i in range(M_OBS):
        D = s[:, i] + np.maximum(D_prev, V[:, i])
        y[:, i] = D - D_prev
        D_prev = D
    return summarize(y).reshape(m, n, 6)


def summarize(y):
    """y: (rows, M_OBS) -> z: (rows, 6) = [q.1, q.3, q.5, q.7, q.9, min]."""
    q = np.quantile(y, [0.1, 0.3, 0.5, 0.7, 0.9], axis=1).T               # (rows, 5)
    mn = y.min(axis=1, keepdims=True)
    return np.concatenate([q, mn], axis=1)


# ----------------------------------------------------------------------------------
# Feature maps on the 6-dim summary. basis(z, anchor) -> Phi; anchor unused by 'linear'.
# ----------------------------------------------------------------------------------
def basis_linear(z, anchor):
    ones = np.ones(z.shape[:-1] + (1,))
    return np.concatenate([ones, z], axis=-1)


def basis_patp(z, anchor, alpha=ALPHA_PATP):
    p2 = patp_power(2, alpha)
    zc = z - anchor
    ones = np.ones(z.shape[:-1] + (1,))
    return np.concatenate([ones, z, np.abs(zc) ** p2], axis=-1)


# ----------------------------------------------------------------------------------
# Closed-form FSM fit + Robbins-Monro/Polyak-Ruppert MLE, on summaries (mirrors
# fsm_fit/fsm_mle in run.py; the "raw x" of run.py becomes the 6-dim summary z here).
# ----------------------------------------------------------------------------------
def fsm_fit_mg1(theta_t, basis, sigma, J, n, ridge, rng):
    """Return score_fn: z -> (...,2). Closed-form, moment-free (Eq. 4-5 on summaries)."""
    theta_t = np.asarray(theta_t, float)
    thetas = theta_t + sigma * rng.standard_normal((J, 2))                 # per-coord proposals
    u = -(thetas - theta_t) / sigma ** 2                                   # (J,2) targets
    Z = simulate_mg1(thetas, n, rng)                                       # (J,n,6)
    anchor = Z.reshape(-1, 6).mean(axis=0)                                # proposal-batch mean
    Phi = basis(Z, anchor).reshape(J * n, -1)                             # (J*n, d)
    U = np.repeat(u, n, axis=0)                                            # (J*n, 2)
    A = Phi.T @ Phi + ridge * np.eye(Phi.shape[1])
    W = -np.linalg.solve(A, Phi.T @ U)                                    # (d, 2)
    return lambda z: basis(z, anchor) @ W


def fsm_mle_mg1(theta0, theta_star, basis, sigma, J, n, T, eta, ridge, rng):
    """Robbins-Monro ascent + Polyak-Ruppert averaging over the 2nd half. theta=(theta1,theta2)."""
    z_obs = simulate_mg1(np.array([theta_star]), 1, rng)[0, 0]            # one observed dataset
    theta = np.asarray(theta0, float).copy()
    traj = []
    for _ in range(T):
        score_fn = fsm_fit_mg1(theta, basis, sigma, J, n, ridge, rng)
        theta = theta + eta * score_fn(z_obs)
        theta[0] = max(theta[0], 0.1)                                     # theta1 >= 0.1
        theta[1] = max(theta[1], theta[0] + 0.2)                          # theta2 >= theta1+0.2
        traj.append(theta.copy())
    return np.mean(traj[T // 2:], axis=0)


def run_experiment(basis):
    """FSM-MLE over R_REPS replications; each rep re-seeds rng at SEED+100*r (run.py pattern)."""
    ests = []
    for r in range(R_REPS):
        rng = np.random.default_rng(SEED + 100 * r)
        ests.append(fsm_mle_mg1(THETA0, THETA_STAR, basis, SIGMA, J_PROP, N_SIM, T_STEPS,
                                 ETA, RIDGE, rng))
    return np.array(ests)


def check(name, cond, detail):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    return bool(cond)


# ==================================================================================
if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "results"), exist_ok=True)
    t0 = time.time()

    print("=" * 78)
    print("R2  M/G/1 queue FSM-PATP estimation (non-smooth simulator, summaries only)")
    print("=" * 78)
    print(f"  truth theta=(theta1,theta2)=({THETA_STAR[0]},{THETA_STAR[1]}), "
          f"theta0=({THETA0[0]},{THETA0[1]}), lam={LAM}, M={M_OBS}, R={R_REPS} reps")

    ests_patp = run_experiment(basis_patp)
    ests_lin = run_experiment(basis_linear)

    err1_patp = float(np.mean(np.abs(ests_patp[:, 0] - THETA_STAR[0]))) / THETA_STAR[0]
    err2_patp = float(np.mean(np.abs(ests_patp[:, 1] - THETA_STAR[1]))) / THETA_STAR[1]
    err1_lin = float(np.mean(np.abs(ests_lin[:, 0] - THETA_STAR[0]))) / THETA_STAR[0]
    err2_lin = float(np.mean(np.abs(ests_lin[:, 1] - THETA_STAR[1]))) / THETA_STAR[1]

    print(f"\n  PATP basis   theta1_hat = {ests_patp[:,0].mean():.3f} +/- {ests_patp[:,0].std():.3f}"
          f"   theta2_hat = {ests_patp[:,1].mean():.3f} +/- {ests_patp[:,1].std():.3f}")
    print(f"  linear basis theta1_hat = {ests_lin[:,0].mean():.3f} +/- {ests_lin[:,0].std():.3f}"
          f"   theta2_hat = {ests_lin[:,1].mean():.3f} +/- {ests_lin[:,1].std():.3f}")

    ok = []
    ok.append(check("R2a theta1 (PATP basis)",
                     err1_patp < REL_ERR_TOL,
                     f"mean rel err = {err1_patp:.4f} < {REL_ERR_TOL}"))
    ok.append(check("R2b theta2 (PATP basis)",
                     err2_patp < REL_ERR_TOL,
                     f"mean rel err = {err2_patp:.4f} < {REL_ERR_TOL}"))
    print(f"  [info] linear-summary basis (no gate): theta1 rel err = {err1_lin:.4f}, "
          f"theta2 rel err = {err2_lin:.4f}")

    all_pass = all(ok)
    elapsed = time.time() - t0
    print("\n" + "=" * 78)
    print(f"  => R2-MG1 {'PASS' if all_pass else 'FAIL'}   (runtime {elapsed:.1f}s)")

    with open(os.path.join(here, "results", "r2_mg1.json"), "w") as f:
        json.dump({
            "seed": SEED,
            "theta_star": THETA_STAR.tolist(),
            "theta0": THETA0.tolist(),
            "R_reps": R_REPS,
            "M_obs": M_OBS,
            "lam": LAM,
            "hyperparams": {"sigma": SIGMA, "J": J_PROP, "n": N_SIM, "T": T_STEPS,
                            "eta": ETA.tolist(), "ridge": RIDGE, "alpha_patp": ALPHA_PATP},
            "patp_basis": {
                "theta1_hat_mean": float(ests_patp[:, 0].mean()),
                "theta1_hat_sd": float(ests_patp[:, 0].std()),
                "theta2_hat_mean": float(ests_patp[:, 1].mean()),
                "theta2_hat_sd": float(ests_patp[:, 1].std()),
                "rel_err_theta1": err1_patp,
                "rel_err_theta2": err2_patp,
                "estimates": ests_patp.tolist(),
            },
            "linear_basis_info": {
                "theta1_hat_mean": float(ests_lin[:, 0].mean()),
                "theta1_hat_sd": float(ests_lin[:, 0].std()),
                "theta2_hat_mean": float(ests_lin[:, 1].mean()),
                "theta2_hat_sd": float(ests_lin[:, 1].std()),
                "rel_err_theta1": err1_lin,
                "rel_err_theta2": err2_lin,
                "estimates": ests_lin.tolist(),
            },
            "R2a_pass": bool(ok[0]),
            "R2b_pass": bool(ok[1]),
            "all_pass": bool(all_pass),
            "runtime_s": elapsed,
        }, f, indent=2)
    print("  [written] results/r2_mg1.json")

    sys.exit(0 if all_pass else 1)
