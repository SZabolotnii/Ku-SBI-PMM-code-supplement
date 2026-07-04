"""
R1 -- parity obstruction and its PATP repair on the g-and-k distribution.

Thesis under test (paper Sec. 6, Lemma parity): for a SYMMETRIC scale family the score of
the scale parameter is EVEN in x, so the linear surrogate [1, x] is structurally blind to
it (g_m = 0), while a single even PATP feature |x|^{p2(alpha)} repairs the blindness.  The
g-and-k distribution is the canonical SBI benchmark where the density is unavailable but
sampling is trivial (quantile sampler); with skewness g_skew = 0 its scale parameter B is a
pure scale parameter of a symmetric density, so the parity theory applies EXACTLY; with
g_skew > 0 the parities mix and the mixed even+odd basis becomes necessary (paper,
"Beyond symmetry and scale").

g-and-k sampler (A = 0 location, k = 0.3 kurtosis fixed; theta = B scale, truth 2.0):
    x = Q(z) = B * (1 + 0.8*tanh(g_skew*z/2)) * z * (1 + z^2)^k,   z ~ N(0,1).

Sub-gates (pre-registered; thresholds fixed BEFORE the runs):
    R1a-blind  (g_skew=0):   mean_r |B_hat_linear - 2.0| > 0.5   (linear surrogate fails)
    R1a-repair (g_skew=0):   mean_r |B_hat_patp   - 2.0| < 0.3   (even PATP term repairs)
    R1b-mixed  (g_skew=0.5): mean_r |B_hat_mixed  - 2.0| < 0.4   (mixed basis survives skew)
    [info]     (g_skew=0.5): even-only basis error reported, ungated (parities mix).

FSM machinery mirrors run.py (fsm_fit / fsm_mle), specialised to scalar theta = B:
proposals B_j ~ N(B_t, sigma^2), n draws per proposal, features Phi(x), IBP targets
u_j = -(B_j - B_t)/sigma^2, closed-form ridge solve (Phi^T Phi + lambda I) W = -Phi^T u,
Robbins-Monro ascent on the mean estimated score over the observed data, Polyak-Ruppert
averaging over the second half.

Run:  python r1_gandk_gate.py   (writes results/r1_gandk.json; exit 1 on FAIL)
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
B_STAR = 2.0
B0 = 1.0
K_GK = 0.3
C_GK = 0.8
N_OBS = 500
R_REPS = 40
ALPHA_PATP = 0.6

THR_BLIND = 0.5      # pre-registered (do not alter)
THR_REPAIR = 0.3
THR_MIXED = 0.4

# budget knobs (tuned only for runtime/convergence; house recipe of run.py:
# large proposal batch + sigma ~ 0.3 keeps Var(W_hat) ~ 1/(J sigma^2) small enough
# for a stable Robbins-Monro drift)
SIGMA = 0.30
J_PROP = 200
N_SIM = 20
T_STEPS = 150
ETA = 0.30
RIDGE = 1e-8
B_CLIP = 0.2


# ----------------------------------------------------------------------------------
def sample_gandk(B, g_skew, n, rng):
    """B: (J,) scale values -> x: (J, n) draws from g-and-k with A=0, k=K_GK."""
    z = rng.standard_normal((B.shape[0], n))
    return B[:, None] * (1.0 + C_GK * np.tanh(g_skew * z / 2.0)) * z * (1.0 + z ** 2) ** K_GK


def get_basis_r1(spec):
    """Feature maps on raw scalar x. 'linear'=[1,x]; 'patp'=[1,x,|x|^p2]; 'even'=[1,|x|^p2]."""
    p2 = patp_power(2, ALPHA_PATP)
    if spec == "linear":
        return lambda x: np.stack([np.ones_like(x), x], axis=-1)
    if spec == "patp":
        return lambda x: np.stack([np.ones_like(x), x, np.abs(x) ** p2], axis=-1)
    if spec == "even":
        return lambda x: np.stack([np.ones_like(x), np.abs(x) ** p2], axis=-1)
    raise ValueError(spec)


def fsm_fit_r1(B_t, g_skew, basis, rng):
    """Closed-form FSM fit at scalar B_t (Eq. 4-5); returns score_fn: x -> scalar score."""
    Bj = B_t + SIGMA * rng.standard_normal(J_PROP)
    Bj = np.maximum(Bj, B_CLIP)
    u = -(Bj - B_t) / SIGMA ** 2                                   # (J,)
    x = sample_gandk(Bj, g_skew, N_SIM, rng)                       # (J, n)
    Phi = basis(x).reshape(J_PROP * N_SIM, -1)                     # (J*n, d)
    U = np.repeat(u, N_SIM)                                        # (J*n,)
    A = Phi.T @ Phi + RIDGE * np.eye(Phi.shape[1])
    W = -np.linalg.solve(A, Phi.T @ U)                             # (d,)
    return lambda xx: basis(xx) @ W


def fsm_mle_r1(g_skew, basis, rng):
    """One replication: fresh data at truth, RM ascent from B0, PR average (2nd half)."""
    x_obs = sample_gandk(np.array([B_STAR]), g_skew, N_OBS, rng)[0]
    B = B0
    traj = []
    for _ in range(T_STEPS):
        score_fn = fsm_fit_r1(B, g_skew, basis, rng)
        B = max(B + ETA * float(np.mean(score_fn(x_obs))), B_CLIP)
        traj.append(B)
    return float(np.mean(traj[T_STEPS // 2:]))


def run_experiment(g_skew, spec):
    ests = []
    for r in range(R_REPS):
        rng = np.random.default_rng(SEED + 100 * r)
        ests.append(fsm_mle_r1(g_skew, get_basis_r1(spec), rng))
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
    print("R1  g-and-k scale parameter: parity blindness of [1,x] and its PATP repair")
    print("=" * 78)
    print(f"  truth B={B_STAR}, start B0={B0}, A=0, k={K_GK}, N={N_OBS}, R={R_REPS} reps, "
          f"alpha={ALPHA_PATP} (p2={patp_power(2, ALPHA_PATP):.3f})")

    e_lin = run_experiment(0.0, "linear")
    e_patp = run_experiment(0.0, "patp")
    e_mix = run_experiment(0.5, "patp")
    e_even = run_experiment(0.5, "even")

    err_lin = float(np.mean(np.abs(e_lin - B_STAR)))
    err_patp = float(np.mean(np.abs(e_patp - B_STAR)))
    err_mix = float(np.mean(np.abs(e_mix - B_STAR)))
    err_even = float(np.mean(np.abs(e_even - B_STAR)))

    print(f"\n  g=0.0  linear [1,x]        B_hat = {e_lin.mean():.3f} +/- {e_lin.std():.3f}")
    print(f"  g=0.0  patp  [1,x,|x|^p2]  B_hat = {e_patp.mean():.3f} +/- {e_patp.std():.3f}")
    print(f"  g=0.5  mixed [1,x,|x|^p2]  B_hat = {e_mix.mean():.3f} +/- {e_mix.std():.3f}")
    print(f"  g=0.5  even  [1,|x|^p2]    B_hat = {e_even.mean():.3f} +/- {e_even.std():.3f}")

    ok = []
    ok.append(check("R1a-blind  (linear fails on symmetric scale)",
                    err_lin > THR_BLIND, f"mean |B_hat-2| = {err_lin:.4f} > {THR_BLIND}"))
    ok.append(check("R1a-repair (even PATP term recovers B)",
                    err_patp < THR_REPAIR, f"mean |B_hat-2| = {err_patp:.4f} < {THR_REPAIR}"))
    ok.append(check("R1b-mixed  (mixed basis under skew g=0.5)",
                    err_mix < THR_MIXED, f"mean |B_hat-2| = {err_mix:.4f} < {THR_MIXED}"))
    print(f"  [info] R1b even-only basis under skew: mean |B_hat-2| = {err_even:.4f} (ungated)")

    all_pass = all(ok)
    elapsed = time.time() - t0
    print("\n" + "=" * 78)
    print(f"  => R1-GANDK {'PASS' if all_pass else 'FAIL'}   (runtime {elapsed:.1f}s)")

    with open(os.path.join(here, "results", "r1_gandk.json"), "w") as f:
        json.dump({
            "seed": SEED, "B_star": B_STAR, "B0": B0, "k": K_GK, "N_obs": N_OBS,
            "R_reps": R_REPS,
            "hyperparams": {"sigma": SIGMA, "J": J_PROP, "n": N_SIM, "T": T_STEPS,
                            "eta": ETA, "ridge": RIDGE, "alpha_patp": ALPHA_PATP},
            "g0_linear": {"mean": float(e_lin.mean()), "sd": float(e_lin.std()),
                          "abs_err": err_lin, "estimates": e_lin.tolist()},
            "g0_patp": {"mean": float(e_patp.mean()), "sd": float(e_patp.std()),
                        "abs_err": err_patp, "estimates": e_patp.tolist()},
            "g05_mixed": {"mean": float(e_mix.mean()), "sd": float(e_mix.std()),
                          "abs_err": err_mix, "estimates": e_mix.tolist()},
            "g05_even_only_info": {"mean": float(e_even.mean()), "sd": float(e_even.std()),
                                   "abs_err": err_even, "estimates": e_even.tolist()},
            "R1a_blind_pass": bool(ok[0]), "R1a_repair_pass": bool(ok[1]),
            "R1b_mixed_pass": bool(ok[2]), "all_pass": bool(all_pass),
            "runtime_s": elapsed,
        }, f, indent=2)
    print("  [written] results/r1_gandk.json")

    sys.exit(0 if all_pass else 1)
