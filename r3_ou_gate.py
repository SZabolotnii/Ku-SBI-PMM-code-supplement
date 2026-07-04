r"""
r3_ou_gate.py -- Gate R3: FSM score-surrogate estimation of an AR(1)/OU
autoregression parameter, benchmarked against the EXACT conditional MLE.

Model: Gaussian AR(1),  x_{t+1} | x_t ~ N(a x_t, 1),  t = 0..L-1 (L transitions,
so one series has L+1 points x_0..x_L), x_0 ~ N(0, 1/(1-a^2)) (stationary).
Truth a = 0.6, L = 200.  One "observation" is one whole series.

The 2-dim sufficient summary of a series is z(x) = (m1, m2),
    m1 = (1/L) sum_{t=0}^{L-1} x_t x_{t+1},   m2 = (1/L) sum_{t=0}^{L-1} x_t^2,
whose ratio IS the exact conditional MLE (the anchor):
    a_hat_MLE = sum(x_t x_{t+1}) / sum(x_t^2) = m1 / m2.
Per-series Fisher information I ~ L * E[x_t^2] = L/(1-a^2); theoretical sd of
the MLE is 1/sqrt(I).

FSM fit (closed form, Eq. 4-5 of run.py's fsm_fit, specialised to a scalar
parameter and z-features instead of point features): draw J proposals
a_j = a_hat + sigma*N(0,1), simulate n series per proposal, extract z_j =
(m1_j, m2_j); the IBP target u_j = -(a_j - a_hat)/sigma^2 is replicated over
the n series sharing that proposal; solve
    (Phi^T Phi + lambda I) W = -Phi^T u
for the feature matrix Phi (rows = simulated series, columns = the basis).
The fitted score at the data is phi(z_data)^T W.  Wrapped in a Robbins-Monro
loop with Polyak-Ruppert averaging over the second half of T iterations
(run.py's fsm_mle), fully vectorised over the R=100 independent replications
via numpy's batched linalg.solve (one (d,d) system per replication, solved
together as a (R,d,d) batch).

Two bases contrast information content:
  R3-full  [1, m1, m2]  spans the sufficient statistic -> should track the MLE.
  R3-blind [1, xbar]    series mean carries ~no information about a (AR(1) is
                        mean-zero at every t) -> should fail to move off a_0.

Gates (PRE-REGISTERED -- do not alter):
  R3a consistency:  |mean_r(a_hat_full) - 0.6| < 0.03
  R3b efficiency:   Var_r(a_hat_full) / Var_r(a_hat_MLE) <= 1.6
  R3c blindness:    mean_r(|a_hat_blind - 0.6|) > 0.15

Pure numpy, SEED = 2026, deterministic (single rng consumed sequentially:
data -> full-basis RM -> blind-basis RM).  Runtime well under the 4 min budget.
Run:  python r3_ou_gate.py     (writes results/r3_ou.json)
"""
import json
import os
import sys
import time

import numpy as np

# ----------------------------------------------------------------------------------
# Fixed experiment design
# ----------------------------------------------------------------------------------
SEED = 2026
A_TRUE = 0.6
L = 200            # transitions t=0..L-1  (series has L+1 points x_0..x_L)
R = 100            # independent replications (data series)
A0 = 0.3           # FSM start point

# Budget knobs (tuned only for runtime/convergence, within the pre-registered ranges)
SIGMA = 0.08       # proposal sd,  in [0.05, 0.15]
J = 30             # proposals/iter,  in [20, 40]
N_SIM = 2          # series per proposal,  in [1, 3]
T = 100            # RM iterations,  in [60, 150]
ETA = 0.0025       # RM step (near the Newton step 1/I ~ 1/312 for fast, stable descent)
LAMBDA_RIDGE = 1e-8
A_CLIP = 0.97      # keep proposals / iterates inside the AR(1) stability region |a|<1

# Pre-registered thresholds (do not alter)
THR_R3A = 0.03
THR_R3B = 1.6
THR_R3C = 0.15


def simulate_series(a, n_sim, rng):
    """a: array of AR(1) coefficients, any shape S. Returns X, shape S+(n_sim,L+1),
    points x_0..x_L of n_sim independent series per entry of a (stationary x_0)."""
    a_c = np.clip(a, -A_CLIP, A_CLIP)
    shape = a.shape + (n_sim,)
    x0_sd = np.sqrt(1.0 / (1.0 - a_c**2))
    X = np.empty(shape + (L + 1,))
    X[..., 0] = x0_sd[..., None] * rng.standard_normal(shape)
    eps = rng.standard_normal(shape + (L,))
    for t in range(L):
        X[..., t + 1] = a_c[..., None] * X[..., t] + eps[..., t]
    return X


def summarize(X):
    """Sufficient summary (m1, m2) from series X, last axis = L+1 points."""
    x_t, x_t1 = X[..., :-1], X[..., 1:]              # t=0..L-1
    m1 = (x_t * x_t1).mean(axis=-1)
    m2 = (x_t * x_t).mean(axis=-1)
    return m1, m2


def fsm_mle_ou(basis, phi_data, rng):
    """Robbins-Monro FSM-MLE of a, vectorised over R replications (batched linalg).
    basis in {'full','blind'}; phi_data: (R,d) data features for that basis."""
    d = phi_data.shape[1]
    eye_d = np.eye(d)[None]
    a_hat = np.full(R, A0)
    traj = np.empty((T, R))
    for k in range(T):
        props = a_hat[:, None] + SIGMA * rng.standard_normal((R, J))     # (R,J)
        X = simulate_series(props, N_SIM, rng)                          # (R,J,N_SIM,L+1)
        m1, m2 = summarize(X)
        if basis == "full":
            Phi = np.stack([np.ones_like(m1), m1, m2], axis=-1)
        else:
            xbar = X.mean(axis=-1)
            Phi = np.stack([np.ones_like(xbar), xbar], axis=-1)
        Phi = Phi.reshape(R, J * N_SIM, d)
        u = -(props - a_hat[:, None]) / SIGMA**2                        # (R,J) grad_a log q
        u_rep = np.repeat(u[:, :, None], N_SIM, axis=2).reshape(R, J * N_SIM, 1)
        A = Phi.transpose(0, 2, 1) @ Phi + LAMBDA_RIDGE * eye_d          # (R,d,d)
        rhs = -(Phi.transpose(0, 2, 1) @ u_rep)                         # (R,d,1)
        W = np.linalg.solve(A, rhs)                                     # (R,d,1)
        score = (phi_data[:, None, :] @ W).reshape(R)                   # (R,)
        a_hat = np.clip(a_hat + ETA * score, -A_CLIP, A_CLIP)
        traj[k] = a_hat
    return traj[T // 2:].mean(axis=0)                                   # Polyak-Ruppert


# ======================================================================================
if __name__ == "__main__":
    t_start = time.time()
    here = os.path.dirname(os.path.abspath(__file__))
    rng = np.random.default_rng(SEED)                # single rng, consumed sequentially

    print("=" * 78)
    print(f"GATE R3-OU  --  FSM score-surrogate vs exact conditional MLE, AR(1)/OU")
    print(f"  a_true={A_TRUE}, L={L} transitions, R={R} replications, a_0={A0}")
    print("=" * 78)

    # --- data: R independent series at the truth, and the exact conditional MLE -------
    a_true_vec = np.full(R, A_TRUE)
    Xd = simulate_series(a_true_vec, 1, rng)[:, 0, :]              # (R, L+1)
    m1_data, m2_data = summarize(Xd[:, None, :])
    m1_data, m2_data = m1_data[:, 0], m2_data[:, 0]
    xbar_data = Xd.mean(axis=-1)
    a_mle = m1_data / m2_data

    I_theory = L / (1.0 - A_TRUE**2)
    sd_theory = 1.0 / np.sqrt(I_theory)
    print(f"\n  [info] Fisher info I = L/(1-a^2) = {I_theory:.2f}, "
          f"theoretical MLE sd 1/sqrt(I) = {sd_theory:.4f}")
    print(f"  [info] exact MLE: mean={a_mle.mean():.4f}, sd={a_mle.std():.4f} "
          f"(empirical vs theory: {a_mle.std():.4f} vs {sd_theory:.4f})")

    # --- FSM-MLE, basis R3-full [1, m1, m2] --------------------------------------------
    phi_full = np.stack([np.ones(R), m1_data, m2_data], axis=-1)
    a_fsm_full = fsm_mle_ou("full", phi_full, rng)
    print(f"\n  [info] FSM-full  [1,m1,m2]: mean={a_fsm_full.mean():.4f}, "
          f"sd={a_fsm_full.std():.4f}")

    # --- FSM-MLE, basis R3-blind [1, xbar] ----------------------------------------------
    phi_blind = np.stack([np.ones(R), xbar_data], axis=-1)
    a_fsm_blind = fsm_mle_ou("blind", phi_blind, rng)
    print(f"  [info] FSM-blind [1,xbar] : mean={a_fsm_blind.mean():.4f}, "
          f"sd={a_fsm_blind.std():.4f}")

    # --- criteria (PRE-REGISTERED; report honestly, do not tune to green) --------------
    r3a_measured = float(abs(a_fsm_full.mean() - A_TRUE))
    r3a_ok = r3a_measured < THR_R3A

    r3b_measured = float(a_fsm_full.var() / a_mle.var())
    r3b_ok = r3b_measured <= THR_R3B

    r3c_measured = float(np.abs(a_fsm_blind - A_TRUE).mean())
    r3c_ok = r3c_measured > THR_R3C

    runtime = time.time() - t_start

    print("\n" + "=" * 78)
    print("CRITERIA")
    print("=" * 78)
    print(f"  [{'PASS' if r3a_ok else 'FAIL'}] R3a consistency: "
          f"|mean(a_hat_full) - {A_TRUE}| = {r3a_measured:.4f}  (thr < {THR_R3A})")
    print(f"  [{'PASS' if r3b_ok else 'FAIL'}] R3b efficiency:  "
          f"Var(a_hat_full)/Var(a_hat_MLE) = {r3b_measured:.4f}  (thr <= {THR_R3B})")
    print(f"  [{'PASS' if r3c_ok else 'FAIL'}] R3c blindness:   "
          f"mean(|a_hat_blind - {A_TRUE}|) = {r3c_measured:.4f}  (thr > {THR_R3C})")

    ok = r3a_ok and r3b_ok and r3c_ok
    print(f"\n  runtime: {runtime:.1f} s")
    print("\n" + "=" * 78)
    print(f"  => R3-OU {'PASS' if ok else 'FAIL'}  (full basis tracks the exact conditional "
          f"MLE; mean-only blind basis fails to move off a_0)")

    out = dict(
        seed=SEED, a_true=A_TRUE, L=L, R=R, a0=A0,
        hyperparams=dict(sigma=SIGMA, J=J, n_sim=N_SIM, T=T, eta=ETA,
                         lambda_ridge=LAMBDA_RIDGE, a_clip=A_CLIP),
        exact_mle=dict(mean=float(a_mle.mean()), var=float(a_mle.var()),
                       sd=float(a_mle.std())),
        fisher_info_theory=float(I_theory), mle_sd_theory=float(sd_theory),
        fsm_full=dict(mean=float(a_fsm_full.mean()), var=float(a_fsm_full.var()),
                     sd=float(a_fsm_full.std())),
        fsm_blind=dict(mean=float(a_fsm_blind.mean()), var=float(a_fsm_blind.var()),
                      sd=float(a_fsm_blind.std())),
        criteria=dict(
            R3a=dict(measured=r3a_measured, threshold=THR_R3A, op="<", passed=r3a_ok),
            R3b=dict(measured=r3b_measured, threshold=THR_R3B, op="<=", passed=r3b_ok),
            R3c=dict(measured=r3c_measured, threshold=THR_R3C, op=">", passed=r3c_ok),
        ),
        all_pass=bool(ok), runtime_seconds=round(runtime, 2),
    )
    os.makedirs(os.path.join(here, "results"), exist_ok=True)
    with open(os.path.join(here, "results", "r3_ou.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("  [written] results/r3_ou.json")

    sys.exit(0 if ok else 1)
