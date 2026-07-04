"""
sgld_calibration.py -- prototype gates for the planned "score-surrogate posterior"
companion paper (SGLD calibration of the generalized posterior built from the
FSM/Kunchenko score surrogate).

Theory under test (cf. paper/sections/efficiency.tex, Theorem 1):
  psi(x; theta) = W*^T phi_tilde(x),  W* = G^{-1} b,  b_j = Cov_theta(phi_j(X), s),
  G = Cov_theta(phi), is the L2(p_theta) projection of the Fisher score s onto
  span{phi_tilde}.  With g = b^T G^{-1} b / I (captured Fisher fraction), psi is an
  unbiased estimating function (E_theta[psi] = 0 for all theta); differentiating
  that identity gives
      H := -E_theta[d psi/d theta] = <psi, s> = ||proj s||^2 = I*g,
      J := Var_theta(psi)          =            ||proj s||^2 = I*g,
  i.e. the information-matrix equality H = J holds for the projected score.
  Consequence: the generalized (Gibbs) posterior whose log-gradient is
  sum_i psi(y_i; theta), sampled by SGLD at learning rate w = 1, has asymptotic
  variance (N*I*g)^{-1} -- exactly the frequentist sandwich variance J/(N*H^2) of
  the corresponding M-estimator -- hence asymptotically correct frequentist
  coverage, with posterior sd inflated by g^{-1/2} relative to the exact-likelihood
  posterior sd 1/sqrt(N*I).

Gates (each prints PASS/FAIL; exit code 1 if any fails):
  B1  H = J = I*g by simulation, three models with analytic anchors:
      (a) Gaussian location x~N(theta,1),   phi={x}:    I = 1,          g = 1
      (b) Laplace  location x~Lap(theta,1), phi={x}:    I = 1,          g = 1/2
      (c) Gaussian scale    x~N(0,v),       phi={x^2}:  I = 1/(2 v^2),  g = 1
      W* is fit from a large sample at theta_0; J = Var(psi) on a fresh sample;
      H by central finite difference with common random numbers (psi frozen at
      theta_0, sampling distribution shifted to theta_0 +/- h; by the
      unbiasedness identity this equals -E[d psi/d theta]).
  B2  Frequentist coverage of the surrogate posterior (headline): Laplace
      location, N = 200 obs, R replications = R parallel SGLD chains, flat prior,
      psi_hat re-fit at every iterate by the FSM recipe (run.py fsm_fit, Eq. 4-5,
      specialized to scalar theta and vectorized over chains).  Checks: empirical
      90% CI coverage in [0.84, 0.96]; mean posterior sd within 15% of
      1/sqrt(N*I*g) = sqrt(2/200) ~ 0.100; width inflation vs the exact-likelihood
      asymptotic sd 1/sqrt(N*I) = sqrt(1/200) within 15% of sqrt(2).
  B3  Control (g = 1, no inflation): same pipeline on Gaussian location, N = 200:
      coverage in [0.84, 0.96], mean posterior sd within 15% of sqrt(1/200).

Implementation note (W-fit noise): with features [1, x] the fitted intercept
carries the term mean_j[(theta_j - theta_k)/sigma^2], a zero-mean noise of sd
1/(sigma*sqrt(m)) per refit.  Injected into the drift it would exceed the
Langevin noise sqrt(eta) and visibly inflate the chain (measured ~3.4x variance
at m = 50).  Drawing the proposals as antithetic pairs theta_k +/- delta --
marginally still N(theta_k, sigma^2) -- cancels that term exactly at no extra
simulation cost; this is the same class of legitimate variance reduction of the
W-fit as increasing m*n or averaging W over refits.

Pure numpy, SEED = 2026, runtime well under 5 min on a laptop CPU.
Run:  python sgld_calibration.py     (writes results/sgld_calibration.json)
"""
import json
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------------
# Hyperparameters (all final; see header for rationale)
# ---------------------------------------------------------------------------------
SEED = 2026

# -- Gate B1 (information-matrix equality) --
N_FIT = 2_000_000     # draws at theta_0 to fit W* = b/G (and measure I, g)
N_CHECK = 1_000_000   # fresh draws for J = Var(psi) and for each side of the FD
H_FD = 0.1            # finite-difference step (E[psi] is exactly linear in theta
                      # for all three models, so h only sets the CRN noise scale)
V0 = 4.0              # theta_0 for the Gaussian-scale family (v = variance)
TOL_B1 = 0.05         # gate: |H - I*g|/(I*g) < 5%  and  |J - I*g|/(I*g) < 5%

# -- Gates B2/B3 (SGLD surrogate posterior) --
N_OBS = 200           # observations per replication
R_REPS = 200          # replications = parallel SGLD chains (band below quoted
                      # for R = 100; R = 200 only tightens the binomial noise)
THETA_STAR = 1.0      # true location parameter
M_PROP = 50           # proposals per FSM refit (as 25 antithetic pairs)
N_SIM = 10            # simulator draws per proposal
SIGMA_PROP = 0.3      # proposal sd; smears the fitted slope to 1/(Var_noise+sigma^2)
                      # -> known O(sigma^2) upward bias in posterior var (~4-9%)
LAMBDA_RIDGE = 1e-8   # same ridge as run.py fsm_fit
ETA = 1e-3            # SGLD step; stationary-var inflation of the unadjusted
                      # Langevin chain is 1/(1 - eta*N*I*g/4) ~ +2.6% (B2) / +4.8% (B3)
T_STEPS = 2500        # SGLD iterations per chain
BURN_IN = 500         # discarded; relaxation time is ~20 steps at this eta
CI_LO, CI_HI = 0.05, 0.95           # central 90% credible interval
COV_BAND = (0.84, 0.96)             # binomial noise band around 0.90 (R = 100)
TOL_SD = 0.15                       # posterior-sd and width-ratio tolerance

gates = {}   # gate name -> bool
summary = [] # printed at the end


def gate(name, ok, detail):
    gates[name] = bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# ==================================================================================
# Gate B1 -- information-matrix equality  H = J = I*g
# ==================================================================================
# Each model: draw(theta, z) maps base noise z (CRN across theta) to x ~ p_theta;
# the analytic score is used ONLY to build W* (b = Cov(phi, s) is its definition;
# the moment-free IBP route for b is gate T5, e5_currency_identity.py).
B1_MODELS = [
    dict(name="gauss_location", theta0=1.0, Ig=1.0, g=1.0,
         base=lambda rng, n: rng.standard_normal(n),
         draw=lambda th, z: th + z,
         feat=lambda x: x,
         score=lambda x, th: x - th),
    dict(name="laplace_location", theta0=2.0, Ig=0.5, g=0.5,
         base=lambda rng, n: rng.laplace(0.0, 1.0, n),
         draw=lambda th, z: th + z,
         feat=lambda x: x,
         score=lambda x, th: np.sign(x - th)),
    dict(name="gauss_scale", theta0=V0, Ig=1.0 / (2 * V0**2), g=1.0,
         base=lambda rng, n: rng.standard_normal(n),
         draw=lambda th, z: np.sqrt(th) * z,
         feat=lambda x: x**2,
         score=lambda x, th: -1.0 / (2 * th) + x**2 / (2 * th**2)),
]


def b1_case(model, rng):
    th0 = model["theta0"]
    # (1) fit W* at theta_0 (population projection, estimated at N_FIT draws)
    x = model["draw"](th0, model["base"](rng, N_FIT))
    p, s = model["feat"](x), model["score"](x, th0)
    c, G = p.mean(), p.var()
    b = np.mean((p - c) * s)
    I_hat = np.mean(s * s)
    W = b / G
    psi = lambda xx: W * (model["feat"](xx) - c)       # frozen at theta_0
    # (2) J = Var(psi) on a fresh sample at theta_0
    J = psi(model["draw"](th0, model["base"](rng, N_CHECK))).var()
    # (3) H = -E[d psi/d theta] = d/dtheta E_theta[psi_frozen], central FD + CRN
    z = model["base"](rng, N_CHECK)
    H = (psi(model["draw"](th0 + H_FD, z)).mean()
         - psi(model["draw"](th0 - H_FD, z)).mean()) / (2 * H_FD)
    Ig = model["Ig"]
    return dict(H=float(H), J=float(J), Ig_analytic=float(Ig),
                relerr_H=float(abs(H - Ig) / Ig), relerr_J=float(abs(J - Ig) / Ig),
                g_hat=float(b * b / (G * I_hat)), g_analytic=model["g"],
                I_hat=float(I_hat))


# ==================================================================================
# Gates B2/B3 -- SGLD over the surrogate posterior, R chains vectorized
# ==================================================================================
def run_surrogate_posterior(noise, rng):
    """R_REPS replications, one SGLD chain each, updated jointly as an (R,) vector.
    noise in {'laplace','gauss'}: observation & simulator noise (location family).
    Returns (coverage, mean posterior sd)."""
    lap = (noise == "laplace")
    draw = (lambda size: rng.laplace(0.0, 1.0, size)) if lap \
        else (lambda size: rng.standard_normal(size))
    Y = THETA_STAR + draw((R_REPS, N_OBS))             # fresh data per replication
    Ysum = Y.sum(axis=1)
    theta = Y.mean(axis=1).copy()                      # start near the mode
    half = M_PROP // 2
    mn = M_PROP * N_SIM
    kept = np.empty((R_REPS, T_STEPS - BURN_IN))
    for k in range(T_STEPS):
        # --- FSM refit (mirrors run.py fsm_fit Eq. 4-5, scalar theta, R chains).
        # Antithetic pairs theta_k +/- delta ~ N(theta_k, sigma^2) marginally;
        # cancels the intercept noise of the W-fit (see header note).
        delta = SIGMA_PROP * rng.standard_normal((R_REPS, half))
        thp = np.concatenate([theta[:, None] + delta,
                              theta[:, None] - delta], axis=1)      # (R, m)
        X = thp[:, :, None] + draw((R_REPS, M_PROP, N_SIM))         # simulator
        t = (thp - theta[:, None]) / SIGMA_PROP**2                  # -grad log q
        # ridge normal equations for features [1, x]:
        #   (Phi^T Phi + lambda I) W = -Phi^T grad_log_q = Phi^T t, per chain
        Sx = X.sum(axis=(1, 2))
        Sxx = (X * X).sum(axis=(1, 2))
        St = N_SIM * t.sum(axis=1)                     # = 0 exactly (antithetic)
        Sxt = (X.sum(axis=2) * t).sum(axis=1)
        a11, a22 = mn + LAMBDA_RIDGE, Sxx + LAMBDA_RIDGE
        det = a11 * a22 - Sx * Sx
        w0 = (a22 * St - Sx * Sxt) / det
        w1 = (a11 * Sxt - Sx * St) / det
        # --- SGLD step (flat prior): drift = sum_i psi_hat(y_i; theta_k)
        drift = N_OBS * w0 + w1 * Ysum
        theta = theta + 0.5 * ETA * drift + np.sqrt(ETA) * rng.standard_normal(R_REPS)
        if k >= BURN_IN:
            kept[:, k - BURN_IN] = theta
    lo = np.quantile(kept, CI_LO, axis=1)
    hi = np.quantile(kept, CI_HI, axis=1)
    coverage = float(np.mean((lo <= THETA_STAR) & (THETA_STAR <= hi)))
    mean_sd = float(kept.std(axis=1).mean())
    return coverage, mean_sd


# ==================================================================================
if __name__ == "__main__":
    t_start = time.time()
    rng = np.random.default_rng(SEED)
    here = os.path.dirname(os.path.abspath(__file__))

    print("=" * 78)
    print("GATE B1  information-matrix equality  H = J = I*g   (three analytic anchors)")
    print("=" * 78)
    b1 = {}
    for model in B1_MODELS:
        r = b1_case(model, rng)
        b1[model["name"]] = r
        ok = r["relerr_H"] < TOL_B1 and r["relerr_J"] < TOL_B1
        r["pass"] = bool(ok)
        gate(f"B1 {model['name']}", ok,
             f"H={r['H']:.5f}  J={r['J']:.5f}  I*g={r['Ig_analytic']:.5f}  "
             f"relerr(H)={r['relerr_H']:.4f}  relerr(J)={r['relerr_J']:.4f}  "
             f"(tol {TOL_B1}); g_hat={r['g_hat']:.4f} vs g={r['g_analytic']}")

    sd_exact = 1.0 / np.sqrt(N_OBS)                    # 1/sqrt(N*I), I = 1 (both)
    print("\n" + "=" * 78)
    print(f"GATE B2  surrogate-posterior coverage, Laplace location "
          f"(g=1/2), N={N_OBS}, R={R_REPS}")
    print("=" * 78)
    cov2, sd2 = run_surrogate_posterior("laplace", rng)
    sd2_theory = np.sqrt(2.0 / N_OBS)                  # 1/sqrt(N*I*g), g = 1/2
    ratio2 = sd2 / sd_exact                            # width inflation vs exact-lik.
    gate("B2 coverage", COV_BAND[0] <= cov2 <= COV_BAND[1],
         f"90% CI empirical coverage = {cov2:.3f}  (band {COV_BAND})")
    gate("B2 posterior sd", abs(sd2 / sd2_theory - 1) < TOL_SD,
         f"mean sd = {sd2:.4f}  vs  1/sqrt(N*I*g) = {sd2_theory:.4f}  "
         f"(relerr {abs(sd2 / sd2_theory - 1):.3f}, tol {TOL_SD})")
    gate("B2 width inflation", abs(ratio2 / np.sqrt(2) - 1) < TOL_SD,
         f"sd/(1/sqrt(N*I)) = {ratio2:.3f}  vs  sqrt(2) = {np.sqrt(2):.3f}  "
         f"(relerr {abs(ratio2 / np.sqrt(2) - 1):.3f}, tol {TOL_SD})")

    print("\n" + "=" * 78)
    print(f"GATE B3  control, Gaussian location (g=1, no inflation), "
          f"N={N_OBS}, R={R_REPS}")
    print("=" * 78)
    cov3, sd3 = run_surrogate_posterior("gauss", rng)
    ratio3 = sd3 / sd_exact
    gate("B3 coverage", COV_BAND[0] <= cov3 <= COV_BAND[1],
         f"90% CI empirical coverage = {cov3:.3f}  (band {COV_BAND})")
    gate("B3 posterior sd", abs(sd3 / sd_exact - 1) < TOL_SD,
         f"mean sd = {sd3:.4f}  vs  1/sqrt(N*I) = {sd_exact:.4f}  "
         f"(relerr {abs(sd3 / sd_exact - 1):.3f}, tol {TOL_SD})")

    runtime = time.time() - t_start

    # ------------------------------------------------------------------ summary
    print("\n" + "=" * 78)
    print("SUMMARY  (all measured numbers)")
    print("=" * 78)
    print(f"  {'B1 model':18s} {'H':>9s} {'J':>9s} {'I*g':>9s} {'g_hat':>8s} {'g':>6s}")
    for name, r in b1.items():
        print(f"  {name:18s} {r['H']:9.5f} {r['J']:9.5f} {r['Ig_analytic']:9.5f} "
              f"{r['g_hat']:8.4f} {r['g_analytic']:6.2f}")
    print(f"\n  {'gate':4s} {'coverage':>9s} {'mean sd':>9s} {'theory sd':>10s} "
          f"{'width ratio':>12s} {'target':>8s}")
    print(f"  {'B2':4s} {cov2:9.3f} {sd2:9.4f} {sd2_theory:10.4f} "
          f"{ratio2:12.3f} {np.sqrt(2):8.3f}")
    print(f"  {'B3':4s} {cov3:9.3f} {sd3:9.4f} {sd_exact:10.4f} "
          f"{ratio3:12.3f} {1.0:8.3f}")
    print(f"\n  runtime: {runtime:.1f} s")

    out = dict(
        seed=SEED, runtime_seconds=round(runtime, 2),
        hyperparams=dict(N_FIT=N_FIT, N_CHECK=N_CHECK, H_FD=H_FD, V0=V0,
                         TOL_B1=TOL_B1, N_OBS=N_OBS, R_REPS=R_REPS,
                         THETA_STAR=THETA_STAR, M_PROP=M_PROP, N_SIM=N_SIM,
                         SIGMA_PROP=SIGMA_PROP, LAMBDA_RIDGE=LAMBDA_RIDGE,
                         ETA=ETA, T_STEPS=T_STEPS, BURN_IN=BURN_IN,
                         CI=[CI_LO, CI_HI], COV_BAND=list(COV_BAND),
                         TOL_SD=TOL_SD, antithetic_proposals=True),
        B1=b1,
        B2=dict(model="laplace_location", coverage=cov2, mean_post_sd=sd2,
                theory_sd=float(sd2_theory),
                sd_relerr=float(abs(sd2 / sd2_theory - 1)),
                width_ratio=float(ratio2), width_ratio_target=float(np.sqrt(2)),
                width_relerr=float(abs(ratio2 / np.sqrt(2) - 1)),
                passes={k: v for k, v in gates.items() if k.startswith("B2")}),
        B3=dict(model="gauss_location", coverage=cov3, mean_post_sd=sd3,
                theory_sd=float(sd_exact),
                sd_relerr=float(abs(sd3 / sd_exact - 1)),
                width_ratio=float(ratio3),
                passes={k: v for k, v in gates.items() if k.startswith("B3")}),
        gates=gates, all_pass=all(gates.values()),
    )
    os.makedirs(os.path.join(here, "results"), exist_ok=True)
    with open(os.path.join(here, "results", "sgld_calibration.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("  [written] results/sgld_calibration.json")

    print("\n" + "=" * 78)
    if all(gates.values()):
        print("ALL PASS  (SGLD calibration of the surrogate posterior verified)")
    else:
        failed = [k for k, v in gates.items() if not v]
        print(f"FAIL: {failed}")
        sys.exit(1)
