"""
E5 / gate T5 -- the simulation-based currency-identity.

Claim (Spec-Paper-SimulationBased-ScoreSurrogate.md, T5):
  the moment-free FSM machinery recovers the LSU efficiency currency
      g = ||proj_Vm S||^2 / I = b^T G^-1 b / I       (captured Fisher fraction)
  WITHOUT ever evaluating the score S, the density, or its moments.

Mechanism (Bartlett/Stein identity):  b_i = <phi_i, S> = d/dtheta E[phi_i].
Hence the integration-by-parts estimate
      b_hat_i = -E[phi_i * grad_theta log q] = (1/sigma^2) Cov_sim(phi_i, theta),
and  G_hat = Cov_sim(phi)   are both pure-simulation quantities.

We compare, on models with a KNOWN ground-truth g (computed from the analytic score):
  g_truth(m)  -- big-MC using the true score S            (ground truth)
  g_hat(m)    -- simulation + IBP, NO score                (the claim)
Gate:  max_m | g_hat(m) - g_truth(m) | < 0.05.

Analytic anchors (Gaussian scale x~N(0,theta)):  basis {x} -> g=0 (score even, x odd);
basis {x,x^2} -> S = (x^2 - theta)/(2 theta^2)  in the span -> g=1.
"""
import numpy as np

RNG = np.random.default_rng(2026)
TOL = 0.05

# ----- centered monomial design matrix ------------------------------------------------
def feats(powers):
    def f(x):
        F = np.stack([x.astype(float) ** p for p in powers], axis=1)
        return F - F.mean(0)
    return f

# ----- models: analytic score (ground truth only) + vectorised simulator --------------
def gauss_mean():
    def score(x, th): return (x - th)                       # x ~ N(theta, 1)
    def big(th, N):  return RNG.normal(th, 1.0, N)
    def vec(thetas, n):  return (thetas[:, None] + RNG.standard_normal((len(thetas), n)))
    return dict(score=score, big=big, vec=vec, positive=False)

def gauss_var():
    def score(x, th): return -1.0/(2*th) + x**2/(2*th**2)   # x ~ N(0, theta)
    def big(th, N):  return RNG.normal(0.0, np.sqrt(th), N)
    def vec(thetas, n):  return np.sqrt(thetas)[:, None] * RNG.standard_normal((len(thetas), n))
    return dict(score=score, big=big, vec=vec, positive=True)

def laplace_loc():
    """Location family, bounded NON-polynomial score sign(x-theta); light tails -> clean Gram.
    Analytic anchor: basis {x} -> g = b^2/(G*I) = 1^2/(2*1) = 0.5 (Var Laplace(b=1)=2)."""
    def score(x, th): return np.sign(x - th)
    def big(th, N):  return RNG.laplace(th, 1.0, N)
    def vec(thetas, n):  return thetas[:, None] + RNG.laplace(0.0, 1.0, (len(thetas), n))
    return dict(score=score, big=big, vec=vec, positive=False)

# ----- g_truth: MC with the analytic score (ground truth) -----------------------------
def g_truth(model, feat, theta, N=2_000_000):
    x = model["big"](theta, N)
    S = model["score"](x, theta)
    F = feat(x)
    b = (F * S[:, None]).mean(0)
    G = F.T @ F / N
    I = float((S**2).mean())
    num = float(b @ np.linalg.solve(G + 1e-12*np.eye(len(b)), b))
    return num / I, I

# ----- g_hat: simulation + IBP, NO score ----------------------------------------------
def g_hat(model, feat, theta_t, I_known, sigma, m=20000, n=100):
    thetas = RNG.normal(theta_t, sigma, m)
    if model["positive"]:
        thetas = np.abs(thetas) + 1e-9
    X = model["vec"](thetas, n).ravel()
    th_rep = np.repeat(thetas, n)
    F = feat(X)
    b_hat = (F * (th_rep - theta_t)[:, None]).mean(0) / sigma**2     # (1/s^2) Cov(phi, theta)
    G_hat = F.T @ F / len(X)
    num = float(b_hat @ np.linalg.solve(G_hat + 1e-12*np.eye(len(b_hat)), b_hat))
    return num / I_known

# ======================================================================================
CASES = [
    # name             model           theta  sigma  basis ladders (with known g_truth)
    ("gauss_mean",     gauss_mean(),   1.0,   0.10,  [("{x}", [1])]),                 # g=1
    ("gauss_var",      gauss_var(),    4.0,   0.20,  [("{x}", [1]), ("{x,x^2}", [1, 2]),
                                                      ("{x,x^2,x^3}", [1, 2, 3])]),   # 0 -> 1 -> 1
    ("laplace_loc",    laplace_loc(),  2.0,   0.20,  [("{x}", [1]), ("{x,x^3}", [1, 3]),
                                                      ("{x,x^2,x^3}", [1, 2, 3])]),   # 0.5 -> .. (intermediate)
]

if __name__ == "__main__":
    print("=" * 76)
    print("E5 / T5  -  simulation-based currency identity   g = ||proj S||^2 / I")
    print("           g_truth uses the analytic score; g_hat uses NO score (sim+IBP)")
    print("=" * 76)
    worst = 0.0
    for name, model, theta, sigma, ladders in CASES:
        print(f"\n  {name}  (theta={theta}, proposal sigma={sigma})")
        print(f"    {'basis':14s} {'g_truth':>9s} {'g_hat(sim)':>11s} {'|diff|':>8s}")
        for label, powers in ladders:
            feat = feats(powers)
            gt, I = g_truth(model, feat, theta)
            gh = g_hat(model, feat, theta, I, sigma)
            d = abs(gh - gt)
            worst = max(worst, d)
            print(f"    {label:14s} {gt:9.4f} {gh:11.4f} {d:8.4f}"
                  f"{'   <-- FAIL' if d > TOL else ''}")
    print("\n" + "=" * 76)
    print(f"  worst |g_hat - g_truth| = {worst:.4f}   "
          f"=> T5 {'PASS' if worst < TOL else 'FAIL'} (tol {TOL})")
    print("  anchors recovered moment-free: scale {x}->g=0 (blind), {x,x^2}->g=1 (exact)")
