r"""
V1 -- MULTIVARIATE basis construction, d_x = 2..6  (EJS spec, block R).

The projection framework (Prop. 1) is dimension-free: nothing in the normal equations
requires d_x = 1, only the feature map changes.  The paper's construction for
x in R^d is COORDINATE-WISE sectors,
    odd  : x_1, ..., x_d
    even : |x_1|^{p_2(alpha)}, ..., |x_d|^{p_2(alpha)}        (2d+1 features with const),
plus OPTIONAL cross terms x_a x_b -- needed exactly when a cross parameter
(covariance/correlation) is a target.  This gate verifies the construction on
analytic anchors, x ~ N(mu, diag(v)) with theta = (mu_1..mu_d, v_1..v_d):

  V1a  odd-only coordinate-wise basis {x_i}:      E ~ diag(1_d, 0_d)
       (parity blindness persists coordinate-wise: every v_i block INVISIBLE)
  V1b  mixed coordinate-wise basis {x_i, x_i^2}:  E ~ I_{2d}  for ALL d = 2..6
       (2d features capture all 2d parameters; cross-information ~ 0)
  V1c  conditioning: with per-coordinate STANDARDIZED features the centred Gram is
       blockdiag(I_d, 2 I_d) + O(N^{-1/2})  =>  cond(G) ~ 2, flat in d
       (raw unstandardized features: cond grows with scale heterogeneity -- info only)
  V1d  cross terms: bivariate N(0, [[1,rho],[rho,1]]), target rho, truth rho = 0.
       The rho-score at rho=0 is s = x_1 x_2 exactly:
         coordinate-wise mixed basis (no cross):  g_rho ~ 0   (structurally blind)
         + single cross feature x_1 x_2        :  g_rho ~ 1   (score in the span)
       [info] the blindness is EXACT at every rho, not just rho=0: with
       M = Sigma^{-1} A Sigma^{-1} (A = dSigma/drho, zero diagonal), Isserlis gives
       cov(x_i^2, 1/2 x^T M x) = (Sigma M Sigma)_ii = A_ii = 0, and cov(x_i, s_rho)=0
       by parity -- so b = 0 for ALL coordinate-wise features at every rho; the
       single cross term restores g = 1 exactly (s_rho lies in
       span{x_1^2, x_2^2, x_1 x_2, 1}).  Verified at rho = 0.3, printed ungated.

Efficiency matrix as in block_gm.py (C5):  E = I^{-1/2} (B^T G^{-1} B) I^{-1/2},
B via direct MC of E[phi~ s]; Fisher I via MC of E[s s^T]; scores analytic.

PRE-REGISTERED THRESHOLDS (fixed before the first full run; budget knobs = N only):
  V1a: mean-block diag within 0.02 of 1; v-block diag <= 0.02; |off-diag| <= 0.02
  V1b: all 2d diag within 0.02 of 1;                          |off-diag| <= 0.02
  V1c: cond(G_std) <= 10 for every d in {2..6}   (analytic value 2)
  V1d: g_rho(no cross) <= 0.02;  g_rho(cross) >= 0.98

Pure numpy, seed 2026, ~10 s.  Writes results/v1_multivariate.json, prints PASS/FAIL.
"""
import json
import os
import numpy as np

SEED = 2026
N = 2_000_000
DS = (2, 3, 4, 5, 6)
RHO_INFO = 0.3

# pre-registered thresholds -- never adjust to make a run pass
TOL_E = 0.02
COND_MAX = 10.0
G_BLIND_MAX = 0.02
G_CROSS_MIN = 0.98


def inv_sqrt(M):
    w, Q = np.linalg.eigh(M)
    return Q @ np.diag(w ** -0.5) @ Q.T


def block_eff(F, S):
    """E = I^{-1/2} (B^T G^{-1} B) I^{-1/2} from raw feature/score matrices."""
    F = F - F.mean(0)
    n = len(F)
    G = F.T @ F / n
    B = F.T @ S / n
    I = S.T @ S / n
    K = B.T @ np.linalg.solve(G + 1e-12 * np.eye(G.shape[0]), B)
    Ih = inv_sqrt(I)
    return Ih @ K @ Ih, G


def diag_gauss_case(rng, d):
    """x ~ N(mu, diag(v)); theta = (mu_1..mu_d, v_1..v_d); analytic scores."""
    mu = 0.5 * np.arange(1, d + 1)
    v = 1.0 + 0.5 * np.arange(d)
    x = rng.normal(mu, np.sqrt(v), size=(N, d))
    xc = x - mu
    S = np.concatenate([xc / v, -1.0 / (2 * v) + xc ** 2 / (2 * v ** 2)], axis=1)

    F_odd = x.copy()
    F_mix = np.concatenate([x, x ** 2], axis=1)
    E_odd, _ = block_eff(F_odd, S)
    E_mix, _ = block_eff(F_mix, S)

    # V1c conditioning: per-coordinate standardized vs raw features
    z = (x - x.mean(0)) / x.std(0)
    F_std = np.concatenate([z, z ** 2], axis=1)
    F_std = F_std - F_std.mean(0)
    G_std = F_std.T @ F_std / N
    F_raw = F_mix - F_mix.mean(0)
    G_raw = F_raw.T @ F_raw / N
    return E_odd, E_mix, np.linalg.cond(G_std), np.linalg.cond(G_raw)


def rho_g(rng, rho, with_cross):
    """g for the correlation parameter of bivariate N(0, [[1,rho],[rho,1]])."""
    L = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    x = rng.standard_normal((N, 2)) @ L.T
    Sig_inv = np.linalg.inv(np.array([[1.0, rho], [rho, 1.0]]))
    A = np.array([[0.0, 1.0], [1.0, 0.0]])
    M = Sig_inv @ A @ Sig_inv
    s = 0.5 * np.einsum("ni,ij,nj->n", x, M, x) - 0.5 * np.trace(Sig_inv @ A)

    feats = [x[:, 0], x[:, 1], x[:, 0] ** 2, x[:, 1] ** 2]
    if with_cross:
        feats.append(x[:, 0] * x[:, 1])
    F = np.stack(feats, 1)
    F = F - F.mean(0)
    sc = s - s.mean()
    G = F.T @ F / N
    b = F.T @ sc / N
    I = sc @ sc / N
    return float(b @ np.linalg.solve(G + 1e-12 * np.eye(len(b)), b) / I)


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    print("=" * 78)
    print("V1  -  multivariate coordinate-wise PATP construction, d = 2..6")
    print("=" * 78)

    rows, ok_a, ok_b, ok_c = [], True, True, True
    for d in DS:
        E_odd, E_mix, c_std, c_raw = diag_gauss_case(rng, d)
        mu_diag = np.diag(E_odd)[:d]
        v_diag = np.diag(E_odd)[d:]
        off_odd = np.max(np.abs(E_odd - np.diag(np.diag(E_odd))))
        dev_mix = np.max(np.abs(E_mix - np.eye(2 * d)))
        a = (np.max(np.abs(mu_diag - 1)) <= TOL_E and np.max(np.abs(v_diag)) <= TOL_E
             and off_odd <= TOL_E)
        b = dev_mix <= TOL_E
        c = c_std <= COND_MAX
        ok_a &= a
        ok_b &= b
        ok_c &= c
        rows.append({"d": d,
                     "odd_mu_diag_maxdev": float(np.max(np.abs(mu_diag - 1))),
                     "odd_v_diag_max": float(np.max(np.abs(v_diag))),
                     "odd_offdiag_max": float(off_odd),
                     "mixed_maxdev_from_I": float(dev_mix),
                     "cond_G_standardized": float(c_std),
                     "cond_G_raw": float(c_raw)})
        print(f"  d={d}:  odd E~diag(1_d,0_d) dev {np.max(np.abs(mu_diag-1)):.4f}/"
              f"{np.max(np.abs(v_diag)):.4f}/{off_odd:.4f} | mixed |E-I| "
              f"{dev_mix:.4f} | cond(G) std {c_std:.2f} raw {c_raw:.1f}")

    g_blind = rho_g(rng, 0.0, with_cross=False)
    g_cross = rho_g(rng, 0.0, with_cross=True)
    g_part_nc = rho_g(rng, RHO_INFO, with_cross=False)
    g_part_c = rho_g(rng, RHO_INFO, with_cross=True)
    ok_d = g_blind <= G_BLIND_MAX and g_cross >= G_CROSS_MIN
    print(f"\n  V1d cross terms (rho=0):   no-cross g = {g_blind:.4f}  "
          f"(<= {G_BLIND_MAX}), + x1*x2 g = {g_cross:.4f} (>= {G_CROSS_MIN})")
    print(f"  [info] rho={RHO_INFO}: no-cross g = {g_part_nc:.4f} (blindness is exact "
          f"at every rho -- Isserlis cancellation), cross g = {g_part_c:.4f}")

    all_pass = ok_a and ok_b and ok_c and ok_d
    res = {"seed": SEED, "N": N, "dims": list(DS), "per_d": rows,
           "V1a_odd_blind_pass": bool(ok_a), "V1b_mixed_full_pass": bool(ok_b),
           "V1c_cond_pass": bool(ok_c),
           "V1d": {"rho0_nocross": g_blind, "rho0_cross": g_cross,
                   "rho03_nocross_info": g_part_nc, "rho03_cross_info": g_part_c,
                   "pass": bool(ok_d)},
           "thresholds": {"TOL_E": TOL_E, "COND_MAX": COND_MAX,
                          "G_BLIND_MAX": G_BLIND_MAX, "G_CROSS_MIN": G_CROSS_MIN},
           "all_pass": bool(all_pass)}
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
                exist_ok=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "v1_multivariate.json"), "w") as f:
        json.dump(res, f, indent=2)

    print("\n" + "=" * 78)
    print(f"  => V1 {'PASS' if all_pass else 'FAIL'}  "
          f"(coordinate-wise: odd blind / mixed full for d=2..6; "
          f"cond(G) flat after standardization; cross term <=> cross parameter)")
    raise SystemExit(0 if all_pass else 1)
