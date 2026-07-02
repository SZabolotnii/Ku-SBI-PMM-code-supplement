"""
Ablations for the reviewer's conditioning / basis-design questions (Q4, Q6/D).
Everything is the captured-Fisher fraction

    g = ||proj_V S||^2 / ||S||^2 = R^2 of the score S regressed on the (centered) basis V,

the same currency as g_m (Prop. 2).  Four gates:

  C2  RIDGE lambda-sweep       : cond(Phi^T Phi + lambda I) drops with lambda while g is
                                 stable for small lambda (Tikhonov only regularises, does
                                 not change the projection until lambda bites).
  C3  BASIS-SIZE m-sweep       : nested even basis {x^2},{x^2,x^4},...  ; g is monotone
                                 non-decreasing in m and saturates, while cond(G) blows up
                                 -> keep m small (3-5).
  C6  alpha CROSS-CHECK        : closed-form g_m(alpha) (moments, Prop. 2) vs a
                                 simulation-based g_hat(alpha) that never sees the score
                                 (Bartlett/IBP, as in E5).  They agree across alpha.
  C7  ORTHOGONALISATION        : g is invariant to a QR re-parameterisation of the basis
                                 (projection is basis-free): g_raw == g_qr, yet
                                 cond(raw) >> cond(orthonormal)=1.  This is why we do not
                                 need orthogonal polynomials -- G^{-1} already orthogonalises.

Pure numpy, seed 2026, self-checking.  Prints a single ABLATIONS PASS/FAIL line.
"""
import numpy as np

RNG = np.random.default_rng(2026)
NPOP = 4_000_000


# ---------------------------------------------------------------------------
# Populations (standardised u, Var u = 1) + analytic scale-score at theta=1 (x=u).
# ---------------------------------------------------------------------------
def p2(a):
    return 0.5 + 0.5 * a + 1.0 * a ** 2                      # PATP exponent p_2: 1/2 -> 2


def pop_t(nu, N=NPOP):
    u = RNG.standard_t(nu, N) / np.sqrt(nu / (nu - 2.0))
    c2 = (nu - 2.0) / nu
    uu = u ** 2 / c2
    S = -0.5 + (nu + 1.0) * uu / (2.0 * (nu + uu))
    return u, S


def pop_mix(N=NPOP, w=(0.9, 0.1), c=(1.0, 3.0)):
    """Contaminated normal (scale mixture): all moments finite, x^2 NOT sufficient,
    so g({x^2})<1 and a nested even basis strictly improves it."""
    w = np.array(w, float); c = np.array(c, float)
    sig = c / np.sqrt(float((w * c ** 2).sum()))            # standardise: Var u = 1
    comp = RNG.choice(len(w), size=N, p=w)
    u = RNG.standard_normal(N) * sig[comp]
    # analytic scale score at theta=1 for a mixture of zero-mean Gaussians N(0, sig_k^2):
    Nk = np.stack([wk * np.exp(-u ** 2 / (2 * s ** 2)) / (np.sqrt(2 * np.pi) * s)
                   for wk, s in zip(w, sig)], 0)
    dens = Nk.sum(0)
    num = np.stack([Nk[k] * (u ** 2 / sig[k] ** 2 - 1.0) / 2.0 for k in range(len(sig))], 0).sum(0)
    return u, num / dens


def g_of(F, S):
    """Captured Fisher fraction = ||proj S||^2/||S||^2 via centred features F (N,d)."""
    Fc = F - F.mean(0)
    Sc = S - S.mean()
    G = Fc.T @ Fc
    b = Fc.T @ Sc
    num = float(b @ np.linalg.solve(G + 1e-12 * np.eye(G.shape[0]), b))
    return num / float(Sc @ Sc)


def cond_of(F):
    Fc = F - F.mean(0)
    Z = Fc / (Fc.std(0) + 1e-12)
    return float(np.linalg.cond(Z.T @ Z / Z.shape[0]))


# ===========================================================================
def c2_ridge():
    """cond(G+lambda I) vs lambda on an ill-conditioned even basis; g stable for small lam."""
    u, S = pop_mix()
    F = np.stack([u ** 2, u ** 4, u ** 6], 1)               # nested even, deliberately stiff
    Fc = F - F.mean(0); Sc = S - S.mean()
    G = Fc.T @ Fc / len(u); b = Fc.T @ Sc / len(u); I = float(Sc @ Sc) / len(u)
    lams = [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0]
    rows, g0 = [], None
    for lam in lams:
        A = G + lam * np.eye(3)
        g = float(b @ np.linalg.solve(A, b)) / I
        rows.append((lam, float(np.linalg.cond(A)), g))
        if lam == 0.0:
            g0 = g
    cond_drop = rows[-1][1] < rows[0][1]                    # lambda=1 better conditioned
    g_stable = all(abs(g - g0) < 0.02 for lam, _, g in rows if lam <= 1e-4)
    return rows, (cond_drop and g_stable), g0


def c3_msweep():
    """Nested even basis of growing size -> g monotone up, cond up."""
    u, S = pop_mix()
    rows, prev = [], -1.0
    mono = True
    for k in range(1, 6):
        F = np.stack([u ** (2 * j) for j in range(1, k + 1)], 1)   # {x^2,...,x^{2k}}
        g = g_of(F, S); c = cond_of(F)
        rows.append((k, g, c))
        if g < prev - 1e-3:
            mono = False
        prev = g
    cond_grows = rows[-1][2] > rows[0][2] * 10
    gains = rows[-1][1] > rows[0][1] + 0.3                  # substantial info gain with m
    return rows, (mono and cond_grows and gains)


def c6_alpha_crosscheck():
    """closed-form g_m(alpha) (moments) vs simulation-based g_hat(alpha) (NO score)."""
    nu = 8
    u, S = pop_t(nu)
    I = float(((S - S.mean()) ** 2).mean())
    # simulation harness (no score): x = sqrt(theta) u_t8, theta ~ N(1, sigma^2)>0
    theta_t, sigma, m, n = 1.0, 0.08, 60000, 200
    thetas = np.abs(RNG.normal(theta_t, sigma, m)) + 1e-9
    ut = RNG.standard_t(nu, (m, n)) / np.sqrt(nu / (nu - 2.0))
    X = (np.sqrt(thetas)[:, None] * ut)                     # (m,n); scale target
    th_rep = np.repeat(thetas, n)
    rows, worst = [], 0.0
    for a in [0.0, 0.3, 0.5, 0.6, 0.8, 1.0]:
        q = p2(a)
        # closed form (moments of standardised u):
        M = lambda kk: float((np.abs(u) ** kk).mean())
        v = 0.5 * q * M(q); Sig = M(2 * q) - M(q) ** 2
        g_closed = v * v / (Sig * I)
        # simulation + IBP, no score:
        phi = (np.abs(X) ** q).ravel()
        phi = phi - phi.mean()
        b_hat = float((phi * (th_rep - theta_t)).mean()) / sigma ** 2
        G_hat = float((phi * phi).mean())
        g_hat = b_hat * b_hat / (G_hat * I)
        d = abs(g_closed - g_hat)
        worst = max(worst, d)
        rows.append((a, q, g_closed, g_hat, d))
    return rows, worst, (worst < 0.05)


def c7_orthogonalisation():
    """g invariant under QR re-parameterisation; cond(raw)>>cond(orthonormal)=1."""
    u, S = pop_mix()
    F = np.stack([u ** 2, u ** 4, u ** 6], 1)
    Fc = F - F.mean(0); Sc = S - S.mean()
    g_raw = g_of(F, S)
    Q, _ = np.linalg.qr(Fc)                                 # Q^T Q = I over the sample
    g_qr = float((Q.T @ Sc) @ (Q.T @ Sc)) / float(Sc @ Sc)
    cond_raw = float(np.linalg.cond(Fc.T @ Fc))
    cond_qr = float(np.linalg.cond(Q.T @ Q))
    ok = abs(g_raw - g_qr) < 1e-8 and cond_qr < 1.001 and cond_raw > 1e3
    return g_raw, g_qr, cond_raw, cond_qr, ok


# ===========================================================================
if __name__ == "__main__":
    print("=" * 78)
    print("ABLATIONS  -  conditioning / basis design (reviewer Q4, Q6/D)")
    print("=" * 78)
    ok_all = True

    rows, ok, g0 = c2_ridge()
    ok_all &= ok
    print("\n C2  ridge lambda-sweep  (even basis {x^2,x^4,x^6}, contaminated normal)")
    print(f"    {'lambda':>8} {'cond(G+lI)':>12} {'g_m':>8}")
    for lam, c, g in rows:
        print(f"    {lam:8.0e} {c:12.3e} {g:8.4f}")
    print(f"    -> cond falls with lambda; g_m stable at {g0:.4f} for lambda<=1e-4  [{'OK' if ok else 'OFF'}]")

    rows, ok = c3_msweep()
    ok_all &= ok
    print("\n C3  basis-size m-sweep  (nested even {x^2,...,x^{2k}}, contaminated normal)")
    print(f"    {'k (#feat)':>10} {'g_m':>8} {'cond(G)':>12}")
    for k, g, c in rows:
        print(f"    {k:10d} {g:8.4f} {c:12.3e}")
    print(f"    -> g_m monotone up & saturates; cond blows up -> keep m small  [{'OK' if ok else 'OFF'}]")

    rows, worst, ok = c6_alpha_crosscheck()
    ok_all &= ok
    print("\n C6  alpha cross-check  (Student-t8 scale): closed-form g_m vs sim g_hat (NO score)")
    print(f"    {'alpha':>6} {'q':>6} {'g_closed':>9} {'g_hat(sim)':>11} {'|diff|':>8}")
    for a, q, gc, gh, d in rows:
        print(f"    {a:6.2f} {q:6.3f} {gc:9.4f} {gh:11.4f} {d:8.4f}")
    print(f"    -> worst |g_closed - g_hat| = {worst:.4f} < 0.05  [{'OK' if ok else 'OFF'}]")

    g_raw, g_qr, cr, cq, ok = c7_orthogonalisation()
    ok_all &= ok
    print("\n C7  orthogonalisation invariance  (QR re-parameterisation of {x^2,x^4,x^6})")
    print(f"    g_raw = {g_raw:.6f}   g_qr = {g_qr:.6f}   |diff| = {abs(g_raw-g_qr):.2e}")
    print(f"    cond(raw G) = {cr:.3e}   cond(orthonormal) = {cq:.4f}")
    print(f"    -> g identical; projection is basis-free (G^-1 already orthogonalises)  [{'OK' if ok else 'OFF'}]")

    print("\n" + "=" * 78)
    print(f"  => ABLATIONS {'PASS' if ok_all else 'FAIL'}  (C2 ridge, C3 m-sweep, C6 alpha, C7 QR)")
