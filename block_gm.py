r"""
C5 / A3 -- the MULTI-PARAMETER (block) efficiency currency  (reviewer Q3).

For a vector parameter theta the scalar g_m generalises to a matrix.  With centred
features phi (Gram G) and Fisher matrix I, the captured-information matrix is
    K = B^T G^{-1} B,     B_{j,a} = E[ phi_j * s_a ] = d/dtheta_a E[phi_j]   (Bartlett),
and the block efficiency (matrix ARE relative to the MLE) is
    E = I^{-1/2} K I^{-1/2}   in [0, I]_{Loewner},  eigenvalues in [0,1].
Its diagonal gives the per-parameter captured fraction; off-diagonals are cross-information.

Demonstration on the joint Gaussian, x ~ N(mu, v), theta=(mu, v):
  s_mu = (x-mu)/v is ODD, s_v = -1/(2v)+(x-mu)^2/(2v^2) is EVEN, and E[s_mu s_v]=0
  (location-scale orthogonality) -> I is diagonal.

  basis {x}       (odd only)  : E ~ diag(1, 0)  -- mu captured, v INVISIBLE (parity blindness
                                                    of the linear/odd surrogate, now as a block).
  basis {x, x^2}  (mixed)     : E ~ I_2         -- one even term lifts the v-block from 0 to 1;
                                                    cross-term stays 0.

Pure numpy, seed 2026, self-checking.  Prints a single BLOCK-GM PASS/FAIL line.
"""
import numpy as np

RNG = np.random.default_rng(2026)
MU, V, N = 2.0, 4.0, 6_000_000


def scores(x):
    s_mu = (x - MU) / V
    s_v = -1.0 / (2 * V) + (x - MU) ** 2 / (2 * V ** 2)
    return np.stack([s_mu, s_v], 1)                        # (N,2)


def inv_sqrt(M):
    w, Q = np.linalg.eigh(M)
    return Q @ np.diag(w ** -0.5) @ Q.T


def block_eff(x, feats):
    S = scores(x)                                          # (N,2)
    F = np.stack(feats, 1)
    F = F - F.mean(0)                                      # centre (const absorbed)
    G = F.T @ F / len(x)                                   # (d,d)
    B = F.T @ S / len(x)                                   # (d,2)  Bartlett terms
    I = S.T @ S / len(x)                                   # (2,2)  Fisher matrix
    K = B.T @ np.linalg.solve(G + 1e-12 * np.eye(G.shape[0]), B)   # (2,2) captured info
    Ih = inv_sqrt(I)
    return Ih @ K @ Ih, I                                  # (2,2) efficiency, Fisher


if __name__ == "__main__":
    print("=" * 78)
    print(f"BLOCK-GM  -  multi-parameter efficiency matrix, x~N(mu={MU}, v={V})")
    print("=" * 78)
    x = RNG.normal(MU, np.sqrt(V), N)

    E1, I = block_eff(x, [x])                              # {x}: odd only
    E2, _ = block_eff(x, [x, x ** 2])                      # {x, x^2}: mixed parity

    print(f"\n  Fisher matrix I (empirical) =\n{np.array2string(I, precision=4, prefix='      ')}")
    print(f"  (analytic diag = [1/v, 1/(2v^2)] = [{1/V:.4f}, {1/(2*V**2):.4f}], off-diag 0)")

    print("\n  basis {x}  (odd/location only) -- efficiency matrix E:")
    print(np.array2string(E1, precision=4, prefix='      '))
    print(f"    mu-block = {E1[0,0]:.4f} (captured),  v-block = {E1[1,1]:.4f} (BLIND),  "
          f"cross = {E1[0,1]:+.4f}")

    print("\n  basis {x, x^2}  (mixed parity) -- efficiency matrix E:")
    print(np.array2string(E2, precision=4, prefix='      '))
    print(f"    mu-block = {E2[0,0]:.4f},  v-block = {E2[1,1]:.4f} (LIFTED from 0),  "
          f"cross = {E2[0,1]:+.4f}")

    ok = (abs(E1[0, 0] - 1) < 0.02 and abs(E1[1, 1]) < 0.02 and abs(E1[0, 1]) < 0.02
          and abs(E2[0, 0] - 1) < 0.02 and abs(E2[1, 1] - 1) < 0.02 and abs(E2[0, 1]) < 0.02)
    print("\n" + "=" * 78)
    print(f"  => BLOCK-GM {'PASS' if ok else 'FAIL'}  ({{x}}->diag(1,0); {{x,x^2}}->I_2; "
          f"even term lifts the v-block, cross-info stays 0)")
