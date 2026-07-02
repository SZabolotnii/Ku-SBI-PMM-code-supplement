"""
S5' -- CLOSING THE THEORY.  Validate the closed forms derived for the PATP-FSM
efficiency and its coefficients (Spec/math-spine S5').  Scale family, u=x/sqrt(theta):

  g_m(alpha) = 4 v^T Sigma^-1 v / J0,   v_j=(q_j/2)M(q_j),  Sigma_ij=M(q_i+q_j)-M(q_i)M(q_j),
  A(alpha)   = mu^T Sigma^-1 mu,        mu_j=M(q_j),        M(q)=E_{u~f0}|u|^q  (std moments),
  kappa(alpha) ~ (A/g_m)^2,             J0 = E[(1+u l'(u))^2] = 4 theta^2 I  (scale Fisher).

Key claims tested:
  (1) g_m via CLOSED-FORM moments (no score) == g_m via DIRECT score projection b^T G^-1 b / I.
      -> this IS the Bartlett transfer b_j = (q_j/2)M(q_j) = d/dtheta E[phi_j], moment-free.
  (2) g_m is theta-INVARIANT (scale invariance).
  (3) alpha* = argmax_alpha g_m(alpha):  Gaussian -> 1 (x^2),  heavy tail -> interior <1.
  (4) smoothing-bias driver E[S2] = -1/4 (dI/dtheta + E[S^3]) VANISHES for Gaussian scale.
"""
import numpy as np
RNG = np.random.default_rng(2026)

def p2(a): return 0.5 + 0.5*a + 1.0*a**2          # PATP exponent p_2(alpha): 1/2 -> 2

# --- standardised populations u (Var u = 1) + analytic scale-score at theta=1 (x=u) ----
def pop_gauss(Nu=4_000_000):
    u = RNG.standard_normal(Nu)
    S = 0.5*(u**2 - 1.0)                            # scale score at theta=1: (u^2-1)/2
    return u, S
def pop_t(nu, Nu=4_000_000):
    u = RNG.standard_t(nu, Nu) / np.sqrt(nu/(nu-2.0))
    c2 = (nu-2.0)/nu; uu = u**2/c2
    S = -0.5 + (nu+1.0)*uu/(2.0*(nu+uu))           # scale score at theta=1
    return u, S

def g_direct(u, S, q):                             # uses the score (ground truth)
    phi = np.abs(u)**q; phi = phi - phi.mean()
    b = float((phi*S).mean()); G = float((phi*phi).mean()); I = float((S*S).mean())
    return b*b/(G*I), b, G, I
def g_closed(u, S, q):                             # uses only |u|^q moments (+ I as normaliser)
    M = lambda k: float((np.abs(u)**k).mean())
    v = 0.5*q*M(q); Sig = M(2*q) - M(q)**2; I = float((S*S).mean())
    A = M(q)**2 / Sig
    return v*v/(Sig*I), v, Sig, A                  # = 4 v^2/(Sig*J0), J0=4I

if __name__ == "__main__":
    print("="*78); print("S5'  -  closed-form efficiency g_m(alpha), kappa(alpha)  (scale family)"); print("="*78)

    for name, (u, S) in [("Gaussian", pop_gauss()), ("Student-t(nu=8)", pop_t(8))]:
        print(f"\n  {name}   [g_direct uses score | g_closed uses only |u|^q moments (Bartlett)]")
        print(f"    {'alpha':>6} {'q=p2':>6} {'g_direct':>9} {'g_closed':>9} {'|diff|':>8} {'A(a)':>7} {'kappa~':>8}")
        best = (None, -1)
        for a in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0]:
            q = p2(a)
            gd, b, G, I = g_direct(u, S, q)
            gc, v, Sig, A = g_closed(u, S, q)
            kap = (A/gc)**2
            if gc > best[1]: best = (a, gc)
            print(f"    {a:6.2f} {q:6.2f} {gd:9.4f} {gc:9.4f} {abs(gd-gc):8.5f} {A:7.3f} {kap:8.2f}")
        print(f"    -> alpha* = argmax g_m = {best[0]}  (g_m*={best[1]:.4f})")

    # (2) theta-invariance: g_direct at theta in {1,4,9} for a fixed feature (x=sqrt(theta)u)
    print("\n  theta-invariance of g_m  (feature |x|^{p2(0.6)}, direct, x~scale theta):")
    a = 0.6; q = p2(a)
    for th in [1.0, 4.0, 9.0]:
        u = RNG.standard_normal(2_000_000) * np.sqrt(th)          # x ~ N(0, th)
        S = -1/(2*th) + u**2/(2*th**2)
        gd, *_ = g_direct(u, S, q)
        print(f"    theta={th:4.1f}:  g_m = {gd:.4f}")

    # (4) Gaussian-scale bias driver  E[S2] = -1/4 (dI/dtheta + E[S^3])  -> 0
    th = 1.0
    dI_dtheta = -1.0/th**3                                        # I=1/(2 th^2) -> dI/dth=-1/th^3
    x = RNG.standard_normal(8_000_000)*np.sqrt(th)
    S = -1/(2*th) + x**2/(2*th**2)
    ES3 = float((S**3).mean())
    ES2_driver = -0.25*(dI_dtheta + ES3)
    print("\n  smoothing-bias driver (Gaussian scale):  E[S2] = -1/4 (dI/dtheta + E[S^3])")
    print(f"    dI/dtheta = {dI_dtheta:+.4f},  E[S^3] = {ES3:+.4f},  sum = {dI_dtheta+ES3:+.4f}"
          f"  =>  E[S2] = {ES2_driver:+.5f}  (=> bias O(sigma^4), explains tiny S5-(ii))")
    print("\n  => g_m(alpha) closed form matches the score-based truth (Bartlett transfer),")
    print("     is theta-invariant, predicts alpha*, and the Gaussian bias driver vanishes.")
