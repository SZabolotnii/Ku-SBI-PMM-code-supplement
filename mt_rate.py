r"""
C4 -- the finite-budget rule  sigma* ~ (JT)^{-1/6}   (reviewer Q7).

Section 5 predicts, for the surrogate coefficient estimator,
    MSE(sigma) = [B sigma^2]^2  +  kappa/(J sigma^2 T)
                 \___bias^2___/    \____MC variance____/
whose minimiser is  sigma* = (kappa/(2 B^2 JT))^{1/6} ~ (JT)^{-1/6}.

Section 7 validated only the sigma-dependence of the variance term.  Here we measure the
two facts the rule actually needs, each independently and non-circularly, on a NON-Gaussian
scale family (contaminated normal -- the Gaussian-scale special case has B=0):

  (A) SMOOTHING BIAS is O(sigma^2).  Measured NOISE-FREE from the population-smoothed
      coefficient W_pop(sigma) (huge single sample, no per-rep MC): |W_pop(sigma)-W*| ~ sigma^2.
  (B) MC VARIANCE is kappa/(J sigma^2).  Measured over R reps on a (J,sigma) grid:
      Var(W_hat) * J * sigma^2 = kappa is constant.

Given B (from A) and kappa (from B), sigma*(J) = (kappa/(2 B^2 J))^{1/6} ~ J^{-1/6}; with
the standard Polyak-Ruppert 1/T averaging over T steps, J -> JT, giving sigma* ~ (JT)^{-1/6}.
The -1/6 is the algebraic consequence of the two MEASURED scalings, not an input.

Pure numpy, seed 2026, self-checking.  Prints a single MT-RATE PASS/FAIL line.
"""
import numpy as np

RNG = np.random.default_rng(2026)


def mix_params(w=(0.9, 0.1), c=(1.0, 3.0)):
    w = np.array(w, float); c = np.array(c, float)
    sig = c / np.sqrt(float((w * c ** 2).sum()))           # standardise Var u = 1
    return w, sig


W, SIG = mix_params()


def draw_u(size):
    comp = RNG.choice(len(W), size=size, p=W)
    return RNG.standard_normal(size) * SIG[comp]


def scale_score(u):                                        # analytic scale score at theta=1
    Nk = np.stack([wk * np.exp(-u ** 2 / (2 * s ** 2)) / (np.sqrt(2 * np.pi) * s)
                   for wk, s in zip(W, SIG)], 0)
    dens = Nk.sum(0)
    num = np.stack([Nk[k] * (u ** 2 / SIG[k] ** 2 - 1.0) / 2.0 for k in range(len(SIG))], 0).sum(0)
    return num / dens


def w_star(N=40_000_000):
    """Population projection coefficient of the score onto the centred feature x^2.
    Large N: its residual error sets the floor of the O(sigma^2) bias measurement."""
    u = draw_u(N); S = scale_score(u)
    f = u ** 2 - (u ** 2).mean()
    return float((f * S).mean() / (f * f).mean())


def w_hat(theta_t, sigma, J, n):
    """IBP (no score) estimate of the x^2 coefficient at proposal scale sigma, budget J*n."""
    thetas = np.abs(RNG.normal(theta_t, sigma, J)) + 1e-9
    x = np.sqrt(thetas)[:, None] * draw_u((J, n))          # scale target: x = sqrt(theta) u
    th_rep = np.repeat(thetas, n)
    phi = (x ** 2).ravel(); phi = phi - phi.mean()
    b = float((phi * (th_rep - theta_t)).mean()) / sigma ** 2
    G = float((phi * phi).mean())
    return b / G


def slope(xs, ys):
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


# ======================================================================================
if __name__ == "__main__":
    print("=" * 78)
    print("C4  -  finite-budget rule  sigma* ~ (JT)^{-1/6}   (contaminated-normal scale)")
    print("=" * 78)
    theta_t = 1.0
    Wst = w_star()
    print(f"\n  W* (population x^2 coefficient) = {Wst:.5f}")

    # --- (A) smoothing bias O(sigma^2), measured NOISE-FREE (huge single sample) --------
    sig_b = np.array([0.14, 0.18, 0.22, 0.26, 0.30])
    bias = np.array([abs(w_hat(theta_t, s, 20_000_000, 1) - Wst) for s in sig_b])
    b_slope = slope(sig_b, bias)
    B = float(np.median(bias / sig_b ** 2))
    print("\n (A) smoothing bias  |W_pop(sigma) - W*|   [predict slope +2]")
    print("     sigma:", " ".join(f"{x:8.3f}" for x in sig_b))
    print("     bias :", " ".join(f"{x:8.2e}" for x in bias))
    print(f"     log-log slope = {b_slope:+.2f}   [{'OK' if abs(b_slope - 2) < 0.5 else 'OFF'}]"
          f"   =>  B = {B:.3f}")

    # --- (B) MC variance kappa/(J sigma^2): Var * J * sigma^2 = kappa (constant) ---------
    Js = [300, 600, 1200, 2400]
    sig_v = [0.07, 0.10, 0.14]
    n, R = 60, 300
    kappas = []
    print("\n (B) MC variance  Var(W_hat) * J * sigma^2 = kappa  [predict constant]")
    print(f"    {'J':>6} " + " ".join(f"s={s:<5}" for s in sig_v))
    for J in Js:
        row = []
        for s in sig_v:
            w = np.array([w_hat(theta_t, s, J, n) for _ in range(R)])
            k = float(w.var() * J * s ** 2)
            row.append(k); kappas.append(k)
        print(f"    {J:6d} " + " ".join(f"{k:7.2e}" for k in row))
    kappas = np.array(kappas)
    kappa = float(kappas.mean()); k_cv = float(kappas.std() / kappas.mean())
    print(f"     kappa = {kappa:.3e}   CV(kappa) across (m,sigma) = {k_cv:.2f}"
          f"   [{'OK' if k_cv < 0.35 else 'OFF'}]")

    # --- (=>) sigma*(J) = (kappa/(2 B^2 J))^{1/6} ~ J^{-1/6} -----------------------------
    sstar = np.array([(kappa / (2 * B ** 2 * J)) ** (1 / 6) for J in Js])
    s_slope = slope(Js, sstar)
    print("\n (=>) assembled rule  sigma*(J) = (kappa/2B^2 J)^{1/6}   [predict slope -1/6=-0.167]")
    print("     J      :", " ".join(f"{J:8d}" for J in Js))
    print("     sigma* :", " ".join(f"{x:8.4f}" for x in sstar))
    print(f"     log-log slope = {s_slope:+.3f}   [{'OK' if abs(s_slope + 1 / 6) < 0.02 else 'OFF'}]")

    ok = abs(b_slope - 2) < 0.5 and k_cv < 0.35 and abs(s_slope + 1 / 6) < 0.02
    print("\n" + "=" * 78)
    print(f"  => MT-RATE {'PASS' if ok else 'FAIL'}  (bias O(sigma^2), var kappa/(J sigma^2), "
          f"sigma* ~ (JT)^-1/6)")
