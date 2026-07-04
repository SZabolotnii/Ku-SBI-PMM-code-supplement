"""
D1/D2 -- PATP exponent-domain gates for the EJS hardening spec.

D1 checks exactly the schedule claims used in the manuscript:
    * p2(alpha) is positive and strictly increasing on [0, 1];
    * p3(alpha) is positive and nondecreasing on [0, 1];
    * high-order counterexamples exist, so no global monotonicity/positivity claim is allowed.

D2 checks the infinite-fourth-moment edge for the single even PATP feature on a standardized
Student-t4 scale family.  For q = p2(alpha), the closed-form captured-Fisher numerator uses
M(q), while the denominator uses M(2q).  At alpha -> 1, q -> 2 and M(4) diverges, so the
integer edge degenerates.  The reported h(alpha) omits the common Fisher constant J0 because
it is independent of alpha and irrelevant to the edge and argmax checks.
"""
import json
import math
import os

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))


def patp_power(i, alpha):
    return 1.0 / i + (4.0 - i - 3.0 / i) * alpha + (2.0 * i - 4.0 + 2.0 / i) * alpha**2


def std_t_abs_moment(nu, r):
    """E|U|^r for U = standardized Student-t_nu with Var(U)=1."""
    if r >= nu:
        return math.inf
    log_m = (
        (r / 2.0) * math.log(nu - 2.0)
        + math.lgamma((r + 1.0) / 2.0)
        + math.lgamma((nu - r) / 2.0)
        - 0.5 * math.log(math.pi)
        - math.lgamma(nu / 2.0)
    )
    return math.exp(log_m)


def t4_truncated_m4(cap):
    """E[U^4 1{|U|<=cap}] for standardized Student-t4.

    With density C(1+u^2/2)^(-5/2), the integral reduces to
    6 * [log((sqrt(L^2+2)+L)/sqrt(2)) - s - s^3/3], s=L/sqrt(L^2+2).
    """
    s = cap / math.sqrt(cap * cap + 2.0)
    return 6.0 * (
        math.log((math.sqrt(cap * cap + 2.0) + cap) / math.sqrt(2.0))
        - s
        - s**3 / 3.0
    )


def t4_single_even_h(alpha):
    q = patp_power(2, alpha)
    m_q = std_t_abs_moment(4.0, q)
    m_2q = std_t_abs_moment(4.0, 2.0 * q)
    if not math.isfinite(m_2q):
        return 0.0
    sigma = m_2q - m_q * m_q
    v = (q / 2.0) * m_q
    return 4.0 * v * v / sigma


def load_phase3_c_sweep():
    path = os.path.join(HERE, "phase3_real", "results", "phase3_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f).get("C_sweep")


def main():
    alphas = np.linspace(0.0, 1.0, 10001)

    p2 = patp_power(2, alphas)
    p3 = patp_power(3, alphas)
    p2_ok = float(p2.min()) > 0.0 and bool(np.all(np.diff(p2) > 0.0))
    p3_ok = float(p3.min()) > 0.0 and bool(np.all(np.diff(p3) >= -1e-14))

    nonmonotone = []
    nonpositive = []
    for i in range(1, 9):
        vals = patp_power(i, alphas)
        if np.any(np.diff(vals) < -1e-12):
            nonmonotone.append((i, float(vals.min()), float(vals.max())))
        if float(vals.min()) <= 0.0:
            nonpositive.append((i, float(vals.min())))

    d1_ok = p2_ok and p3_ok and any(i >= 4 for i, _, _ in nonmonotone) and any(
        i >= 6 for i, _ in nonpositive
    )

    alpha_probe = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99])
    h_vals = np.array([t4_single_even_h(float(a)) for a in alpha_probe])
    best_alpha = float(alpha_probe[int(np.argmax(h_vals))])
    m4_infinite = math.isinf(std_t_abs_moment(4.0, 4.0))

    caps = np.array([20.0, 100.0, 1000.0, 1_000_000.0])
    edge_h = np.array([4.0 / (t4_truncated_m4(float(c)) - 1.0) for c in caps])
    edge_degenerates = bool(np.all(np.diff(edge_h) < 0.0) and edge_h[-1] < 0.15 * h_vals.max())

    c_sweep = load_phase3_c_sweep()
    phase3_ok = False
    phase3_msg = "phase3 C_sweep unavailable"
    if c_sweep:
        phase3_ok = c_sweep.get("alpha_star_t", 1.0) < 1.0 and c_sweep.get("t_gain_vs_x2", 0.0) > 0.1
        phase3_msg = (
            f"phase3 t4 alpha*={c_sweep.get('alpha_star_t')}, "
            f"gain over x2={100.0 * c_sweep.get('t_gain_vs_x2', 0.0):.0f}%"
        )

    d2_ok = m4_infinite and best_alpha < 1.0 and edge_degenerates and phase3_ok

    print("=" * 78)
    print("D1/D2 - PATP domain and infinite-fourth-moment edge")
    print("=" * 78)
    print("\n D1 schedule claims on alpha grid [0,1]")
    print(
        f"    p2: min={p2.min():.4f}, strictly increasing={np.all(np.diff(p2) > 0.0)}"
    )
    print(
        f"    p3: min={p3.min():.4f}, nondecreasing={np.all(np.diff(p3) >= -1e-14)}"
    )
    print(
        "    high-order nonmonotone examples: "
        + ", ".join(f"i={i} min={mn:.3f}" for i, mn, _ in nonmonotone[:3])
    )
    print(
        "    high-order nonpositive examples: "
        + ", ".join(f"i={i} min={mn:.3f}" for i, mn in nonpositive[:3])
    )
    print(f"    => D1 {'PASS' if d1_ok else 'FAIL'}")

    print("\n D2 Student-t4 edge, h(alpha)=4v^2/Sigma (common J0 omitted)")
    print("    alpha:", " ".join(f"{a:7.2f}" for a in alpha_probe))
    print("    h    :", " ".join(f"{h:7.3f}" for h in h_vals))
    print(
        "    truncated integer-edge h_L(alpha=1), cap L: "
        + " ".join(f"{int(c):>7d}" for c in caps)
    )
    print("                                      h_L: " + " ".join(f"{h:7.3f}" for h in edge_h))
    print(f"    M_t4(4) is infinite={m4_infinite}; best finite probe alpha={best_alpha:.2f}")
    print(f"    {phase3_msg}")
    print(f"    => D2 {'PASS' if d2_ok else 'FAIL'}")

    ok = d1_ok and d2_ok
    print("\n" + "=" * 78)
    print(f"  => PATP-DOMAIN {'PASS' if ok else 'FAIL'}  (D1 schedule scope, D2 t4 edge)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
