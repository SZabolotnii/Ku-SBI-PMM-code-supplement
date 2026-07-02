"""
Verification gate for Ku-SBI-PMM (run BEFORE citing any number).
Recomputes the three experiments from run.py (fixed seed) and asserts the
workflow's verification standards.  Prints a PASS/FAIL report; exit code != 0 on FAIL.

    python verify.py
"""
import sys
import run

def check(name, cond, detail):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    return bool(cond)

def main():
    print("Recomputing experiments (seed = %d) ..." % run.SEED)
    e1 = run.exp_score_recovery()
    e2 = run.exp_joint_mle()
    e3 = run.exp_patp_sweep()
    ok = []

    print("\n--- Score recovery (Exp 1) ---")
    lin, p2 = e1["linear [1,x]"], e1["poly2 [1,x,x^2]"]
    ok.append(check("G1 linear is blind to the EVEN scale score",
                    lin["mse_v"] > 10 * p2["mse_v"],
                    f"MSE(s_v): linear={lin['mse_v']:.3f} vs poly2={p2['mse_v']:.3f} "
                    f"(ratio {lin['mse_v']/p2['mse_v']:.0f}x)"))
    ok.append(check("G2 polynomial does no harm on the ODD location score",
                    p2["mse_mu"] < 0.1 and lin["mse_mu"] < 0.1,
                    f"MSE(s_mu): linear={lin['mse_mu']:.3f}, poly2={p2['mse_mu']:.3f}"))

    print("\n--- Joint FSM-MLE of (mu, v) (Exp 2) ---")
    L, P = e2["linear [1,x]"], e2["poly2 [1,x,x^2]"]
    ok.append(check("G3 linear FAILS the variance; poly2 recovers it",
                    abs(L["v"] - 4.0) > 1.0 and abs(P["v"] - 4.0) < 0.5,
                    f"v_hat: linear={L['v']:.2f} (err {abs(L['v']-4):.2f}), "
                    f"poly2={P['v']:.2f} (err {abs(P['v']-4):.2f}); truth 4.0"))
    ok.append(check("G4 ALL bases recover the mean",
                    abs(L["mu"] - 2.0) < 0.2 and abs(P["mu"] - 2.0) < 0.2,
                    f"mu_hat: linear={L['mu']:.2f}, poly2={P['mu']:.2f}; truth 2.0"))

    print("\n--- PATP alpha-sweep (Exp 3, pattern N) ---")
    g, t = e3["gaussian"], e3["student_t_df4"]
    ok.append(check("G5 Gaussian: alpha*~1 (x^2 exact -> fixed basis suffices)",
                    g["alpha_star"] >= 0.9,
                    f"alpha*_gauss = {g['alpha_star']:.2f} (boundary, as expected)"))
    gain = (t["mse_v"][-1] - t["mse_at_star"]) / t["mse_v"][-1]
    ok.append(check("G6 heavy tail: alpha* INTERIOR and beats pure x^2",
                    t["alpha_star"] < 1.0 and gain > 0.10,
                    f"alpha*_t = {t['alpha_star']:.2f}, gain over alpha=1 is {100*gain:.0f}%"))
    max_cond = max(max(g["cond"]), max(t["cond"]))
    ok.append(check("G7 conditioning stable across alpha (no blow-up)",
                    max_cond < 100,
                    f"max cond(F_standardised) over the sweep = {max_cond:.1f}"))

    n = sum(ok)
    print(f"\n=== {n}/{len(ok)} checks passed ===")
    if n != len(ok):
        sys.exit(1)

if __name__ == "__main__":
    main()
