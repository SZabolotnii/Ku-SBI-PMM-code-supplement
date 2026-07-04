"""
COSMO gate -- paired analysis of the sigma_8 experiment arms.

PRE-REGISTERED criteria (fixed BEFORE any full run; do not alter after seeing data):
  COSMO-1 (direction): mean over paired runs of
      D_r = |sigma8_err(linear, r)| - |sigma8_err(poly2, r)|
    is > 0  (the even-augmented basis is more accurate on the amplitude parameter).
  COSMO-2 (strength): one-sided paired t-statistic on {D_r} >= 1.70.
  [info, ungated]: same paired contrast for patp05/patp07 vs poly2 (fractional
    exponent on heavy-tailed summaries), per-component error table, aggregate
    unbound-space MSE per arm.

An honest FAIL is a publishable boundary result; report the numbers either way.

Usage: python analyze_results.py [--results-dir results]
Writes cosmo_gate_verdict.json next to the run files.
"""
import argparse
import glob
import json
import os

import numpy as np

PARAM_NAMES = ["omega_c", "omega_b", "sigma_8", "h_0", "n_s", "w_0"]
T_THRESHOLD = 1.70   # pre-registered


def load_arm(results_dir, arm):
    recs = {}
    for p in sorted(glob.glob(os.path.join(results_dir, f"{arm}_run*.json"))):
        with open(p) as f:
            d = json.load(f)
        recs[d["run"]] = d
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = ap.parse_args()

    arms = {}
    for arm in ("linear", "poly2", "patp05", "patp07"):
        recs = load_arm(args.results_dir, arm)
        if recs:
            arms[arm] = recs
            print(f"[load] {arm}: {len(recs)} runs")

    if "linear" not in arms or "poly2" not in arms:
        raise SystemExit("[FATAL] need finished 'linear' and 'poly2' arms for the gate.")

    common = sorted(set(arms["linear"]) & set(arms["poly2"]))
    if len(common) < 8:
        raise SystemExit(f"[FATAL] only {len(common)} paired runs; need >= 8 for the t-test.")
    print(f"[pairs] {len(common)} paired runs (common seeds)")

    # per-component table
    print(f"\n{'component':10s}" + "".join(f"{a:>16s}" for a in arms))
    for i, name in enumerate(PARAM_NAMES):
        row = f"{name:10s}"
        for a, recs in arms.items():
            errs = np.array([recs[r]["abs_err_natural"][i] for r in sorted(recs)])
            row += f"  {errs.mean():7.4f}±{errs.std():.4f}"
        print(row)
    print(f"{'mse(unb.)':10s}" + "".join(
        f"  {np.mean([recs[r]['mse_unbound'] for r in sorted(recs)]):7.4f}        "
        for a, recs in arms.items()))

    # pre-registered paired gate: linear vs poly2 on sigma_8
    D = np.array([arms["linear"][r]["sigma8_abs_err"] - arms["poly2"][r]["sigma8_abs_err"]
                  for r in common])
    mean_d, sd_d = float(D.mean()), float(D.std(ddof=1))
    t_stat = mean_d / (sd_d / np.sqrt(len(D))) if sd_d > 0 else float("inf")
    frac_pos = float((D > 0).mean())

    c1 = mean_d > 0
    c2 = t_stat >= T_THRESHOLD
    print(f"\nCOSMO-1 direction : mean paired Delta(sigma8 err, linear-poly2) = "
          f"{mean_d:+.4f}  -> {'PASS' if c1 else 'FAIL'}")
    print(f"COSMO-2 strength  : paired one-sided t = {t_stat:.2f} (thr {T_THRESHOLD}), "
          f"{100*frac_pos:.0f}% runs positive -> {'PASS' if c2 else 'FAIL'}")

    info = {}
    for a in ("patp05", "patp07"):
        if a in arms:
            cm = sorted(set(arms[a]) & set(arms["poly2"]))
            if cm:
                Dp = np.array([arms["poly2"][r]["sigma8_abs_err"] - arms[a][r]["sigma8_abs_err"]
                               for r in cm])
                info[a] = {"n_pairs": len(cm), "mean_delta_vs_poly2": float(Dp.mean()),
                           "sd": float(Dp.std(ddof=1))}
                print(f"[info] {a} vs poly2 on sigma8: mean Delta = {Dp.mean():+.4f} "
                      f"± {Dp.std(ddof=1):.4f} over {len(cm)} pairs (ungated)")

    verdict = {
        "n_pairs": len(common),
        "COSMO_1_direction": {"mean_paired_delta": mean_d, "pass": bool(c1)},
        "COSMO_2_strength": {"t_stat": t_stat, "threshold": T_THRESHOLD,
                             "frac_positive": frac_pos, "pass": bool(c2)},
        "patp_info": info,
        "all_pass": bool(c1 and c2),
    }
    out = os.path.join(args.results_dir, "cosmo_gate_verdict.json")
    with open(out, "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"\n=> COSMO GATE {'PASS' if verdict['all_pass'] else 'FAIL'}   [written] {out}")


if __name__ == "__main__":
    main()
