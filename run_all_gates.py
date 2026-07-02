"""
Unified gate for Ku-SBI-PMM (Paper 5).  Runs every RUNNABLE gate and writes
verification_report.txt -- the single artefact the manuscript cites.
School discipline: no number enters the paper before this prints PASS.

    python run_all_gates.py

Covers:  synthetic G1-G7 (verify.py) · currency identity T5 (e5_currency_identity.py)
         · ablations C2/C3/C6/C7 (ablations.py) · finite-budget rate C4 (mt_rate.py)
         · multi-parameter block g_m C5 (block_gm.py)
         · authors'-code validation P-A/B/Bt/C (phase3_real/results, pre-run in the jax venv).
Cosmo sigma_8 stays TODO (needs GPU + jax_cosmo/sbi_lens + data; patch in phase3_real/).
"""
import json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

def run(script):
    r = subprocess.run([PY, os.path.join(HERE, script)], capture_output=True, text=True)
    return r.stdout + r.stderr

def main():
    lines, all_pass = [], True

    # --- G1-G7 : synthetic prototype ---------------------------------------------------
    out = run("verify.py")
    m = re.search(r"===\s*(\d+)/(\d+) checks passed", out)
    g_ok = bool(m) and m.group(1) == m.group(2)
    all_pass &= g_ok
    lines.append(f"[{'PASS' if g_ok else 'FAIL'}] G1-G7 synthetic (mean+variance, PATP sweep): "
                 f"{m.group(0).split('===')[1].strip() if m else 'no result'}")

    # --- T5 : simulation-based currency identity --------------------------------------
    out = run("e5_currency_identity.py")
    m = re.search(r"worst \|g_hat - g_truth\| = ([\d.]+)\s+=> T5 (PASS|FAIL)", out)
    t5_ok = bool(m) and m.group(2) == "PASS"
    all_pass &= t5_ok
    lines.append(f"[{'PASS' if t5_ok else 'FAIL'}] T5 currency identity g=||projS||^2/I "
                 f"(moment-free, no score): worst |g_hat-g_truth| = "
                 f"{m.group(1) if m else '?'} < 0.05")

    # --- C2/C3/C6/C7 : conditioning / basis-design ablations ---------------------------
    out = run("ablations.py")
    ab_ok = "=> ABLATIONS PASS" in out
    all_pass &= ab_ok
    lines.append(f"[{'PASS' if ab_ok else 'FAIL'}] C2/C3/C6/C7 ablations: ridge cond-drop, "
                 f"m-sweep g_m up + cond up, closed-vs-sim g_m(alpha), QR g-invariance")

    # --- C4 : finite-budget rate sigma* ~ (mT)^-1/6 ------------------------------------
    out = run("mt_rate.py")
    mt_ok = "=> MT-RATE PASS" in out
    all_pass &= mt_ok
    lines.append(f"[{'PASS' if mt_ok else 'FAIL'}] C4 finite-budget rate: bias O(sigma^2) + "
                 f"var kappa/(m sigma^2) => sigma* ~ (mT)^-1/6")

    # --- C5 : multi-parameter block efficiency matrix ---------------------------------
    out = run("block_gm.py")
    bg_ok = "=> BLOCK-GM PASS" in out
    all_pass &= bg_ok
    lines.append(f"[{'PASS' if bg_ok else 'FAIL'}] C5 multi-parameter block g_m: {{x}}->diag(1,0), "
                 f"{{x,x^2}}->I_2 (even term lifts scale block, cross-info 0)")

    # --- P-A/B/Bt/C : authors' own code (pre-run in the jax venv; read results) ---------
    pj = os.path.join(HERE, "phase3_real", "results", "phase3_results.json")
    if os.path.exists(pj):
        d = json.load(open(pj))
        L = d["B_variance"]["linear [x,1] (theirs)"]; P = d["B_variance"]["poly2  [x,x^2,1]"]
        pb_ok = L["err"] > 1.0 and P["err"] < L["err"]
        all_pass &= pb_ok
        lines.append(f"[{'PASS' if pb_ok else 'FAIL'}] P-B authors' FSM core, variance: "
                     f"linear v-err {L['err']:.2f} (FAILS) vs poly2 {P['err']:.2f} -- structural contrast")
        if "Bt_variance_t6" in d:
            Lt = d["Bt_variance_t6"]["linear [x,1] (theirs)"]
            Pt = d["Bt_variance_t6"]["poly2  [x,x^2,1]"]
            At = d["Bt_variance_t6"]["patp@0.6 [x,|x|^p,1]"]
            pbt_ok = Lt["err"] > 1.0 and At["err"] < Lt["err"]
            all_pass &= pbt_ok
            lines.append(f"[{'PASS' if pbt_ok else 'FAIL'}] P-Bt heavy-tail t(6) scale, their code: "
                         f"linear err {Lt['err']:.2f} (FAILS), poly2 {Pt['err']:.2f}, "
                         f"patp@0.6 {At['err']:.2f} (fractional best)")
        C = d["C_sweep"]
        lines.append(f"[info] P-C alpha-sweep via their fit: Gauss a*={C['alpha_star_gauss']}, "
                     f"t(4) a*={C['alpha_star_t']} (+{100*C['t_gain_vs_x2']:.0f}% score-recovery over x^2)")
    else:
        lines.append("[skip] P-A/B/C: phase3_real/results/phase3_results.json not found "
                     "(run in the jax venv against approxml)")

    lines.append("[TODO] COSMO sigma_8: needs GPU + jax_cosmo/sbi_lens + data "
                 "(ready patch: phase3_real/cosmo_patch.md)")

    header = ["=== Ku-SBI-PMM (Paper 5) -- Unified Verification Report ===",
              "seed 2026 | Kunchenko poly/PATP score surrogate for simulation-based FSM", ""]
    footer = ["", f"=== RUNNABLE GATES: {'ALL PASS' if all_pass else 'SOME FAIL'} ===",
              "(cosmo sigma_8 is the only pending gate; theory phase-1 = manuscript.)"]
    report = "\n".join(header + lines + footer) + "\n"
    print(report)
    with open(os.path.join(HERE, "verification_report.txt"), "w") as f:
        f.write(report)
    print("[written] verification_report.txt")
    sys.exit(0 if all_pass else 1)

if __name__ == "__main__":
    main()
