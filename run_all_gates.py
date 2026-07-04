"""
Unified gate for Ku-SBI-PMM (Paper 5).  Runs every RUNNABLE gate and writes
verification_report.txt -- the single artefact the manuscript cites.
School discipline: no number enters the paper before this prints PASS.

    python run_all_gates.py

Covers:  synthetic G1-G7 (verify.py) · currency identity T5 (e5_currency_identity.py)
         · PATP domain D1/D2 (patp_domain_gates.py)
         · ablations C2/C3/C6/C7 (ablations.py) · finite-budget rate C4/S5
         · multi-parameter block g_m C5 (block_gm.py)
         · SGLD calibration B1-B3 (sgld_calibration.py)
         · SBI benchmarks R1-R3 (r1_gandk_gate.py, r2_mg1_gate.py, r3_ou_gate.py)
         · authors'-code validation P-A/B/Bt/C (phase3_real/results, pre-run in the jax venv).
Cosmo sigma_8 (COSMO-1/2) is read from phase3_real/cosmo_m3max/results/ when present.
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

    # --- D1/D2 : PATP schedule scope and infinite-fourth-moment edge -------------------
    out = run("patp_domain_gates.py")
    d_ok = "=> PATP-DOMAIN PASS" in out
    all_pass &= d_ok
    m = re.search(r"phase3 t4 alpha\*=([^,]+), gain over x2=([^\n]+)", out)
    phase3 = f"; {m.group(0)}" if m else ""
    lines.append(f"[{'PASS' if d_ok else 'FAIL'}] D1/D2 PATP domain: p2/p3 scoped; "
                 f"high-order counterexamples documented; Student-t4 integer edge degenerates"
                 f"{phase3}")

    # --- C2/C3/C6/C7 : conditioning / basis-design ablations ---------------------------
    out = run("ablations.py")
    ab_ok = "=> ABLATIONS PASS" in out
    all_pass &= ab_ok
    lines.append(f"[{'PASS' if ab_ok else 'FAIL'}] C2/C3/C6/C7 ablations: ridge cond-drop, "
                 f"m-sweep g_m up + cond up, closed-vs-sim g_m(alpha), QR g-invariance")

    # --- C4 : finite-budget rate sigma* ~ (JT)^-1/6 ------------------------------------
    out = run("mt_rate.py")
    mt_ok = "=> MT-RATE PASS" in out
    all_pass &= mt_ok
    lines.append(f"[{'PASS' if mt_ok else 'FAIL'}] C4 finite-budget rate: bias O(sigma^2) + "
                 f"var kappa/(J sigma^2) => sigma* ~ (JT)^-1/6")

    # --- S5 : MSE decomposition component scalings -------------------------------------
    out = run("s5_mse_decomposition.py")
    s5_ok = "=> S5-MSE PASS" in out
    all_pass &= s5_ok
    lines.append(f"[{'PASS' if s5_ok else 'FAIL'}] S5 MSE decomposition: "
                 f"approximation term + smoothing bias + MC variance wired")

    # --- C5 : multi-parameter block efficiency matrix ---------------------------------
    out = run("block_gm.py")
    bg_ok = "=> BLOCK-GM PASS" in out
    all_pass &= bg_ok
    lines.append(f"[{'PASS' if bg_ok else 'FAIL'}] C5 multi-parameter block g_m: {{x}}->diag(1,0), "
                 f"{{x,x^2}}->I_2 (even term lifts scale block, cross-info 0)")

    # --- B1-B3 : SGLD surrogate-posterior calibration (backs the hedged Discussion
    # direction: Cor. H=J=I*g exact, coverage/width empirically calibrated) -------------
    out = run("sgld_calibration.py")
    sg_ok = "ALL PASS" in out
    all_pass &= sg_ok
    sj = os.path.join(HERE, "results", "sgld_calibration.json")
    sg_detail = ""
    if sg_ok and os.path.exists(sj):
        d = json.load(open(sj))
        worst_b1 = max(max(v["relerr_H"], v["relerr_J"]) for v in d["B1"].values())
        sg_detail = (f": H=J=I*g worst relerr {worst_b1:.4f}; coverage B2 "
                     f"{d['B2']['coverage']:.2f} (width x{d['B2']['width_ratio']:.2f} ~ sqrt2), "
                     f"B3 {d['B3']['coverage']:.2f}")
    lines.append(f"[{'PASS' if sg_ok else 'FAIL'}] B1-B3 SGLD surrogate-posterior calibration "
                 f"(information-matrix equality + coverage){sg_detail}")

    # --- R1-R3 : standard SBI benchmarks (g-and-k / M/G/1 / OU-AR(1) vs exact MLE) ------
    out = run("r1_gandk_gate.py")
    r1_ok = "=> R1-GANDK PASS" in out
    all_pass &= r1_ok
    r1j = os.path.join(HERE, "results", "r1_gandk.json")
    d1 = json.load(open(r1j)) if os.path.exists(r1j) else {}
    lines.append(f"[{'PASS' if r1_ok else 'FAIL'}] R1 g-and-k scale: linear blind "
                 f"|B-2|={d1.get('g0_linear', {}).get('abs_err', float('nan')):.2f}, "
                 f"even-PATP repairs {d1.get('g0_patp', {}).get('abs_err', float('nan')):.3f}, "
                 f"mixed under skew {d1.get('g05_mixed', {}).get('abs_err', float('nan')):.3f}")

    out = run("r2_mg1_gate.py")
    r2_ok = "=> R2-MG1 PASS" in out
    all_pass &= r2_ok
    r2j = os.path.join(HERE, "results", "r2_mg1.json")
    d2 = json.load(open(r2j)) if os.path.exists(r2j) else {}
    p2b = d2.get("patp_basis", {}); l2b = d2.get("linear_basis_info", {})
    lines.append(f"[{'PASS' if r2_ok else 'FAIL'}] R2 M/G/1 quantile summaries: PATP rel err "
                 f"theta1 {p2b.get('rel_err_theta1', float('nan')):.2f} / theta2 "
                 f"{p2b.get('rel_err_theta2', float('nan')):.2f}; linear theta2 "
                 f"{l2b.get('rel_err_theta2', float('nan')):.2f} (unstable)")

    out = run("r3_ou_gate.py")
    r3_ok = "=> R3-OU PASS" in out
    all_pass &= r3_ok
    r3j = os.path.join(HERE, "results", "r3_ou.json")
    d3 = json.load(open(r3j)) if os.path.exists(r3j) else {}
    c3 = d3.get("criteria", {})
    lines.append(f"[{'PASS' if r3_ok else 'FAIL'}] R3 OU/AR(1) vs exact MLE: "
                 f"Var(FSM)/Var(MLE)={c3.get('R3b', {}).get('measured', float('nan')):.3f} "
                 f"(ARE=g_m=1 end-to-end); mean-only summary blind, err "
                 f"{c3.get('R3c', {}).get('measured', float('nan')):.2f}")

    # --- V1 : multivariate coordinate-wise construction, d = 2..6 ------------------------
    out = run("v1_multivariate_gate.py")
    v1_ok = "=> V1 PASS" in out
    all_pass &= v1_ok
    v1j = os.path.join(HERE, "results", "v1_multivariate.json")
    dv = json.load(open(v1j)) if os.path.exists(v1j) else {}
    cond6 = next((r["cond_G_standardized"] for r in dv.get("per_d", []) if r["d"] == 6),
                 float("nan"))
    v1d = dv.get("V1d", {})
    lines.append(f"[{'PASS' if v1_ok else 'FAIL'}] V1 multivariate d=2..6: coordinate-wise "
                 f"odd->diag(1,0) / mixed->I_2d; cond(G) std {cond6:.2f} at d=6 (flat); "
                 f"cross param needs cross term: g {v1d.get('rho0_nocross', float('nan')):.3f}"
                 f"->{v1d.get('rho0_cross', float('nan')):.3f}")

    # --- R4 : recipe-blind alpha* from pilot moments (pre-run selection) -----------------
    out = run("r4_alpha_recipe_gate.py")
    r4_ok = "=> R4 PASS" in out
    all_pass &= r4_ok
    r4j = os.path.join(HERE, "results", "r4_alpha_recipe.json")
    d4 = json.load(open(r4j)) if os.path.exists(r4j) else {}
    f4 = d4.get("families", {})
    worst4 = min((v["releff_worst"] for v in f4.values()), default=float("nan"))
    med = {k: (sorted(v["alpha_hat_picks"])[len(v["alpha_hat_picks"]) // 2])
           for k, v in f4.items()}
    lines.append(f"[{'PASS' if r4_ok else 'FAIL'}] R4 recipe-blind alpha* (pilot moments, "
                 f"no score): median picks gauss {med.get('gauss', float('nan')):.2f} / t8 "
                 f"{med.get('t8', float('nan')):.2f} / t5 {med.get('t5', float('nan')):.2f} "
                 f"track oracle; worst releff {worst4:.3f} >= 0.90")

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

    # --- COSMO-1/2 : sigma_8 on the authors' weak-lensing benchmark (pre-run on the
    # M3 Max box via phase3_real/cosmo_m3max/; read the pre-registered verdict) ----------
    cj = os.path.join(HERE, "phase3_real", "cosmo_m3max", "results", "cosmo_gate_verdict.json")
    if os.path.exists(cj):
        v = json.load(open(cj))
        c_ok = v["all_pass"]
        all_pass &= c_ok
        lines.append(f"[{'PASS' if c_ok else 'FAIL'}] COSMO-1/2 sigma_8 (LSST Y10, authors' "
                     f"pipeline, {v['n_pairs']} paired runs): paired Delta="
                     f"{v['COSMO_1_direction']['mean_paired_delta']:+.4f}, one-sided t="
                     f"{v['COSMO_2_strength']['t_stat']:.2f} (thr "
                     f"{v['COSMO_2_strength']['threshold']}), "
                     f"{100*v['COSMO_2_strength']['frac_positive']:.0f}% pairs positive")
    else:
        lines.append("[skip] COSMO sigma_8: ready-to-run package phase3_real/cosmo_m3max/ "
                     "(paired linear-vs-poly2/PATP protocol, pre-registered COSMO-1/2; "
                     "needs data + jax_cosmo/haiku/tfp deps -- see its README_UA.md)")

    header = ["=== Ku-SBI-PMM (Paper 5) -- Unified Verification Report ===",
              "seed 2026 | Kunchenko poly/PATP score surrogate for simulation-based FSM", ""]
    footer = ["", f"=== RUNNABLE GATES: {'ALL PASS' if all_pass else 'SOME FAIL'} ===",
              "(theory phase-1 = manuscript.)"]
    report = "\n".join(header + lines + footer) + "\n"
    print(report)
    with open(os.path.join(HERE, "verification_report.txt"), "w") as f:
        f.write(report)
    print("[written] verification_report.txt")
    sys.exit(0 if all_pass else 1)

if __name__ == "__main__":
    main()
