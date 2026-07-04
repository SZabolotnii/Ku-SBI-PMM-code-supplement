r"""
R4 -- RECIPE-BLIND alpha* selection from estimated moments, BEFORE the run.

Closes the "how do you pick alpha in practice?" gap.  The S5' closed form makes the
efficiency of the even PATP feature |x|^{p_2(alpha)} a pure MOMENT functional: with
q = p_2(alpha) and standardized moments M(k) = E|u|^k,

    g(alpha)  =  v^2 / (Sigma * I),   v = (q/2) M(q),   Sigma = M(2q) - M(q)^2,

where the Fisher term I does not depend on alpha.  Hence  argmax_alpha g(alpha)  is
computable from moments alone -- no density, no score, no Fisher information.  The
pre-run RECIPE is:

    1. draw a PILOT of n_pilot simulations at theta_0 (before any estimation run),
    2. standardize, estimate M-hat(q) on the alpha grid,
    3. alpha-hat = argmax_alpha  v-hat^2 / Sigma-hat.

This gate runs the recipe on three scale families (Gaussian, Student-t8, Student-t5;
all with the A4 moments finite for q in [1/2, 2]) x N_REP independent pilots, then
scores each pick against the ORACLE curve g_oracle(alpha) computed with the analytic
score on a large fresh sample:

    releff(alpha-hat) = g_oracle(alpha-hat) / max_alpha g_oracle(alpha).

PRE-REGISTERED THRESHOLD (fixed before the first full run):
    R4: min over families and replicates of releff(alpha-hat)  >=  0.90.
Budget knobs (adjustable, documented): n_pilot, N_REP, oracle sample size, grid step.
The g(alpha) curve is flat near its optimum, so the recipe only has to land in the
flat region -- it does not have to identify alpha* exactly; the alpha-hat spread is
logged as ungated info.

Pure numpy, seed 2026, ~30 s.  Writes results/r4_alpha_recipe.json, prints PASS/FAIL.
"""
import json
import os
import numpy as np

SEED = 2026
N_PILOT = 2_000
N_REP = 20
N_ORACLE = 4_000_000
GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)

RELEFF_MIN = 0.90          # pre-registered -- never adjust to make a run pass


def p2(a):
    return 0.5 + 0.5 * a + 1.0 * a ** 2


def sample_std(rng, fam, n):
    """Standardized draw u (Var u = 1) + analytic scale score at theta=1."""
    if fam == "gauss":
        u = rng.standard_normal(n)
        S = 0.5 * (u ** 2 - 1.0)
    else:
        nu = int(fam[1:])
        u = rng.standard_t(nu, n) / np.sqrt(nu / (nu - 2.0))
        c2 = (nu - 2.0) / nu
        uu = u ** 2 / c2
        S = -0.5 + (nu + 1.0) * uu / (2.0 * (nu + uu))
    return u, S


def moment_objective(u, a):
    """Moment-only objective v^2/Sigma (g up to the alpha-free Fisher factor)."""
    q = p2(a)
    au = np.abs(u)
    Mq = float(np.mean(au ** q))
    M2q = float(np.mean(au ** (2 * q)))
    Sig = M2q - Mq ** 2
    v = 0.5 * q * Mq
    return v * v / Sig if Sig > 0 else 0.0


def g_oracle_curve(u, S):
    """g(alpha) on the grid via direct score projection (ground truth)."""
    I = float(np.mean(S ** 2))
    out = []
    for a in GRID:
        q = p2(a)
        phi = np.abs(u) ** q
        phi = phi - phi.mean()
        b = float(np.mean(phi * S))
        G = float(np.mean(phi * phi))
        out.append(b * b / (G * I))
    return np.array(out)


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    print("=" * 78)
    print("R4  -  recipe-blind alpha* from pilot moments vs post-hoc oracle sweep")
    print(f"       n_pilot={N_PILOT}, {N_REP} replicates/family, grid step 0.05")
    print("=" * 78)

    fams = ["gauss", "t8", "t5"]
    res_f, ok = {}, True
    for fam in fams:
        u_big, S_big = sample_std(rng, fam, N_ORACLE)
        g_or = g_oracle_curve(u_big, S_big)
        a_star = float(GRID[np.argmax(g_or)])
        g_max = float(np.max(g_or))

        picks, reles = [], []
        for _ in range(N_REP):
            up, _ = sample_std(rng, fam, N_PILOT)
            up = up / up.std()
            obj = [moment_objective(up, a) for a in GRID]
            a_hat = float(GRID[int(np.argmax(obj))])
            picks.append(a_hat)
            reles.append(float(g_or[int(np.argmax(obj))] / g_max))
        worst = min(reles)
        ok &= worst >= RELEFF_MIN
        res_f[fam] = {"alpha_oracle": a_star, "g_max": g_max,
                      "alpha_hat_picks": picks,
                      "releff_per_rep": reles, "releff_worst": worst,
                      "releff_mean": float(np.mean(reles))}
        print(f"\n  {fam:6s}: oracle alpha*={a_star:.2f} (g*={g_max:.4f}) | recipe "
              f"alpha-hat in [{min(picks):.2f},{max(picks):.2f}] "
              f"median {np.median(picks):.2f}")
        print(f"          releff: worst {worst:.4f}, mean {np.mean(reles):.4f}  "
              f"(threshold >= {RELEFF_MIN})")

    res = {"seed": SEED, "n_pilot": N_PILOT, "n_rep": N_REP,
           "grid": [float(a) for a in GRID], "releff_min": RELEFF_MIN,
           "families": res_f, "all_pass": bool(ok)}
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "results"), exist_ok=True)
    with open(os.path.join(here, "results", "r4_alpha_recipe.json"), "w") as f:
        json.dump(res, f, indent=2)

    print("\n" + "=" * 78)
    print(f"  => R4 {'PASS' if ok else 'FAIL'}  (moment-only pre-run recipe lands in "
          f"the flat region of g(alpha): worst releff >= {RELEFF_MIN} on "
          f"Gaussian/t8/t5)")
    raise SystemExit(0 if ok else 1)
