# Code supplement — reproducible verification of the calculations

Reproducibility package for the paper **"An Efficiency-Quantified Score Surrogate
for Simulation-Based Inference via Kunchenko Stochastic Polynomials."**

Every numerical claim in the manuscript is produced by a small, self-contained gate script
with a **fixed seed (2026)** and an **analytic anchor**: each script recomputes a quantity
and checks it against a value known in closed form, so the reader can confirm the numbers
independently rather than trust them. The core scripts depend only on **NumPy**.

## Quick start

```bash
pip install -r requirements.txt      # numpy only, for the core gates
python run_all_gates.py              # runs every core gate, writes verification_report.txt
```

Expected tail of the report: `=== RUNNABLE GATES: ALL PASS ===`. Total runtime is around one
to two minutes on a laptop CPU. A stored copy of the report ships as `verification_report.txt`.

## What each script verifies

| Script | Manuscript object | Verifies (with its analytic anchor) |
|---|---|---|
| `run.py` / `verify.py` | §3, §6, §7 | G1–G7 synthetic: linear surrogate blind to a scale score; poly/PATP recover it; PATP α-sweep (Gaussian α\*=1, Student-t interior) |
| `e5_currency_identity.py` | §4 (Thm 1) | T5 currency identity g = ‖proj S‖²/I recovered **moment-free** (Bartlett/Stein, no score); anchors g=0 on {x} and g=1 on {x,x²} for a scale parameter |
| `patp_domain_gates.py` | §3, §4 | D1/D2 PATP domain checks: p2/p3 positivity/monotonicity only; high-order counterexamples documented; Student-t4 infinite-fourth-moment edge |
| `s5_mse_decomposition.py` | §5 (Prop finite-budget MSE) | MC variance ∝ 1/σ² (log-log slope ≈ −1.9); Gaussian-scale bias cancellation ∂θI+E[s³]=0 |
| `s5prime_closed_forms.py` | §4 (Prop 2) | closed-form g_m(α) (moments) = score-based g_m to <0.0013; θ-invariance; α\* Gaussian 1.0 / t8 0.6 |
| `ablations.py` | §3, §7 | C2 ridge λ vs cond(G) with g_m stable; C3 basis-size m-sweep (g_m↑, cond↑); C6 closed-form vs simulation-based g_m(α) on t8 (worst 0.021); C7 QR orthogonalization leaves g_m invariant (Δ ≈ 2e-15) |
| `mt_rate.py` | §5 (Prop finite-budget MSE) | C4 finite-budget rule: bias O(σ²) + variance κ/(Jσ²) ⇒ σ\* ∝ (JT)^(−1/6) (fitted exponent −0.167) |
| `block_gm.py` | §4 (vector parameters) | C5 multi-parameter block efficiency: {x}→diag(1,0), {x,x²}→I₂ (even term lifts the scale block; cross-info 0) |
| `sgld_calibration.py` | §8 (Bayesian direction) | B1–B3 SGLD surrogate posterior: information-matrix equality H=J=I·g on three analytic anchors; frequentist coverage and g^(−1/2) width inflation of the generalized posterior |
| `r1_gandk_gate.py` | §7 (benchmarks) | R1 g-and-k scale: linear surrogate blind (mean err 0.80), one even PATP feature repairs (0.09); mixed basis survives skew g=0.5 (0.08) |
| `r2_mg1_gate.py` | §7 (benchmarks) | R2 M/G/1 queue, 6-dim quantile summaries, non-smooth simulator: PATP rel. err 0.13/0.19; linear basis unstable on θ₂ (0.40) |
| `r3_ou_gate.py` | §7 (benchmarks) | R3 OU/AR(1) vs the exact conditional MLE: variance ratio 0.98 (end-to-end ARE=g_m=1); mean-only summary blind (err 0.31) |
| `v1_multivariate_gate.py` | §3 (multivariate), §7 | V1 coordinate-wise construction d=2..6: odd basis → diag(1,0) blocks, mixed → I_2d; cond(G)≈2 flat after per-coordinate standardization; cross parameter (bivariate ρ) exactly blind without / fully captured with the x₁x₂ feature |
| `r4_alpha_recipe_gate.py` | §5 (recipe), §7 | R4 recipe-blind α\* from a 2,000-sample pilot (standardized moments only, no score): median picks track the oracle on Gaussian/t8/t5; worst-replicate oracle efficiency ≥ 0.94 |
| `figs/make_fig_gm.py` | Figure 1 | regenerates `figs/gm_alpha.pdf` (captured-Fisher g_m(α), Gaussian vs Student-t8) |

`run_all_gates.py` runs the core (NumPy) gates live and reads the **stored** phase-3 results
(next section) into one report.

## Phase 3 — validation inside the original FSM implementation

`phase3_real/` checks the thesis against the **authors' own** score-matching code
(`approxml`, from *Direct Fisher Score Estimation for Likelihood Maximization*, Khoo et al.,
NeurIPS 2025). Our `fit_feature_sm` is their `fit_linear_sm` with a **single line changed**
(the feature augmentation), proving the closed form is agnostic to the feature map. Because
`approxml` is third-party, it is **not** vendored here; obtain it from the public repository
`github.com/Shermjj/Direct_FSM` and run:

```bash
pip install jax jaxlib optax equinox jaxtyping
PYTHONPATH=/path/to/Direct_FSM JAX_PLATFORMS=cpu python phase3_real/phase3_scale_experiment.py
```

Stored outputs are in `phase3_real/results/phase3_results.json` (reproduced bit-for-bit under
jax 0.10): a mean is recovered by their linear surrogate (2.99) but a variance is not (1.46);
poly2/PATP recover it (3.30); on a heavy-tailed Student-t(6) scale target the linear surrogate
fails while the fractional PATP is best.

The cosmological σ₈ benchmark (LSST Y10 weak lensing, the authors' own pipeline) **was run**
with the `phase3_real/cosmo_m3max/` package on a CPU-only Apple M3 Max: 3 arms × 20 paired
runs on common seeds, pre-registered COSMO-1/2 verdict **PASS** (σ₈ error −26%, paired
t = 2.41; aggregate 6-parameter MSE unchanged). Per-run records and the verdict are stored in
`phase3_real/cosmo_m3max/results/*.json`; the three data artefacts are third-party
(`DifferentiableUniverseInitiative/sbi_lens`) and are fetched by
`phase3_real/cosmo_m3max/fetch_data.sh`, not redistributed — see that package's README.

## Contents

```
run_all_gates.py            unified driver -> verification_report.txt
run.py, verify.py           G1-G7 synthetic gates
e5_currency_identity.py     T5 currency identity (moment-free)
patp_domain_gates.py        D1/D2 PATP schedule scope + Student-t4 edge
s5_mse_decomposition.py     S5 MSE scalings
s5prime_closed_forms.py     closed-form g_m(alpha)
ablations.py                C2/C3/C6/C7 conditioning & basis-design ablations
mt_rate.py                  C4 (JT)^{-1/6} budget rule
block_gm.py                 C5 multi-parameter block efficiency
sgld_calibration.py         B1-B3 SGLD surrogate-posterior calibration
r1_gandk_gate.py            R1 g-and-k parity blindness + PATP repair
r2_mg1_gate.py              R2 M/G/1 quantile-summary estimation
r3_ou_gate.py               R3 OU/AR(1) vs exact MLE (ARE=g_m=1)
v1_multivariate_gate.py     V1 multivariate coordinate-wise construction d=2..6
r4_alpha_recipe_gate.py     R4 recipe-blind pre-run alpha* from pilot moments
figs/make_fig_gm.py         Figure 1 generator (+ stored gm_alpha.pdf)
results/results.json        stored synthetic outputs
verification_report.txt     stored unified report
phase3_real/                authors'-code validation (harness + stored results + cosmo_m3max sigma_8 package)
```

## Notes

- Reproducibility: fixed seed 2026 throughout; NumPy is the only dependency for the core gates.
- License: MIT, included as `LICENSE`.
- The companion PATP manuscript referenced by the paper (oPMM, arXiv:2605.14610) is cited,
  not included here.
