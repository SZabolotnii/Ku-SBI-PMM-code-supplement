# Code supplement — reproducible verification of the calculations

Reproducibility package for the paper **"A Closed-Form, Efficiency-Optimal Score Surrogate
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

Expected tail of the report: `=== RUNNABLE GATES: ALL PASS ===`. Total runtime is under a
minute on a laptop CPU. A stored copy of the report ships as `verification_report.txt`.

## What each script verifies

| Script | Manuscript object | Verifies (with its analytic anchor) |
|---|---|---|
| `run.py` / `verify.py` | §3, §6, §7 | G1–G7 synthetic: linear surrogate blind to a scale score; poly/PATP recover it; PATP α-sweep (Gaussian α\*=1, Student-t interior) |
| `e5_currency_identity.py` | §4 (Thm 1) | T5 currency identity g = ‖proj S‖²/I recovered **moment-free** (Bartlett/Stein, no score); anchors g=0 on {x} and g=1 on {x,x²} for a scale parameter |
| `s5_mse_decomposition.py` | §5 (Thm 2) | MC variance ∝ 1/σ² (log-log slope ≈ −1.9); Gaussian-scale bias cancellation ∂θI+E[s³]=0 |
| `s5prime_closed_forms.py` | §4 (Prop 2) | closed-form g_m(α) (moments) = score-based g_m to <0.0013; θ-invariance; α\* Gaussian 1.0 / t8 0.6 |
| `ablations.py` | §3, §7 | C2 ridge λ vs cond(G) with g_m stable; C3 basis-size m-sweep (g_m↑, cond↑); C6 closed-form vs simulation-based g_m(α) on t8 (worst 0.021); C7 QR orthogonalization leaves g_m invariant (Δ ≈ 2e-15) |
| `mt_rate.py` | §5 (Thm 2) | C4 finite-budget rule: bias O(σ²) + variance κ/(mσ²) ⇒ σ\* ∝ (mT)^(−1/6) (fitted exponent −0.167) |
| `block_gm.py` | §4 (vector parameters) | C5 multi-parameter block efficiency: {x}→diag(1,0), {x,x²}→I₂ (even term lifts the scale block; cross-info 0) |
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
fails while the fractional PATP is best. The cosmological σ₈ benchmark needs a GPU and the
cosmological simulation stack (`jax_cosmo`, `sbi_lens`) and data that are not redistributable;
`phase3_real/cosmo_patch.md` gives the ready one-line patch.

## Contents

```
run_all_gates.py            unified driver -> verification_report.txt
run.py, verify.py           G1-G7 synthetic gates
e5_currency_identity.py     T5 currency identity (moment-free)
s5_mse_decomposition.py     S5 MSE scalings
s5prime_closed_forms.py     closed-form g_m(alpha)
ablations.py                C2/C3/C6/C7 conditioning & basis-design ablations
mt_rate.py                  C4 (mT)^{-1/6} budget rule
block_gm.py                 C5 multi-parameter block efficiency
figs/make_fig_gm.py         Figure 1 generator (+ stored gm_alpha.pdf)
results/results.json        stored synthetic outputs
verification_report.txt     stored unified report
phase3_real/                authors'-code validation (harness + stored results + cosmo patch)
```

## Notes

- Reproducibility: fixed seed 2026 throughout; NumPy is the only dependency for the core gates.
- License: **add a LICENSE before publishing** (MIT is recommended for reproducibility code).
- Companion manuscripts referenced by the paper (Kunchenko-school LSU / oPMM) are cited, not
  included here.
