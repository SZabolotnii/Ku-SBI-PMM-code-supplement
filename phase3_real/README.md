# Phase 3 — validation inside the original FSM code

This checks the paper's thesis not against our reimplementation but against the **authors'
own** machinery: `approxml`, from *Direct Fisher Score Estimation for Likelihood Maximization*
(Khoo, Wang, Liu, Beaumont; NeurIPS 2025).

`phase3_scale_experiment.py` imports their core verbatim and adds **one generalized line**:
their `fit_linear_sm` hardcodes the feature augmentation `concat([x, 1])`; our
`fit_feature_sm` replaces it with `feature_fn(x)`. Everything downstream (Gram, regularized
inverse, weights, prediction) is untouched — which is exactly what proves the closed-form
least-squares solution is agnostic to the feature map.

## How to run

`approxml` is third-party and is **not** included here. Obtain it from the public repository
and put it on the path:

```bash
git clone https://github.com/Shermjj/Direct_FSM
pip install jax jaxlib optax equinox jaxtyping
PYTHONPATH=/path/to/Direct_FSM JAX_PLATFORMS=cpu python phase3_scale_experiment.py
```

## Stored results (`results/phase3_results.json`)

| block | test | result |
|---|---|---|
| A (sanity) | estimate a **mean** with their linear surrogate | 2.99 (truth 3.0) ✓ harness faithful |
| B (scale) | estimate a **variance** — linear vs poly2 vs PATP | linear **1.46** ✗ fails; poly2 **3.30**; PATP **3.29** |
| B-t (heavy tail) | estimate a Student-t(6) variance end-to-end | linear **1.47** ✗ fails; poly2 **3.29**; **PATP@0.6 3.35** (smaller error) |
| C (α-sweep) | scale-score recovery | Gaussian α\*=1.0 (x² exact); Student-t(4) α\*=0.0, **+91%** over pure x² |

The residual downward bias in B/B-t is the documented Gaussian-smoothing + strong Tikhonov
(λ=1.0) effect; the **structural contrast** (linear fails, polynomial/PATP recover) is the
point, and it is stable across seeds and jax versions.

## Cosmological σ₈ — run and PASSED (CPU-only, Apple M3 Max, 2026-07-03)

The full LSST Y10 weak-lensing benchmark was run with the `cosmo_m3max/` package (driver,
environment probe, data fetcher, pre-registered analysis — see its `README_UA.md`): 3 arms
(linear / poly2 / patp05) × 20 paired runs on common seeds `PRNGKey(2026000+r)`,
n_obs = 250, the authors' hyperparameters unchanged, CPU-only jax (no GPU needed;
8.1 h wall-clock total). The three data artefacts are third-party
(`DifferentiableUniverseInitiative/sbi_lens`) and are **not redistributed** here — fetch
them with `cosmo_m3max/fetch_data.sh`; their sha256 hashes are recorded in
`cosmo_m3max/results/` provenance and the repository history.

**Pre-registered verdict (thresholds fixed before any full run): PASS.**
COSMO-1 (direction): mean paired Δ|σ̂₈−σ₈| (linear−poly2) = **+0.0034 > 0**.
COSMO-2 (strength): one-sided paired **t = 2.41 ≥ 1.70** (75% of pairs positive).
σ₈ error: linear 0.0130 → poly2 **0.0096 (−26%)**; the aggregate six-parameter MSE is
unchanged (0.449 both arms) — the even feature buys amplitude accuracy at no cost
elsewhere. patp05 matches poly2 (Δ = +0.0000 ± 0.0043): the learned compressor output is
close enough to Gaussian that the integer basis is already optimal, so the gain comes from
**parity**, not tail adaptation. Per-run records and the verdict live in
`cosmo_m3max/results/*.json`; `cosmo_m3max/analyze_results.py` recomputes the verdict.

`cosmo_patch.md` (the original one-line patch to the authors' `cosmo_sm_n_obs.py`) is kept
for reference; the full protocol is the `cosmo_m3max/` package.
