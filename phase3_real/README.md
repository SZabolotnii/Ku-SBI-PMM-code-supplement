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

## Cosmological σ₈

Not runnable without a GPU and the cosmological stack (`jax_cosmo`, `sbi_lens`) plus
non-redistributable data. `cosmo_patch.md` contains the ready one-line patch to the authors'
`cosmo_sm_n_obs.py` for a machine with the full stack; σ₈ is an amplitude (scale) parameter,
so the even PATP features apply.
