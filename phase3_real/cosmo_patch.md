# Патч cosmo σ_8 під poly/PATP score-сурогат

Готова зміна FSM-драйвера LSST weak lensing — локально:
`../camera_ready_nips_2025_direct_fsm/experimental_results_CR/4_cosmo/cosmo_sm_n_obs.py`.
**Запускати на GPU-боксі з повним стеком** (`jax[cuda12]`, `jax_cosmo`, `sbi_lens`,
`tensorflow_probability`) **і даними** (`../data/lognormal_shifts_LSSTY10_*.npy`,
`../data/params_compressor/*resnet*vmim*.pkl`). У вашій camera-ready копії цих даних і пакетів
**немає** (перевірено) — тому це ready-to-run артефакт, не прогін.

**Чому має спрацювати саме тут:** параметри = `[Ω_m, Ω_b, σ_8, h, n_s, w]`; **σ_8 — параметр №2,
амплітуда спектра потужності = масштабний**. Карти конвергенції негаусові (log-normal). Отже
score по σ_8 має парну компоненту по даних, якої лінійний сурогат $[x,1]$ не бачить — рівно як у
синтетиці фази 3. Працюють **парні** фічі $|x_i|^{p_2(\alpha)}$.

## Зміни (3 місця)

**1. Додати фічемапу + узагальнений fit** (вгорі файлу, після імпортів). 6-вим. summary →
покоординатні парні+непарні степені (13 фіч; без кросів — дешево й цілить у масштаб):

```python
from approxml.utils import gen_simulation_samples, grad_log_normal
import jax, jax.numpy as jnp

PATP_ALPHA = 1.0   # 1.0 = [x, x^2, 1] (poly2);  <1 = дробовий парний |x|^{p2(a)} для важких хвостів
def _p2(a): return 0.5 + 0.5*a + 1.0*a**2           # p_2(alpha): 1/2 -> 2
def feature_fn(x):                                   # x: (..., 6) -> (..., 13)
    p = _p2(PATP_ALPHA)
    return jnp.concatenate([x, jnp.abs(x)**p, jnp.ones_like(x[..., :1])], axis=-1)

def fit_feature_sm(key, theta_t, gen_sim_fn, grad_log_prop_fn, n_prop, n_sim_dst,
                   feature_fn, lamb=1e-3, thetas_q=None, sims_q=None):
    theta_dim = theta_t.shape[0]
    if thetas_q is None and sims_q is None and gen_sim_fn is not None:
        thetas_q, sims_q, _ = gen_sim_fn(key, theta_t)
    grad_log_q_1 = jax.vmap(grad_log_prop_fn, in_axes=(0, None))(thetas_q, theta_t)
    grad_log_q_2 = jnp.repeat(grad_log_q_1, n_sim_dst, axis=0).reshape(n_prop, n_sim_dst, theta_dim)
    sims_q_aug = feature_fn(sims_q)                  # <<< the only change vs fit_linear_sm
    G_j = jax.vmap(lambda x: x.T @ x, in_axes=0)(sims_q_aug).sum(0)
    reg_term = lamb * jnp.eye(G_j.shape[0], M=G_j.shape[1])
    W = - jnp.linalg.inv(G_j + reg_term) @ jax.vmap(
        jax.vmap(jnp.outer, in_axes=(0, 0)), in_axes=(0, 0))(sims_q_aug, grad_log_q_2).sum(0).sum(0)
    return W, sims_q, sims_q_aug, thetas_q
```

**2. У `run_sgd`** — замінити augmentation спостережень:

```python
# було:  obs_aug = jnp.concatenate([obs, jnp.ones_like(obs[..., :1])], axis=-1)
obs_aug = feature_fn(obs)
```

**3. У closure `update`** — замінити fit (решта рядка `grads = jnp.einsum('mk,ik->im', W.T, obs_aug)...`
лишається без змін, бо `obs_aug` уже у новій фічемапі, а `W` має узгоджену розмірність):

```python
# було:  current_grad_fn = partial(fit_linear_sm, gen_sim_fn=..., grad_log_prop_fn=..., n_sim_dst=..., n_prop=..., lamb=1.0)
current_grad_fn = partial(fit_feature_sm,
    gen_sim_fn=current_gen_sim_fn,
    grad_log_prop_fn=partial(grad_log_normal, cov=current_prop_cov),
    n_sim_dst=N_SIM_DST, n_prop=N_PROP, feature_fn=feature_fn, lamb=1.0)
```

## Протокол порівняння (gate фази 3-real)

Прогнати їхній pipeline **тричі** при тому самому симуляційному бюджеті:
- `PATP_ALPHA` нерелевантний → baseline = оригінал (`fit_linear_sm`, $[x,1]$);
- `feature_fn` poly2 (`PATP_ALPHA=1.0`);
- `feature_fn` PATP з $\alpha\in\{0.3,0.5,0.7\}$ (крос-валідація як у їхньому `cross_val_sm`).

Метрика — їхня ж: `mse = ||theta_hat - params||` (вони друкують її), окремо по компоненті **σ_8**.
**Очікування / gate:** poly/PATP $\le$ linear за MSE(σ_8) при рівному бюджеті; на негаусових
картах ефект найсильніший саме на амплітудних (σ_8, w) компонентах. Звітувати MSE покомпонентно,
а не лише сумарну норму (сумарна може маскувати виграш на σ_8).

## Застороги
- 13 фіч на 6-вим. summary — додати Тихонова (вони вже мають `lamb=1.0`); за потреби кросів —
  `vech(x xᵀ)` дає 28 фіч, але гірша обумовленість.
- Якщо хвости summary важкі — `PATP_ALPHA<1` (нижчі моменти, стабільніша Gram-матриця).
- Зберегти їхній seed/розбивку, міняти **лише** фічемапу — інакше порівняння нечесне.
