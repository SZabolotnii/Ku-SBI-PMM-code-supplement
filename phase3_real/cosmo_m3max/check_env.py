"""
COSMO gate -- environment / data / timing probe for the M3 Max box.
Run BEFORE anything else:  python check_env.py [--data-dir data]

Checks: (1) all imports + versions; (2) jax backend is CPU; (3) the three data
files exist (prints sha256 for provenance); (4) one compiled simulator call and
one compiled SGD update, with a wall-clock extrapolation for the full protocol.
Exit code 0 = READY.
"""
import argparse
import hashlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    args = ap.parse_args()
    ok = True

    print("== imports ==")
    mods = ["numpy", "jax", "jaxlib", "optax", "haiku", "jax_cosmo",
            "tensorflow_probability", "sklearn"]
    for m in mods:
        try:
            mod = __import__(m)
            print(f"  [ok] {m:26s} {getattr(mod, '__version__', '?')}")
        except Exception as e:
            print(f"  [MISSING] {m}: {e}")
            ok = False
    if not ok:
        sys.exit("[FATAL] install missing packages (see requirements-cosmo.txt / README_UA.md)")

    import jax
    print(f"\n== jax backend ==\n  default_backend={jax.default_backend()}  devices={jax.devices()}")
    if jax.default_backend() != "cpu":
        print("  [WARN] the pre-registered protocol is CPU-only; unset jax-metal for final runs.")

    print("\n== data files ==")
    files = ["lognormal_shifts_LSSTY10_om_s8_w_bin.npy",
             "opt_state_resnet_vmim.pkl", "params_nd_compressor_vmim.pkl"]
    for fn in files:
        p = os.path.join(args.data_dir, fn)
        if os.path.exists(p):
            print(f"  [ok] {fn}  ({os.path.getsize(p)/1e6:.1f} MB)  sha256={sha256(p)[:16]}…")
        else:
            print(f"  [MISSING] {p}")
            ok = False
    if not ok:
        sys.exit("[FATAL] fetch the data first: ./fetch_data.sh  (see README_UA.md §3)")

    print("\n== timing probe (compiles the simulator + one SGD update) ==")
    sys.path.insert(0, HERE)
    from cosmo_gate_experiment import build, run_sgd, N_ITER  # noqa: E402
    sim_fn, params, theta_init = build(args.data_dir)

    t0 = time.time()
    _ = jax.block_until_ready(sim_fn(jax.random.PRNGKey(1), params, 4))
    t_compile = time.time() - t0
    t0 = time.time()
    _ = jax.block_until_ready(sim_fn(jax.random.PRNGKey(2), params, 4))
    t_sim4 = time.time() - t0
    print(f"  simulator n_sim=4 : compile {t_compile:.1f}s, steady {t_sim4:.2f}s")

    t0 = time.time()
    obs = jax.block_until_ready(sim_fn(jax.random.PRNGKey(3), params, 50))
    t_obs = time.time() - t0
    print(f"  simulator n_sim=50: {t_obs:.1f}s  (observed-data draw)")

    t0 = time.time()
    _theta, _ = run_sgd(jax.random.PRNGKey(4), theta_init, obs, sim_fn, "poly2", 3)
    t_3updates = time.time() - t0
    per_update = t_3updates / 3
    per_run = per_update * N_ITER
    print(f"  SGD update (poly2, incl. compile amortized over 3): {per_update:.1f}s "
          f"-> ~{per_run/60:.0f} min per full {N_ITER}-iter run")
    for plan, runs, arms in (("A  (2 arms x 20)", 20, 2), ("A+ (3 arms x 20)", 20, 3),
                             ("B  (2 arms x 10)", 10, 2)):
        print(f"    plan {plan}: ~{per_run*runs*arms/3600:.1f} h total")

    print("\n== READY ==  next: python cosmo_gate_experiment.py --smoke")


if __name__ == "__main__":
    main()
