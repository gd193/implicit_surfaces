"""Per-primitive cost benchmark across surface modes and batch sizes.

For each (surface mode, primitive, batch size) cell:
  * timing: 2 warmup + 5 timed calls of `jax.jit(jax.vmap(primitive))`,
    report best ms and throughput (calls/sec, where one "call" = one
    sample evaluated by the primitive).
  * memory:  `device.memory_stats()['bytes_in_use']` after `block_until_ready`.
    Records the working-set at this batch size; peak is monotonic in JAX
    so we read current after sync.

Three primitives covered today:
  * forward       — `surface(params, x)`  (raw MLP / fn)
  * classify_box  — `surface.classify_box(params, lower, upper)`
  * empty_sphere  — `surface.empty_sphere(params, x, tr)`
  * ray_step      — `surface.ray_step(params, x, dir, tr)`

Network selection mirrors the other benches: `--network <basename>` resolves
both `assets/equinox_models/<name>.eqx` and `assets/niq_models/<name>.npz`.

Usage:
  scripts/benchmark_primitives.py                    # all defaults
  scripts/benchmark_primitives.py --primitives forward classify_box
  scripts/benchmark_primitives.py --batch_sizes 1 1024 1048576 --no-d3
  scripts/benchmark_primitives.py --modes interval affine_all --d2
"""
import argparse, gc, os, sys, time

import numpy as np
# fp64 enable must happen BEFORE first jax import.
if "fp64" in sys.argv:
    os.environ["JAX_ENABLE_X64"] = "True"
import jax
# fp32 matmul (mirror surface modules — relevant for autobound polynomial constant).
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr
_X64 = bool(jax.config.read("jax_enable_x64"))
DTYPE = jnp.float64 if _X64 else jnp.float32

sys.path.insert(0, '/system/user/volkmann/implicit_surfaces_project')

import jnp_bounds.autobounds as jb
from implicitSurfaces.niImplicitSurface import niImplicitSurface
from implicitSurfaces.jnpboundedImplicit import jnpboundedImplicit


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
DEFAULT_MODES = ["interval", "affine_fixed", "slope_interval",
                 "affine_truncate", "affine_all"]
DEFAULT_PRIMITIVES = ["forward", "classify_box", "empty_sphere", "ray_step"]
DEFAULT_BATCH_SIZES = [1, 32, 128, 1024, 8192, 65536, 262144, 1048576]

p = argparse.ArgumentParser()
p.add_argument("--network", default="bunny_softplus")
p.add_argument("--eqx_path", default=None,
               help="explicit eqx path; overrides --network resolution")
p.add_argument("--npz_path", default=None,
               help="explicit niq npz path; overrides --network resolution")
p.add_argument("--activation", default="softplus")
p.add_argument("--modes", nargs="*", default=DEFAULT_MODES,
               help="NIQ modes; pass empty (--modes) to skip NIQ entirely.")
p.add_argument("--d2", action=argparse.BooleanOptionalAction, default=True)
p.add_argument("--d3", action=argparse.BooleanOptionalAction, default=False)
p.add_argument("--ab_modes", nargs="+", default=["default"],
               choices=["default", "fp32", "mixed", "fp64"],
               help="autobound precision_mode list")
p.add_argument("--shape", default="bunny")
p.add_argument("--depth", type=int, default=None)
p.add_argument("--width", type=int, default=None)
p.add_argument("--loss", default="SDF")
p.add_argument("--no_record", action="store_true")
p.add_argument("--primitives", nargs="+", default=DEFAULT_PRIMITIVES,
               choices=["forward", "classify_box", "empty_sphere", "ray_step"])
p.add_argument("--batch_sizes", nargs="+", type=int, default=DEFAULT_BATCH_SIZES)
p.add_argument("--n_warmup", type=int, default=2)
p.add_argument("--n_timed", type=int, default=5)
p.add_argument("--bbox_half", type=float, default=0.2,
               help="cube side / sphere radius / point sampling extent")
p.add_argument("--init_tr", type=float, default=0.1)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--affine_n_truncate", type=int, default=8)
p.add_argument("--affine_truncate_policy", default="absolute")
args = p.parse_args()

ASSETS = "/system/user/volkmann/implicit_surfaces_project/assets"
EQX_PATH = args.eqx_path or f"{ASSETS}/equinox_models/{args.network}.eqx"
NPZ_PATH = args.npz_path or f"{ASSETS}/niq_models/{args.network}.npz"

print(f"JAX: {[d.platform for d in jax.devices()]}", flush=True)
print(f"network={args.network}  primitives={args.primitives}  "
      f"batch_sizes={args.batch_sizes}", flush=True)


# ----------------------------------------------------------------------------
# MLP load (generic)
# ----------------------------------------------------------------------------
def load_eqx_mlp(path, activation_name):
    arrs = []
    with open(path, "rb") as f:
        while True:
            try: arrs.append(np.lib.format.read_array(f))
            except (EOFError, ValueError): break
    weights = [jnp.asarray(a, dtype=DTYPE) for a in arrs[0::2]]
    biases  = [jnp.asarray(a, dtype=DTYPE) for a in arrs[1::2]]
    act = getattr(jax.nn, activation_name)
    def mlp(x):
        h = x
        for i in range(len(weights) - 1):
            h = act(weights[i] @ h + biases[i])
        return weights[-1] @ h + biases[-1]
    return mlp


mlp_fn = load_eqx_mlp(EQX_PATH, args.activation)
def fn_2arg(_p, x): return mlp_fn(x)

# Infer depth/width from the eqx file if not provided.
def _peek_arch(path):
    arrs = []
    with open(path, "rb") as f:
        while True:
            try: arrs.append(np.lib.format.read_array(f))
            except (EOFError, ValueError): break
    weights = arrs[0::2]
    return len(weights) - 1, (weights[0].shape[0] if len(weights) > 1 else weights[-1].shape[0])

if args.depth is None or args.width is None:
    d_, w_ = _peek_arch(EQX_PATH)
    args.depth = args.depth or d_
    args.width = args.width or w_


# ----------------------------------------------------------------------------
# Surface registry
# ----------------------------------------------------------------------------
def make_niq(mode):
    kw = {}
    if mode == "affine_truncate":
        kw["affine_n_truncate"] = args.affine_n_truncate
        kw["affine_truncate_policy"] = args.affine_truncate_policy
    return niImplicitSurface(mode=mode, npz_path=NPZ_PATH, level_set=0.0, dim=3, **kw)

def make_autobound(max_degree, precision_mode=None):
    s = jnpboundedImplicit(autobounds_max_degree=max_degree, fn=fn_2arg,
                            level_set=0.0, dim=3, use_warp=True)
    s.taylor_bounds = jb.taylor_bounds(mlp_fn, max_degree=max_degree, use_warp=True,
                                        precision_mode=precision_mode)
    return s


variants = []
for mode in args.modes:
    label = f"niq:{mode}{f'({args.affine_n_truncate})' if mode=='affine_truncate' else ''}"
    variants.append((label, lambda m=mode: make_niq(m)))
for pm in args.ab_modes:
    pm_arg = None if pm == "default" else pm
    if args.d2:
        variants.append((f"autobound:d2[{pm}]", lambda d=2, p=pm_arg: make_autobound(d, p)))
    if args.d3:
        variants.append((f"autobound:d3[{pm}]", lambda d=3, p=pm_arg: make_autobound(d, p)))


# ----------------------------------------------------------------------------
# Inputs per primitive
# ----------------------------------------------------------------------------
def make_inputs(prim, B, key):
    if prim == "forward":
        x = jr.uniform(key, (B, 3), minval=-args.bbox_half, maxval=args.bbox_half, dtype=DTYPE)
        return (x,)
    if prim == "classify_box":
        c = jr.uniform(key, (B, 3), minval=-args.bbox_half + args.init_tr,
                                     maxval=args.bbox_half - args.init_tr, dtype=DTYPE)
        s = DTYPE(args.init_tr)
        lo = c - s
        hi = c + s
        return (lo, hi)
    if prim == "empty_sphere":
        x = jr.uniform(key, (B, 3), minval=-args.bbox_half, maxval=args.bbox_half, dtype=DTYPE)
        tr = jnp.full((B,), args.init_tr, dtype=DTYPE)
        return (x, tr)
    if prim == "ray_step":
        k1, k2 = jr.split(key)
        x = jr.uniform(k1, (B, 3), minval=-args.bbox_half, maxval=args.bbox_half, dtype=DTYPE)
        d = jr.normal(k2, (B, 3), dtype=DTYPE)
        d = d / (jnp.linalg.norm(d, axis=-1, keepdims=True) + 1e-12)
        tr = jnp.full((B,), args.init_tr, dtype=DTYPE)
        return (x, d, tr)


def make_callable(surface, prim):
    """Build (jit + vmap)'d callable for the chosen primitive."""
    params = getattr(surface, 'params', None)
    if prim == "forward":
        f = jax.vmap(lambda x: surface(params, x))
    elif prim == "classify_box":
        f = jax.vmap(lambda lo, hi: surface.classify_box(params, lo, hi))
    elif prim == "empty_sphere":
        f = jax.vmap(lambda x, tr: surface.empty_sphere(params, x, tr))
    elif prim == "ray_step":
        f = jax.vmap(lambda x, d, tr: surface.ray_step(params, x, d, tr))
    return jax.jit(f)


# ----------------------------------------------------------------------------
# Memory helper
# ----------------------------------------------------------------------------
_dev = jax.devices()[0]
def gpu_bytes_in_use() -> int:
    try:
        return int(_dev.memory_stats().get('bytes_in_use', 0))
    except Exception:
        return -1


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
results = {}   # (variant_label, primitive, B) → dict
key0 = jr.PRNGKey(args.seed)

for variant_label, factory in variants:
    print(f"\n=== variant: {variant_label} ===", flush=True)
    surface = factory()
    for prim in args.primitives:
        # Some primitives might not be defined on a surface — skip.
        method_name = {"forward": "__call__"}.get(prim, prim)
        if not hasattr(surface, method_name) and prim != "forward":
            print(f"  [{prim}]: skipped (no method on surface)")
            continue
        try:
            cfn = make_callable(surface, prim)
        except Exception as e:
            print(f"  [{prim}]: build failed: {type(e).__name__}: {e}")
            continue

        for B in args.batch_sizes:
            try:
                key = jr.fold_in(key0, B)
                inputs = make_inputs(prim, B, key)
                # Warmup
                for _ in range(args.n_warmup):
                    out = cfn(*inputs)
                    jax.block_until_ready(out)
                # Memory snapshot after warmup steady state
                mem_bytes = gpu_bytes_in_use()
                # Timed
                times = []
                for _ in range(args.n_timed):
                    t0 = time.time()
                    out = cfn(*inputs)
                    jax.block_until_ready(out)
                    times.append(time.time() - t0)
                t_min = min(times) * 1000
                t_med = float(np.median(times)) * 1000
                throughput = B / (t_min / 1000)
                results[(variant_label, prim, B)] = dict(
                    ms_min=t_min, ms_med=t_med, throughput=throughput,
                    mem_mb=mem_bytes / (1024 * 1024) if mem_bytes >= 0 else float('nan'),
                )
                print(f"  [{prim:>13}] B={B:>7d}  best={t_min:8.2f} ms  "
                      f"med={t_med:8.2f} ms  thru={throughput:>10.2e}/s  "
                      f"mem={mem_bytes/1024/1024:8.1f} MB", flush=True)
            except Exception as e:
                results[(variant_label, prim, B)] = dict(error=type(e).__name__)
                print(f"  [{prim:>13}] B={B:>7d}  FAIL ({type(e).__name__}: {str(e)[:80]})",
                      flush=True)
                # Try to recover from OOM by clearing caches.
                try:
                    jax.clear_caches()
                    gc.collect()
                except Exception:
                    pass


# ----------------------------------------------------------------------------
# Reporting — three tables (best ms, calls/sec, MB)
# ----------------------------------------------------------------------------
def print_table(title, value_key, fmt):
    print(f"\n{title}")
    header = f"{'variant':<26}  {'primitive':<14}  " + "  ".join(f"{B:>11d}" for B in args.batch_sizes)
    print(header)
    print("-" * len(header))
    for variant_label, _ in variants:
        for prim in args.primitives:
            row = f"{variant_label:<26}  {prim:<14}  "
            cells = []
            for B in args.batch_sizes:
                r = results.get((variant_label, prim, B), {})
                if 'error' in r:
                    cells.append(f"{r['error']:>11s}")
                elif value_key in r:
                    cells.append(fmt.format(r[value_key]))
                else:
                    cells.append(f"{'-':>11s}")
            print(row + "  ".join(cells), flush=True)

print_table("=== best wall time (ms) ===", 'ms_min', "{:>11.2f}")
print_table("=== throughput (samples / sec) ===", 'throughput', "{:>11.2e}")
print_table("=== device memory in use (MB) ===", 'mem_mb', "{:>11.1f}")


# ----------------------------------------------------------------------------
# Record to paper_utils.results / single_query_bound.jsonl (one row per
# variant×primitive — batch_sizes folded into the metrics dict). Keyed by
# (shape, network, variant, algorithm='single_query_bound') so primitive
# benches and quality benches can sit in the same JSONL file and be
# disambiguated by `metrics.primitive`.
# ----------------------------------------------------------------------------
if not args.no_record:
    import re as _re
    sys.path.insert(0, "/system/user/volkmann/implicit_surfaces_project")
    from utils.paper_utils.results import record, make_id
    net = dict(depth=args.depth, width=args.width,
               activation=args.activation, loss=args.loss)
    for variant_label, _ in variants:
        variant = _re.sub(r'[^a-z0-9]+', '_', variant_label.lower()).strip('_')
        for prim in args.primitives:
            cells = {
                str(B): {k: r[k] for k in ('ms_min', 'ms_med', 'throughput', 'mem_mb')
                          if k in r}
                for B in args.batch_sizes
                for r in [results.get((variant_label, prim, B), {})]
                if r and 'ms_min' in r
            }
            try:
                record({
                    "id": make_id(args.shape, net, f"{variant}__{prim}",
                                  "single_query_bound", seed=args.seed),
                    "shape": args.shape,
                    "network": net,
                    "variant": variant,
                    "algorithm": "single_query_bound",
                    "metrics": {
                        "primitive": prim,
                        "batch_sizes": list(args.batch_sizes),
                        "by_batch_size": cells,
                        "bbox_half": float(args.bbox_half),
                        "init_tr": float(args.init_tr),
                    },
                    "system": {"gpu_index": os.environ.get("CUDA_VISIBLE_DEVICES")},
                    "seed": int(args.seed),
                })
            except Exception as e:
                print(f"  [warn] record({variant}/{prim}) failed: {type(e).__name__}: {e}",
                      flush=True)
    print(f"  recorded {len(variants)*len(args.primitives)} rows → "
          f"paper_utils.results/single_query_bound.jsonl", flush=True)
