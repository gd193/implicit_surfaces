"""Quality benchmark for classify_box across all bound modes.

Mirrors NIQ's `benchmark_quality.py` Table-1 protocol restricted to the 3D
axis-aligned case: for each bound mode and each region size s, sample N
random AABBs of half-width s, classify each as POS/NEG/UNKNOWN, and report
the fraction that came back conclusively-signed. Bigger s with f >= threshold
= tighter bound.

Methods compared (subset selectable via --modes):
  * niq:interval, niq:affine_fixed, niq:slope_interval,
    niq:affine_truncate_8, niq:affine_all  — via niImplicitSurface
  * autobound:d2  — via jnpboundedImplicit(max_degree=2, use_warp=True)
  * autobound:d3  — same with max_degree=3 (opt-in via --d3)

Network selection: pass `--network <basename>` and the script looks up
  assets/equinox_models/<basename>.eqx        (for autobound surfaces)
  assets/niq_models/<basename>.npz            (for niq:* surfaces)

Both files must exist (use scripts/convert_models.py if only one form is
available). Activation must match what the .eqx was trained with —
softplus by default; override with `--activation`.

Usage:
  python scripts/benchmark_classify_quality.py
  python scripts/benchmark_classify_quality.py --network bunny_softplus --modes interval affine_fixed
  python scripts/benchmark_classify_quality.py --d3 --n_regions 5000
"""
import argparse
import os
import sys
import time

import numpy as np
import jax
# Force fp32 matmul (mirror jnpboundedImplicit / niImplicitSurface — important
# for the autobound polynomial constant to be correct under vmap; see
# bench_raycast_clean.py for the full diagnostic).
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp

sys.path.insert(0, '/system/user/volkmann/implicit_surfaces_project')

import jnp_bounds.autobounds as jb
from implicitSurfaces.niImplicitSurface import niImplicitSurface
from implicitSurfaces.jnpboundedImplicit import jnpboundedImplicit


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
DEFAULT_MODES = [
    "interval", "affine_fixed", "slope_interval",
    "affine_truncate", "affine_all",
]

p = argparse.ArgumentParser()
p.add_argument("--network", default="bunny_softplus",
               help="basename; resolves to assets/equinox_models/<n>.eqx and "
                    "assets/niq_models/<n>.npz")
p.add_argument("--activation", default="softplus",
               help="MLP activation (jax.nn.* name); must match the .eqx file")
p.add_argument("--modes", nargs="+", default=DEFAULT_MODES,
               help="NIQ bound modes to include")
p.add_argument("--d2", action=argparse.BooleanOptionalAction, default=True,
               help="include autobound deg-2 (warp=True)")
p.add_argument("--d3", action=argparse.BooleanOptionalAction, default=False,
               help="include autobound deg-3 (warp=True)")
p.add_argument("--n_regions", type=int, default=10_000)
p.add_argument("--n_sizes", type=int, default=24)
p.add_argument("--s_min", type=float, default=1e-4)
p.add_argument("--s_max", type=float, default=1.0)
p.add_argument("--data_bound", type=float, default=1.0)
p.add_argument("--threshold", type=float, default=0.5)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--chunk", type=int, default=8192,
               help="batch size for classify_box vmap; lower if OOM")
p.add_argument("--affine_n_truncate", type=int, default=8)
p.add_argument("--affine_truncate_policy", default="absolute")
args = p.parse_args()

ASSETS = "/system/user/volkmann/implicit_surfaces_project/assets"
EQX_PATH = f"{ASSETS}/equinox_models/{args.network}.eqx"
NPZ_PATH = f"{ASSETS}/niq_models/{args.network}.npz"
for path in [EQX_PATH, NPZ_PATH]:
    if not os.path.exists(path):
        print(f"ERROR: missing {path}; run scripts/convert_models.py first.")
        sys.exit(2)

print(f"JAX: {[d.platform for d in jax.devices()]}", flush=True)
print(f"network={args.network}  activation={args.activation}  "
      f"n_regions={args.n_regions}  n_sizes={args.n_sizes}  "
      f"s∈[{args.s_min:.0e},{args.s_max:.0e}]  data_bound={args.data_bound}", flush=True)


# ----------------------------------------------------------------------------
# Generic eqx MLP loader (alternating W, b arrays; works for any depth)
# ----------------------------------------------------------------------------
def load_eqx_mlp(path, activation_name):
    arrs = []
    with open(path, "rb") as f:
        while True:
            try:
                arrs.append(np.lib.format.read_array(f))
            except (EOFError, ValueError):
                break
    if len(arrs) % 2 != 0:
        raise ValueError(f"expected even number of arrays in {path}, got {len(arrs)}")
    weights = [jnp.asarray(a, dtype=jnp.float32) for a in arrs[0::2]]
    biases  = [jnp.asarray(a, dtype=jnp.float32) for a in arrs[1::2]]
    act = getattr(jax.nn, activation_name)

    def mlp(x):
        h = x
        for i in range(len(weights) - 1):
            h = act(weights[i] @ h + biases[i])
        return weights[-1] @ h + biases[-1]
    return mlp, weights, biases


mlp_fn, _W, _b = load_eqx_mlp(EQX_PATH, args.activation)
def fn_2arg(_p, x): return mlp_fn(x)
print(f"loaded {EQX_PATH}: {len(_W)} layers, in={_W[0].shape[1]} out={_W[-1].shape[0]} "
      f"hidden={tuple(W.shape[0] for W in _W[:-1])}", flush=True)


# ----------------------------------------------------------------------------
# Build classify_box callables per mode
# ----------------------------------------------------------------------------
def build_niq(mode):
    kw = {}
    if mode == "affine_truncate":
        kw["affine_n_truncate"] = args.affine_n_truncate
        kw["affine_truncate_policy"] = args.affine_truncate_policy
    surf = niImplicitSurface(mode=mode, npz_path=NPZ_PATH, level_set=0.0, dim=3, **kw)
    return surf

def build_autobound(max_degree):
    surf = jnpboundedImplicit(autobounds_max_degree=max_degree, fn=fn_2arg,
                               level_set=0.0, dim=3, use_warp=True)
    surf.taylor_bounds = jb.taylor_bounds(mlp_fn, max_degree=max_degree, use_warp=True)
    return surf


methods = []                      # list[(label, surface, params)]
for mode in args.modes:
    label = f"niq:{mode}{'_'+str(args.affine_n_truncate) if mode=='affine_truncate' else ''}"
    surf = build_niq(mode)
    methods.append((label, surf, surf.params))
if args.d2:
    methods.append(("autobound:d2", build_autobound(2), None))
if args.d3:
    methods.append(("autobound:d3", build_autobound(3), None))


# ----------------------------------------------------------------------------
# Sampling + classify (chunked)
# ----------------------------------------------------------------------------
def sample_centers(key, n, half_extent_max):
    return jax.random.uniform(
        key, (n, 3), minval=-half_extent_max, maxval=half_extent_max, dtype=jnp.float32,
    )


def make_classify(surface):
    return jax.jit(jax.vmap(surface.classify_box, in_axes=(None, 0, 0)))


def classify_chunked(classify_fn, params, lowers, uppers, chunk):
    out = []
    for k in range(0, lowers.shape[0], chunk):
        out.append(classify_fn(params, lowers[k:k+chunk], uppers[k:k+chunk]))
    return jnp.concatenate(out, axis=0)


sizes = np.geomspace(args.s_min, args.s_max, args.n_sizes).astype(np.float32)
key = jax.random.PRNGKey(args.seed)

results = {label: {} for (label, _, _) in methods}

# Cache jit'd callables once per surface (capture by class).
classifiers = {label: make_classify(surf) for (label, surf, _) in methods}

t_total = time.time()
for s in sizes:
    s_margin = float(s) * 1.01
    keep_max = max(args.data_bound - s_margin, 0.0)
    if keep_max <= 0.0:
        for label, _, _ in methods:
            results[label][float(s)] = float("nan")
        continue
    kc, key = jax.random.split(key)
    centers = sample_centers(kc, args.n_regions, keep_max)
    lowers = centers - float(s)
    uppers = centers + float(s)

    for label, _surf, params in methods:
        try:
            signs = classify_chunked(classifiers[label], params, lowers, uppers, args.chunk)
            jax.block_until_ready(signs)
            f = float(jnp.mean(signs != 0))
        except Exception as e:
            print(f"  [warn] {label} @ s={float(s):.3e} failed: {type(e).__name__}: {str(e)[:80]}")
            f = float("nan")
        results[label][float(s)] = f
print(f"sweep took {time.time() - t_total:.1f}s", flush=True)


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def largest_s_above(s_list, f_list, threshold):
    candidates = [s for s, f in zip(s_list, f_list)
                  if (not (f != f)) and f >= threshold]
    return max(candidates) if candidates else None


print()
print(f"Largest 3D half-width with >= {args.threshold*100:.0f}% classified  "
      f"({args.n_regions} regions per cell):")
print(f"{'mode':>26}  {'half-width ↑':>14}  {'volume (2s)^3 ↑':>18}")
print("-" * 68)
for label, _, _ in methods:
    s_keys = list(results[label].keys())
    f_vals = list(results[label].values())
    s_best = largest_s_above(s_keys, f_vals, args.threshold)
    s_str = f"{s_best:.3e}" if s_best is not None else "  none "
    v_str = f"{(2*s_best)**3:.3e}" if s_best is not None else "  none "
    print(f"{label:>26}  {s_str:>14}  {v_str:>18}")

print()
print("Fraction classified vs region half-width:")
header = "  ".join(f"{label:>22}" for (label, _, _) in methods)
print(f"{'s':>10}  | " + header)
for s in sizes:
    cells = [f"{results[label][float(s)]*100:>20.1f}%"
             for (label, _, _) in methods]
    print(f"{float(s):>10.3e}  | " + "  ".join(cells))
