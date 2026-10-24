"""Quality benchmark for classify across all bound modes (1D + 3D).

Mirrors NIQ's `benchmark_quality.py` Table-1 protocol for both query shapes:
  * 3D: random axis-aligned cubes of half-width s.
  * 1D: random unit-direction segments of half-length s.
For each (mode, region_dim, s) cell, sample N regions, classify each as
POS/NEG/UNKNOWN, and report the fraction conclusively-signed. Bigger s with
f >= threshold = tighter bound.

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
# fp64 enable must happen BEFORE first jax import. Inspect raw argv (argparse
# runs after) so users can request fp64 by listing it in --ab_modes.
if "fp64" in sys.argv:
    os.environ["JAX_ENABLE_X64"] = "True"
import jax
# Force fp32 matmul (mirror jnpboundedImplicit / niImplicitSurface — important
# for the autobound polynomial constant to be correct under vmap; see
# bench_raycast_clean.py for the full diagnostic).
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
_X64 = bool(jax.config.read("jax_enable_x64"))
DTYPE = jnp.float64 if _X64 else jnp.float32

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
p.add_argument("--eqx_path", default=None,
               help="explicit eqx path; overrides --network resolution")
p.add_argument("--npz_path", default=None,
               help="explicit niq npz path; overrides --network resolution")
p.add_argument("--activation", default="softplus",
               help="MLP activation (jax.nn.* name); must match the .eqx file")
p.add_argument("--modes", nargs="*", default=DEFAULT_MODES,
               help="NIQ bound modes to include. Pass nothing (--modes) to skip "
                    "NIQ entirely — useful for swish networks NIQ doesn't support.")
p.add_argument("--d2", action=argparse.BooleanOptionalAction, default=True,
               help="include autobound deg-2 (warp=True)")
p.add_argument("--d3", action=argparse.BooleanOptionalAction, default=False,
               help="include autobound deg-3 (warp=True)")
p.add_argument("--ab_modes", nargs="+", default=["default"],
               choices=["default", "fp32", "mixed", "fp64"],
               help="precision_mode(s) for autobound: 'default' = compute matches "
                    "storage (fp32×fp32); 'mixed' = fp32 storage + fp64 compute "
                    "(tighter bounds, ~30-50%% slower).")
p.add_argument("--region_dims", nargs="+", type=int, default=[1, 3],
               choices=[1, 3],
               help="which query shapes to evaluate: 1=random-direction segments, "
                    "3=axis-aligned cubes. Default: both.")
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
p.add_argument("--shape", default="bunny",
               help="paper_utils.results 'shape' field")
p.add_argument("--depth", type=int, default=None,
               help="MLP depth for results metadata (auto-inferred from .eqx if None)")
p.add_argument("--width", type=int, default=None,
               help="MLP width for results metadata (auto-inferred from .eqx if None)")
p.add_argument("--loss", default="SDF",
               help="loss the network was trained with")
p.add_argument("--no_record", action="store_true",
               help="skip writing to paper_utils.results")
args = p.parse_args()

ASSETS = "/system/user/volkmann/implicit_surfaces_project/assets"
EQX_PATH = args.eqx_path or f"{ASSETS}/equinox_models/{args.network}.eqx"
NPZ_PATH = args.npz_path or f"{ASSETS}/niq_models/{args.network}.npz"
required = [EQX_PATH] + ([NPZ_PATH] if args.modes else [])
for path in required:
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
    weights = [jnp.asarray(a, dtype=DTYPE) for a in arrs[0::2]]
    biases  = [jnp.asarray(a, dtype=DTYPE) for a in arrs[1::2]]
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

# Infer depth/width from the eqx file if not provided.
if args.depth is None:
    args.depth = len(_W) - 1   # hidden layers = #weight matrices - 1 (output)
if args.width is None:
    args.width = _W[0].shape[0] if len(_W) > 1 else _W[-1].shape[0]


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

def build_autobound(max_degree, precision_mode=None):
    """precision_mode: None ('fp32 storage + fp32 compute' default), 'mixed'
    ('fp32 storage + fp64 compute'), 'fp32', 'fp64'. See jb.taylor_bounds."""
    surf = jnpboundedImplicit(autobounds_max_degree=max_degree, fn=fn_2arg,
                               level_set=0.0, dim=3, use_warp=True)
    surf.taylor_bounds = jb.taylor_bounds(
        mlp_fn, max_degree=max_degree, use_warp=True,
        precision_mode=precision_mode,
    )
    return surf


methods = []                      # list[(label, surface, params)]
for mode in args.modes:
    label = f"niq:{mode}{'_'+str(args.affine_n_truncate) if mode=='affine_truncate' else ''}"
    surf = build_niq(mode)
    methods.append((label, surf, surf.params))
for pm in args.ab_modes:
    pm_key = "default" if pm == "default" else pm
    pm_arg = None if pm == "default" else pm
    if args.d2:
        methods.append((f"autobound:d2[{pm_key}]", build_autobound(2, pm_arg), None))
    if args.d3:
        methods.append((f"autobound:d3[{pm_key}]", build_autobound(3, pm_arg), None))


# ----------------------------------------------------------------------------
# Sampling + classify (chunked, 1D + 3D)
# ----------------------------------------------------------------------------
def sample_centers(key, n, half_extent_max):
    return jax.random.uniform(
        key, (n, 3), minval=-half_extent_max, maxval=half_extent_max, dtype=DTYPE,
    )


def sample_unit_dirs(key, n):
    raw = jax.random.normal(key, (n, 3), dtype=DTYPE)
    return raw / jnp.linalg.norm(raw, axis=-1, keepdims=True)


def _build_3d_classifier(surf):
    """3D: classify axis-aligned cube [c-s, c+s]^3 via classify_box(lower, upper)."""
    @jax.jit
    @jax.vmap
    def fn(c, s):
        return surf.classify_box(None, c - s, c + s)
    return fn


def _build_1d_classifier_niq(surf):
    """1D NIQ: classify ±s·d segment via _classify_general_box on (1,3) box_vec."""
    @jax.jit
    @jax.vmap
    def fn(c, d, s):
        # NIQ box_vec convention: shape (k,3) — half-extents along k axes.
        box_vec = (s * d)[None, :]
        niq_sign = surf._classify_general_box(surf.params, c, box_vec)
        # Map NIQ {0:UNKNOWN, 1:POS, 2:NEG} → our {0:unknown, +1:pos, -1:neg}
        return jnp.where(niq_sign == 1, jnp.int8(1),
                jnp.where(niq_sign == 2, jnp.int8(-1), jnp.int8(0)))
    return fn


def _build_1d_classifier_ab(surf):
    """1D autobound: ray_step(±d, tr=2s) twice; classified iff min(t_pf, t_nf) >= s.

    Mirrors NIQ benchmark_quality._build_abroots_classifiers. The trust region is
    the AABB of edge length 2s containing the segment; the polynomial enclosure
    over that AABB is what decides the segment.
    """
    @jax.jit
    @jax.vmap
    def fn(c, d, s):
        tr = 2.0 * s
        fn_val_pf, t_pf, _ = surf.ray_step(None, c, +d, tr)
        _,         t_nf, _ = surf.ray_step(None, c, -d, tr)
        safe = jnp.minimum(t_pf, t_nf)
        classified = safe >= s
        return jnp.where(classified,
                jnp.where(fn_val_pf > 0, jnp.int8(1), jnp.int8(-1)),
                jnp.int8(0))
    return fn


def build_classifiers(surf):
    """Return (classify_1d, classify_3d) for one surface."""
    c3 = _build_3d_classifier(surf)
    if isinstance(surf, niImplicitSurface):
        c1 = _build_1d_classifier_niq(surf)
    else:
        c1 = _build_1d_classifier_ab(surf)
    return c1, c3


def chunked_apply_3d(cfn, centers, s_arr, chunk):
    out = []
    for k in range(0, centers.shape[0], chunk):
        out.append(cfn(centers[k:k+chunk], s_arr[k:k+chunk]))
    return jnp.concatenate(out, axis=0)


def chunked_apply_1d(cfn, centers, dirs, s_arr, chunk):
    out = []
    for k in range(0, centers.shape[0], chunk):
        out.append(cfn(centers[k:k+chunk], dirs[k:k+chunk], s_arr[k:k+chunk]))
    return jnp.concatenate(out, axis=0)


sizes = np.geomspace(args.s_min, args.s_max, args.n_sizes).astype(np.float32)
key = jax.random.PRNGKey(args.seed)

# results[label][region_dim][s] -> fraction
DIMS = sorted(set(args.region_dims))
results = {label: {dim: {} for dim in DIMS} for (label, _, _) in methods}

# Cache jit'd callables once per surface (capture by class).
classifiers = {}
for label, surf, _ in methods:
    c1, c3 = build_classifiers(surf)
    classifiers[label] = {1: c1, 3: c3}

t_total = time.time()
for s in sizes:
    s_margin = float(s) * 1.01
    keep_max = max(args.data_bound - s_margin, 0.0)
    if keep_max <= 0.0:
        for label, _, _ in methods:
            for dim in DIMS:
                results[label][dim][float(s)] = float("nan")
        continue
    kc, kd, key = jax.random.split(key, 3)
    centers = sample_centers(kc, args.n_regions, keep_max)
    s_arr   = jnp.full((args.n_regions,), float(s), dtype=DTYPE)
    dirs    = sample_unit_dirs(kd, args.n_regions) if 1 in DIMS else None

    for label, _surf, _params in methods:
        for dim in DIMS:
            try:
                if dim == 3:
                    signs = chunked_apply_3d(classifiers[label][3], centers, s_arr, args.chunk)
                else:
                    signs = chunked_apply_1d(classifiers[label][1], centers, dirs, s_arr, args.chunk)
                jax.block_until_ready(signs)
                f = float(jnp.mean(signs != 0))
            except Exception as e:
                print(f"  [warn] {label} dim={dim} @ s={float(s):.3e} failed: "
                      f"{type(e).__name__}: {str(e)[:80]}")
                f = float("nan")
            results[label][dim][float(s)] = f
print(f"sweep took {time.time() - t_total:.1f}s", flush=True)


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def largest_s_above(s_list, f_list, threshold):
    candidates = [s for s, f in zip(s_list, f_list)
                  if (not (f != f)) and f >= threshold]
    return max(candidates) if candidates else None


for dim in DIMS:
    label_dim = "3D cube" if dim == 3 else "1D random-direction segment"
    print()
    print(f"Largest {label_dim} half-width with >= {args.threshold*100:.0f}% "
          f"classified  ({args.n_regions} regions per cell):")
    print(f"{'mode':>26}  {'half-width ↑':>14}")
    print("-" * 50)
    for label, _, _ in methods:
        s_keys = list(results[label][dim].keys())
        f_vals = list(results[label][dim].values())
        s_best = largest_s_above(s_keys, f_vals, args.threshold)
        s_str = f"{s_best:.3e}" if s_best is not None else "  none "
        print(f"{label:>26}  {s_str:>14}")


# ----------------------------------------------------------------------------
# Record to paper_utils.results / single_query_bound.jsonl
# ----------------------------------------------------------------------------
if not args.no_record:
    import re as _re
    sys.path.insert(0, "/system/user/volkmann/implicit_surfaces_project")
    from utils.paper_utils.results import record, make_id
    net = dict(depth=args.depth, width=args.width,
               activation=args.activation, loss=args.loss)
    n_recorded = 0
    for label, _, _ in methods:
        variant = _re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')
        for dim in DIMS:
            s_keys = list(results[label][dim].keys())
            f_vals = list(results[label][dim].values())
            s_best = largest_s_above(s_keys, f_vals, args.threshold)
            base_id = make_id(args.shape, net, variant, "single_query_bound", seed=args.seed)
            # Inject region_dim segment into id so 1d/3d records are distinct.
            rec_id = base_id.replace(
                "__single_query_bound__",
                f"__r{dim}d__single_query_bound__",
            )
            try:
                record({
                    "id": rec_id,
                    "shape": args.shape,
                    "network": net,
                    "variant": variant,
                    "algorithm": "single_query_bound",
                    "metrics": {
                        "region_dim": int(dim),
                        "halfwidth_50pct": float(s_best) if s_best is not None else None,
                        "n_regions_per_cell": int(args.n_regions),
                        "n_sizes": int(args.n_sizes),
                        "s_min": float(args.s_min),
                        "s_max": float(args.s_max),
                        "data_bound": float(args.data_bound),
                        "threshold": float(args.threshold),
                        "fraction_classified": {
                            f"{s:.6e}": float(results[label][dim][float(s)]) for s in sizes
                        },
                    },
                    "seed": int(args.seed),
                })
                n_recorded += 1
            except Exception as e:
                print(f"  [warn] record({variant}, dim={dim}) failed: "
                      f"{type(e).__name__}: {e}", flush=True)
    print(f"  recorded {n_recorded} rows → paper_utils.results/single_query_bound.jsonl",
          flush=True)
