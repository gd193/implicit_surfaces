"""WoS benchmark — bunny_softplus interior, Δu=0 with Dirichlet g(x)=‖x‖² on f=0.

Mirrors bench_raycast_clean.py's structure: same surface registry, same network
selection, fp32 matmul forced via the surface modules. Each method runs through
WoSIntegrator with its `empty_sphere`. Per method we report:

  * warm best/median ms (5 timed runs)
  * mean walker steps to absorption
  * per-test-point u(p) estimates
  * RMS / max disagreement vs a high-walker-count reference (n_walkers_high)

Per-method output saved to assets/wos_solutions/<method>/u.npz so a separate
checker can compare any two runs after the fact.

Usage:
  bench_wos_clean.py
  bench_wos_clean.py --variant autobound
  bench_wos_clean.py --network bunny_softplus --n_test_points 256 --n_walkers 1024
"""
import argparse, os, re, subprocess, sys, time

sys.path.insert(0, '/system/user/volkmann/implicit_surfaces_project')

import numpy as np
import jax, jax.numpy as jnp
import jax.random as jr
import jnp_bounds.autobounds as jb

# Surface module imports also set jax_default_matmul_precision='highest'.
from implicitSurfaces.niImplicitSurface import niImplicitSurface
from implicitSurfaces.jnpboundedImplicit import jnpboundedImplicit
from sphereWalker.integrator import WoSIntegrator
from utils.rejection_sampling import sample_points_batched

print(f"JAX: {[d.platform for d in jax.devices()]}", flush=True)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
DEFAULT_MODES = ["interval", "affine_fixed", "slope_interval", "affine_truncate", "affine_all"]

p = argparse.ArgumentParser()
p.add_argument("--network", default="bunny_softplus")
p.add_argument("--activation", default="softplus")
p.add_argument("--variant", default=None,
               help="substring filter on variant labels (comma-separated for multiple)")
p.add_argument("--modes", nargs="+", default=DEFAULT_MODES,
               help="NIQ bound modes to include")
p.add_argument("--d2", action=argparse.BooleanOptionalAction, default=True)
p.add_argument("--d3", action=argparse.BooleanOptionalAction, default=False)
p.add_argument("--n_test_points", type=int, default=256)
p.add_argument("--n_walkers", type=int, default=1024)
p.add_argument("--n_walkers_high", type=int, default=4096)
p.add_argument("--max_steps", type=int, default=256)
p.add_argument("--eps", type=float, default=1e-3)
p.add_argument("--init_tr", type=float, default=0.1)
p.add_argument("--isolevel", type=float, default=0.0)
p.add_argument("--boundary_fn", default="norm_sq", choices=["norm_sq", "one"])
p.add_argument("--bbox_half", type=float, default=0.2,
               help="rejection-sample interior test points from [-bbox_half, bbox_half]^3")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--n_warmup", type=int, default=5,
               help="warmup integrate_batch calls with diverse RNG keys; bigger N "
                    "saturates the JIT cache for compaction static-arg buffer sizes "
                    "so timed runs don't pay first-compile costs.")
p.add_argument("--n_timed", type=int, default=5)
p.add_argument("--steps_between_refill", type=int, default=50)
p.add_argument("--max_active_states", type=int, default=1_000_000)
p.add_argument("--n_compactions", type=int, default=20)
p.add_argument("--affine_n_truncate", type=int, default=8)
p.add_argument("--affine_truncate_policy", default="absolute")
args = p.parse_args()


ASSETS = "/system/user/volkmann/implicit_surfaces_project/assets"
EQX_PATH = f"{ASSETS}/equinox_models/{args.network}.eqx"
NPZ_PATH = f"{ASSETS}/niq_models/{args.network}.npz"
TEST_POINTS_DIR = f"{ASSETS}/wos_test_points"
SOLUTION_DIR = f"{ASSETS}/wos_solutions"


def _slug(label: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')


def gpu_peak_bytes():
    """Peak GPU bytes in use since process start. Monotonic — does NOT reset
    when JAX frees buffers. For accurate per-variant peak, run each variant
    in a separate process (use --variant <one>). Mirrors NIQ's
    `benchmark_wos.py:_peak_gb`."""
    try:
        return int(jax.devices()[0].memory_stats()['peak_bytes_in_use'])
    except Exception:
        return -1


def gpu_in_use_bytes():
    try:
        return int(jax.devices()[0].memory_stats()['bytes_in_use'])
    except Exception:
        return -1


# ----------------------------------------------------------------------------
# MLP load (generic, mirrors quality bench)
# ----------------------------------------------------------------------------
def load_eqx_mlp(path, activation_name):
    arrs = []
    with open(path, "rb") as f:
        while True:
            try: arrs.append(np.lib.format.read_array(f))
            except (EOFError, ValueError): break
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
    return mlp


mlp_fn = load_eqx_mlp(EQX_PATH, args.activation)
def fn_2arg(_p, x): return mlp_fn(x)
print(f"loaded {EQX_PATH}", flush=True)


# ----------------------------------------------------------------------------
# Test points: rejection-sample inside iso-surface, cache for reproducibility
# ----------------------------------------------------------------------------
def sample_interior_points(n, bbox_half, isolevel, mlp, seed):
    """Reject-sample uniform points in [-bbox_half,bbox_half]^3 with f(x) < isolevel.
    Uses the project's jit-traceable batched rejection sampler."""
    aa = -jnp.full((3,), bbox_half, dtype=jnp.float32)
    bb =  jnp.full((3,), bbox_half, dtype=jnp.float32)
    pts, ok = sample_points_batched(
        fn=lambda _p, x: jnp.squeeze(mlp(x)),
        params=None,
        accept_sample_fn=lambda v: v < isolevel,
        init_buffer=None,
        key=jr.PRNGKey(seed),
        aa=aa, bb=bb,
        num_points=n,
        batch_size=max(8 * n, 4096),
    )
    if not bool(ok):
        raise RuntimeError(f"rejection sampling exhausted before producing {n} interior points")
    return pts


os.makedirs(TEST_POINTS_DIR, exist_ok=True)
points_path = f"{TEST_POINTS_DIR}/{args.network}_n{args.n_test_points}_seed{args.seed}.npy"
if os.path.exists(points_path):
    test_points = jnp.asarray(np.load(points_path))
    print(f"loaded test points: {points_path}  shape={test_points.shape}", flush=True)
else:
    test_points = sample_interior_points(args.n_test_points, args.bbox_half,
                                          args.isolevel, mlp_fn, args.seed)
    np.save(points_path, np.asarray(test_points))
    print(f"sampled + saved test points → {points_path}  shape={test_points.shape}", flush=True)


# ----------------------------------------------------------------------------
# Boundary condition + source
# ----------------------------------------------------------------------------
if args.boundary_fn == "norm_sq":
    def boundary_fn(x): return jnp.sum(x * x)
elif args.boundary_fn == "one":
    def boundary_fn(x): return jnp.asarray(1.0)

def source_fn(x): return jnp.asarray(0.0)


# ----------------------------------------------------------------------------
# Surface factories
# ----------------------------------------------------------------------------
def make_niq(mode):
    kw = {}
    if mode == "affine_truncate":
        kw["affine_n_truncate"] = args.affine_n_truncate
        kw["affine_truncate_policy"] = args.affine_truncate_policy
    return niImplicitSurface(mode=mode, npz_path=NPZ_PATH, level_set=args.isolevel, dim=3, **kw)

def make_autobound(max_degree):
    s = jnpboundedImplicit(autobounds_max_degree=max_degree, fn=fn_2arg,
                            level_set=args.isolevel, dim=3, use_warp=True)
    s.taylor_bounds = jb.taylor_bounds(mlp_fn, max_degree=max_degree, use_warp=True)
    return s


variants = []  # list[(label, factory, params_attr)]
for mode in args.modes:
    label = f"NIQ {mode}{f'({args.affine_n_truncate})' if mode=='affine_truncate' else ''}"
    variants.append((label, lambda m=mode: make_niq(m), 'params'))
if args.d2:
    variants.append(("autobound deg-2 (warp=True)", lambda: make_autobound(2), None))
if args.d3:
    variants.append(("autobound deg-3 (warp=True)", lambda: make_autobound(3), None))

if args.variant:
    needles = [n.strip().lower() for n in args.variant.split(',') if n.strip()]
    variants = [(l, f, p) for (l, f, p) in variants
                if any(n in l.lower() for n in needles)]
    if not variants:
        print(f"No variants match {args.variant!r}"); sys.exit(1)


# ----------------------------------------------------------------------------
# Benchmark loop
# ----------------------------------------------------------------------------
def run_method(label, surface, params):
    integrator = WoSIntegrator(
        empty_sphere_fn=surface.empty_sphere,
        boundary_condition_fn=boundary_fn,
        source_term_fn=source_fn,
        dim=3, level_set=args.isolevel,
        max_steps=args.max_steps, eps=args.eps,
        print_progress=True,
    )

    def integrate(n_walkers, key):
        return integrator.integrate_batch(
            params=params,
            initial_position_batch=test_points,
            rng_key=key,
            init_trust_region_size=args.init_tr,
            n_samples=n_walkers,
            steps_between_refill=args.steps_between_refill,
            max_active_states=args.max_active_states,
            n_compactions=args.n_compactions,
        )

    base_key = jr.PRNGKey(args.seed)

    # Explicit per-bucket pre-warm: trigger one JIT compile per distinct
    # `active_buffer_size` that timed runs and the high-walker run will hit.
    # This forecloses the "first-call pays the JIT compile cost" failure mode
    # where random-key warmups don't deterministically visit all bucket sizes.
    print(f'  [pre-warm buckets — n_walkers / n_walkers_high]', flush=True)
    pre_warm_key = jr.fold_in(base_key, 10000)
    integrator.pre_warm_compaction_buckets(
        params=params,
        initial_position_batch=test_points,
        rng_key=pre_warm_key,
        init_trust_region_size=args.init_tr,
        n_samples=args.n_walkers,
        steps_between_refill=args.steps_between_refill,
        max_active_states=args.max_active_states,
        n_compactions=args.n_compactions,
    )
    integrator.pre_warm_compaction_buckets(
        params=params,
        initial_position_batch=test_points,
        rng_key=jr.fold_in(base_key, 10001),
        init_trust_region_size=args.init_tr,
        n_samples=args.n_walkers_high,
        steps_between_refill=args.steps_between_refill,
        max_active_states=args.max_active_states,
        n_compactions=args.n_compactions,
    )
    # Trajectory warmup: lets the integrator's JIT cache also see the
    # accumulator/refill graphs hit by integrate_batch (different jaxpr from
    # the standalone _process_active_states calls above).
    print(f'  [warmup × {args.n_warmup} + 1×high]', end='', flush=True)
    for k in range(args.n_warmup):
        out = integrate(args.n_walkers, jr.fold_in(base_key, 11000 + k))
        jax.block_until_ready(out.w)
        print('.', end='', flush=True)
    out = integrate(args.n_walkers_high, jr.fold_in(base_key, 11000 + args.n_warmup))
    jax.block_until_ready(out.w)
    print('h done', flush=True)

    # Timed
    times = []
    last_out = None
    for k in range(args.n_timed):
        t0 = time.time()
        out = integrate(args.n_walkers, jr.fold_in(base_key, k))
        jax.block_until_ready(out.w)
        times.append(time.time() - t0)
        last_out = out
    t_min = min(times) * 1000
    t_med = float(np.median(times)) * 1000

    # High-walker reference (single run)
    out_high = integrate(args.n_walkers_high, jr.fold_in(base_key, 999))
    jax.block_until_ready(out_high.w)

    u_low = np.asarray(last_out.w).reshape(-1)
    u_high = np.asarray(out_high.w).reshape(-1)
    n_steps_mean = float(np.mean(np.asarray(last_out.nSteps)))
    diff = u_low - u_high
    rms = float(np.sqrt(np.mean(diff ** 2)))
    max_abs = float(np.max(np.abs(diff)))

    peak_gb = gpu_peak_bytes() / (1024 ** 3)
    in_use_gb = gpu_in_use_bytes() / (1024 ** 3)
    print(f'  warm: best {t_min:7.1f} ms | median {t_med:7.1f} ms | '
          f'mean nSteps {n_steps_mean:6.1f} | '
          f'RMS(u_low − u_high) {rms:.4e} | max {max_abs:.4e} | '
          f'peak {peak_gb:5.2f} GB (in_use {in_use_gb:5.2f} GB)', flush=True)

    method_dir = f"{SOLUTION_DIR}/{_slug(label)}"
    os.makedirs(method_dir, exist_ok=True)
    np.savez(f"{method_dir}/u.npz",
             u_low=u_low, u_high=u_high,
             n_steps=np.asarray(last_out.nSteps),
             test_points=np.asarray(test_points),
             warm_ms=t_min, median_ms=t_med,
             peak_bytes=gpu_peak_bytes(),
             n_walkers=args.n_walkers, n_walkers_high=args.n_walkers_high)
    print(f'  saved → {method_dir}/u.npz', flush=True)


for label, factory, params_attr in variants:
    print(f'\n[{label}]', flush=True)
    try:
        surface = factory()
        params = getattr(surface, params_attr) if params_attr else None
        run_method(label, surface, params)
    except Exception as e:
        import traceback
        print(f'  FAILED: {type(e).__name__}: {e}', flush=True)
        traceback.print_exc()
