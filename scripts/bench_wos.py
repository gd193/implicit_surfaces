"""WoS benchmark — bunny_softplus interior, Δu=0 with Dirichlet g(x)=‖x‖² on f=0.

The degree-two autobounded surface supplies its empty_sphere query to WoSIntegrator.

  * warm best/median ms (5 timed runs)
  * mean walker steps to absorption
  * per-test-point u(p) estimates
  * RMS / max disagreement vs a high-walker-count reference (n_walkers_high)

Per-method output saved to outputs/wos_solutions/<method>/u.npz so a separate
checker can compare any two runs after the fact.

Usage:
  python scripts/bench_wos.py
  python scripts/bench_wos.py --network bunny_softplus --n_test_points 256 --n_walkers 1024
"""
from pathlib import Path
import argparse, os, re, sys, time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import jax, jax.numpy as jnp
import jax.random as jr

# Surface module imports also set jax_default_matmul_precision='highest'.
from scripts.eqx_model import load_eqx_mlp
from surfaces.implicit.jnp_bounded import jnpboundedImplicit
from sphereWalker import WoSIntegrator
from utils.rejection_sampling import sample_points_batched

print(f"JAX: {[d.platform for d in jax.devices()]}", flush=True)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

p = argparse.ArgumentParser()
p.add_argument("--network", default="bunny_softplus")
p.add_argument("--activation", default="softplus")
p.add_argument("--shape", default="bunny",
               help="paper_utils.results 'shape' field (e.g. bunny, fox, hammer)")
p.add_argument("--depth", type=int, default=3,
               help="MLP hidden-layer count (for results metadata)")
p.add_argument("--width", type=int, default=32,
               help="MLP hidden width (for results metadata)")
p.add_argument("--loss", default="SDF",
               help="loss the network was trained with (for results metadata)")
p.add_argument("--no_record", action="store_true",
               help="skip writing to paper_utils.results")
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
args = p.parse_args()


ASSETS = ROOT / "assets"
OUTPUTS = ROOT / "outputs"
EQX_PATH = f"{ASSETS}/models/{args.network}.eqx"
TEST_POINTS_DIR = f"{OUTPUTS}/wos_test_points"
SOLUTION_DIR = f"{OUTPUTS}/wos_solutions"


def _slug(label: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')


def gpu_peak_bytes():
    """Peak GPU bytes in use since process start."""
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
mlp_fn, _ = load_eqx_mlp(EQX_PATH, args.activation, jnp.float32)
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
        batch_size=min(max(4096, n), n),
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
# Benchmark loop
# ----------------------------------------------------------------------------
def run_method(label, surface):
    integrator = WoSIntegrator(
        empty_sphere_fn=surface.empty_sphere,
        boundary_condition_fn=boundary_fn,
        source_term_fn=source_fn,
        dim=3,
        max_steps=args.max_steps, eps=args.eps,
        print_progress=True,
    )

    def integrate(n_walkers, key):
        return integrator.integrate_batch(
            params=None,
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
        params=None,
        initial_position_batch=test_points,
        rng_key=pre_warm_key,
        init_trust_region_size=args.init_tr,
        n_samples=args.n_walkers,
        steps_between_refill=args.steps_between_refill,
        max_active_states=args.max_active_states,
        n_compactions=args.n_compactions,
    )
    integrator.pre_warm_compaction_buckets(
        params=None,
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

    # Record to paper_utils.results JSONL.
    if not args.no_record:
        from utils.paper_utils.results import record, make_id
        net = dict(depth=args.depth, width=args.width,
                   activation=args.activation, loss=args.loss)
        variant_id = _slug(label)
        try:
            record({
                "id": make_id(args.shape, net, variant_id, "wos", seed=args.seed),
                "shape": args.shape,
                "network": net,
                "variant": variant_id,
                "algorithm": "wos",
                "metrics": {
                    "warm_best_ms": float(t_min),
                    "warm_median_ms": float(t_med),
                    "mean_n_steps": float(n_steps_mean),
                    "rms_u_low_minus_high": float(rms),
                    "max_u_low_minus_high": float(max_abs),
                    "peak_gb": float(peak_gb),
                    "in_use_gb": float(in_use_gb),
                    "n_test_points": int(test_points.shape[0]),
                    "n_walkers": int(args.n_walkers),
                    "n_walkers_high": int(args.n_walkers_high),
                    "max_steps": int(args.max_steps),
                    "init_tr": float(args.init_tr),
                    "eps": float(args.eps),
                    "boundary_fn": args.boundary_fn,
                    "n_compactions": int(args.n_compactions),
                    "n_warmup": int(args.n_warmup),
                    "n_timed": int(args.n_timed),
                },
                "system": {"gpu_index": os.environ.get("CUDA_VISIBLE_DEVICES")},
                "seed": int(args.seed),
            })
            print(f'  recorded → paper_utils.results / wos.jsonl', flush=True)
        except Exception as e:
            print(f'  [warn] record() failed: {type(e).__name__}: {e}', flush=True)
    print(f'  saved → {method_dir}/u.npz', flush=True)


surface = jnpboundedImplicit(autobounds_max_degree=2, fn=fn_2arg, level_set=args.isolevel,
                             dim=3, use_warp=jax.default_backend() == "gpu")
label = f"autobound deg-2 ({surface.range_implementation})"
print(f"\n[{label}]", flush=True)
run_method(label, surface)
