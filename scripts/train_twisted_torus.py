"""Train a small Softplus MLP on a twisted torus; save a portable NPZ checkpoint.

The target is the original notebooks' twisted field (not an exact SDF after
warping). The loader is also a factory for render_orbit.py:
  --surface scripts/train_twisted_torus.py:load_surface
  --surface-kwargs '{"checkpoint":"assets/models/twisted_torus_softplus.npz"}'
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check_inference(params, beta):
    """Check compiled and single-point inference against a small NumPy reference."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    xs = np.random.default_rng(123).uniform(-1., 1., (16, 3)).astype(np.float32)
    reference = xs
    for w, b in params[:-1]:
        reference = np.logaddexp(0., beta*(reference@np.asarray(w)+np.asarray(b)))/beta
    reference = reference@np.asarray(params[-1][0])+np.asarray(params[-1][1])
    predictions = jax.jit(lambda x: network(params, x, beta))(jnp.asarray(xs))
    dynamic = jax.jit(network, static_argnums=2)(params, jnp.asarray(xs), beta)
    point_fn = jax.jit(lambda x: network(params, x, beta))
    single = jnp.stack([point_fn(jnp.asarray(x)) for x in xs])
    for result in (predictions, dynamic, single):
        np.testing.assert_allclose(result, reference, atol=2e-5, rtol=2e-5,
            err_msg='Compiled MLP inference disagrees with the NumPy reference')
    return max(float(np.max(np.abs(np.asarray(result)-reference)))
               for result in (predictions, dynamic, single))


def network(params, x, beta=10.):
    import jax
    for weight, bias in params[:-1]:
        # Scale both affine terms explicitly: XLA 0.11 GPU GEMM rewriting
        # can turn beta*(x@weight+bias) into beta*(x@weight)+bias.
        x = jax.nn.softplus(x @ (beta * weight) + beta * bias) / beta
    return x @ params[-1][0] + params[-1][1]


def twisted_torus(x, major=0.5, minor=0.2, twist=5.):
    import jax.numpy as jnp
    angle = twist*x[..., 1]
    c, s = jnp.cos(angle), jnp.sin(angle)
    u = c*x[..., 0] - s*x[..., 2]
    v = s*x[..., 0] + c*x[..., 2]
    radial = jnp.sqrt(u*u + x[..., 1]**2) - major
    return jnp.sqrt(radial*radial + v*v) - minor


def load_surface(checkpoint: str, use_warp: bool = True):
    """Load a frozen checkpoint into the Taylor-bounded surface interface."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    import jnp_bounds.autobounds as jb
    jax.config.update('jax_default_matmul_precision', 'highest')
    from surfaces.implicit.jnp_bounded import jnpboundedImplicit
    with np.load(checkpoint, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata']))
        weights = tuple((jnp.asarray(data[f'w{i}']), jnp.asarray(data[f'b{i}']))
                        for i in range(metadata['depth']+1))
    check_inference(weights, metadata['beta'])
    # Absorb the Softplus scale into affine weights. This is algebraically
    # identical but avoids scalar interval broadcasts between bound layers.
    beta = metadata['beta']
    affine = tuple((w*beta if i == 0 else w/beta if i == len(weights)-1 else w,
                    b*beta if i < len(weights)-1 else b)
                   for i, (w, b) in enumerate(weights))
    def field(x):
        for w, b in affine[:-1]:
            x = jax.nn.softplus(x @ w + b)
        return x @ affine[-1][0] + affine[-1][1]
    surface = jnpboundedImplicit(fn=lambda unused, x: field(x), dim=3, level_set=0.,
                                 autobounds_max_degree=2, use_warp=use_warp)
    # The bounding backend traces a function of position only. Freeze this
    # checkpoint in both the field and the bound, rather than tracing params.
    surface.taylor_bounds = jb.taylor_bounds(
        field, max_degree=2, composition_algorithm='softplus_cheap',
        composition_implementation='warp_fused' if use_warp else 'jax')
    return surface, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='gpu', help='gpu, gpu:N, or cpu')
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--beta', type=float, default=10., help='softplus(beta*x)/beta')
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--domain', type=float, default=1.)
    parser.add_argument('--major', type=float, default=0.5)
    parser.add_argument('--minor', type=float, default=0.2)
    parser.add_argument('--twist', type=float, default=5.)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--out', type=Path, default=Path('assets/models/twisted_torus_softplus.npz'))
    args = parser.parse_args()
    import math
    for name in ('hidden', 'depth', 'steps', 'batch_size', 'beta', 'lr', 'domain', 'major', 'minor'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f'{name} must be positive and finite')
    if not math.isfinite(args.twist) or args.major <= args.minor:
        parser.error('twist must be finite and major must exceed minor')
    if args.device == 'cpu':
        os.environ['JAX_PLATFORMS'] = 'cpu'
    elif args.device == 'gpu' or (args.device.startswith('gpu:') and args.device[4:].isdigit()):
        os.environ['JAX_PLATFORMS'] = 'cuda'
        if ':' in args.device:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(':')[1]
    else:
        parser.error('--device must be gpu, gpu:N, or cpu')
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    jax.config.update('jax_default_matmul_precision', 'highest')
    device = jax.devices()[0]
    if args.device.startswith('gpu') and device.platform != 'gpu':
        raise RuntimeError('GPU training requested but JAX did not select a GPU')
    print(f'Training on {device}', flush=True)
    widths = [3] + [args.hidden]*args.depth + [1]
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.depth+2)
    params = tuple((jax.nn.initializers.glorot_uniform()(keys[i], (a, b)), jnp.zeros(b))
                   for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])))
    optimizer = optax.adam(optax.cosine_decay_schedule(args.lr, args.steps, alpha=0.1))
    state = optimizer.init(params)
    target = lambda x: twisted_torus(x, args.major, args.minor, args.twist)

    def loss_fn(p, xs, ys):
        error = network(p, xs, args.beta)[..., 0] - ys
        # Preserve the global field while giving the zero set more influence.
        weights = 1. + 9.*jnp.exp(-(ys/0.06)**2)
        return jnp.mean(weights*error**2) / jnp.mean(weights)

    @jax.jit
    def step(p, opt_state, key):
        key, subkey = jax.random.split(key)
        xs = jax.random.uniform(subkey, (args.batch_size, 3), minval=-args.domain, maxval=args.domain)
        loss, grads = jax.value_and_grad(loss_fn)(p, xs, target(xs))
        updates, opt_state = optimizer.update(grads, opt_state, p)
        return optax.apply_updates(p, updates), opt_state, key, loss

    key = keys[-1]
    started = time.perf_counter()
    for i in range(args.steps):
        params, state, key, loss = step(params, state, key)
        if i % max(1, args.steps//20) == 0 or i == args.steps-1:
            print(f'Step {i+1}/{args.steps}: weighted MSE {float(loss):.7g}', flush=True)
    jax.block_until_ready(params)
    elapsed = time.perf_counter()-started
    inference_error = check_inference(params, args.beta)
    eval_key, surface_key = jax.random.split(jax.random.PRNGKey(args.seed+1000))
    xs = jax.random.uniform(eval_key, (100000, 3), minval=-args.domain, maxval=args.domain)
    error = network(params, xs, args.beta)[..., 0] - target(xs)
    # Invert the twist on exact torus points for an independent surface check.
    angles = jax.random.uniform(surface_key, (10000, 2), minval=0., maxval=2*jnp.pi)
    a, b = angles[:, 0], angles[:, 1]
    u = (args.major+args.minor*jnp.cos(b))*jnp.cos(a)
    v = args.minor*jnp.sin(b)
    y = (args.major+args.minor*jnp.cos(b))*jnp.sin(a)
    c, s = jnp.cos(args.twist*y), jnp.sin(args.twist*y)
    points = jnp.stack((c*u+s*v, y, -s*u+c*v), axis=-1)
    surface_error = jnp.abs(network(params, points, args.beta)[..., 0])
    metadata = {k: v for k, v in vars(args).items() if k != 'out'}
    metadata.update(activation='scaled_softplus', target='twisted_torus', format_version=1,
                    parameter_count=sum(x.size for x in jax.tree.leaves(params)),
                    training_seconds=elapsed, jax_device=str(device),
                    inference_max_abs_error=inference_error, xla_flags=os.environ.get("XLA_FLAGS", ""),
                    device_kind=device.device_kind,
                    validation_mse=float(jnp.mean(error**2)), validation_mae=float(jnp.mean(jnp.abs(error))),
                    surface_residual_mae=float(jnp.mean(surface_error)),
                    surface_residual_p95=float(jnp.quantile(surface_error, .95)))
    arrays = {f'{name}{i}': np.asarray(value) for i, pair in enumerate(params)
              for name, value in zip(('w', 'b'), pair)}
    if not all(np.all(np.isfinite(v)) for v in arrays.values()):
        raise RuntimeError('training produced nonfinite weights')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('wb') as output:
        np.savez_compressed(output, **arrays, metadata=json.dumps(metadata))
    args.out.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2), flush=True)
    print(f'Saved {args.out}', flush=True)


if __name__ == '__main__':
    main()
