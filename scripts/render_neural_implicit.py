"""Render a neural implicit surface using jnpboundedImplicit.

A small swish MLP is fit (with optax Adam) to a target SDF in [-1,1]^3,
then rendered via the Taylor-bounded ray marcher. Default target is a twisted
torus (matches notebooks/test_tracing_autobounded_3d_fn.ipynb).
"""
import argparse
import math
import os
import sys
from pathlib import Path
from typing import Callable, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Render a neural implicit surface.")
    parser.add_argument("--device", default="cpu", help="'cpu' or 'gpu' or 'gpu:N'.")
    parser.add_argument("--target", default="torus", choices=["torus", "sphere"])
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=64, help="Hidden layer width.")
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--train-batch", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--out", default="neural_implicit.png")
    return parser.parse_args()


def configure_device(device: str) -> None:
    """Must run before importing jax."""
    device = device.lower()
    if device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    elif device == "gpu" or device.startswith("gpu:"):
        gpu_id = device.split(":", 1)[1] if ":" in device else "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        # Don't set JAX_PLATFORMS=cuda; the auto-discovered cuda plugin handles it.
    else:
        raise ValueError(f"Unknown device spec: {device!r}")


class MLPParams(NamedTuple):
    w1: "jnp.ndarray"; b1: "jnp.ndarray"
    w2: "jnp.ndarray"; b2: "jnp.ndarray"
    w3: "jnp.ndarray"; b3: "jnp.ndarray"
    w4: "jnp.ndarray"; b4: "jnp.ndarray"


def main():
    args = parse_args()
    configure_device(args.device)

    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import matplotlib.pyplot as plt
    import optax

    from implicitSurfaces.jnpboundedImplicit import jnpboundedImplicit
    from rayMarcher.camera import Camera
    from rayMarcher.marcher import SimpleIntegrator

    print(f"JAX running on: {[d.platform for d in jax.devices()]}")

    # --- 4-layer swish MLP (mirrors notebook architecture) ------------------
    def neural_network(params: MLPParams, x: jnp.ndarray) -> jnp.ndarray:
        # Returns shape (1,). No .squeeze() / [...] reduction inside; jnp_bounds
        # chokes on the squeeze primitive when tracing Taylor bounds.
        x = jax.nn.swish(x @ params.w1 + params.b1)
        x = jax.nn.swish(x @ params.w2 + params.b2)
        x = jax.nn.swish(x @ params.w3 + params.b3)
        return x @ params.w4 + params.b4

    def init_params(key, d_in: int, d_hidden: int, d_out: int) -> MLPParams:
        keys = jr.split(key, 8)
        u_in, u_h, u_out = 1.0 / math.sqrt(d_in), 1.0 / math.sqrt(d_hidden), 1.0 / math.sqrt(d_hidden)
        return MLPParams(
            w1=jr.uniform(keys[0], (d_in, d_hidden), minval=-u_in, maxval=u_in),
            b1=jr.uniform(keys[1], (d_hidden,), minval=-u_h, maxval=u_h),
            w2=jr.uniform(keys[2], (d_hidden, d_hidden), minval=-u_h, maxval=u_h),
            b2=jr.uniform(keys[3], (d_hidden,), minval=-u_h, maxval=u_h),
            w3=jr.uniform(keys[4], (d_hidden, d_hidden), minval=-u_h, maxval=u_h),
            b3=jr.uniform(keys[5], (d_hidden,), minval=-u_h, maxval=u_h),
            w4=jr.uniform(keys[6], (d_hidden, d_out), minval=-u_out, maxval=u_out),
            b4=jr.uniform(keys[7], (d_out,), minval=-1.0 / math.sqrt(d_out), maxval=1.0 / math.sqrt(d_out)),
        )

    # --- Target SDFs --------------------------------------------------------
    def sdf_sphere(x: jnp.ndarray, radius: float = 0.6) -> jnp.ndarray:
        return jnp.linalg.norm(x, axis=-1) - radius

    def sdf_torus(x: jnp.ndarray, r: float = 0.2, R: float = 0.5) -> jnp.ndarray:
        q = jnp.sqrt(x[0] ** 2 + x[2] ** 2) - R
        return jnp.sqrt(q ** 2 + x[1] ** 2) - r

    def twist(fn: Callable, k: float) -> Callable:
        def fn_twisted(x: jnp.ndarray) -> jnp.ndarray:
            c, s = jnp.cos(k * x[1]), jnp.sin(k * x[1])
            m = jnp.array([[c, 0.0, -s], [s, 0.0, c], [0.0, 1.0, 0.0]])
            return fn(m @ x)
        return fn_twisted

    if args.target == "sphere":
        target_fn = sdf_sphere
    else:
        target_fn = twist(sdf_torus, k=5.0)

    # --- Train --------------------------------------------------------------
    key = jr.PRNGKey(args.seed)
    key, init_key = jr.split(key)
    params = init_params(init_key, d_in=3, d_hidden=args.hidden, d_out=1)
    optim = optax.adam(args.lr)
    opt_state = optim.init(params)

    def loss_fn(params, xs):
        # Pick scalar [0] outside any autobound trace to avoid the squeeze primitive.
        preds = jax.vmap(lambda x: neural_network(params, x)[0])(xs)
        targets = jax.vmap(target_fn)(xs)
        return jnp.mean((preds - targets) ** 2)

    @jax.jit
    def train_step(params, opt_state, xs):
        loss, grads = jax.value_and_grad(loss_fn)(params, xs)
        updates, opt_state = optim.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    print(f"training MLP (target={args.target}, hidden={args.hidden}, steps={args.train_steps}, lr={args.lr})")
    log_every = max(1, args.train_steps // 20)
    for step in range(args.train_steps):
        key, sub = jr.split(key)
        xs = jr.uniform(sub, (args.train_batch, 3), minval=-1.0, maxval=1.0)
        params, opt_state, loss = train_step(params, opt_state, xs)
        if step % log_every == 0 or step == args.train_steps - 1:
            print(f"  step {step:5d}  loss={float(loss):.6f}")

    trained_params = params

    # --- Wrap the trained network in jnpboundedImplicit ---------------------
    def neural_fn_1arg(x: jnp.ndarray) -> jnp.ndarray:
        return neural_network(trained_params, x)

    # implicitSurface.__call__ assumes (params, x); accept and ignore.
    def neural_fn_2arg(_dummy, x):
        return neural_fn_1arg(x)

    surface = jnpboundedImplicit(
        autobounds_max_degree=2,
        fn=neural_fn_2arg,
        level_set=0.0,
        dim=3,
    )
    # Rebuild taylor_bounds against the 1-arg variant (jnp_bounds expects single input).
    import jnp_bounds.autobounds as jb
    surface.taylor_bounds = jb.taylor_bounds(neural_fn_1arg, max_degree=2, use_warp=True)

    def neural_normal(_params, x):
        grad = jax.grad(lambda y: neural_fn_1arg(y)[0])(x)
        return grad / (jnp.linalg.norm(grad) + 1e-12)

    # --- Camera + render ----------------------------------------------------
    camera = Camera.setup_Camera(
        image_width=args.width,
        image_height=args.height,
        vfov=40.0,
        lookfrom=jnp.array([2.0, 1.5, 2.0]),
        lookat=jnp.array([0.0, 0.0, 0.0]),
        vup=jnp.array([0.0, 1.0, 0.0]),
    )

    integrator = SimpleIntegrator(
        camera=camera,
        background_color=jnp.array([0.5, 0.7, 1.0]),
        ray_march_fn=surface.ray_step,
        surface_normal_fn=neural_normal,
        image_height=args.height,
        image_width=args.width,
        max_ray_march_steps=1000,
        max_ray_distance=20.0,
        level_set=0.0,
        epsilon=1e-3,
        n_compactions=10,
    )

    print("rendering (first call JIT-compiles the Taylor-bounded marcher; can take minutes)")
    image, niter = integrator.render(
        params={},
        key=jax.random.PRNGKey(1),
        init_trust_region_size=0.5,
    )

    plt.imsave(args.out, jnp.clip(image, 0.0, 1.0))
    print(f"wrote {args.out}  (target={args.target}, mean niter = {float(jnp.mean(niter)):.1f})")


if __name__ == "__main__":
    main()
