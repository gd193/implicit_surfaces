"""Render a rotating video of a neural implicit surface.

Trains a small MLP on a target SDF (default: twisted torus), then orbits the
camera around the lookat point on the plane perpendicular to a user-chosen axis
and writes a gif/mp4.
"""
import argparse
import math
import os
import sys
from pathlib import Path
from typing import Callable, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Render a rotating video of a neural implicit surface.")
    parser.add_argument("--device", default="cpu", help="'cpu' or 'gpu' or 'gpu:N'.")
    parser.add_argument("--target", default="torus", choices=["torus", "sphere"])
    parser.add_argument(
        "--axis",
        default="y",
        help="Rotation axis: 'x', 'y', 'z', or three comma-separated floats e.g. '0,1,0.5'.",
    )
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--radius", type=float, default=2.5, help="Camera orbit radius around lookat.")
    parser.add_argument("--height-offset", type=float, default=0.8,
                        help="Camera offset along the rotation axis (0 = orbit in the axis-perpendicular plane).")
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--train-batch", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--out", default="neural_implicit.gif",
                        help="Output path. .gif uses imageio's gif writer; .mp4 needs ffmpeg.")
    return parser.parse_args()


def configure_device(device: str) -> None:
    """Must run before importing jax."""
    device = device.lower()
    if device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    elif device == "gpu" or device.startswith("gpu:"):
        gpu_id = device.split(":", 1)[1] if ":" in device else "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    else:
        raise ValueError(f"Unknown device spec: {device!r}")


def parse_axis(axis_arg: str):
    import jax.numpy as jnp
    presets = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}
    if axis_arg.lower() in presets:
        v = jnp.array(presets[axis_arg.lower()])
    else:
        parts = [float(p) for p in axis_arg.replace(" ", ",").split(",") if p]
        if len(parts) != 3:
            raise ValueError(f"--axis must be 'x'/'y'/'z' or three floats (got {axis_arg!r})")
        v = jnp.array(parts)
    norm = jnp.linalg.norm(v)
    if float(norm) < 1e-8:
        raise ValueError("--axis must be non-zero")
    return v / norm


def orbit_basis(axis):
    """Return (e1, e2) — an orthonormal basis spanning the plane perpendicular to `axis`."""
    import jax.numpy as jnp
    # Pick a reference vector not parallel to axis.
    ref = jnp.array([0.0, 1.0, 0.0])
    if abs(float(jnp.dot(axis, ref))) > 0.9:
        ref = jnp.array([1.0, 0.0, 0.0])
    e1 = ref - jnp.dot(ref, axis) * axis
    e1 = e1 / jnp.linalg.norm(e1)
    e2 = jnp.cross(axis, e1)
    return e1, e2


class MLPParams(NamedTuple):
    w1: "jnp.ndarray"; b1: "jnp.ndarray"
    w2: "jnp.ndarray"; b2: "jnp.ndarray"
    w3: "jnp.ndarray"; b3: "jnp.ndarray"
    w4: "jnp.ndarray"; b4: "jnp.ndarray"


def main():
    args = parse_args()
    configure_device(args.device)

    import imageio.v2 as imageio
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import numpy as np
    import optax

    from implicitSurfaces.jnpboundedImplicit import jnpboundedImplicit
    from rayMarcher.camera import Camera
    from rayMarcher.marcher import SimpleIntegrator

    print(f"JAX running on: {[d.platform for d in jax.devices()]}")

    # --- Network definition + init -------------------------------------------
    def neural_network(params: MLPParams, x: jnp.ndarray) -> jnp.ndarray:
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

    def sdf_sphere(x): return jnp.linalg.norm(x, axis=-1) - 0.6

    def sdf_torus(x, r=0.2, R=0.5):
        q = jnp.sqrt(x[0] ** 2 + x[2] ** 2) - R
        return jnp.sqrt(q ** 2 + x[1] ** 2) - r

    def twist(fn, k):
        def fn_twisted(x):
            c, s = jnp.cos(k * x[1]), jnp.sin(k * x[1])
            m = jnp.array([[c, 0.0, -s], [s, 0.0, c], [0.0, 1.0, 0.0]])
            return fn(m @ x)
        return fn_twisted

    target_fn = sdf_sphere if args.target == "sphere" else twist(sdf_torus, k=5.0)

    # --- Train ---------------------------------------------------------------
    key = jr.PRNGKey(args.seed)
    key, init_key = jr.split(key)
    params = init_params(init_key, d_in=3, d_hidden=args.hidden, d_out=1)
    optim = optax.adam(args.lr)
    opt_state = optim.init(params)

    def loss_fn(params, xs):
        preds = jax.vmap(lambda x: neural_network(params, x)[0])(xs)
        targets = jax.vmap(target_fn)(xs)
        return jnp.mean((preds - targets) ** 2)

    @jax.jit
    def train_step(params, opt_state, xs):
        loss, grads = jax.value_and_grad(loss_fn)(params, xs)
        updates, opt_state = optim.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    print(f"training MLP (target={args.target}, hidden={args.hidden}, steps={args.train_steps}, lr={args.lr})")
    log_every = max(1, args.train_steps // 10)
    for step in range(args.train_steps):
        key, sub = jr.split(key)
        xs = jr.uniform(sub, (args.train_batch, 3), minval=-1.0, maxval=1.0)
        params, opt_state, loss = train_step(params, opt_state, xs)
        if step % log_every == 0 or step == args.train_steps - 1:
            print(f"  step {step:5d}  loss={float(loss):.6f}")
    trained_params = params

    # --- Wrap in jnpboundedImplicit ------------------------------------------
    def neural_fn_1arg(x):
        return neural_network(trained_params, x)

    def neural_fn_2arg(_dummy, x):
        return neural_fn_1arg(x)

    surface = jnpboundedImplicit(
        autobounds_max_degree=2,
        fn=neural_fn_2arg,
        level_set=0.0,
        dim=3,
    )
    import jnp_bounds.autobounds as jb
    surface.taylor_bounds = jb.taylor_bounds(neural_fn_1arg, max_degree=2, use_warp=True)

    def neural_normal(_params, x):
        grad = jax.grad(lambda y: neural_fn_1arg(y)[0])(x)
        return grad / (jnp.linalg.norm(grad) + 1e-12)

    # --- Camera orbit setup --------------------------------------------------
    axis = parse_axis(args.axis)
    e1, e2 = orbit_basis(axis)
    lookat = jnp.array([0.0, 0.0, 0.0])
    # vup must not be parallel to the view direction; use the rotation axis as "up"
    # when feasible, otherwise fall back to world y.
    vup = axis if abs(float(jnp.dot(axis, jnp.array([0.0, 1.0, 0.0])))) < 0.99 else jnp.array([1.0, 0.0, 0.0])

    # Reuse a single SimpleIntegrator across frames. Static marcher config is set
    # once in __init__ so the JIT-compiled step kernel and post-process cache hit
    # across frames; only camera (mutable) is swapped per frame.
    init_camera = Camera.setup_Camera(
        image_width=args.width, image_height=args.height,
        vfov=40.0,
        lookfrom=jnp.array([args.radius, args.height_offset, 0.0]) + lookat,
        lookat=lookat, vup=vup,
    )
    integrator = SimpleIntegrator(
        camera=init_camera,
        background_color=jnp.array([0.5, 0.7, 1.0]),
        ray_march_fn=surface.ray_step,
        surface_normal_fn=neural_normal,
        image_height=args.height,
        image_width=args.width,
        max_ray_march_steps=128,
        max_ray_distance=20.0,
        level_set=0.0,
        epsilon=1e-3,
        n_compactions=4,
    )

    print(f"rendering {args.frames} frames around axis={args.axis} "
          f"(first frame JIT-compiles the Taylor-bounded marcher; can take minutes)")
    frames = []
    for i in range(args.frames):
        theta = 2.0 * math.pi * i / args.frames
        offset = args.radius * (math.cos(theta) * e1 + math.sin(theta) * e2) + args.height_offset * axis
        lookfrom = lookat + offset
        integrator.camera = Camera.setup_Camera(
            image_width=args.width, image_height=args.height,
            vfov=40.0,
            lookfrom=lookfrom, lookat=lookat, vup=vup,
        )
        image, _ = integrator.render(
            params={},
            key=jax.random.PRNGKey(1 + i),
            init_trust_region_size=0.5,
        )
        frame = np.asarray(jnp.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        frames.append(frame)
        print(f"  frame {i + 1}/{args.frames}")

    if args.out.lower().endswith(".gif"):
        # loop=0 -> infinite loop in viewers/browsers
        imageio.mimsave(args.out, frames, duration=1.0 / args.fps, loop=0)
    else:
        imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"wrote {args.out}  ({args.frames} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
