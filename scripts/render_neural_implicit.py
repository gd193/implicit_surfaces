"""Render a neural implicit surface using jnpboundedImplicit.

A small swish MLP is fit (with optax Adam) to a target SDF in [-1,1]^3,
then rendered via the Taylor-bounded ray marcher. Default target is a twisted torus.
"""
import argparse
import math
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


class MLPParams(NamedTuple):
    w1: "jnp.ndarray"; b1: "jnp.ndarray"
    w2: "jnp.ndarray"; b2: "jnp.ndarray"
    w3: "jnp.ndarray"; b3: "jnp.ndarray"
    w4: "jnp.ndarray"; b4: "jnp.ndarray"


def train_surface(*, target="torus", hidden=64, train_steps=2000, train_batch=10000,
                  lr=1e-2, seed=2):
    """Fit the target field and return a surface and its trained parameters."""
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import optax
    from surfaces.implicit.jnp_bounded import jnpboundedImplicit

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

    if target not in ("torus", "sphere"):
        raise ValueError("target must be torus or sphere")
    if target == "sphere":
        target_fn = sdf_sphere
    else:
        target_fn = twist(sdf_torus, k=5.0)

    # --- Train --------------------------------------------------------------
    key = jr.PRNGKey(seed)
    key, init_key = jr.split(key)
    params = init_params(init_key, d_in=3, d_hidden=hidden, d_out=1)
    optim = optax.adam(lr)
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

    print(f"training MLP (target={target}, hidden={hidden}, steps={train_steps}, lr={lr})")
    log_every = max(1, train_steps // 20)
    for step in range(train_steps):
        key, sub = jr.split(key)
        xs = jr.uniform(sub, (train_batch, 3), minval=-1.0, maxval=1.0)
        params, opt_state, loss = train_step(params, opt_state, xs)
        if step % log_every == 0 or step == train_steps - 1:
            print(f"  step {step:5d}  loss={float(loss):.6f}")

    surface = jnpboundedImplicit(
        autobounds_max_degree=2,
        fn=neural_network,
        level_set=0.0,
        dim=3,
    )
    return surface, params


def main():
    args = parse_args()
    from scripts.render_sphere import configure_device
    configure_device(args.device)

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from rayMarcher.camera import Camera
    from rayMarcher.integrator import SimpleIntegrator

    surface, params = train_surface(target=args.target, hidden=args.hidden,
                                    train_steps=args.train_steps, train_batch=args.train_batch,
                                    lr=args.lr, seed=args.seed)
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
        surface=surface,
        max_ray_march_steps=1000,
        max_ray_distance=20.0,
        epsilon=1e-3,
        n_compactions=10,
    )

    print("rendering (first call JIT-compiles the Taylor-bounded marcher; can take minutes)")
    image, niter = integrator.render(
        params=params,
        key=jax.random.PRNGKey(1),
        init_trust_region_size=0.5,
    )

    plt.imsave(args.out, jnp.clip(image, 0.0, 1.0))
    print(f"wrote {args.out}  (target={args.target}, mean niter = {float(jnp.mean(niter)):.1f})")


if __name__ == "__main__":
    main()
