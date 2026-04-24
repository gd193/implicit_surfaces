"""Minimal demo: render an SDF in 3D via the ray marcher."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Render a 3D SDF.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device: 'cpu' or 'gpu' (default 0) or 'gpu:N' for GPU index N.",
    )
    parser.add_argument(
        "--sdf",
        default="sphere",
        choices=["sphere", "bbox"],
        help="Which SDF primitive to render.",
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--out", default="sdf.png")
    return parser.parse_args()


def configure_device(device: str) -> None:
    """Must be called before importing jax."""
    device = device.lower()
    if device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    elif device == "gpu" or device.startswith("gpu:"):
        gpu_id = device.split(":", 1)[1] if ":" in device else "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        # Don't set JAX_PLATFORMS=cuda; the auto-discovered cuda plugin handles it.
    else:
        raise ValueError(f"Unknown device spec: {device!r} (expected 'cpu', 'gpu', or 'gpu:N')")


def build_scene(sdf_name: str):
    """Returns (sdf_fn, params) for the named primitive. All entries assume Lipschitz constant 1."""
    import jax.numpy as jnp
    from implicitSurfaces.sdf_primitives import sphere_sdf, bbox_sdf

    if sdf_name == "sphere":
        return sphere_sdf, {
            "center": jnp.array([0.0, 0.0, 0.0]),
            "radius": jnp.array(1.0),
        }
    if sdf_name == "bbox":
        return bbox_sdf, {
            "min_corner": jnp.array([-0.7, -0.7, -0.7]),
            "max_corner": jnp.array([0.7, 0.7, 0.7]),
        }
    raise ValueError(f"Unknown sdf: {sdf_name!r}")


def main():
    args = parse_args()
    configure_device(args.device)

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt

    from implicitSurfaces.lipschitzImplicit import sdfImplicit
    from rayMarcher.camera import Camera
    from rayMarcher.marcher import SimpleIntegrator

    print(f"JAX running on: {[d.platform for d in jax.devices()]}")

    sdf_fn, params = build_scene(args.sdf)
    surface = sdfImplicit(fn=sdf_fn, dim=3)

    camera = Camera.setup_Camera(
        image_width=args.width,
        image_height=args.height,
        vfov=40.0,
        lookfrom=jnp.array([3.0, 2.0, 3.0]),
        lookat=jnp.array([0.0, 0.0, 0.0]),
        vup=jnp.array([0.0, 1.0, 0.0]),
    )

    integrator = SimpleIntegrator(
        camera=camera,
        background_color=jnp.array([0.5, 0.7, 1.0]),
        ray_march_fn=surface.ray_step,
        surface_normal_fn=surface.normal_at,
        image_height=args.height,
        image_width=args.width,
        max_ray_march_steps=128,
        max_ray_distance=20.0,
        level_set=0.0,
        epsilon=1e-3,
        n_compactions=4,
    )

    image, niter = integrator.render(
        params=params,
        key=jax.random.PRNGKey(0),
        init_trust_region_size=1.0,
    )

    plt.imsave(args.out, jnp.clip(image, 0.0, 1.0))
    print(f"wrote {args.out}  (sdf={args.sdf}, mean niter = {float(jnp.mean(niter)):.1f})")


if __name__ == "__main__":
    main()
