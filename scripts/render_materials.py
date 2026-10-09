"""Render diffuse, metal, and glass implicit surfaces with the shared marcher.

Example: python scripts/render_materials.py --device cpu --out materials.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--passes", type=int, default=1, help="Average independent sample batches")
    parser.add_argument("--scene", choices=("spheres", "primitives"), default="spheres")
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("materials.png"))
    args = parser.parse_args()
    from render_sphere import configure_device
    configure_device(args.device)

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from rayMarcher.camera import Camera
    from rayMarcher.integrator import MaterialIntegrator
    from rayMarcher.materials import Material
    from surfaces.implicit.lipschitz import sdfImplicit

    ball = sdfImplicit(fn=lambda p, x: jnp.linalg.norm(x - p["center"]) - p["radius"], dim=3)
    ground = sdfImplicit(fn=lambda p, x: x[1] + 0.5, dim=3)

    def rounded_box(p, x):
        x = x - p["center"]
        angle = 0.45
        c, s = jnp.cos(angle), jnp.sin(angle)
        local = jnp.array([c*x[0] - s*x[2], x[1], s*x[0] + c*x[2]])
        q = jnp.abs(local) - 0.42
        return jnp.linalg.norm(jnp.maximum(q, 0.)) + jnp.minimum(jnp.max(q), 0.) - 0.08

    def torus(p, x):
        x = x - p["center"]
        q = jnp.array([jnp.linalg.norm(x[:2]) - 0.34, x[2]])
        return jnp.linalg.norm(q) - 0.16

    surfaces = [ball, ball, ball, ground] if args.scene == "spheres" else [
        sdfImplicit(fn=rounded_box, dim=3), sdfImplicit(fn=torus, dim=3), ball, ground,
    ]
    camera = Camera.setup_Camera(
        args.width, args.height, 35., jnp.array([0., 1.05, 4.7]),
        jnp.array([0., 0., 0.]), jnp.array([0., 1., 0.]),
    )
    materials = Material(
        type=jnp.array([0, 1, 2, 0]),
        albedo=jnp.array([[0.7, 0.15, 0.08], [0.92, 0.7, 0.3], [0.98, 0.99, 1.], [0.55, 0.6, 0.67]]),
        fuzz=jnp.array([0., 0.035, 0., 0.]),
        ref_idx=jnp.array([1., 1., 1.5, 1.]),
    )
    renderer = MaterialIntegrator(
        camera, materials, jnp.array([0.5, 0.7, 1.]), surfaces,
        samples_per_pixel=args.samples, max_depth=args.depth,
        scene_bounds=((-20., -2., -20.), (20., 8., 20.)), max_ray_distance=60.,
    )
    scene_params = [{"center": jnp.array([x, 0., 0.]), "radius": jnp.array(0.5)}
                    for x in (-1.1, 0., 1.1)] + [None]
    if args.passes < 1:
        parser.error("--passes must be positive")
    accumulated = jnp.zeros((args.height, args.width, 3))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {args.scene}: {args.width}x{args.height}, {args.samples * args.passes} spp on {jax.devices()[0].platform}", flush=True)
    for sample_pass in range(args.passes):
        image = renderer.render(scene_params, jax.random.PRNGKey(sample_pass))
        if not bool(jnp.all(jnp.isfinite(image))):
            raise RuntimeError("render produced nonfinite colors")
        accumulated = accumulated + image
        plt.imsave(args.out, jnp.sqrt(jnp.clip(accumulated / (sample_pass + 1), 0., 1.)))
        print(f"Pass {sample_pass + 1}/{args.passes}: {args.out}", flush=True)



if __name__ == "__main__":
    main()
