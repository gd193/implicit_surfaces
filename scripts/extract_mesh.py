"""Extract a mesh from an implicit field and write it to OBJ.

Usage:
    python scripts/extract_mesh.py --surface sphere --resolution 64 --out sphere.obj
    python scripts/extract_mesh.py --surface torus --extractor dc --resolution 64 --out torus.obj
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--surface", default="sphere", choices=["sphere", "torus"])
    p.add_argument("--extractor", default="dc", choices=["mc", "dc"])
    p.add_argument("--resolution", type=int, default=64, help="Cells per axis")
    p.add_argument("--sparse", action="store_true", help="Enable Lipschitz-sparse pruning")
    p.add_argument("--bounds", type=float, default=1.5, help="Half-width of root box")
    p.add_argument("--out", default="mesh.obj")
    p.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"

    import jax
    import jax.numpy as jnp

    from meshExtractor import make_dual_contouring, make_marching_cubes, write_obj
    from surfaces.implicit.primitives import sphere_sdf, torus_sdf

    print(f"JAX devices: {[d.platform for d in jax.devices()]}")

    if args.surface == "sphere":
        field = sphere_sdf
        params = {"center": jnp.zeros(3, dtype=jnp.float32), "radius": jnp.float32(1.0)}
    elif args.surface == "torus":
        field = torus_sdf
        params = {
            "center": jnp.zeros(3, dtype=jnp.float32),
            "major": jnp.float32(0.65),
            "minor": jnp.float32(0.25),
        }
    else:
        raise ValueError(args.surface)

    b = float(args.bounds)
    bounds = ((-b, -b, -b), (b, b, b))
    lipschitz_bound = 1.0 if args.sparse else None
    factory = make_marching_cubes if args.extractor == "mc" else make_dual_contouring
    extract = factory(field, bounds, resolution=args.resolution, lipschitz_bound=lipschitz_bound)

    t0 = time.time()
    verts, tris = extract(params)
    dt = time.time() - t0

    print(f"extracted {tris.shape[0]} triangles, {verts.shape[0]} vertices in {dt:.2f}s")
    write_obj(args.out, verts, tris)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
