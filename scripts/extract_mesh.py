"""Extract a mesh from a bounded implicit surface and write it to OBJ.

Usage:
    python scripts/extract_mesh.py --surface sphere --depth 5 --out sphere.obj
    python scripts/extract_mesh.py --surface poisson --depth 5 --subcell 2 --out poisson.obj
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--surface", default="sphere", choices=["sphere", "poisson"])
    p.add_argument("--tree", default="octree", choices=["octree", "kdtree"])
    p.add_argument("--extractor", default="mc", choices=["mc", "surface_nets", "dc"])
    p.add_argument("--depth", type=int, default=5, help="Tree max_depth")
    p.add_argument("--subcell", type=int, default=0, help="MC subcell_depth")
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

    from implicitSurfaces.lipschitzImplicit import sdfImplicit
    from implicitSurfaces.sdf_primitives import sphere_sdf
    from implicitSurfaces.harmonicImplicit import poissonSurface
    from meshExtractor import extract_mesh
    from meshExtractor.trees import KdTree, Octree
    from meshExtractor.extractors import DualContouring, MarchingCubes, SurfaceNets
    from meshExtractor.mesh_io import write_obj

    print(f"JAX devices: {[d.platform for d in jax.devices()]}")

    if args.surface == "sphere":
        surface = sdfImplicit(fn=sphere_sdf, dim=3)
        params = {"center": jnp.array([0.0, 0.0, 0.0]),
                  "radius": jnp.array(1.0)}
    elif args.surface == "poisson":
        surface = poissonSurface(scale=1.0, offset=0.0, level_set=5.0, alpha=1.25, dim=3)
        params = {"p": jnp.array([
            [-0.4, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.0, 0.4, 0.0],
        ])}
    else:
        raise ValueError(args.surface)

    b = args.bounds
    bounds = (jnp.array([-b, -b, -b]), jnp.array([b, b, b]))
    tree = Octree(max_depth=args.depth) if args.tree == "octree" else KdTree(max_depth=3 * args.depth)
    extractor = {
        "mc": MarchingCubes(subcell_depth=args.subcell),
        "surface_nets": SurfaceNets(),
        "dc": DualContouring(),
    }[args.extractor]

    t0 = time.time()
    verts, tris = extract_mesh(surface, params, bounds, tree, extractor)
    dt = time.time() - t0

    print(f"extracted {tris.shape[0]} triangles, {verts.shape[0]} vertices in {dt:.2f}s")
    write_obj(args.out, verts, tris)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
