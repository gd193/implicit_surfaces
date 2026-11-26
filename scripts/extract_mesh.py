"""Extract a mesh from an implicit field and write it to OBJ.

Usage:
    python scripts/extract_mesh.py --surface sphere --extractor mc --tolerance .15 --max-depth 6 --out sphere.obj
    python scripts/extract_mesh.py --surface torus --extractor dc --tolerance .15 --max-depth 6 --out torus.obj
"""
import argparse
import os
import sys
import time
from pathlib import Path

if __name__ == "__main__" and not __package__:
    scripts_dir = Path(__file__).resolve().parent
    # Remove scripts/ so its profile package cannot shadow the standard library.
    if str(scripts_dir) in sys.path:
        sys.path.remove(str(scripts_dir))
    sys.path.insert(0, str(scripts_dir.parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--surface", default="sphere", choices=["sphere", "torus"])
    p.add_argument("--extractor", default="dc", choices=["mc", "dc"])
    p.add_argument("--max-depth", type=int, default=6, help="Subdivision limit (default 6)")
    p.add_argument("--tolerance", type=float, default=.15, help="World-space distance tolerance (default 0.15)")
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
    else:
        field = torus_sdf
        params = {
            "center": jnp.zeros(3, dtype=jnp.float32),
            "major": jnp.float32(0.65),
            "minor": jnp.float32(0.25),
        }

    b = float(args.bounds)
    bounds = ((-b, -b, -b), (b, b, b))
    t0 = time.time()
    from surfaces.implicit.lipschitz import sdfImplicit
    factory = make_marching_cubes if args.extractor == "mc" else make_dual_contouring
    extract = factory(sdfImplicit(fn=field, dim=3), bounds,
                      tolerance=args.tolerance, max_depth=args.max_depth)
    mesh, unresolved = extract(params)
    print(f"{len(unresolved)} unresolved boxes")
    dt = time.time() - t0

    print(f"extracted {len(mesh.elements)} triangles, {len(mesh.vertices)} vertices in {dt:.2f}s")
    write_obj(args.out, mesh)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
