"""Unified CLI runner for the Walk-on-Spheres (WoS) showcase scenes.

Usage:
  python -m examples.wos.run <scene> [--solve | --render]

2D scenes:
  poisson_square  Poisson -Lap u = 1, u = 0 on [-1, 1]^2 vs. Fourier series
  mixed_star      Laplace on a 5-point star with VoxelTreePartition per-arm BCs
  two_wires       Magnetostatics: two opposite-current wires in a grounded disk

3D volume scenes (support --solve / --render):
  torus           Scalar Laplace g = cos(3 phi) in an analytic torus
  saturn          Vector harmonic Laplace on sphere + tilted ring (smooth union)
  twisted_torus   Vector harmonic Laplace on a neural Softplus MLP (twist 10)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from examples.wos import common, shapes
from examples.wos.common import (
    compare_panels,
    grid2d,
    load_volume,
    plot,
    plot_bc,
    report,
    solve,
    solve_volume,
    volume_slice,
)
from surfaces.implicit.lipschitz import sdfImplicit
from surfaces.implicit.primitives import bbox_sdf, sphere_sdf, star2d_sdf


def _stages(name: str, solve_only: bool, render_only: bool) -> tuple[bool, bool]:
    if solve_only:
        return True, False
    if render_only:
        return False, True
    return not (common.out_dir(name) / "volume.npz").exists(), True


def _run_vector_volume(name, surface, params, g, lo, hi, *, title, info, slices,
                       cut_normal, cut_camera, solve_now, render_now, n=97, **solve_kwargs):
    """Shared pipeline for 3D vector scenes with harmonic data g (so u = g)."""
    if solve_now:
        solve_volume(name, surface, g, lo, hi, n, params=params, **solve_kwargs)
    if not render_now:
        return

    data = load_volume(name)
    u, mid = data["values"], n // 2
    axes = [np.linspace(lo[i], hi[i], n) for i in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    exact = np.asarray(jax.vmap(g)(jnp.asarray(points, jnp.float32))).reshape(u.shape)
    err, (max_err, _) = report(name, u, np.where(np.isnan(u), np.nan, exact))

    panels = []
    for axis, label in slices:
        i, j = [k for k in range(3) if k != axis]
        s, e = volume_slice(u, axis, mid), volume_slice(err, axis, mid)
        panels += [
            {"title": f"slice {label}", "image": np.linalg.norm(s, axis=-1),
             "quiver": (s[..., i], s[..., j]), "cmap": "plasma"},
            {"title": f"|error|  (max {np.nanmax(e):.3f})", "image": e, "cmap": "magma"},
        ]
    plot(name, panels, (lo[0], hi[0], lo[1], hi[1]),
         f"{title}: color norm(u), arrows in-plane u, {int(data['n_walks'])} walks/pt, "
         f"solve {float(data['seconds']):.1f}s, volume max error {max_err:.3f}", name_suffix="slices")

    from examples.wos import render_volume as rv
    rv.bc_overview(name, surface, params, g, lo, hi, cmap="plasma", info=info, normal=cut_normal)
    grid, _ = rv.load(name)
    smooth, _ = rv.load(name, sigma=1.)
    rv.vector_views(name, grid, smooth, title, normal=cut_normal, cut_camera=cut_camera)


# ------------------------------------------------------------------ 2D scenes
def scene_poisson_square(**_kwargs):
    params = {"min_corner": -jnp.ones(2), "max_corner": jnp.ones(2)}
    square = sdfImplicit(fn=bbox_sdf, dim=2)
    g = lambda x: jnp.asarray(0., x.dtype)
    f = lambda x: jnp.asarray(1., x.dtype)

    def series(x, y, terms=40):
        u = (1. - x ** 2) / 2.
        for k in range(1, 2 * terms, 2):
            u -= (16. / np.pi ** 3 * np.sin(k * np.pi * (1. + x) / 2.) / (k ** 3 * np.sinh(k * np.pi))
                  * (np.sinh(k * np.pi * (1. + y) / 2.) + np.sinh(k * np.pi * (1. - y) / 2.)))
        return u

    n, lo, hi = 128, (-1.05, -1.05), (1.05, 1.05)
    points, (X, Y) = grid2d(lo, hi, n)
    u, seconds = solve(square, g, points, source=f, params=params, n_walks=256)
    p = np.asarray(points, np.float64)
    panels, _ = compare_panels("poisson_2d_square", u, series(p[:, 0], p[:, 1]), n,
                               "exact (Fourier series)", cmap="inferno")
    extent = (lo[0], hi[0], lo[1], hi[1])
    outline = (X, Y, np.maximum(np.abs(X), np.abs(Y)) - 1.)
    plot("poisson_2d_square", panels, extent, f"Poisson -Δu = 1, square, 256 walks/pt, {seconds:.2f}s", outline)
    plot_bc("poisson_2d_square", bbox_sdf, g, extent, params=params, source=f,
            info=["Poisson: -Δu = f in [-1, 1]²", "f = 1 (uniform heating)",
                  "Dirichlet: u = g = 0 on the walls"])


def scene_mixed_star(**_kwargs):
    from boundaryConditions import (
        BoundaryConditions,
        BoundaryKind,
        FunctionBoundaryData,
        VoxelTreePartition,
    )

    params = {"center": jnp.zeros(2), "radius": jnp.asarray(0.95), "m": jnp.asarray(2.6)}
    star = sdfImplicit(fn=star2d_sdf, dim=2)
    lo, hi, cells = (-1., -1.), (1., 1.), 256
    c = (np.arange(cells) + .5) / cells * 2. - 1.
    CX, CY = np.meshgrid(c, c, indexing="ij")
    angle = np.degrees(np.arctan2(CY, CX)) - 90. + 36.
    labels = (np.floor(np.mod(angle, 360.) / 72.)).astype(np.int32)
    partition = VoxelTreePartition.from_grid(labels, lo, hi)
    values = {"v": jnp.array([1., -1., .5, -.5, 0.])}
    data = FunctionBoundaryData([BoundaryKind.DIRICHLET] * 5, lambda p, r, x, n: p["v"][r])
    g = BoundaryConditions(partition, data).dirichlet_fn(None, values)

    n = 160
    points, (X, Y) = grid2d((-1.02, -1.02), (1.02, 1.02), n)
    u, seconds = solve(star, g, points, params=params, n_walks=512)
    sdf = np.asarray(jax.vmap(lambda q: star2d_sdf(params, q))(points)).reshape(n, n)
    extent = (-1.02, 1.02, -1.02, 1.02)
    plot("mixed_regions_star", [
        {"title": "WoS estimate", "image": u.reshape(n, n), "cmap": "coolwarm"},
    ], extent, f"Star, per-arm Dirichlet values, 512 walks/pt, {seconds:.2f}s", (X, Y, sdf))
    tips = [(0.6 * np.cos(np.radians(90 + 72 * k)), 0.6 * np.sin(np.radians(90 + 72 * k))) for k in range(5)]
    plot_bc("mixed_regions_star", star2d_sdf, g, extent, params=params, cmap="coolwarm",
            labels=[(tip, f"region {k}\ng = {v:g}") for k, (tip, v) in
                    enumerate(zip(tips, np.asarray(values["v"]).tolist()))],
            info=["Laplace: -Δu = 0 in a 5-point star", "Dirichlet: u = g, constant per arm",
                  "regions: VoxelTreePartition (sectors)", "values: FunctionBoundaryData gather"])


def scene_two_wires(**_kwargs):
    R, rw, I, mu0 = 4., 1., 1., 1.
    a1, a2 = np.array([-2., 0.]), np.array([2., 0.])
    j = I / (np.pi * rw ** 2)
    params = {"center": jnp.zeros(2), "radius": jnp.asarray(R)}
    domain = sdfImplicit(fn=sphere_sdf, dim=2)
    g = lambda x: jnp.asarray(0., x.dtype)

    def source(x):
        in1 = jnp.linalg.norm(x - a1) <= rw
        in2 = jnp.linalg.norm(x - a2) <= rw
        return jnp.where(in1, mu0 * j, jnp.where(in2, -mu0 * j, 0.)).astype(x.dtype)

    def wire(r):
        return np.where(r >= rw, -np.log(r) / (2 * np.pi),
                        -np.log(rw) / (2 * np.pi) + (1. - r ** 2 / rw ** 2) / (4 * np.pi))

    def exact_A(p):
        d = lambda a: np.linalg.norm(p - a, axis=1)
        image = lambda a: R ** 2 * a / np.dot(a, a)
        return mu0 * I * (wire(d(a1)) - wire(d(a2))
                          + (np.log(d(image(a1))) - np.log(d(image(a2)))) / (2 * np.pi))

    n, half = 96, 1.02 * R
    lo, hi = (-half, -half), (half, half)
    points, (X, Y) = grid2d(lo, hi, n)
    A, seconds = solve(domain, g, points, source=source, params=params, n_walks=8192, eps=1e-2)
    exact = exact_A(np.asarray(points, np.float64))
    dy, dx = np.gradient(exact.reshape(n, n), Y[:, 0], X[0])
    panels, _ = compare_panels("two_wires", A, exact, n, "exact A_z, field lines of B", cmap="RdBu_r",
                               exact_extra={"streams": (X, Y, dy, -dx)})
    extent = (lo[0], hi[0], lo[1], hi[1])
    wires = (X, Y, np.minimum(np.hypot(X - a1[0], Y), np.hypot(X - a2[0], Y)) - rw)
    plot("two_wires", panels, extent, f"Two wires, grounded disk R={R:g}, 8192 walks/pt, {seconds:.2f}s", wires)
    plot_bc("two_wires", sphere_sdf, g, extent, params=params, source=source,
            labels=[(tuple(a1 + [0., 1.6]), "+I"), (tuple(a2 + [0., 1.6]), "-I")],
            info=[f"Magnetostatics: -ΔA_z = μ0 j in a disk of radius {R:g}",
                  f"j = ±I / (π rw²) in wires of radius {rw:g} at x = ±2",
                  "Dirichlet: A_z = g = 0 on the circle (grounded)"])


# ------------------------------------------------------------------ 3D scenes
def scene_torus(*, solve_only=False, render_only=False):
    name, n, lo, hi = "torus", 97, (-1., -1., -1.), (1., 1., 1.)
    surface, params, g = shapes.torus()
    solve_now, render_now = _stages(name, solve_only, render_only)
    if solve_now:
        solve_volume(name, surface, g, lo, hi, n, params=params, n_walks=256)
    if not render_now:
        return
    data = load_volume(name)
    u, mid = data["values"], n // 2
    plot(name, [
        {"title": "slice z = 0  (x, y)", "image": volume_slice(u, 2, mid), "cmap": "coolwarm", "clim": (-1., 1.)},
        {"title": "slice y = 0  (x, z)", "image": volume_slice(u, 1, mid), "cmap": "coolwarm", "clim": (-1., 1.)},
    ], (lo[0], hi[0], lo[1], hi[1]),
        f"Torus, g = cos(3φ), {int(data['n_walks'])} walks/pt, solve {float(data['seconds']):.1f}s",
        name_suffix="slices")

    from examples.wos import render_volume as rv
    rv.bc_overview(name, surface, params, g, lo, hi, cmap="coolwarm", clim=(-1., 1.),
                   info=["Laplace: -Lap u = 0 in a torus (R = 0.6, r = 0.3)",
                         "Dirichlet: u = g on the surface", "g = cos(3 phi), phi = atan2(y, x)"],
                   normal=(0., 0., 1.))
    grid, _ = rv.load(name)
    smooth, _ = rv.load(name, sigma=1.)
    rv.scalar_views(name, grid, smooth, "Torus, g = cos(3 phi)", normal=(0., 0., 1.),
                    levels=[-.75, -.5, -.25, .25, .5, .75], clim=(-1., 1.),
                    cut_camera=(2.3, -2.7, 1.3), iso_camera=(1.8, -2.2, 2.4))


def scene_saturn(*, solve_only=False, render_only=False):
    name = "saturn"
    surface, params, g = shapes.saturn()
    solve_now, render_now = _stages(name, solve_only, render_only)
    _run_vector_volume(
        name, surface, params, g, (-.95,) * 3, (.95,) * 3, title="Saturn",
        slices=[(0, "x = 0  (y, z): tilted ring"), (1, "y = 0  (x, z)")],
        cut_normal=(1., 0., 0.), cut_camera=(2.8, -1.8, 1.4),
        info=["Vector Laplace on Saturn (sphere + tilted ring, smooth union)",
              "Dirichlet: u = g on the surface (arrows)",
              "g = grad h, h = x^3 - 3xy^2 + z^3 - 1.5 z (x^2 + y^2)"],
        solve_now=solve_now, render_now=render_now, n_walks=256, max_steps=5000,
    )


def scene_twisted_torus(*, solve_only=False, render_only=False):
    name = "twisted_torus"
    surface, params, g = shapes.twisted_torus("assets/models/twisted_torus_twist10_softplus_64k.npz")
    solve_now, render_now = _stages(name, solve_only, render_only)
    _run_vector_volume(
        name, surface, params, g, (-1.,) * 3, (1.,) * 3, title="Neural twisted torus (twist 10)",
        slices=[(2, "z = 0  (x, y)"), (1, "y = 0  (x, z)")],
        cut_normal=(0., 0., 1.), cut_camera=(2.3, -2.7, 1.6),
        info=["Vector Laplace in a neural twisted torus, twist 10 (Softplus MLP)",
              "Dirichlet: u = g on the learned zero level set (arrows)",
              "g = grad h, h = x^3 - 3xy^2 + z^3 - 1.5 z (x^2 + y^2)"],
        solve_now=solve_now, render_now=render_now, n_walks=128, max_steps=2000,
    )


SCENES = {
    "poisson_square": scene_poisson_square,
    "mixed_star": scene_mixed_star,
    "two_wires": scene_two_wires,
    "torus": scene_torus,
    "saturn": scene_saturn,
    "twisted_torus": scene_twisted_torus,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", choices=sorted(SCENES), help="WoS showcase scene to run")
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--solve", action="store_true", help="3D scenes: solve volume only")
    stage.add_argument("--render", action="store_true", help="3D scenes: render saved volume only")
    args = parser.parse_args(argv)
    SCENES[args.scene](solve_only=args.solve, render_only=args.render)


if __name__ == "__main__":
    main()
