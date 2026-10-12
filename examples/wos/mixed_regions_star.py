"""Laplace in a 5-point star with a different Dirichlet value on each arm.

The boundary data comes from the boundaryConditions package: a voxel-tree
partition labels the five angular sectors, and per-region values are a
gather from data params. No closed form; the plot shows the partition too.
"""
import jax
import jax.numpy as jnp
import numpy as np
from common import grid2d, plot, plot_bc, solve

from boundaryConditions import (BoundaryConditions, BoundaryKind, FunctionBoundaryData,
                                VoxelTreePartition)
from surfaces.implicit.lipschitz import sdfImplicit
from surfaces.implicit.primitives import star2d_sdf

params = {"center": jnp.zeros(2), "radius": jnp.asarray(0.95), "m": jnp.asarray(2.6)}
star = sdfImplicit(fn=star2d_sdf, dim=2)

# Sector k spans the arm whose tip is at angle 90 + 72k degrees.
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
