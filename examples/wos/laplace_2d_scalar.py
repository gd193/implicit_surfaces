"""Laplace in the unit disk with g = 1 + 2 sin(theta). Exact: u = 1 + 2y."""
import jax.numpy as jnp
import numpy as np
from common import compare_panels, grid2d, plot, plot_bc, solve

from surfaces.implicit.lipschitz import sdfImplicit
from surfaces.implicit.primitives import sphere_sdf

params = {"center": jnp.zeros(2), "radius": jnp.asarray(1.)}
disk = sdfImplicit(fn=sphere_sdf, dim=2)
g = lambda x: 1. + 2. * jnp.sin(jnp.arctan2(x[1], x[0]))

n, lo, hi = 128, (-1.05, -1.05), (1.05, 1.05)
points, (X, Y) = grid2d(lo, hi, n)
u, seconds = solve(disk, g, points, params=params, n_walks=256)
panels, _ = compare_panels("laplace_2d_scalar", u, 1. + 2. * np.asarray(points[:, 1]), n,
                           "exact  u = 1 + 2y")

extent = (lo[0], hi[0], lo[1], hi[1])
outline = (X, Y, np.hypot(X, Y) - 1.)
plot("laplace_2d_scalar", panels, extent, f"Laplace, unit disk, 256 walks/pt, {seconds:.2f}s", outline)

plot_bc("laplace_2d_scalar", sphere_sdf, g, extent, params=params,
        info=["Laplace: -Δu = 0 in the unit disk", "Dirichlet: u = g on the circle",
              "g = 1 + 2 sin θ"])
