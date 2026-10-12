"""Poisson in [-1, 1]^2: uniform heating f = 1, cold walls g = 0.
Exact: the classical Fourier series for -Δu = 1 in a square."""
import jax.numpy as jnp
import numpy as np
from common import compare_panels, grid2d, plot, plot_bc, solve

from surfaces.implicit.lipschitz import sdfImplicit
from surfaces.implicit.primitives import bbox_sdf

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
