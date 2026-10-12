"""3D Laplace in the unit ball, scalar and vector, shown on the slice z = 0.3.

Scalar g = z + 2xy and vector g = (-y, x, z) are harmonic, so u = g inside.
"""
import jax.numpy as jnp
import numpy as np
from common import plot, report, slice3d, solve

from surfaces.implicit.lipschitz import sdfImplicit
from surfaces.implicit.primitives import sphere_sdf

params = {"center": jnp.zeros(3), "radius": jnp.asarray(1.)}
ball = sdfImplicit(fn=sphere_sdf, dim=3)
z0, n, lo, hi = 0.3, 112, (-1.05, -1.05), (1.05, 1.05)
points, (X, Y) = slice3d(lo, hi, n, axis=2, value=z0)
p = np.asarray(points)
extent = (lo[0], hi[0], lo[1], hi[1])
outline = (X, Y, np.hypot(X, Y) - np.sqrt(1. - z0 ** 2))

scalar = lambda x: x[2] + 2. * x[0] * x[1]
u, t_s = solve(ball, scalar, points, params=params, n_walks=256)
exact = np.where(np.isnan(u), np.nan, p[:, 2] + 2. * p[:, 0] * p[:, 1])
err, (max_s, _) = report("laplace_3d scalar", u, exact)

vector = lambda x: jnp.stack([-x[1], x[0], x[2]])
v, t_v = solve(ball, vector, points, params=params, n_walks=256)
exact_v = np.where(np.isnan(v), np.nan, np.stack([-p[:, 1], p[:, 0], p[:, 2]], 1))
err_v, (max_v, _) = report("laplace_3d vector", v, exact_v)

plot("laplace_3d", [
    {"title": "scalar estimate  (z + 2xy)", "image": u.reshape(n, n)},
    {"title": f"scalar |error|  (max {max_s:.3f})", "image": err.reshape(n, n), "cmap": "magma"},
    {"title": "vector estimate  |(u_x, u_y)|, arrows", "image": np.hypot(v[:, 0], v[:, 1]).reshape(n, n),
     "quiver": (v[:, 0].reshape(n, n), v[:, 1].reshape(n, n))},
    {"title": f"vector |error|  (max {max_v:.3f})", "image": err_v.reshape(n, n), "cmap": "magma"},
], extent, f"3D Laplace, unit ball, slice z = {z0}, 256 walks/pt, {t_s + t_v:.2f}s", outline)

import render_volume as rv  # noqa: E402

lo3, hi3 = (-1.05,) * 3, (1.05,) * 3
rv.bc_overview("laplace_3d", ball, params, scalar, lo3, hi3, name="bc_scalar",
               info=["Laplace: -Lap u = 0 in the unit ball", "Dirichlet: u = g on the sphere",
                     "g = z + 2xy"])
rv.bc_overview("laplace_3d", ball, params, vector, lo3, hi3, name="bc_vector",
               info=["Vector Laplace in the unit ball", "Dirichlet: u = g on the sphere",
                     "g = (-y, x, z)  (arrows)"])
