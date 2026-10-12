"""Magnetostatics: two wires with opposite currents inside a grounded disk.

A_z solves -ΔA = mu0 j, with j = +-I / (pi rw^2) inside the wires and A = 0 on
the disk of radius R. Exact: wire potentials plus image line currents at
a* = R^2 a / |a|^2, which cancel on the circle. B = (dA/dy, -dA/dx).
"""
import jax.numpy as jnp
import numpy as np
from common import compare_panels, grid2d, plot, plot_bc, solve

from surfaces.implicit.lipschitz import sdfImplicit
from surfaces.implicit.primitives import sphere_sdf

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


def wire(r):  # unit-current potential of a uniformly filled wire
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
# Each step draws one source sample in its empty ball; only samples inside a
# wire contribute. A small disk keeps balls small, so more samples hit.
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
