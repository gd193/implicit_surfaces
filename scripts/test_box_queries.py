"""Smoke test for jnpboundedImplicit box queries.

Builds a jnpboundedImplicit, monkey-patches its `taylor_bounds` to return a
hand-constructed polynomial enclosure of a known quadratic
  f(x) = x0^2 + 2*x1^2 + 3*x2^2 + 4*x0 + 5*x1 + 6*x2 + 7
and checks that the new box queries (classify_box, box_bounds, box_grad_bounds)
return enclosures that contain the analytic answer.

This sidesteps autobound's restricted primitive set — the API contract being
tested is the surface methods + helper, not the autobound tracer.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from jnp_bounds.types import Interval, MixedIntervalPolynomial

from implicitSurfaces.jnpboundedImplicit import jnpboundedImplicit


# Quadratic f and its analytics.
def f_np(x):
    return x[..., 0]**2 + 2 * x[..., 1]**2 + 3 * x[..., 2]**2 \
           + 4 * x[..., 0] + 5 * x[..., 1] + 6 * x[..., 2] + 7


def grad_f_np(x):
    return np.stack([
        2 * x[..., 0] + 4,
        4 * x[..., 1] + 5,
        6 * x[..., 2] + 6,
    ], axis=-1)


def make_quadratic_poly(center, half_extent):
    """Hand-build the deg-2 Taylor polynomial of f around `center` over `±half_extent`.

    For our exact quadratic, the deg-2 Taylor at any point is exact — interval coefs
    collapse to point values.

    Exponent order for (dim=3, deg=2) per dim_degree_to_exponents_cache:
        idx 0: [0,0,0]  (const)
        idx 1: [0,0,1]  (x_2)
        idx 2: [0,1,0]  (x_1)
        idx 3: [1,0,0]  (x_0)
        idx 4: [0,0,2]  (x_2^2)   coef = 3 (interval, no uncertainty)
        idx 5: [0,1,1]  (x_1*x_2) coef = 0
        idx 6: [0,2,0]  (x_1^2)   coef = 2
        idx 7: [1,0,1]  (x_0*x_2) coef = 0
        idx 8: [1,1,0]  (x_0*x_1) coef = 0
        idx 9: [2,0,0]  (x_0^2)   coef = 1
    Real part: const + linear (4 entries).
    """
    c = np.asarray(center)
    f_c = float(f_np(c))
    g_c = grad_f_np(c)  # shape (3,)
    real = jnp.array([[f_c], [float(g_c[2])], [float(g_c[1])], [float(g_c[0])]])
    ilo = jnp.array([[3.0], [0.0], [2.0], [0.0], [0.0], [1.0]])
    ihi = ilo
    return MixedIntervalPolynomial(real_coeffs=real, interval_coeffs=Interval(lower=ilo, upper=ihi))


class _PatchedTaylor:
    """Mimics the autobound TaylorBounds object: exposes `.polynomial_enclosure`."""
    def __init__(self, poly):
        self.polynomial_enclosure = poly


def fake_taylor_bounds(x, x_trust_region=None, trust_region_size=None):
    half = jnp.asarray(trust_region_size) / 2.0
    poly = make_quadratic_poly(np.asarray(x), np.asarray(half))
    return _PatchedTaylor(poly)


def main():
    # Surface that *would* use autobound. We monkey-patch taylor_bounds since the
    # autobound tracer doesn't support `slice` ops on raw jnp polynomials.
    def fn_2arg(_p, x):
        return jnp.array([f_np(x)])  # shape (1,) to match MLP convention

    surface = jnpboundedImplicit(
        autobounds_max_degree=2,
        fn=fn_2arg,
        level_set=0.0,
        dim=3,
    )
    surface.taylor_bounds = fake_taylor_bounds

    lower = jnp.array([0.4, -0.6, -0.1])
    upper = jnp.array([0.6, -0.4, 0.1])

    # Analytic reference over the box (dense grid)
    N = 31
    g = [np.linspace(float(lower[i]), float(upper[i]), N) for i in range(3)]
    xx, yy, zz = np.meshgrid(*g, indexing='ij')
    pts = np.stack([xx, yy, zz], axis=-1)
    f_grid = f_np(pts)
    g_grid = grad_f_np(pts)

    # Test box_bounds
    lo, hi = surface.box_bounds(None, lower, upper)
    print(f"box_bounds:        [{float(lo):.4f}, {float(hi):.4f}]")
    print(f"  analytic:        [{f_grid.min():.4f}, {f_grid.max():.4f}]")
    assert float(lo) <= f_grid.min() + 1e-4, "box_bounds lower not provably-valid"
    assert float(hi) >= f_grid.max() - 1e-4, "box_bounds upper not provably-valid"

    # Test classify_box: function is ~13 over this box, well above level_set=0
    sign = surface.classify_box(None, lower, upper)
    print(f"classify_box:      {int(sign)}  (analytic +1)")
    assert int(sign) == 1

    # Move to a box straddling the level set
    lower2 = jnp.array([-3.0, -3.0, -3.0])
    upper2 = jnp.array([3.0, 3.0, 3.0])
    f_grid2 = f_np(np.stack(np.meshgrid(*[np.linspace(-3, 3, 21)]*3, indexing='ij'), axis=-1))
    sign2 = surface.classify_box(None, lower2, upper2)
    print(f"classify_box big:  {int(sign2)}  (f range [{f_grid2.min():.2f},{f_grid2.max():.2f}], analytic 0)")
    assert int(sign2) == 0

    # Test box_grad_bounds
    glo, ghi = surface.box_grad_bounds(None, lower, upper)
    print(f"box_grad_bounds:")
    for i in range(3):
        gmin = g_grid[..., i].min()
        gmax = g_grid[..., i].max()
        print(f"  axis {i}: [{float(glo[i]):.4f}, {float(ghi[i]):.4f}]  analytic: [{gmin:.4f}, {gmax:.4f}]")
        assert float(glo[i]) <= gmin + 1e-4
        assert float(ghi[i]) >= gmax - 1e-4

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
