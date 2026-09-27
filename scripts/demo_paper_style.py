"""Gallery script: produce one example of each paper-style plot type.

Run:
    python scripts/demo_paper_style.py

Outputs to figures/demo_*.{pdf,png}.

Five demos:
  1. 1-D function with intervals + Taylor polynomials.
  2. 2-D scalar field with zero level set.
  3. k-d tree (octree slice) subdivision.
  4. Walk on Spheres trajectory.
  5. 3-D schematic of trust-region box + neural surface mesh.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

from utils.paper_utils import (
    use_style, COLORS, CMAP_FIELD, CMAP_DIVERGING,
    figure_single_col, figure_double_col, figure_grid,
    plot_function, plot_interval, plot_taylor, annotate_anchor,
    plot_field2d, plot_levelset2d,
    plot_kdtree, TreeBox2D,
    plot_wos_walk,
    axes3d_clean, plot_box3d,
    paper_savefig, tight_legend,
)


# ---------------------------------------------------------------------------
# 1. 1-D function with bound interval + Taylor polynomial
# ---------------------------------------------------------------------------

def demo_1d():
    fig, ax = figure_single_col(height_ratio=0.65)

    # A nontrivial function with curvature
    f = lambda x: 0.7 * np.sin(2.5 * x) + 0.4 * np.cos(1.3 * x) - 0.15 * x
    fp = lambda x: 0.7 * 2.5 * np.cos(2.5 * x) - 0.4 * 1.3 * np.sin(1.3 * x) - 0.15
    fpp = lambda x: -0.7 * 2.5**2 * np.sin(2.5 * x) - 0.4 * 1.3**2 * np.cos(1.3 * x)

    x = np.linspace(-2.5, 2.5, 400)
    plot_function(ax, x, f(x), color=COLORS["ink"], label=r"$f(x)$")

    # An interval enclosure (placeholder bound, illustrative)
    box_lo, box_hi = -1.0, 1.0
    mask = (x >= box_lo) & (x <= box_hi)
    f_lo = np.full_like(x[mask], f(0.0) - 1.05)
    f_hi = np.full_like(x[mask], f(0.0) + 1.05)
    plot_interval(ax, x[mask], f_lo, f_hi, color=COLORS["secondary"],
                  label="affine arithmetic (interval)")

    # Taylor polynomial of degree 2 anchored at x0 = 0
    x0 = 0.0
    coeffs = [f(x0), fp(x0), fpp(x0)]
    plot_taylor(ax, x[mask], x0, coeffs, color=COLORS["accent"],
                label="autobound deg-2 polynomial")
    annotate_anchor(ax, x0, f(x0), label=r"$x_0$")

    # Trust-region brackets
    for xb in (box_lo, box_hi):
        ax.axvline(xb, color=COLORS["neutral_light"], linewidth=0.6, linestyle=":")

    ax.set_xlim(-2.5, 2.5)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$f(x)$")
    ax.set_title("1-D: function, interval bound, Taylor polynomial")
    tight_legend(ax, loc="upper right", ncol=1)
    paper_savefig(fig, "demo_1d_bounds")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. 2-D scalar field with zero level set
# ---------------------------------------------------------------------------

def demo_2d_field():
    fig, axes = figure_grid(1, 2, single_col=False)
    n = 240
    x = np.linspace(-1.4, 1.4, n)
    y = np.linspace(-1.0, 1.0, n)
    X, Y = np.meshgrid(x, y)

    # Signed-distance-like field of a peanut shape
    F = (X**2 + Y**2 - 0.6**2) * (((X - 0.55)**2 + Y**2 - 0.4**2)
                                  * ((X + 0.55)**2 + Y**2 - 0.4**2))
    F = F / np.max(np.abs(F))

    im = plot_field2d(axes[0], X, Y, F, signed=True, contour=True)
    axes[0].set_title("signed field with zero level set")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    # Right: just the level set in a single primary color
    plot_levelset2d(axes[1], X, Y, F, level=0.0,
                    color=COLORS["primary"], linewidth=1.6, label="level set")
    axes[1].set_xlim(-1.4, 1.4); axes[1].set_ylim(-1.0, 1.0)
    axes[1].set_aspect("equal")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[1].set_title("isocontour in primary color")

    cbar = fig.colorbar(im, ax=axes[0], shrink=0.85, pad=0.02)
    cbar.set_label(r"$f(x, y)$ (normalized)", fontsize=8)
    cbar.outline.set_visible(False)

    paper_savefig(fig, "demo_2d_field")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. k-d tree subdivision (2-D slice)
# ---------------------------------------------------------------------------

def _classify(box, fn, level=0.0):
    """Classify a 2-D box against an analytic level set fn (vectorized over corners)."""
    (xlo, ylo), (xhi, yhi) = box
    pts = np.array([[xlo, ylo], [xhi, ylo], [xhi, yhi], [xlo, yhi],
                    [(xlo + xhi) / 2, (ylo + yhi) / 2]])
    vals = fn(pts[:, 0], pts[:, 1]) - level
    if np.all(vals > 0): return "exterior"
    if np.all(vals < 0): return "interior"
    return "boundary"


def _subdivide(lo, hi, depth, fn, max_depth=5):
    """Recursive 2-D quadtree subdivision until classified or depth budget."""
    s = _classify((lo, hi), fn)
    if s in ("interior", "exterior") or depth >= max_depth:
        return [TreeBox2D(lo, hi, s)]
    # Mark as split, then recurse
    boxes = [TreeBox2D(lo, hi, "split")]
    mx = (lo[0] + hi[0]) / 2
    my = (lo[1] + hi[1]) / 2
    quads = [
        ((lo[0], lo[1]), (mx, my)),
        ((mx, lo[1]), (hi[0], my)),
        ((mx, my), (hi[0], hi[1])),
        ((lo[0], my), (mx, hi[1])),
    ]
    for q_lo, q_hi in quads:
        boxes.extend(_subdivide(q_lo, q_hi, depth + 1, fn, max_depth))
    return boxes


def demo_kdtree():
    fig, ax = figure_single_col(height_ratio=1.0)

    # A flower-like level set so the tree adapts non-trivially
    fn = lambda x, y: np.sqrt(x**2 + y**2) - (0.55 + 0.18 * np.cos(5 * np.arctan2(y, x)))

    boxes = _subdivide((-1.0, -1.0), (1.0, 1.0), 0, fn, max_depth=6)
    plot_kdtree(ax, boxes, fill=True, draw_split=True)

    # Overlay the true level set
    n = 400
    x = np.linspace(-1.0, 1.0, n)
    X, Y = np.meshgrid(x, x)
    plot_levelset2d(ax, X, Y, fn(X, Y), level=0.0,
                    color=COLORS["ink"], linewidth=1.0)

    ax.set_xlim(-1.0, 1.0); ax.set_ylim(-1.0, 1.0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("k-d tree subdivision: green=interior, red=exterior, yellow=boundary")
    paper_savefig(fig, "demo_kdtree")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Walk on Spheres trajectory
# ---------------------------------------------------------------------------

def demo_wos():
    fig, ax = figure_single_col(height_ratio=1.0)

    # Domain boundary: a peanut shape; we'll fake a distance function as
    # the (signed) distance to the convex hull of two circles.
    def dist_to_boundary(p):
        c1 = np.array([-0.45, 0.0]); r1 = 0.55
        c2 = np.array([0.45, 0.0]);  r2 = 0.55
        d1 = np.linalg.norm(p - c1) - r1
        d2 = np.linalg.norm(p - c2) - r2
        return min(d1, d2)

    rng = np.random.default_rng(2)
    p = np.array([0.0, 0.05])
    walk = [p.copy()]
    radii = []
    eps = 0.01
    for _ in range(20):
        r = -dist_to_boundary(p)  # positive distance to boundary from inside
        radii.append(r)
        if r < eps:
            break
        theta = rng.uniform(0, 2 * np.pi)
        p = p + r * np.array([np.cos(theta), np.sin(theta)])
        walk.append(p.copy())
    walk = np.array(walk); radii = np.array(radii + [eps])

    # Draw the boundary first
    th = np.linspace(0, 2 * np.pi, 200)
    for c, r in [((-0.45, 0.0), 0.55), ((0.45, 0.0), 0.55)]:
        ax.plot(c[0] + r * np.cos(th), c[1] + r * np.sin(th),
                color=COLORS["ink"], linewidth=1.0)

    plot_wos_walk(ax, walk, radii, hit_point=walk[-1])

    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-0.75, 0.75)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Walk on Spheres: empty-ball steps on a 2-D domain")
    paper_savefig(fig, "demo_wos")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. 3-D schematic: trust-region box + a sample surface
# ---------------------------------------------------------------------------

def demo_3d():
    fig = plt.figure(figsize=(figure_single_col.__defaults__[0] * 0 + 3.33,
                              3.33))
    ax = fig.add_subplot(111, projection="3d")
    axes3d_clean(ax)

    # Implicit surface: f(x,y,z) = x^2 + y^2 + z^2 - 0.36 = 0  (sphere)
    u = np.linspace(0, 2 * np.pi, 36)
    v = np.linspace(0, np.pi, 18)
    R = 0.6
    X = R * np.outer(np.cos(u), np.sin(v))
    Y = R * np.outer(np.sin(u), np.sin(v))
    Z = R * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(X, Y, Z, color=COLORS["primary"], alpha=0.35,
                    linewidth=0, antialiased=True, rstride=2, cstride=2)
    ax.plot_wireframe(X, Y, Z, color=COLORS["primary"], linewidth=0.4,
                      alpha=0.6, rstride=4, cstride=4)

    # Trust-region box surrounding part of the surface
    plot_box3d(ax, lo=(-0.85, -0.85, -0.85), hi=(0.85, 0.85, 0.85),
               color=COLORS["accent"], linewidth=0.9, alpha=0.9)

    # Cell center marker
    ax.plot([0.0], [0.0], [0.0], "o", markersize=4.5,
            color=COLORS["accent"], markeredgecolor="white",
            markeredgewidth=0.6)

    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.view_init(elev=22, azim=-58)
    ax.set_box_aspect([1, 1, 1])
    ax.set_title("3-D: trust-region box around the surface", pad=2)

    paper_savefig(fig, "demo_3d_box")
    plt.close(fig)


def main():
    use_style()
    demo_1d()
    demo_2d_field()
    demo_kdtree()
    demo_wos()
    demo_3d()
    print("Wrote demo_*.pdf and demo_*.png to figures/")


if __name__ == "__main__":
    main()
