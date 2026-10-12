"""Shared helpers for the WoS examples: grids, inside masks, solving, plots.

Conventions: fields are negative inside the domain; the integrator solves
Laplace / Poisson with  -Δu = f  and Dirichlet data g. Grid points outside the
domain are not solved (walks started there do not terminate) and show as NaN.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sphereWalker.integrator import WoSIntegrator

OUT = ROOT / "assets" / "renders" / "wos"


def out_dir(example):
    """assets/renders/wos/<example>/: every output of one example lives here."""
    path = OUT / example
    path.mkdir(parents=True, exist_ok=True)
    return path


def grid2d(lower, upper, n):
    """(n*n, 2) points of an n x n grid; reshape values to (n, n) for imshow."""
    xs = np.linspace(lower[0], upper[0], n)
    ys = np.linspace(lower[1], upper[1], n)
    X, Y = np.meshgrid(xs, ys)  # image[row=y, col=x]
    return jnp.asarray(np.stack([X.ravel(), Y.ravel()], 1), jnp.float32), (X, Y)


def slice3d(lower, upper, n, axis=2, value=0.):
    """(n*n, 3) points on the plane x[axis] = value; also the in-plane grids."""
    plane, (A, B) = grid2d(lower, upper, n)
    cols = [plane[:, 0], plane[:, 1]]
    cols.insert(axis, jnp.full(n * n, value, jnp.float32))
    return jnp.stack(cols, 1), (A, B)


def solve(surface, g, points, *, source=None, params=None, n_walks=256, eps=1e-3,
          max_steps=1000, seed=0):
    """WoS estimate at every grid point inside the surface, NaN outside.

    Returns (values, seconds): values is (N,) for scalar g, (N, m) for vector g.
    """
    inside = np.asarray(jax.vmap(lambda x: surface.signed_value(params, x))(points) < 0)
    zero = lambda x: jnp.asarray(0., x.dtype)
    solver = WoSIntegrator(surface.empty_sphere, g, source or zero,
                           dim=points.shape[1], eps=eps, max_steps=max_steps)
    pts = points[inside]
    options = dict(init_trust_region_size=1., n_samples=n_walks, steps_between_refill=16,
                   max_active_states=min(pts.shape[0] * n_walks, 1 << 20), jit_buck_size=1024)
    start = time.perf_counter()
    w = solver.integrate_batch(params, pts, jax.random.PRNGKey(seed), **options).w
    w = np.asarray(jax.block_until_ready(w))
    seconds = time.perf_counter() - start
    w = w[:, 0] if w.shape[1] == 1 else w
    values = np.full((points.shape[0],) + w.shape[1:], np.nan, np.float32)
    values[inside] = w
    return values, seconds


def report(name, est, exact):
    """Print and return the max / mean absolute error over solved points."""
    err = np.abs(est - exact)
    if err.ndim > 1:
        err = np.linalg.norm(err, axis=-1)
    stats = float(np.nanmax(err)), float(np.nanmean(err))
    print(f"{name}: max error {stats[0]:.4f}, mean error {stats[1]:.4f}")
    return err, stats


def plot(name, panels, extent, title, outline=None, name_suffix=None):
    """Save a row of panels to assets/renders/wos/<name>/<name_suffix or name>.png.

    panels: dicts with 'title', 'image' (n x n, NaN outside) and optional
    'cmap', 'clim' (vmin, vmax), 'quiver' (U, V), 'streams' (X, Y, U, V).
    outline: (X, Y, field) drawn as the zero contour on every panel.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.2),
                             squeeze=False, constrained_layout=True)
    for ax, panel in zip(axes[0], panels):
        vmin, vmax = panel.get("clim", (None, None))
        img = ax.imshow(panel["image"], origin="lower", extent=extent,
                        cmap=panel.get("cmap", "viridis"), vmin=vmin, vmax=vmax)
        fig.colorbar(img, ax=ax, shrink=0.85)
        if "quiver" in panel:
            U, V = panel["quiver"]
            n = U.shape[0]
            step = max(n // 20, 1)
            X, Y = np.meshgrid(np.linspace(extent[0], extent[1], n),
                               np.linspace(extent[2], extent[3], n))
            ax.quiver(X[::step, ::step], Y[::step, ::step], U[::step, ::step],
                      V[::step, ::step], color="white", width=0.004)
        if "streams" in panel:
            X, Y, U, V = panel["streams"]
            ax.streamplot(X, Y, U, V, color="white", density=1.2, linewidth=0.7)
        if outline is not None:
            ax.contour(*outline, levels=[0.], colors="black", linewidths=1.)
        ax.set_title(panel["title"])
        ax.set_aspect("equal")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    fig.suptitle(title)
    path = out_dir(name) / f"{name_suffix or name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"saved {path.relative_to(ROOT)}")
    return path


def solve_volume(name, surface, g, lower, upper, n, *, params=None, **solve_kwargs):
    """Solve on an n^3 grid over the box [lower, upper] and save
    assets/renders/wos/<name>/volume.npz with arrays indexed [ix, iy, iz]:

      values  (n, n, n) or (n, n, n, m), NaN outside the domain
      sdf     (n, n, n) surface field (negative inside)
      lower, upper, plus walk metadata.
    """
    axes = [np.linspace(lower[i], upper[i], n) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    points = jnp.asarray(grid, jnp.float32)
    values, seconds = solve(surface, g, points, params=params, **solve_kwargs)
    sdf = np.asarray(jax.vmap(lambda x: surface.signed_value(params, x))(points))
    path = volume_path(name)
    np.savez_compressed(path, values=values.reshape((n, n, n) + values.shape[1:]),
                        sdf=sdf.reshape(n, n, n), lower=np.asarray(lower, np.float32),
                        upper=np.asarray(upper, np.float32), seconds=seconds,
                        n_walks=solve_kwargs.get("n_walks", 256))
    inside = int(np.sum(sdf < 0))
    print(f"solved {name}: {n}^3 grid, {inside} inside points, {seconds:.1f}s -> "
          f"{path.relative_to(ROOT)}")
    return path


def volume_path(name):
    return out_dir(name) / "volume.npz"


def volume_slice(array, axis, index):
    """2D image [row, col] of a volume array indexed [ix, iy, iz, ...] on the
    plane array[axis] = index: columns follow the first remaining axis, rows
    the second (so x-y, x-z and y-z slices display the usual way)."""
    return np.swapaxes(np.take(array, index, axis=axis), 0, 1)


def load_volume(name):
    return np.load(volume_path(name))


def plot_bc(name, field, g, extent, *, params=None, source=None, info=(), labels=(), n=600,
            cmap="viridis"):
    """Boundary-condition overview of a 2D example: assets/renders/wos/<name>/bc.png.

    Shows the domain {field < 0}, its boundary colored by scalar g (or arrows
    colored by norm(g) for vector g), the source f as background when given,
    optional text labels [(xy, text)], and an info box (PDE, data).
    """
    from matplotlib.collections import LineCollection
    xs, ys = np.linspace(extent[0], extent[1], n), np.linspace(extent[2], extent[3], n)
    X, Y = np.meshgrid(xs, ys)
    pts = jnp.asarray(np.stack([X.ravel(), Y.ravel()], 1), jnp.float32)
    sdf = np.asarray(jax.vmap(lambda x: field(params, x))(pts)).reshape(n, n)
    fig, ax = plt.subplots(figsize=(7.4, 6.2), constrained_layout=True)
    if source is not None:
        f = np.where(sdf < 0, np.asarray(jax.vmap(source)(pts)).reshape(n, n), np.nan)
        lim = float(np.nanmax(np.abs(f))) or 1.
        img = ax.imshow(f, origin="lower", extent=extent, cmap="RdBu_r", vmin=-lim, vmax=lim)
        fig.colorbar(img, ax=ax, shrink=0.8, label="source f")
    else:
        ax.contourf(X, Y, sdf, levels=[sdf.min() - 1., 0.], colors=["#ececec"])
    segments = [s for s in ax.contour(X, Y, sdf, levels=[0.], alpha=0.).allsegs[0] if len(s) > 1]
    values = [np.asarray(jax.vmap(g)(jnp.asarray(s, jnp.float32))) for s in segments]
    flat = np.concatenate([v.reshape(len(v), -1) for v in values])
    if values[0].ndim == 1 and np.ptp(flat) < 1e-9:  # constant data: plain line, value in legend
        for k, s in enumerate(segments):
            ax.plot(s[:, 0], s[:, 1], color="black", lw=4, label=f"g = {flat[0, 0]:g}" if k == 0 else None)
        ax.legend(loc="lower right")
    elif values[0].ndim == 1:  # scalar data: thick boundary colored by g
        lines = np.concatenate([np.stack([s[:-1], s[1:]], 1) for s in segments])
        colors = np.concatenate([(v[:-1] + v[1:]) / 2 for v in values])
        collection = LineCollection(lines, array=colors, cmap=cmap, linewidths=6)
        ax.add_collection(collection)
        fig.colorbar(collection, ax=ax, shrink=0.8, label="Dirichlet data g on boundary")
    else:  # vector data: boundary line plus arrows colored by norm(g)
        for s in segments:
            ax.plot(s[:, 0], s[:, 1], color="gray", lw=1.)
        pts_b, vec = np.concatenate(segments), np.concatenate(values)
        step = max(len(pts_b) // 40, 1)
        q = ax.quiver(pts_b[::step, 0], pts_b[::step, 1], vec[::step, 0], vec[::step, 1],
                      np.linalg.norm(vec[::step], axis=1), cmap=cmap, width=0.009,
                      edgecolor="black", linewidth=0.4, zorder=3)
        fig.colorbar(q, ax=ax, shrink=0.8, label="norm of Dirichlet data g")
    for xy, text in labels:
        ax.annotate(text, xy, ha="center", va="center", fontsize=9,
                    bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
    ax.text(0.02, 0.98, "\n".join(info), transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.9))
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_title(f"{name}: boundary conditions")
    path = out_dir(name) / "bc.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"saved {path.relative_to(ROOT)}")


def compare_panels(name, u, exact, n, exact_title, cmap="viridis", exact_extra=None):
    """Estimate / exact / |error| panels for an n x n grid, plus the max error.

    Scalar u: images, estimate on the exact color range. Vector u (N, 2):
    norm images with in-plane arrows. exact is masked to the solved points;
    exact_extra adds keys (e.g. 'streams') to the exact panel.
    """
    exact = np.where(np.isnan(u), np.nan, exact)
    err, (max_err, _) = report(name, u, exact)
    square = lambda a: a.reshape(n, n)
    if u.ndim == 1:
        est_panel = {"title": "WoS estimate", "image": square(u), "cmap": cmap,
                     "clim": (np.nanmin(exact), np.nanmax(exact))}
        exact_panel = {"title": exact_title, "image": square(exact), "cmap": cmap}
    else:
        arrows = lambda a: (square(a[:, 0]), square(a[:, 1]))
        est_panel = {"title": "WoS estimate: norm(u), arrows u",
                     "image": square(np.linalg.norm(u, axis=1)), "quiver": arrows(u), "cmap": cmap}
        exact_panel = {"title": exact_title, "image": square(np.linalg.norm(exact, axis=1)),
                       "quiver": arrows(exact), "cmap": cmap}
    exact_panel.update(exact_extra or {})
    error_panel = {"title": f"|error|  (max {max_err:.3f})", "image": square(err), "cmap": "magma"}
    return [est_panel, exact_panel, error_panel], max_err


def volume_stages(name):
    """--solve / --render handling for volume examples: returns (solve, render).
    Default: solve only if no saved volume exists, then render."""
    import argparse
    parser = argparse.ArgumentParser(description=f"{name}: solve the volume, then plot and render")
    parser.add_argument("--solve", action="store_true", help="force a new solve")
    parser.add_argument("--render", action="store_true", help="render the saved volume only")
    args = parser.parse_args()
    solve_now = args.solve or not (args.render or volume_path(name).exists())
    return solve_now, (not args.solve) or args.render
