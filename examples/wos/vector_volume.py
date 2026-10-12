"""Shared pipeline for 3D vector examples with harmonic data g (so u = g):
solve the volume, check the exact error everywhere, plot slices with error
panels, and render the boundary overview, a cut, and streamlines."""
import jax
import jax.numpy as jnp
import numpy as np

from common import load_volume, plot, report, solve_volume, volume_slice, volume_stages


def run(name, surface, params, g, lo, hi, *, title, info, slices, cut_normal, cut_camera,
        n=97, **solve_kwargs):
    """slices: [(axis, label)] planes through 0 (n odd). info: ASCII lines."""
    solve_now, render_now = volume_stages(name)
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

    import render_volume as rv
    rv.bc_overview(name, surface, params, g, lo, hi, cmap="plasma", info=info)
    grid, _ = rv.load(name)
    smooth, _ = rv.load(name, sigma=1.)
    axis_name = "xyz"[int(np.argmax(np.abs(cut_normal)))]
    rv.view(name, "cut", f"{title}, norm(u): cut open at {axis_name} = 0",
            lambda p: rv.add_cutaway(p, grid, "mag", "plasma", normal=cut_normal), camera=cut_camera)
    rv.view(name, "streamlines", f"{title}, field lines of u (smoothed, sigma = 1 voxel)",
            lambda p: rv.add_streamlines(p, smooth))
