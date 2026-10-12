"""Torus R = 0.6, r = 0.3 with scalar g = cos(3 phi): one volume solve, all views.

  python torus.py [--solve | --render]

Outputs in assets/renders/wos/torus/: volume.npz, slices.png (z = 0, y = 0),
bc.png, and a still plus 20 s orbit (shared sweeping cut along z) for: bc,
cut, iso.
"""
from common import load_volume, plot, solve_volume, volume_slice, volume_stages
from shapes import torus

NAME, N, LO, HI = "torus", 97, (-1., -1., -1.), (1., 1., 1.)  # odd N: planes at 0
surface, params, g = torus()
solve_now, render_now = volume_stages(NAME)

if solve_now:
    solve_volume(NAME, surface, g, LO, HI, N, params=params, n_walks=256)

if render_now:
    data = load_volume(NAME)
    u, mid = data["values"], N // 2
    plot(NAME, [
        {"title": "slice z = 0  (x, y)", "image": volume_slice(u, 2, mid), "cmap": "coolwarm",
         "clim": (-1., 1.)},
        {"title": "slice y = 0  (x, z)", "image": volume_slice(u, 1, mid), "cmap": "coolwarm",
         "clim": (-1., 1.)},
    ], (LO[0], HI[0], LO[1], HI[1]),
        f"Torus, g = cos(3φ), {int(data['n_walks'])} walks/pt, solve {float(data['seconds']):.1f}s",
        name_suffix="slices")

    import render_volume as rv
    rv.bc_overview(NAME, surface, params, g, LO, HI, cmap="coolwarm", clim=(-1., 1.),
                   info=["Laplace: -Lap u = 0 in a torus (R = 0.6, r = 0.3)",
                         "Dirichlet: u = g on the surface", "g = cos(3 phi), phi = atan2(y, x)"],
                   normal=(0., 0., 1.))
    grid, _ = rv.load(NAME)
    smooth, _ = rv.load(NAME, sigma=1.)
    rv.scalar_views(NAME, grid, smooth, "Torus, g = cos(3 phi)", normal=(0., 0., 1.),
                    levels=[-.75, -.5, -.25, .25, .5, .75], clim=(-1., 1.),
                    cut_camera=(2.3, -2.7, 1.3), iso_camera=(1.8, -2.2, 2.4))
