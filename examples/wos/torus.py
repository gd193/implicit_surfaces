"""Torus R = 0.6, r = 0.3 with scalar g = cos(3 phi): one volume solve, all views.

  python torus.py [--solve | --render]

Outputs in assets/renders/wos/torus/: volume.npz, slices.png (z = 0, y = 0),
and a still plus 12 s orbit for each 3D view: cut, iso.
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
                         "Dirichlet: u = g on the surface", "g = cos(3 phi), phi = atan2(y, x)"])
    grid, _ = rv.load(NAME)
    smooth, _ = rv.load(NAME, sigma=1.)
    # Keep the bottom half; the cut face at z = 0 looks up at the camera.
    rv.view(NAME, "cut", "Torus, g = cos(3 phi): cut open at z = 0",
            lambda p: rv.add_cutaway(p, grid, "u", "coolwarm", normal=(0., 0., 1.), clim=(-1., 1.)),
            camera=(2.3, -2.7, 1.3))
    rv.view(NAME, "iso", "Torus, isosurfaces u = +-0.25, +-0.5, +-0.75 (smoothed, sigma = 1 voxel)",
            lambda p: rv.add_isosurfaces(p, smooth, [-.75, -.5, -.25, .25, .5, .75], "coolwarm"),
            camera=(1.8, -2.2, 2.4))
