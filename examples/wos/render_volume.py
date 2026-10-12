"""Render solved WoS volumes (from common.solve_volume) with PyVista.

Solving and rendering are separate: this module only reads
assets/renders/wos/<name>/volume.npz and writes next to it. Values are NaN outside the domain; for
interpolation near the boundary they are filled with the nearest inside value,
and the sdf channel clips every view back to the domain.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from scipy import ndimage

from common import ROOT, out_dir, volume_path

pv.OFF_SCREEN = True


def load(name, sigma=0.):
    """ImageData with point arrays 'sdf', 'u' (scalar or vector) and, for
    vectors, 'mag'. sigma > 0 smooths inside the domain (in voxels)."""
    data = np.load(volume_path(name))
    values, sdf = data["values"], data["sdf"]
    lower, upper = data["lower"], data["upper"]
    n = sdf.shape[0]
    inside = sdf < 0
    if sigma > 0:  # normalized smoothing: never mixes in outside values
        weight = ndimage.gaussian_filter(inside.astype(np.float64), sigma)
        smooth = lambda v: ndimage.gaussian_filter(np.where(inside, v, 0.), sigma) / np.maximum(weight, 1e-12)
        values = np.stack([smooth(values[..., k]) for k in range(values.shape[-1])], -1) \
            if values.ndim == 4 else smooth(values)
    # Fill outside with the nearest inside value so interpolation at the
    # boundary sees the boundary data instead of NaN.
    idx = ndimage.distance_transform_edt(~inside, return_distances=False, return_indices=True)
    values = values[tuple(idx)]

    grid = pv.ImageData(dimensions=(n, n, n), origin=tuple(lower),
                        spacing=tuple((upper - lower) / (n - 1)))
    flat = lambda a: a.reshape((n ** 3,) + a.shape[3:], order="F")  # VTK: x fastest
    grid.point_data["sdf"] = flat(sdf)
    if values.ndim == 4:
        grid.point_data["u"] = flat(values)
        grid.point_data["mag"] = np.linalg.norm(flat(values), axis=1)
    else:
        grid.point_data["u"] = flat(values)
    return grid, data


def shell(grid):
    """Boundary surface (sdf = 0) with the field sampled onto it."""
    return grid.contour([0.], scalars="sdf").sample(grid)


def inside_only(mesh):
    """Keep the part of a mesh inside the domain (sdf < 0)."""
    return mesh.clip_scalar(scalars="sdf", value=0., invert=True)


def plotter(title, shape=(1, 1)):
    p = pv.Plotter(off_screen=True, shape=shape, window_size=(900 * shape[1], 800 * shape[0]),
                   border=False)
    p.set_background("white")
    p.add_text(title, font_size=10, color="black")
    return p


def add_cutaway(p, grid, scalars, cmap, normal=(0., 1., 0.), origin=(0., 0., 0.), clim=None):
    """Cut the shape open along a plane: keep the half the normal points away
    from (so the cut face looks along +normal), color shell and face by the
    field. The font has no Greek glyphs, so titles stay ASCII."""
    surface = shell(grid)
    half = surface.clip(normal=normal, origin=origin)
    face = inside_only(grid.slice(normal=normal, origin=origin))
    args = dict(scalars=scalars, cmap=cmap, clim=clim, smooth_shading=True)
    p.add_mesh(half, **args, show_scalar_bar=False)
    label = "norm(u)" if scalars == "mag" else scalars  # font lacks "|"
    p.add_mesh(face, **args, scalar_bar_args={"title": label, "color": "black"})


def add_isosurfaces(p, grid, levels, cmap):
    """Translucent nested level sets of u inside a faint shell."""
    surfaces = inside_only(grid.contour(levels, scalars="u"))
    p.add_mesh(shell(grid), color="lightgray", opacity=0.12)
    p.add_mesh(surfaces, scalars="u", cmap=cmap, opacity=0.55, smooth_shading=True,
               scalar_bar_args={"title": "u", "color": "black"})


def add_streamlines(p, grid, n_seeds=400, seed=0):
    """Field lines of the vector field u as tubes colored by |u|, seeded at
    random inside points, inside a faint shell."""
    rng = np.random.default_rng(seed)
    inside = grid.points[grid.point_data["sdf"] < -0.02]
    seeds = pv.PolyData(inside[rng.choice(len(inside), min(n_seeds, len(inside)), replace=False)])
    lines = grid.streamlines_from_source(seeds, vectors="u", integration_direction="both",
                                         max_length=4., initial_step_length=0.5)
    lines = inside_only(lines)
    p.add_mesh(shell(grid), color="lightgray", opacity=0.15)
    p.add_mesh(lines.tube(radius=0.006), scalars="mag", cmap="plasma",
               scalar_bar_args={"title": "norm(u)", "color": "black"})


def view(example, name, title, build, camera=(2.2, -2.6, 1.6), frames=360, still=True):
    """Render one view: <example>/<name>.png at `camera` (if still), then
    <example>/<name>_orbit.mp4, one full turn at the camera's radius and
    height over `frames` frames at 30 fps (default 12 s)."""
    folder = out_dir(example)
    p = plotter(title)
    build(p)
    p.camera_position = [camera, (0., 0., 0.), (0., 0., 1.)]
    png = folder / f"{name}.png"
    if still:
        p.screenshot(png)
    radius, height = float(np.hypot(camera[0], camera[1])), float(camera[2])
    start = float(np.arctan2(camera[1], camera[0]))
    mp4 = folder / f"{name}_orbit.mp4"
    p.open_movie(str(mp4), framerate=30)
    for t in start + np.linspace(0., 2 * np.pi, frames, endpoint=False):
        p.camera_position = [(radius * np.cos(t), radius * np.sin(t), height), (0., 0., 0.), (0., 0., 1.)]
        p.write_frame()
    p.close()
    print(f"saved {(png if still else mp4).relative_to(ROOT)}" + (f" and {mp4.name}" if still else ""))


def boundary_mesh(surface, params, lower, upper, n=97):
    """Zero level set of the surface field, extracted on an n^3 grid. Built
    from the surface definition, independent of any solve."""
    axes = [np.linspace(lower[i], upper[i], n) for i in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    sdf = np.asarray(jax.vmap(lambda x: surface.signed_value(params, x))(jnp.asarray(pts, jnp.float32)))
    grid = pv.ImageData(dimensions=(n, n, n), origin=tuple(lower),
                        spacing=tuple((np.asarray(upper) - np.asarray(lower)) / (n - 1)))
    grid.point_data["sdf"] = sdf.reshape(n, n, n).reshape(-1, order="F")
    return grid.contour([0.], scalars="sdf")


def bc_overview(example, surface, params, g, lower, upper, info, *, name="bc", cmap="viridis",
                clim=None, n_arrows=400, seed=0):
    """Boundary-condition overview of a 3D example: <example>/<name>.png shows
    the boundary colored by scalar g (for vector g: a faint translucent shell
    colored by norm(g), with arrows colored and scaled by norm(g))
    from four angles plus the info lines; <example>/<name>_orbit.mp4 orbits it.
    The font has no Greek glyphs, so info lines stay ASCII."""
    mesh = boundary_mesh(surface, params, lower, upper)
    values = np.asarray(jax.vmap(g)(jnp.asarray(mesh.points, jnp.float32)))
    if values.ndim == 1:
        mesh.point_data["g"] = values
        scalars, arrows = "g", None
    else:
        mesh.point_data["g"] = values
        mesh.point_data["norm(g)"] = np.linalg.norm(values, axis=1)
        scalars = "norm(g)"
        rng = np.random.default_rng(seed)
        pick = rng.choice(mesh.n_points, min(n_arrows, mesh.n_points), replace=False)
        seeds = pv.PolyData(mesh.points[pick])
        seeds.point_data["g"] = values[pick]
        seeds.point_data["norm(g)"] = mesh.point_data["norm(g)"][pick]
        extent = float(np.max(np.asarray(upper) - np.asarray(lower)))
        # Thick arrows, long enough to dominate the view; length ~ norm(g).
        head = pv.Arrow(tip_length=0.28, tip_radius=0.07, shaft_radius=0.025)
        norms = mesh.point_data["norm(g)"]
        arrows = seeds.glyph(orient="g", scale="norm(g)", geom=head,
                             factor=0.12 * extent / float(norms.max() or 1.))

    def build(p, bar=True):
        bar_args = {"title": scalars, "color": "black"}
        if arrows is None:
            p.add_mesh(mesh, scalars=scalars, cmap=cmap, clim=clim, smooth_shading=True,
                       show_scalar_bar=bar, scalar_bar_args=bar_args)
        else:  # vectors: faint translucent shell colored by norm(g); arrows carry the data
            p.add_mesh(mesh, scalars="norm(g)", cmap=cmap, clim=clim, opacity=0.3,
                       smooth_shading=True, show_scalar_bar=False)
            p.add_mesh(arrows, scalars="norm(g)", cmap=cmap, clim=clim, smooth_shading=True,
                       show_scalar_bar=bar, scalar_bar_args=bar_args)

    cameras = {"iso": (2.2, -2.6, 1.6), "front (-y)": (0., -3.6, 0.), "side (+x)": (3.6, 0., 0.),
               "top (+z)": (0., 0.001, 3.6)}
    p = pv.Plotter(off_screen=True, shape=(2, 2), window_size=(1400, 1250), border=False)
    for k, (label, camera) in enumerate(cameras.items()):
        p.subplot(k // 2, k % 2)
        p.set_background("white")
        build(p, bar=(k == 0))
        p.add_text(label if k else "\n".join(info) + "\n\nview: iso", font_size=12 if k else 11,
                   color="black")
        p.camera_position = [camera, (0., 0., 0.), (0., 0., 1.) if "top" not in label else (0., 1., 0.)]
    folder = out_dir(example)
    p.screenshot(folder / f"{name}.png")
    p.close()
    print(f"saved {(folder / f'{name}.png').relative_to(ROOT)}")
    view(example, name, info[0], build, camera=cameras["iso"], still=False)
