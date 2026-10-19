"""Render solved WoS volumes (from common.solve_volume) with PyVista.

Solving and rendering are separate: this module only reads
assets/renders/wos/<name>/volume.npz and writes next to it. Values are NaN outside the domain; for
interpolation near the boundary they are filled with the nearest inside value,
and the sdf channel clips every view back to the domain.

Every video uses one mechanism (`orbit`): the camera makes one turn while a
cut plane sweeps back and forth through the shape. A view is a *layer*,
layer(p, normal, origin) -> actors, redrawn each frame and clipped to the
half-space behind the plane (the side the normal points away from). A faint
"ghost" of the full boundary stays fixed and carries the color bar; colors and
arrow scales are fixed for the whole video. The font has no Greek glyphs or
"|", so titles and labels stay ASCII.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from scipy import ndimage

from common import ROOT, out_dir, volume_path

pv.OFF_SCREEN = True
HEAD = pv.Arrow(tip_length=0.28, tip_radius=0.07, shaft_radius=0.025)


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


def _size(dataset):
    return float(np.max(np.ptp(np.asarray(dataset.bounds).reshape(3, 2), axis=1)))


def field_range(grid, scalars):
    """Color range of a grid array over inside points (0..max for norms)."""
    v = np.asarray(grid.point_data[scalars])[grid.point_data["sdf"] < 0]
    return (0., float(v.max())) if scalars == "mag" else (float(v.min()), float(v.max()))


# --------------------------------------------------------------------- layers
def ghost(mesh, scalars, cmap, clim, label):
    """Static faint full boundary; owns the color bar."""
    def add(p):
        p.add_mesh(mesh, scalars=scalars, cmap=cmap, clim=clim, opacity=0.12, smooth_shading=True,
                   scalar_bar_args={"title": label, "color": "black"})
    return add


def combine(*layers):
    return lambda p, normal, origin: [a for layer in layers for a in layer(p, normal, origin)]


def clipped(mesh, scalars, cmap, clim, opacity=1.):
    """A fixed mesh, clipped to the back half-space each frame."""
    def layer(p, normal, origin):
        part = mesh.clip(normal=normal, origin=origin)
        return [p.add_mesh(part, scalars=scalars, cmap=cmap, clim=clim, opacity=opacity,
                           smooth_shading=True, show_scalar_bar=False)] if part.n_points else []
    return layer


def cut_face(grid, scalars, cmap, clim, opacity=1.):
    """The cut face (plane inside the domain), colored by the field."""
    def layer(p, normal, origin):
        face = inside_only(grid.slice(normal=normal, origin=origin))
        return [p.add_mesh(face, scalars=scalars, cmap=cmap, clim=clim, opacity=opacity,
                           show_scalar_bar=False)] if face.n_points else []
    return layer


def arrows(points, vectors, cmap, clim, size, name="u"):
    """Fixed arrow seeds; each frame draws those in the back half-space.
    Arrow length ~ norm, one scale for all frames."""
    norms = np.linalg.norm(vectors, axis=1)
    factor = 0.1 * size / float(clim[1] or norms.max() or 1.)

    def layer(p, normal, origin):
        keep = (points - origin) @ normal <= 0
        if not keep.any():
            return []
        seeds = pv.PolyData(points[keep])
        seeds.point_data[name] = vectors[keep]
        seeds.point_data["norm"] = norms[keep]
        glyphs = seeds.glyph(orient=name, scale="norm", geom=HEAD, factor=factor)
        return [p.add_mesh(glyphs, scalars="norm", cmap=cmap, clim=clim, smooth_shading=True,
                           show_scalar_bar=False)]
    return layer


def plane_arrows(grid, cmap, clim, spacing=None):
    """Arrows of the raw solved u on a fixed lattice in the moving plane."""
    size = _size(grid)
    spacing = spacing or size / 28
    factor = 0.1 * size / float(clim[1] or 1.)

    def layer(p, normal, origin):
        e1 = np.cross(normal, [0., 0., 1.] if abs(normal[2]) < 0.9 else [1., 0., 0.])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(normal, e1)
        ticks = np.arange(-size / 2, size / 2 + spacing, spacing)
        A, B = np.meshgrid(ticks, ticks)
        probes = pv.PolyData(A.reshape(-1, 1) * e1 + B.reshape(-1, 1) * e2 + origin).sample(grid)
        keep = np.asarray(probes.point_data["sdf"]) < -0.01
        if not keep.any():
            return []
        glyphs = probes.extract_points(keep, adjacent_cells=False).glyph(
            orient="u", scale="mag", geom=HEAD, factor=factor)
        return [p.add_mesh(glyphs, scalars="mag", cmap=cmap, clim=clim, smooth_shading=True,
                           show_scalar_bar=False)]
    return layer


def isosurfaces(grid, levels):
    return inside_only(grid.contour(levels, scalars="u"))


def streamline_tubes(grid, n_seeds=400, seed=0):
    """Field lines of u as tubes, seeded at random inside points."""
    rng = np.random.default_rng(seed)
    inside = grid.points[grid.point_data["sdf"] < -0.02]
    seeds = pv.PolyData(inside[rng.choice(len(inside), min(n_seeds, len(inside)), replace=False)])
    lines = grid.streamlines_from_source(seeds, vectors="u", integration_direction="both",
                                         max_length=4., initial_step_length=0.5)
    return inside_only(lines).tube(radius=0.006)


def volume_samples(grid, n=500, seed=0):
    """Random inside points with the raw solved vectors there."""
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(grid.point_data["sdf"] < -0.01)
    pick = rng.choice(idx, min(n, len(idx)), replace=False)
    return np.asarray(grid.points)[pick], np.asarray(grid.point_data["u"])[pick]


# ---------------------------------------------------------------------- orbit
def orbit(example, name, title, layer, *, normal, extent_points, static=None,
          camera=(2.2, -2.6, 1.6), frames=600, sweeps=2, still=True, on_frame=None, still_t=0.):
    """Shared video mechanism for every view.

    The camera makes one full turn at its radius and height over `frames`
    frames at 30 fps (default 20 s). The plane x . normal = s starts beyond
    the shape (nothing clipped) and sweeps back and forth `sweeps` times down
    to the 10% depth quantile of `extent_points`. Writes <example>/<name>_orbit.mp4 and,
    if still, <example>/<name>.png with the plane through the origin.
    on_frame(t), t in [0, 1), runs before each frame's layer (still_t for the
    still): time-dependent views update their data there. sweeps=0 keeps the
    plane beyond the shape, so nothing is clipped.
    """
    normal = np.asarray(normal, float) / np.linalg.norm(normal)
    depth = np.asarray(extent_points) @ normal
    margin = 0.01 * float(np.ptp(depth))
    # Far end: everything shown. Near end: keep the back 10% so frames never
    # show only the ghost.
    lo_s, hi_s = float(np.quantile(depth, 0.1)), float(depth.max()) + margin
    folder = out_dir(example)
    p = plotter(title)
    if static is not None:
        static(p)
    up = (0., 0., 1.)
    actors = []
    if still:
        if on_frame is not None:
            on_frame(still_t)
        actors = layer(p, normal, np.zeros(3) if sweeps else hi_s * normal)
        p.camera_position = [tuple(camera), (0., 0., 0.), up]
        p.screenshot(folder / f"{name}.png")
    radius, height = float(np.hypot(camera[0], camera[1])), float(camera[2])
    start = float(np.arctan2(camera[1], camera[0]))
    mp4 = folder / f"{name}_orbit.mp4"
    p.open_movie(str(mp4), framerate=30)
    for k in range(frames):
        t = k / frames
        s = hi_s - (hi_s - lo_s) * 0.5 * (1. - np.cos(2 * np.pi * sweeps * t))  # back and forth
        for actor in actors:
            p.remove_actor(actor)
        if on_frame is not None:
            on_frame(t)
        actors = layer(p, normal, s * normal)
        a = start + 2 * np.pi * t
        p.camera_position = [(radius * np.cos(a), radius * np.sin(a), height), (0., 0., 0.), up]
        p.write_frame()
    p.close()
    print(f"saved {(folder / (f'{name}.png' if still else mp4.name)).relative_to(ROOT)}"
          + (f" and {mp4.name}" if still else ""))


# ------------------------------------------------------------ standard views
def scalar_views(example, grid, smooth, title, *, normal, levels, cmap="coolwarm", clim=None,
                 cut_camera=(2.2, -2.6, 1.6), iso_camera=(2.2, -2.6, 1.6)):
    """cut and iso orbits for a scalar volume."""
    clim = clim or field_range(grid, "u")
    surface = shell(grid)
    static = ghost(surface, "u", cmap, clim, "u")
    pts = grid.points[grid.point_data["sdf"] < 0]
    orbit(example, "cut", f"{title}: cut sweeping through the volume",
          combine(clipped(surface, "u", cmap, clim), cut_face(grid, "u", cmap, clim)),
          normal=normal, extent_points=pts, static=static, camera=cut_camera)
    orbit(example, "iso", f"{title}: isosurfaces {', '.join(f'{v:g}' for v in levels)} "
          "(smoothed, sigma = 1 voxel)",
          clipped(isosurfaces(smooth, levels), "u", cmap, clim, opacity=0.6),
          normal=normal, extent_points=pts, static=static, camera=iso_camera)


def vector_views(example, grid, smooth, title, *, normal, cmap="plasma", cut_camera=(2.2, -2.6, 1.6)):
    """cut, streamlines, arrows and cut_arrows orbits for a vector volume."""
    clim = field_range(grid, "mag")
    surface = shell(grid)
    static = ghost(surface, "mag", cmap, clim, "norm(u)")
    pts = grid.points[grid.point_data["sdf"] < 0]
    common = dict(normal=normal, extent_points=pts, static=static)
    orbit(example, "cut", f"{title}: norm(u) on a sweeping cut",
          combine(clipped(surface, "mag", cmap, clim), cut_face(grid, "mag", cmap, clim)),
          camera=cut_camera, **common)
    orbit(example, "streamlines", f"{title}: field lines of u (smoothed, sigma = 1 voxel)",
          clipped(streamline_tubes(smooth), "mag", cmap, clim), **common)
    points, vectors = volume_samples(grid)
    orbit(example, "arrows", f"{title}: solved vectors u in the volume (raw)",
          arrows(points, vectors, cmap, clim, _size(grid)), **common)
    orbit(example, "cut_arrows", f"{title}: solved vectors u on the sweeping cut (raw)",
          combine(cut_face(grid, "mag", cmap, clim, opacity=0.45), plane_arrows(grid, cmap, clim)),
          camera=cut_camera, **common)


# --------------------------------------------------------- boundary overview
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
                clim=None, normal=(0., 0., 1.), n_arrows=400, seed=0):
    """Boundary-condition overview of a 3D example: <example>/<name>.png shows
    the boundary colored by scalar g (for vector g: a faint translucent shell
    colored by norm(g), with arrows colored and scaled by norm(g)) from four
    angles plus the info lines; <example>/<name>_orbit.mp4 uses the shared
    sweeping orbit, clipping the boundary to the back half-space."""
    mesh = boundary_mesh(surface, params, lower, upper)
    values = np.asarray(jax.vmap(g)(jnp.asarray(mesh.points, jnp.float32)))
    mesh.point_data["g"] = values
    if values.ndim == 1:
        scalars = "g"
        clim = clim or (float(values.min()), float(values.max()))
        surface_layer = clipped(mesh, scalars, cmap, clim)
        layer = surface_layer
    else:
        scalars = "norm(g)"
        mesh.point_data[scalars] = np.linalg.norm(values, axis=1)
        clim = clim or (0., float(mesh.point_data[scalars].max()))
        rng = np.random.default_rng(seed)
        pick = rng.choice(mesh.n_points, min(n_arrows, mesh.n_points), replace=False)
        surface_layer = clipped(mesh, scalars, cmap, clim, opacity=0.3)
        arrow_layer = arrows(np.asarray(mesh.points)[pick], values[pick], cmap, clim,
                             float(np.max(np.asarray(upper) - np.asarray(lower))) * 1.2, name="g")
        layer = combine(surface_layer, arrow_layer)
    far = np.full(3, 1e3)  # a plane beyond the shape: nothing clipped

    cameras = {"iso": (2.2, -2.6, 1.6), "front (-y)": (0., -3.6, 0.), "side (+x)": (3.6, 0., 0.),
               "top (+z)": (0., 0.001, 3.6)}
    p = pv.Plotter(off_screen=True, shape=(2, 2), window_size=(1400, 1250), border=False)
    unit = np.asarray(normal, float) / np.linalg.norm(normal)
    for k, (label, camera) in enumerate(cameras.items()):
        p.subplot(k // 2, k % 2)
        p.set_background("white")
        layer(p, unit, far)
        if k == 0:
            p.add_scalar_bar(title=scalars, color="black")
        p.add_text(label if k else "\n".join(info) + "\n\nview: iso", font_size=12 if k else 11,
                   color="black")
        p.camera_position = [camera, (0., 0., 0.), (0., 0., 1.) if "top" not in label else (0., 1., 0.)]
    folder = out_dir(example)
    p.screenshot(folder / f"{name}.png")
    p.close()
    print(f"saved {(folder / f'{name}.png').relative_to(ROOT)}")
    orbit(example, name, info[0], layer, normal=normal, extent_points=mesh.points,
          static=ghost(mesh, scalars, cmap, clim, scalars), camera=cameras["iso"], still=False)
