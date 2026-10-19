"""Shared 3D Monte Carlo fluid (vorticity / vector potential) for closed domains.

Each step solves the vector Poisson problem -Lap psi = omega with psi = 0 on
the wall and gets the 3 x 3 Jacobian of psi at every grid point from one
WoSGradientIntegrator walk; u = curl psi. psi = 0 makes u exactly
divergence-free with u . n = 0 on the wall. The full vector-potential
conditions also need div psi = 0 (a Neumann-type condition, not available), so
curl u = omega + grad h with h harmonic: valid flow, approximate vorticity.
psi = 0 also means zero net flux around a handle (no flow around a torus).

wall_psi gives a net flux without Neumann data: u . n on the wall is the
surface curl of the tangential part of psi, so any wall psi = c grad theta with
theta an angle around the tube (closed, not exact on the wall) keeps the wall
impermeable while the loop integral 2 pi c around the tube is the flux through
it (Stokes): a steady flow around the handle, like the 2D inlet profile.

omega is advected semi-Lagrangian (RK2 backtrace, BFECC error correction
clamped to the local range: much less numerical diffusion) and stretched by
(omega . grad) u, with grad u from central differences of u smoothed over
stretch_sigma voxels. omega enters the walks through params (no retrace per step).

Outputs in assets/renders/wos/<name>/: volume.npz (norm(omega) per step, sdf)
and a 20 s orbit of norm(omega) isosurfaces inside the translucent wall.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter
from scipy.ndimage import map_coordinates as map_coordinates_np

from common import load_volume, volume_path, volume_stages
from sphereWalker.integrator import WoSGradientIntegrator


def ring_vorticity(points, center, axis, radius, core, circulation):
    """Gaussian-core vortex ring around `axis` through `center`; it moves along +axis."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    d = points - np.asarray(center)
    h = d @ axis
    radial = d - h[:, None] * axis
    rho = np.linalg.norm(radial, axis=1)
    r_hat = radial / np.maximum(rho, 1e-9)[:, None]
    strength = circulation / (np.pi * core ** 2) * np.exp(-((rho - radius) ** 2 + h ** 2) / core ** 2)
    return strength[:, None] * np.cross(axis, r_hat)


def torus_jet_vorticity(points, major, jet, width):
    """Vorticity of an axial jet w(d) = jet sech^2(d / width) along the centerline
    of a torus around z (d: distance to the centerline circle of radius major),
    in the straight-pipe approximation: omega = -w'(d) e_theta, e_theta = e_phi x e_d."""
    rho = np.hypot(points[:, 0], points[:, 1])
    e_rho = np.stack([points[:, 0], points[:, 1], np.zeros(len(points))], 1) / np.maximum(rho, 1e-9)[:, None]
    e_phi = np.stack([-e_rho[:, 1], e_rho[:, 0], np.zeros(len(points))], 1)
    offset = (rho - major)[:, None] * e_rho + points[:, 2:3] * np.array([0., 0., 1.])
    d = np.linalg.norm(offset, axis=1)
    e_d = offset / np.maximum(d, 1e-9)[:, None]
    s = 1. / np.cosh(d / width)
    return (2. * jet / width * s * s * np.tanh(d / width))[:, None] * np.cross(e_phi, e_d)


def random_vorticity(points, seed=0, modes=12, wavelength=.3, peak=20.):
    """omega = curl A for A a sum of random Fourier modes with wavenumber
    2 pi / wavelength: smooth and divergence-free, scaled to max norm `peak`."""
    rng = np.random.default_rng(seed)
    omega = np.zeros_like(points)
    for _ in range(modes):
        k = rng.normal(size=3)
        k *= 2 * np.pi / wavelength / np.linalg.norm(k)
        a = np.cross(k, rng.normal(size=3))  # amplitude perpendicular to k
        phase = rng.uniform(0, 2 * np.pi)
        # curl(a sin(k . x + phase)) = cos(k . x + phase) k x a
        omega += np.cos(points @ k + phase)[:, None] * np.cross(k, a)
    return omega * peak / np.linalg.norm(omega, axis=1).max()


def run3d(name, surface, params, lo, hi, n, omega0, *, title, steps=300, dt=.02,
          n_walks=32, eps=1e-3, max_steps=2000, levels=(.2, .45), camera=(2.2, -2.6, 1.6),
          stretch_sigma=3., velocity_stride=1, wall_psi=None, control_fn=None):
    """surface: implicit with empty_sphere(params, x, tr) and signed_value, negative inside.
    omega0(points) -> (N, 3) initial vorticity, zeroed outside. levels: isosurfaces as fractions of the initial max.
    stretch_sigma: smoothing (voxels) of u before grad u for stretching; noise in
    grad u multiplies omega every step, so it must be small. None: no stretching.
    velocity_stride: solve u on every k-th grid point and interpolate
    trilinearly (u is one derivative smoother than omega); k^3 fewer walks.
    wall_psi(x) -> (3,): Dirichlet psi on the wall (default 0); see the module doc.
    control_fn(x) -> (3,): control variate, a guess of psi(x) (e.g. wall_psi
    extended inside); near-wall gradient noise scales with psi - control.
    Stretching acts only farther than two voxels from the wall."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    axes = [np.linspace(lo[i], hi[i], n) for i in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    points_j = jnp.asarray(points, jnp.float32)
    spacing = (hi - lo) / (n - 1)

    def simulate():
        sdf = np.asarray(jax.vmap(lambda x: surface.signed_value(params, x))(points_j))
        solid = sdf.reshape(n, n, n) > 0
        deep = jnp.asarray(sdf.reshape(n, n, n) < -2 * spacing.max())[..., None]
        on_coarse = np.all(np.indices((n, n, n)) % velocity_stride == 0, axis=0).ravel()
        solve_mask = (sdf < -2 * eps) & on_coarse  # gradient walks must start off the wall
        coarse_n = (n - 1) // velocity_stride + 1
        fine = np.indices((n, n, n)).reshape(3, -1) / velocity_stride
        omega = omega0(points).reshape(n, n, n, 3)
        omega = jnp.asarray(np.where(solid[..., None], 0., omega), jnp.float32)

        def sample(grid, x):
            """Trilinear lookup of an (n, n, n) grid, clamped at the edges."""
            idx = (x - jnp.asarray(lo, x.dtype)) / jnp.asarray(spacing, x.dtype)
            return map_coordinates(grid, list(idx), order=1, mode="nearest")

        def sample_vec(grid, x):
            return jnp.stack([sample(grid[..., k], x) for k in range(3)])

        def trace(vel, step):
            """Grid points carried by -step along u (RK2): x - step u."""
            def one(x):
                mid = x - .5 * step * sample_vec(vel, x)
                return x - step * sample_vec(vel, mid)
            return jax.vmap(one)(points_j)

        def lookup(field, where):
            return jax.vmap(lambda x: sample_vec(field, x))(where).reshape(n, n, n, 3)

        def window(field, op, init):
            return jax.lax.reduce_window(field, init, op, (3, 3, 3, 1), (1, 1, 1, 1), "SAME")

        @jax.jit
        def advect(omega, vel):
            """BFECC: advect, advect back, correct half the round-trip error,
            advect again; clamp to the local range at the departure point."""
            back, forward = trace(vel, dt), trace(vel, -dt)
            plain = lookup(omega, back)
            error = .5 * (omega - lookup(plain, forward))
            new = lookup(omega + error, back)
            low = lookup(window(omega, jax.lax.min, jnp.inf), back)
            high = lookup(window(omega, jax.lax.max, -jnp.inf), back)
            new = jnp.clip(new, low, high)
            return jnp.where(jnp.asarray(solid)[..., None], 0., new)

        @jax.jit
        def stretch(omega, vel):
            """omega += dt (omega . grad) u; grad u[i, j] = d u_i / d x_j."""
            grad = jnp.stack([jnp.stack(jnp.gradient(vel[..., i], *spacing), -1) for i in range(3)], -2)
            return omega + dt * jnp.where(deep, jnp.einsum("...ij,...j->...i", grad, omega), 0.)

        empty_sphere = lambda p, x, tr: surface.empty_sphere(p["surface"], x, tr)
        boundary = wall_psi if wall_psi is not None else (lambda x: jnp.zeros(3, x.dtype))
        solver = WoSGradientIntegrator(empty_sphere, boundary,
                                       lambda p, x: sample_vec(p["omega"], x), dim=3, eps=eps,
                                       max_steps=max_steps, source_takes_params=True,
                                       control_fn=control_fn)
        pts = points_j[solve_mask]
        options = dict(init_trust_region_size=1., n_samples=n_walks, steps_between_refill=16,
                       max_active_states=min(pts.shape[0] * n_walks, 1 << 20), jit_buck_size=1024)
        print(f"{name}: {n}^3 grid, {pts.shape[0]} solve points, {n_walks} walks, {steps} steps")
        frames = []
        start = time.perf_counter()
        for step in range(steps):
            details = solver.integrate_batch({"surface": params, "omega": omega}, pts,
                                             jax.random.PRNGKey(step), return_details=True, **options)
            # Rare walks over max_steps are dropped: average the completed ones.
            jac = details.contribution_sum / jnp.maximum(details.successful_samples, 1)[:, None, None]
            curl = jnp.stack([jac[:, 2, 1] - jac[:, 1, 2], jac[:, 0, 2] - jac[:, 2, 0],
                              jac[:, 1, 0] - jac[:, 0, 1]], -1)
            vel = np.zeros((n ** 3, 3), np.float32)
            vel[solve_mask] = np.asarray(curl)
            vel = vel.reshape(n, n, n, 3)
            if velocity_stride > 1:
                coarse = vel[::velocity_stride, ::velocity_stride, ::velocity_stride]
                assert coarse.shape[0] == coarse_n
                vel = np.stack([map_coordinates_np(coarse[..., k], fine, order=1, mode="nearest")
                                for k in range(3)], -1).reshape(n, n, n, 3)
                vel[solid] = 0.
            frames.append(np.asarray(jnp.linalg.norm(omega, axis=-1), np.float16))
            omega = advect(omega, jnp.asarray(vel))
            if stretch_sigma is not None:
                smooth = np.stack([gaussian_filter(vel[..., k], stretch_sigma) for k in range(3)], -1)
                omega = stretch(omega, jnp.asarray(smooth))
            if step % 25 == 0:
                print(f"step {step}/{steps}  {time.perf_counter() - start:.0f}s  "
                      f"max norm(u) {float(np.linalg.norm(vel, axis=-1).max()):.2f}  "
                      f"max norm(omega) {float(jnp.linalg.norm(omega, axis=-1).max()):.1f}  "
                      f"failed walks {int(jnp.sum(n_walks - details.successful_samples))}")
        seconds = time.perf_counter() - start
        np.savez_compressed(volume_path(name), omega=np.stack(frames), sdf=sdf.reshape(n, n, n),
                            lower=lo, upper=hi, seconds=seconds, n_walks=n_walks, dt=dt)
        print(f"saved {volume_path(name)}  ({seconds:.0f}s)")

    def render():
        import pyvista as pv
        import render_volume as rv

        data = load_volume(name)
        omega = data["omega"].astype(np.float32)
        top = float(omega[0].max())
        grid = pv.ImageData(dimensions=(n, n, n), origin=tuple(lo), spacing=tuple(spacing))
        flat = lambda a: a.reshape(n ** 3, order="F")  # VTK: x fastest
        grid.point_data["sdf"] = flat(data["sdf"])
        wall = grid.contour([0.], scalars="sdf")
        cmap, clim = "plasma", (levels[0] * top, levels[-1] * top * 1.6)
        state = {}

        def on_frame(t):
            k = min(int(t * len(omega)), len(omega) - 1)
            grid.point_data["w"] = flat(gaussian_filter(omega[k], .7))
            state["iso"] = rv.inside_only(grid.contour([lv * top for lv in levels], scalars="w"))
            state["k"] = k

        def layer(p, normal, origin):
            iso = state["iso"]
            actors = [p.add_text(f"t = {state['k'] * float(data['dt']):.2f}", position="lower_left",
                                 font_size=10, color="black")]
            if iso.n_points:
                actors.append(p.add_mesh(iso, scalars="w", cmap=cmap, clim=clim, opacity=.75,
                                         smooth_shading=True, show_scalar_bar=False))
            return actors

        def static(p):
            p.add_mesh(wall, color="lightsteelblue", opacity=.12, smooth_shading=True)
            bar = pv.PolyData(np.zeros((2, 3)))  # invisible, owns the color bar
            bar.point_data["w"] = np.asarray(clim)
            p.add_mesh(bar, scalars="w", cmap=cmap, clim=clim, opacity=0.,
                       scalar_bar_args={"title": "norm(omega)", "color": "black"})

        rv.orbit(name, "fluid", f"{title}: norm(omega) isosurfaces "
                 f"{', '.join(f'{lv:g}' for lv in levels)} x initial max, {int(data['n_walks'])} walks/pt/step",
                 layer, normal=(0., 0., 1.), extent_points=np.asarray(wall.points), static=static,
                 camera=camera, sweeps=0, on_frame=on_frame, still_t=.5)

    solve_now, render_now = volume_stages(name)
    if solve_now:
        simulate()
    if render_now:
        render()
