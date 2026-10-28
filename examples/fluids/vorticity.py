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

omega is advected semi-Lagrangian and stretched using the vortex-segment
formula of Rioux-Lavoie et al. (2022), Eqs.21/22. Both segment velocities come
from the previous cache at the departure point. omega enters the walks through
params (no retrace per step).

Outputs in assets/renders/wos/<name>/: volume.npz (norm(omega) per step, sdf)
and a 20 s orbit of norm(omega) isosurfaces inside the translucent wall.
"""
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import map_coordinates as map_coordinates_np

from examples.wos.common import load_volume, volume_path, volume_stages
from sphereWalker.integrator import WoSGradientIntegrator
from examples.fluids.dirichlet import make_gradient_velocity_solve
from examples.fluids.transport import make_transport, make_vortex_stretch


def make_wos_gradient_solve(*args, **kwargs) -> Callable:
    """Factories return a batch gradient solve callable with geometry bound."""
    return WoSGradientIntegrator(*args, **kwargs).integrate_batch


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
          stretching=True, velocity_stride=1, wall_psi=None, control_fn=None,
          make_gradient_solve=make_wos_gradient_solve, cell_centered=False,
          bound_clearance=False, max_active_states=1 << 20, stages=volume_stages,
          dye0=None, continuous_dye=False, diagnostics_fn=None, require_complete=False, render_frames=600,
          rotate_camera=True, check_transport=False, smoke=False, obstacle_metadata=(),
          make_velocity_solve=make_gradient_velocity_solve,
          n_compactions=4, compaction_mode='uniform', advection_method='bfecc', refill_rounds=4,
          frame_stride=1):
    """surface: implicit with empty_sphere(params, x, tr) and signed_value, negative inside.
    omega0(points) -> (N, 3) initial vorticity, zeroed outside. levels: isosurfaces as fractions of the initial max.
    stretching: apply the paper's boundary-aware vortex-segment update.
    velocity_stride: solve u on every k-th grid point and interpolate
    trilinearly (u is one derivative smoother than omega); k^3 fewer walks.
    wall_psi(x) -> (3,): Dirichlet psi on the wall (default 0); see the module doc.
    control_fn(x) -> (3,): control variate, a guess of psi(x) (e.g. wall_psi
    extended inside); near-wall gradient noise scales with psi - control.
    make_gradient_solve accepts the WoS constructor arguments and returns a batch
    solve callable. cell_centered keeps samples off enclosing box faces.
    bound_clearance uses conservative empty spheres for the interior mask instead
    of treating the implicit field as a distance. stages selects solve/render.
    dye0(points) optionally supplies two passive dye channels in [0,1].
    continuous_dye replenishes that initial profile after every advection step,
    excluding solids. It adds density only, without driving the velocity field.
    diagnostics_fn(velocity, dye, grid_lower, spacing) returns named scalar
    measurements, saved per step alongside the last solved velocity.
    require_complete rejects truncated walks instead of conditioning on completion.
    check_transport diagnoses surface crossings in raw advection and stops on failure;
    it never rejects individual paths or retains their density.
    refill_rounds groups pending full walker batches between host counter reads;
    the integrator retains its drain checks and compaction opportunities.
    frame_stride saves every k-th visualization frame and the last one;
    physics and scalar diagnostics still run on every step.
    rotate_camera=False keeps the video viewpoint fixed to show advection clearly."""
    if continuous_dye and dye0 is None:
        raise ValueError('continuous_dye requires an initial dye profile')
    if isinstance(frame_stride, bool) or not isinstance(frame_stride, int) or frame_stride < 1:
        raise ValueError('frame_stride must be a positive integer')
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    spacing = (hi - lo) / (n if cell_centered else n - 1)
    grid_lo = lo + .5 * spacing if cell_centered else lo
    grid_hi = hi - .5 * spacing if cell_centered else hi
    axes = [np.linspace(grid_lo[i], grid_hi[i], n) for i in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    points_j = jnp.asarray(points, jnp.float32)

    def simulate():
        sdf = np.asarray(jax.vmap(lambda x: surface.signed_value(params, x))(points_j))
        solid = sdf.reshape(n, n, n) > 0
        clearance = -sdf
        if bound_clearance:
            # TR/2 exceeds the two-voxel threshold. Geometry comes entirely
            # from the existing conservative surface query, not a field scale.
            clearance = np.asarray(jax.jit(lambda xs: jax.lax.map(
                lambda x: surface.empty_sphere(params, x, 6 * spacing.max())[1],
                xs, batch_size=2048))(points_j))
        on_coarse = np.all(np.indices((n, n, n)) % velocity_stride == 0, axis=0).ravel()
        solve_mask = (sdf < -2 * eps) & (clearance > 0) & on_coarse
        coarse_n = (n - 1) // velocity_stride + 1
        fine = np.indices((n, n, n)).reshape(3, -1) / velocity_stride
        omega = omega0(points).reshape(n, n, n, 3)
        omega = jnp.asarray(np.where(solid[..., None], 0., omega), jnp.float32)
        dye = None
        if dye0 is not None:
            initial_dye = np.asarray(dye0(points))
            if (initial_dye.shape != (len(points), 2) or not np.isfinite(initial_dye).all()
                    or np.any((initial_dye < 0) | (initial_dye > 1))):
                raise ValueError("dye0 must return two finite dye channels in [0,1]")
            dye = jnp.asarray(np.where(solid[..., None], 0., initial_dye.reshape(n, n, n, 2)), jnp.float32)
            dye_source = dye

        sample_vec, transport = make_transport(points_j, grid_lo, spacing, dt, method=advection_method)
        if bound_clearance:
            segment_clearance = lambda x: surface.empty_sphere(params, x, float(spacing.min()))[1]
        else:
            segment_clearance = lambda x: -surface.signed_value(params, x)
        # The cache bounds also contain the enclosing box in mixed examples.
        # Keep both segment endpoints inside that boundary as in section4.3.
        def all_clearance(x):
            box_clearance = jnp.min(jnp.minimum(x-jnp.asarray(lo), jnp.asarray(hi)-x))
            return jnp.minimum(segment_clearance(x), box_clearance)
        stretch = make_vortex_stretch(sample_vec, all_clearance, spacing, dt)

        @jax.jit
        def crossing_mask(departures):
            def check(i, crossed):
                x = points_j+(i/16.)*(departures-points_j)
                return crossed | (jnp.asarray(~solid).ravel()
                                  & (jax.vmap(lambda p: surface.signed_value(params, p))(x) >= 0))
            return jax.lax.fori_loop(0, 17, check, jnp.zeros(len(points_j), bool))

        empty_sphere = lambda p, x, tr: surface.empty_sphere(p["surface"], x, tr)
        boundary = wall_psi if wall_psi is not None else (lambda x: jnp.zeros(3, x.dtype))
        solve_velocity = make_velocity_solve(make_gradient_solve, empty_sphere, boundary,
                                       lambda p, x: sample_vec(p["omega"], x), dim=3, eps=eps,
                                       max_steps=max_steps, source_takes_params=True,
                                       control_fn=control_fn)
        pts = points_j[solve_mask]
        if pts.shape[0] == 0:
            raise ValueError("no interior gradient solve points; increase grid resolution")
        prepared_queries = solve_velocity.prepare_queries(pts)
        solve_indices = jnp.asarray(np.flatnonzero(solve_mask))

        @jax.jit
        def prepare_velocity(estimate):
            details = estimate.diagnostics
            velocity = jnp.zeros((n**3, 3), estimate.velocity.dtype).at[solve_indices].set(estimate.velocity)
            counts = details.termination_counts.sum(axis=0)
            summary = jnp.stack(((details.successful_samples == 0).sum(),
                                 details.successful_samples.sum(), jnp.all(details.complete),
                                 jnp.all(jnp.isfinite(velocity))))
            return velocity.reshape(n, n, n, 3), counts, summary

        @jax.jit
        def capture_frames(omega, dye):
            vorticity = jnp.linalg.norm(omega, axis=-1).astype(jnp.float16)
            density = None if dye is None else (jnp.minimum(dye.sum(axis=-1), 1.) if smoke else dye)
            return vorticity, None if density is None else density.astype(jnp.float16)

        @jax.jit
        def speed_max(velocity):
            return jnp.linalg.norm(velocity, axis=-1).max()

        initial_tr = float(2*spacing.min()) if check_transport else 1.
        options = dict(init_trust_region_size=initial_tr, n_samples=n_walks, steps_between_refill=16,
                       max_active_states=min(pts.shape[0] * n_walks, max_active_states), jit_buck_size=1024,
                       n_compactions=n_compactions, compaction_mode=compaction_mode, refill_rounds=refill_rounds)
        print(f"{name}: {n}^3 grid, {pts.shape[0]} velocity queries, {n_walks} walks/site, {steps} steps")
        frames, walk_counts, dye_frames, diagnostics, transport_checks, step_seconds = [], [], [], [], [], []
        frame_steps = []
        zero_success_counts = []
        injected_dye_mass = []
        start = time.perf_counter()
        failure = None
        try:
            for step in range(steps):
                step_start = time.perf_counter()
                estimate = solve_velocity.solve({"surface": params, "omega": omega}, prepared_queries,
                                                jax.random.PRNGKey(step), **options)
                vel, count_values, summary = prepare_velocity(estimate)
                counts, (zero_success, contributed, complete, finite) = jax.device_get((count_values, summary))
                zero_success = int(zero_success)
                walk_counts.append(counts)
                zero_success_counts.append(zero_success)
                if require_complete and not complete:
                    raise RuntimeError(f"step {step}: incomplete walks {counts.tolist()}")
                if counts[3]:
                    raise RuntimeError(f"step {step}: {counts[3]} invalid walks")
                if counts[2]:
                    print(f"step {step}: {counts[2]} walks reached max_steps; "
                          f"{int(contributed)} samples contributed "
                          "(capped source sums require explicit truncation)")
                if not finite:
                    raise RuntimeError(f"step {step}: nonfinite velocity estimate")
                if velocity_stride > 1:
                    coarse = np.asarray(vel)[::velocity_stride, ::velocity_stride, ::velocity_stride]
                    assert coarse.shape[0] == coarse_n
                    vel = np.stack([map_coordinates_np(coarse[..., k], fine, order=1, mode="nearest")
                                    for k in range(3)], -1).reshape(n, n, n, 3)
                    vel[solid] = 0.
                    vel = jnp.asarray(vel)
                maximum_speed = float(speed_max(vel))
                courant = maximum_speed*dt/spacing.min()
                if step % frame_stride == 0 or step == steps-1:
                    vorticity_frame, dye_frame = jax.device_get(capture_frames(omega, dye))
                    frames.append(vorticity_frame)
                    frame_steps.append(step)
                    if dye is not None:
                        dye_frames.append(dye_frame)
                if diagnostics_fn is not None:
                    measurements = diagnostics_fn(vel, dye, grid_lo, spacing)
                    diagnostics.append(list(measurements.values()))
                    if not np.isfinite(diagnostics[-1]).all():
                        raise RuntimeError(f"step {step}: nonfinite diagnostics")
                    print(f"step {step}: " + ", ".join(f"{k}={v:.4g}" for k, v in measurements.items()), flush=True)
                if dye is not None:
                    fields = jnp.concatenate((omega, dye), axis=-1)
                    moved = transport(fields, vel, return_details=True)
                    if check_transport:
                        crossed = crossing_mask(moved.departures)
                        crossings = int(jnp.count_nonzero(crossed))
                        smoke_paths = jnp.any((dye > 0) | (moved.values[..., 3:] > 0), axis=-1).ravel()
                        smoke_crossings = int(jnp.count_nonzero(crossed & smoke_paths))
                        solid_smoke = int(jnp.count_nonzero(jnp.where(
                            jnp.asarray(solid)[..., None], moved.values[..., 3:], 0.)))
                        long_paths = int(jnp.count_nonzero(jnp.any(
                            jnp.abs(moved.departures-points_j) > jnp.asarray(spacing, points_j.dtype), axis=-1)))
                        transport_checks.append([int(np.count_nonzero(~solid)), crossings, smoke_crossings,
                                                 solid_smoke, courant, long_paths])
                        if crossings or solid_smoke:
                            raise RuntimeError(f"step {step}: {crossings} fluid backtraces crossed the surface "
                                               f"({smoke_crossings} smoke paths); {solid_smoke} raw solid smoke nodes")
                    fields = jnp.where(jnp.asarray(solid)[..., None], 0., moved.values)
                    omega, dye = fields[..., :3], fields[..., 3:]
                    if continuous_dye:
                        replenished = jnp.maximum(dye, dye_source)
                        injected_dye_mass.append(float(jnp.sum(replenished-dye)*np.prod(spacing)))
                        dye = replenished
                else:
                    moved = transport(omega, vel, return_details=True)
                    omega = jnp.where(jnp.asarray(solid)[..., None], 0., moved.values)
                if stretching:
                    omega = stretch(omega, vel, moved.departures)
                jax.block_until_ready(omega)
                step_seconds.append(time.perf_counter()-step_start)
                if step % 5 == 0:
                    print(f"step {step}/{steps}  {time.perf_counter() - start:.0f}s  "
                          f"max norm(u) {maximum_speed:.2f}  "
                          f"max norm(omega) {float(jnp.linalg.norm(omega, axis=-1).max()):.1f}  "
                          f"failed walks {int(counts.sum() - contributed)}, "
                          f"{step_seconds[-1]:.3f}s/step", flush=True)
        except RuntimeError as error:
            failure = error
            if not frames:
                raise
        seconds = time.perf_counter() - start
        extra = {}
        if dye_frames:
            extra["smoke" if smoke else "dye"] = np.stack(dye_frames)
            dye_frames.clear()
            extra["last_dye"] = np.asarray(dye)
        if continuous_dye:
            extra['injected_dye_mass'] = np.asarray(injected_dye_mass)
        if diagnostics:
            extra.update(diagnostics=np.asarray(diagnostics), diagnostic_names=np.asarray(list(measurements)),
                         last_velocity=np.asarray(vel))
        if transport_checks:
            extra.update(transport_checks=np.asarray(transport_checks),
                transport_check_names=np.asarray(['fluid_paths', 'sampled_crossings', 'smoke_path_crossings',
                                                   'raw_solid_smoke_nodes', 'courant', 'oversized_paths']))
        frames = np.stack(frames)
        np.savez_compressed(volume_path(name), omega=frames, sdf=sdf.reshape(n, n, n),
                            lower=grid_lo, upper=grid_hi, seconds=seconds, n_walks=n_walks, dt=dt,
                            max_steps=max_steps, eps=eps, zero_success_counts=np.asarray(zero_success_counts),
                            termination_counts=np.stack(walk_counts), step_seconds=np.asarray(step_seconds),
                            frame_times=np.asarray(frame_steps)*dt, frame_stride=frame_stride,
                            complete=failure is None, failure_reason=str(failure or ""),
                            last_vorticity=np.asarray(omega),
                            formulation='vorticity_vector_potential', advection_method=advection_method,
                            continuous_dye=continuous_dye,
                            **dict(obstacle_metadata), **extra)
        print(f"saved {volume_path(name)}  ({seconds:.0f}s)")
        if failure is not None:
            raise failure

    solve_now, render_now = stages(name)
    if solve_now:
        simulate()
    if render_now:
        render3d(name, title=title, levels=levels, camera=camera,
                 render_frames=render_frames, rotate_camera=rotate_camera)


def render3d(name, *, title, levels=(.2, .45), camera=(2.2, -2.6, 1.6),
             render_frames=600, rotate_camera=True):
    """Shared volume/dye renderer for both fluid formulations."""
    import pyvista as pv
    from examples.wos import render_volume as rv
    from scipy.ndimage import gaussian_filter

    data = load_volume(name)
    if not bool(data.get("complete", True)):
        title += " (partial run)"
    quantity = "norm(omega)" if "omega" in data else "speed"
    omega = data["omega" if "omega" in data else "speed"].astype(np.float32)
    dye_frames = data["dye"] if "dye" in data else None
    smoke_frames = data.get("smoke")
    top = float(omega[0].max())
    render_n = data["sdf"].shape[0]
    render_spacing = (data["upper"] - data["lower"]) / (render_n - 1)
    grid = pv.ImageData(dimensions=(render_n,) * 3, origin=tuple(data["lower"]), spacing=tuple(render_spacing))
    flat = lambda a: a.reshape(render_n ** 3, order="F")  # VTK: x fastest
    grid.point_data["sdf"] = flat(data["sdf"])
    if 'torus_major' in data or 'twisted_torus_major' in data:
        prefix = 'twisted_torus' if 'twisted_torus_major' in data else 'torus'
        wall = pv.ParametricTorus(ringradius=float(data[f'{prefix}_major']),
                                 crosssectionradius=float(data[f'{prefix}_minor']),
                                 u_res=256, v_res=128)
        wall.rotate_x(90, inplace=True)
        if prefix == 'twisted_torus':
            points = np.asarray(wall.points)
            angle = float(data['twisted_torus_twist'])*points[:, 2]
            c, s = np.cos(angle), np.sin(angle)
            wall.points = np.column_stack((c*points[:, 0]+s*points[:, 1],
                                          -s*points[:, 0]+c*points[:, 1], points[:, 2]))
            order = np.argsort(data.get('twisted_torus_coordinate_order', np.arange(3)))
            wall.points = wall.points[:, order]
            wall.compute_normals(auto_orient_normals=True, inplace=True)
        wall.translate(data[f'{prefix}_center'], inplace=True)
    elif 'sphere_radius' in data:
        wall = pv.Sphere(radius=float(data['sphere_radius']), center=data['sphere_center'],
                         theta_resolution=64, phi_resolution=64)
    else:
        wall = grid.contour([0.], scalars="sdf")
    cmap, clim = "plasma", (levels[0] * top, levels[-1] * top * 1.6)
    state = {}

    def on_frame(t):
        if 'frame_times' in data:
            times = data['frame_times']
            requested_time = times[0]+t*(times[-1]-times[0])
            k = min(max(int(np.searchsorted(times, requested_time, side='right'))-1, 0), len(omega)-1)
        else:
            k = min(int(t * len(omega)), len(omega) - 1)
        if smoke_frames is not None:
            grid.point_data['smoke'] = flat(smoke_frames[k].astype(np.float32))
            state['k'] = k
            return
        grid.point_data["w"] = flat(gaussian_filter(omega[k], .7))
        state["iso"] = rv.inside_only(grid.contour([lv * top for lv in levels], scalars="w"))
        if dye_frames is not None:
            state["dyes"] = []
            for channel in range(2):
                field = gaussian_filter(dye_frames[k, ..., channel].astype(np.float32), .5)
                grid.point_data["dye"] = flat(field)
                state["dyes"].append(rv.inside_only(grid.contour([.2], scalars="dye")))
        state["k"] = k

    def layer(p, normal, origin):
        iso = state.get("iso")
        frame_time = (float(data['frame_times'][state['k']]) if 'frame_times' in data else
                      state['k']*float(data['dt'])+float(data.get('first_frame_time', 0.)))
        actors = [p.add_text(f"t = {frame_time:.2f}", position="lower_left",
                             font_size=10, color="black")]
        if smoke_frames is not None:
            actors.append(p.add_volume(grid, scalars='smoke', cmap=['#303941', '#303941'], clim=(0., 1.),
                          opacity=[0., .10, .18, .25, .32, .40, .48, .56],
                          opacity_unit_distance=float(min(render_spacing)),
                          show_scalar_bar=False, shade=False))
        elif "dyes" in state:
            for mesh, color in zip(state["dyes"], ("tomato", "royalblue")):
                if mesh.n_points:
                    actors.append(p.add_mesh(mesh, color=color, opacity=.8, smooth_shading=True))
        elif iso.n_points:
            actors.append(p.add_mesh(iso, scalars="w", cmap=cmap, clim=clim, opacity=.75,
                                     smooth_shading=True, show_scalar_bar=False))
        return actors

    def static(p):
        color = 'royalblue' if data.get('scene', '') == 'smoke-column' else 'lightsteelblue'
        p.add_mesh(wall, color=color, opacity=1. if smoke_frames is not None else .12, smooth_shading=True)
        if dye_frames is not None or smoke_frames is not None:
            return
        bar = pv.PolyData(np.zeros((2, 3)))  # invisible, owns the color bar
        bar.point_data["w"] = np.asarray(clim)
        p.add_mesh(bar, scalars="w", cmap=cmap, clim=clim, opacity=0.,
                   scalar_bar_args={"title": quantity, "color": "black"})

    caption = "smoke concentration" if smoke_frames is not None else "red/blue passive dye" if dye_frames is not None else (
        f"{quantity} isosurfaces {', '.join(f'{lv:g}' for lv in levels)} x initial max")
    budget = (f"{int(data['n_samples'])} shared boundary walks" if "n_samples" in data else
              f"{int(data['n_walks'])} walks/pt/step" if "n_walks" in data else
              f"{int(data['boundary_lines'])} boundary lines, series {int(data['path_length'])}")
    rv.orbit(name, "fluid", f"{title}: {caption}, {budget}",
             layer, normal=(0., 0., 1.), extent_points=np.asarray(wall.points), static=static,
             camera=camera, sweeps=0, on_frame=on_frame, still_t=.5, frames=render_frames,
             rotate_camera=rotate_camera,
             view_up=data.get('render_view_up', (0., 1., 0.) if smoke_frames is not None else (0., 0., 1.)))


def main():
    """Interior torus scenes; the exterior smoke scene lives in smoke.py."""
    import argparse
    from functools import partial
    from examples.wos.shapes import torus, twisted_torus

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('scene', choices=('torus', 'twisted-torus'))
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument('--solve', action='store_true')
    stage.add_argument('--render', action='store_true')
    args = parser.parse_args()

    def stages(name):
        solve = args.solve or not (args.render or volume_path(name).exists())
        return solve, (not args.solve) or args.render

    match args.scene:
        case 'torus':
            major, minor, jet, width, background = .6, .3, 1.2, .08, .3
            flux = background*np.pi*minor**2 + 2*np.pi*jet*width**2*np.log(2.)
            c = flux/(2*np.pi)

            def wall_psi(x, floor=0.):
                rho = jnp.hypot(x[0], x[1])
                d2 = jnp.maximum((rho-major)**2+x[2]**2, floor**2)
                grad_theta = jnp.stack([-x[2]*x[0]/rho, -x[2]*x[1]/rho, rho-major])/d2
                return -c*grad_theta

            surface, params, _ = torus()
            run3d('fluid_torus', surface, params, (-.95, -.95, -.35), (.95, .95, .35), 96,
                  partial(torus_jet_vorticity, major=major, jet=jet, width=width),
                  title='Jet circulating in a torus', wall_psi=wall_psi,
                  control_fn=partial(wall_psi, floor=.5*minor), steps=400, n_walks=32,
                  levels=(.35, .7), camera=(1.6, -1.9, 1.5), stages=stages)
        case 'twisted-torus':
            surface, params, _ = twisted_torus('assets/models/twisted_torus_twist10_softplus_64k.npz')
            run3d('fluid_twisted_torus', surface, params, (-.72,)*3, (.72,)*3, 72,
                  partial(random_vorticity, wavelength=.35, peak=20.),
                  title='Neural twisted torus (twist 10)', n_walks=16, steps=200,
                  velocity_stride=2, camera=(1.6, -1.9, 1.3), stages=stages)


if __name__ == '__main__':
    main()
