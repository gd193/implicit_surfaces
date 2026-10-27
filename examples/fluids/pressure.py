"""Velocity advection/forcing + 2024 WoB pressure projection for closed 3D walls.

Reuses transport3d and fluid3d's volume renderer. Finite MC budgets, boundary
series truncation and grid transport are approximations; save projection energy
and dye mass diagnostics, and reject runaway estimates instead of clipping them.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
from examples.wos.common import volume_path
from scipy.ndimage import distance_transform_edt
from sphereWalker.pressure import ClosedPressureProjector
from examples.fluids.transport import make_transport


def run_velocity3d(name, domain, *, field_fn, initial_velocity, initial_dye,
                   body_force, grid, steps, dt, seed, n_samples,
                   path_length, volume_samples, boundary_volume_samples,
                   regularization, diagnostics_fn, max_active_states=65536):
    """Evolve an inviscid stationary-wall example with explicit sampling budgets.

    initial_velocity(points), initial_dye(points) return flattened vector and
    two-channel fields. body_force(velocity_grid,dye_grid) is acceleration.
    field_fn(x) is used only for visualization. Fluid membership is domain-owned.
    Solid velocity samples use nearest-fluid extension for the pressure boundary
    trace. Dye remains zero in solids. RK2/BFECC is not mass-conservative, and
    grid interpolation/backtracing can leak across under-resolved features.
    """
    if grid < 3 or steps < 1 or not np.isfinite(dt) or dt <= 0 or seed < 0:
        raise ValueError("grid >= 3, steps > 0, finite dt > 0 and seed >= 0 required")
    lo, hi = np.asarray(domain.lower), np.asarray(domain.upper)
    spacing = (hi-lo)/grid
    grid_lo, grid_hi = lo+.5*spacing, hi-.5*spacing
    axes = [np.linspace(grid_lo[k], grid_hi[k], grid) for k in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    points_j = jnp.asarray(points, jnp.float32)
    fluid = np.asarray(jax.jit(jax.vmap(domain.contains))(points_j)).reshape((grid,)*3)
    if not fluid.any():
        raise ValueError("no fluid grid points")
    nearest = distance_transform_edt(~fluid, return_distances=False, return_indices=True)
    nearest = tuple(jnp.asarray(i) for i in nearest)
    mask = jnp.asarray(fluid)[..., None]
    extend = jax.jit(lambda values: values[nearest])
    sample, transport = make_transport(points_j, grid_lo, spacing, dt)
    solver = ClosedPressureProjector(
        domain, lambda params, x: sample(params['velocity'], x),
        volume_samples=volume_samples, boundary_volume_samples=boundary_volume_samples,
        regularization=regularization, path_length=path_length)
    initial_u, initial_c = np.asarray(initial_velocity(points)), np.asarray(initial_dye(points))
    if initial_u.shape != (len(points), 3) or not np.isfinite(initial_u).all():
        raise ValueError("initial_velocity must return finite (N,3) values")
    if (initial_c.shape != (len(points), 2) or not np.isfinite(initial_c).all()
            or np.any((initial_c < 0) | (initial_c > 1))):
        raise ValueError("initial_dye must return finite (N,2) values in [0,1]")
    vel = jnp.asarray(initial_u, jnp.float32).reshape((grid,)*3+(3,))
    dye = jnp.asarray(initial_c, jnp.float32).reshape((grid,)*3+(2,))
    print(f"{name}: {grid}^3, {fluid.sum()} fluid points, {n_samples} shared boundary walks, "
          f"series length {path_length}, volume samples {volume_samples}/{boundary_volume_samples}", flush=True)
    start = time.perf_counter()
    key = jax.random.PRNGKey(seed)
    vel, dye = jnp.where(mask, vel, 0.), jnp.where(mask, dye, 0.)
    print(f"initial velocity: max={float(jnp.linalg.norm(vel, axis=-1).max()):.5g}, "
          f"rms={float(jnp.sqrt(jnp.mean(jnp.sum(vel*vel, -1)))):.5g}", flush=True)
    np.savez_compressed(volume_path(name).with_name('initial.npz'), velocity=np.asarray(vel),
                        dye=np.asarray(dye), lower=grid_lo, upper=grid_hi)
    initial_mass = np.asarray(dye).sum(axis=(0, 1, 2))*np.prod(spacing)
    frames, dye_frames, diagnostics, timings, driver_timings = [], [], [], [], []
    def save_accepted_frames():
        np.savez_compressed(
            volume_path(name), speed=np.stack(frames), dye=np.stack(dye_frames),
            sdf=np.asarray(jax.vmap(field_fn)(points_j)).reshape((grid,)*3),
            lower=grid_lo, upper=grid_hi, dt=dt, first_frame_time=dt, seconds=time.perf_counter()-start,
            last_velocity=np.asarray(vel), initial_dye_mass=initial_mass,
            diagnostics=np.asarray(diagnostics),
            diagnostic_names=np.asarray(list(values)), n_samples=n_samples,
            driver_seconds=np.asarray(driver_timings),
            driver_names=np.asarray(["process", "compaction", "sync"]),
            stage_seconds=np.asarray(timings), stage_names=np.asarray(["transport", "projection", "dye"]),
            seed=seed,
            requested_steps=steps, completed_steps=len(frames), complete=len(frames) == steps,
            path_length=path_length,
            volume_samples=volume_samples, boundary_volume_samples=boundary_volume_samples,
            regularization=regularization, formulation='velocity_pressure_wob')

    for step in range(steps):
        key, projection_key = jax.random.split(key)
        t0 = time.perf_counter()
        extended = extend(vel)
        intermediate = transport(extended, extended)+dt*body_force(vel, dye)
        intermediate = extend(intermediate)
        jax.block_until_ready(intermediate)
        t1 = time.perf_counter()
        driver_log = []
        result = solver.project({'velocity': intermediate}, points_j[fluid.ravel()],
                                projection_key, n_samples=n_samples, max_active_states=max_active_states,
                                steps_between_refill=path_length, jit_buck_size=128, n_compactions=2, diag_log=driver_log)
        jax.block_until_ready(result.velocity)
        t2 = time.perf_counter()
        driver_timings.append([sum(row[key] for row in driver_log)/1000
                               for key in ("ms_process", "ms_compact", "ms_sync")])
        before_energy = float(jnp.mean(jnp.sum(intermediate[fluid]**2, -1)))
        after_energy = float(jnp.mean(jnp.sum(result.velocity**2, -1)))
        gain = after_energy/max(before_energy, 1e-20)
        print(f"projection: gain={gain:.5g}, input_max={float(jnp.linalg.norm(intermediate, axis=-1).max()):.5g}, "
              f"output_max={float(jnp.linalg.norm(result.velocity, axis=-1).max()):.5g}, seconds={t2-t1:.4f}", flush=True)
        if gain > 2.:
            raise RuntimeError(f"step {step}: projection energy gain {gain:.3g}; "
                               "increase MC budgets/check series convergence before simulating")
        vel = jnp.zeros((grid**3, 3), jnp.float32).at[fluid.ravel()].set(result.velocity)
        vel = vel.reshape((grid,)*3+(3,))
        courant = float(jnp.max(jnp.linalg.norm(vel, axis=-1))*dt/spacing.min())
        if courant > 1.:
            raise RuntimeError(f"step {step}: grid Courant number {courant:.3g} > 1; reduce dt")
        dye = jnp.where(mask, transport(dye, extend(vel)), 0.)
        jax.block_until_ready(dye)
        timings.append([t1-t0, t2-t1, time.perf_counter()-t2])
        values = diagnostics_fn(np.asarray(vel), np.asarray(dye), grid_lo, spacing)
        values.update(projection_energy_gain=gain, courant=courant)
        diagnostics.append(list(values.values()))
        if not np.isfinite(diagnostics[-1]).all():
            raise RuntimeError(f"step {step}: nonfinite diagnostics")
        frames.append(np.asarray(jnp.linalg.norm(vel, axis=-1), np.float16))
        dye_frames.append(np.asarray(dye, np.float16))
        print(f"step {step+1}/{steps}: "+', '.join(f'{k}={v:.4g}' for k,v in values.items()), flush=True)
        print(f"stage seconds: transport={timings[-1][0]:.4f}, projection={timings[-1][1]:.4f}, dye={timings[-1][2]:.4f}", flush=True)
        save_accepted_frames()
    print(f"saved {volume_path(name)} ({time.perf_counter()-start:.1f}s)", flush=True)


DYE_CENTERS = np.array([[-.25, -.25, -.25], [.75, .75, .75]])

def gyroid_surface():
    """Same degree-2 Warp-backed analytic field used by the gyroid renderer."""
    from surfaces.implicit.jnp_bounded import jnpboundedImplicit

    roll = jnp.asarray(np.roll(np.eye(3), -1, axis=0).T, jnp.float32)
    ones = jnp.ones((3, 1), jnp.float32)

    def field(params, x):
        p = jnp.pi * x
        return (jnp.sin(p) * jnp.cos(p @ roll)) @ ones

    return jnpboundedImplicit(autobounds_max_degree=2, use_warp=True,
                              fn=field, level_set=0., dim=3)

def initial_dye(points: np.ndarray) -> np.ndarray:
    """Two Gaussian pockets centered at interior field minima (F=-1.5)."""
    return np.exp(-np.sum((points[:, None, :] - DYE_CENTERS)**2, axis=-1) / .3**2)

def gyroid_diagnostics(velocity: np.ndarray, dye: np.ndarray, lower: np.ndarray,
                spacing: np.ndarray) -> dict[str, float]:
    dye = np.asarray(dye)
    weights = dye.reshape(-1, 2)
    totals = weights.sum(axis=0)
    masses = totals * np.prod(spacing)
    axes = [lower[k] + spacing[k] * np.arange(dye.shape[k]) for k in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    centers = weights.T @ points / np.maximum(totals[:, None], 1e-20)
    displacement = np.linalg.norm(centers - DYE_CENTERS, axis=-1)
    return {"red_mass": float(masses[0]), "blue_mass": float(masses[1]),
            "red_displacement": float(displacement[0]), "blue_displacement": float(displacement[1]),
            "velocity_rms": float(np.sqrt(np.mean(np.sum(velocity**2, axis=-1))))}

def rotation(points):
    return np.cross(np.array([0., 0., 1.]), points)

def sphere_diagnostics(velocity, dye, lower, spacing):
    axes = [lower[k]+spacing[k]*np.arange(dye.shape[k]) for k in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing='ij'), -1)
    inside = np.linalg.norm(points, axis=-1) < 1.
    error = np.linalg.norm(velocity[inside]-rotation(points[inside]), axis=-1)
    return {'red_mass': float(dye[..., 0].sum()*spacing.prod()),
            'blue_mass': float(dye[..., 1].sum()*spacing.prod()),
            'velocity_rmse': float(np.sqrt(np.mean(error**2)))}


def main():
    """Closed-domain all-Neumann pressure flow: rigid rotation or forced gyroid dye."""
    import argparse
    from examples.fluids.vorticity import render3d
    from surfaceSampler.boundary import ImplicitBoxBoundary, SphereBoundary
    from utils.helpers import update_trust_region_size_autobound

    parser = argparse.ArgumentParser(description=main.__doc__)
    scenes = parser.add_subparsers(dest='scene', required=True)
    for scene in ('sphere', 'gyroid'):
        p = scenes.add_parser(scene)
        p.add_argument('--render', action='store_true')
        p.add_argument('--name', default='fluid_velocity_sphere' if scene == 'sphere' else 'fluid_gyroid_pressure')
        p.add_argument('--grid', type=int, default=16)
        p.add_argument('--steps', type=int, default=5 if scene == 'sphere' else 10)
        p.add_argument('--dt', type=float, default=.02 if scene == 'sphere' else .1)
        p.add_argument('--walks', type=int, default=262144 if scene == 'sphere' else 8192)
        p.add_argument('--path-length', type=int, default=6 if scene == 'sphere' else 4)
        p.add_argument('--volume-samples', type=int, default=512)
        p.add_argument('--boundary-volume-samples', type=int, default=256)
        p.add_argument('--regularization', type=float, default=1e-5)
        p.add_argument('--max-active-states', type=int, default=2048)
        p.add_argument('--seed', type=int, default=0)
        p.add_argument('--render-frames', type=int, default=120 if scene == 'sphere' else 240)
        p.add_argument('--orbit', action='store_true')
        if scene == 'gyroid':
            p.add_argument('--buoyancy', type=float, default=4.)
            p.add_argument('--geometry-epsilon', type=float, default=1e-4)
            p.add_argument('--max-steps', type=int, default=10000)
    args = parser.parse_args()
    if (args.grid < 3 or min(args.steps, args.walks, args.max_active_states, args.render_frames) < 1
            or args.path_length < 3 or args.seed < 0
            or min(args.volume_samples, args.boundary_volume_samples) < 2
            or args.volume_samples % 2 or args.boundary_volume_samples % 2
            or not np.isfinite([args.dt, args.regularization]).all()
            or args.dt <= 0 or args.regularization < 0):
        parser.error('positive counts/dt, length >=3, even sample budgets and finite parameters required')
    if args.render:
        options = dict(camera=(5., -6., 4.)) if args.scene == 'gyroid' else {}
        render3d(args.name, title=f'{args.scene}: all-Neumann pressure flow',
                 render_frames=args.render_frames, rotate_camera=args.orbit, **options)
        return
    match args.scene:
        case 'sphere':
            domain = SphereBoundary()
            centers = np.array([[-.4, 0., 0.], [.4, 0., 0.]])
            velocity = rotation
            dye = lambda p: np.exp(-np.sum((p[:, None, :]-centers)**2, -1)/.2**2)
            force = lambda u, d: jnp.zeros_like(u)
            diagnostic = sphere_diagnostics
            field = lambda x: domain.surface.signed_value(domain.params, x)
        case 'gyroid':
            if (args.max_steps < 1 or not np.isfinite([args.buoyancy, args.geometry_epsilon]).all()
                    or args.geometry_epsilon <= 0):
                parser.error('positive geometry epsilon/step limit and finite buoyancy required')
            surface = gyroid_surface()
            domain = ImplicitBoxBoundary(surface, {}, (-2.,)*3, (2.,)*3,
                lipschitz_bound=2*np.pi*np.sqrt(3), update_tr_fn=update_trust_region_size_autobound,
                epsilon=args.geometry_epsilon, refinement_limit=.002, max_steps=args.max_steps,
                max_active_states=args.max_active_states)
            velocity, dye = lambda points: np.zeros_like(points), initial_dye
            force = lambda vel, dye: args.buoyancy*(dye[..., :1]-dye[..., 1:])*jnp.array([0., 0., 1.])
            diagnostic, field = gyroid_diagnostics, lambda x: surface.signed_value({}, x)
    run_velocity3d(args.name, domain, field_fn=field, initial_velocity=velocity,
        initial_dye=dye, body_force=force, grid=args.grid, steps=args.steps, dt=args.dt,
        seed=args.seed, n_samples=args.walks, path_length=args.path_length,
        volume_samples=args.volume_samples, boundary_volume_samples=args.boundary_volume_samples,
        regularization=args.regularization, diagnostics_fn=diagnostic,
        max_active_states=args.max_active_states)


if __name__ == '__main__':
    main()
