"""Shared buoyant-smoke evolution and source interpolation for 3D examples.

Uses the existing pressure integrator, transport, and volume artifact format.
"""
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from examples.fluids.grid import GridVelocity
from examples.fluids.pressure_prototype.pressure import ExteriorPressureProjector


def _write_npz(path: Path, *, compressed: bool, **arrays) -> None:
    """Replace an archive only after its new contents have been written."""
    temporary = path.with_name(f'.{path.name}.tmp.npz')
    writer = np.savez_compressed if compressed else np.savez
    try:
        writer(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _resume_data(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load the latest state and any previously completed frame history.

    Frames since the last completed history file are intentionally absent after
    an interruption. Their physical times are not invented or replayed.
    """
    history_keys = ('smoke', 'speed', 'step_seconds', 'frame_times')
    history = {key: np.empty(0) for key in history_keys}
    candidates = [p for p in (path, path.with_name('restart.npz')) if p.exists()]
    if not candidates:
        raise FileNotFoundError(f'no volume or restart checkpoint in {path.parent}')

    def completed_steps(candidate):
        with np.load(candidate) as saved:
            return int(saved['completed_steps'])

    # Prefer the completed history archive when both contain the same step.
    latest = max(candidates, key=completed_steps)
    with np.load(latest) as saved:
        state = {key: saved[key] for key in saved.files
                 if key not in (*history_keys, 'sdf')}
        if 'simulated_time' not in state:
            state['simulated_time'] = np.asarray(saved['frame_times'][-1] if 'frame_times' in saved
                else int(saved['completed_steps'])*float(saved['dt']))
    if path.exists():
        with np.load(path) as saved:
            history = {key: saved[key] for key in history_keys if key in saved}
            if 'frame_times' not in history:
                history['frame_times'] = (np.arange(len(history['smoke']))+1)*float(saved['dt'])
    return state, history


def evolve(args, *, geometry, obstacle_metadata):
    """Smoke preview with shared pressure solver and RK3 transport.

    Temperature diffusion uses a stable explicit grid Laplacian (kappa=.001),
    not the paper's MC diffusion estimator. Sampling/resolution are saved.
    """
    started = time.perf_counter()
    io_seconds = 0.
    restart_seconds = 0.
    restart_writes = 0
    from examples.wos.common import volume_path
    from examples.fluids.transport import make_transport
    if jax.default_backend() != 'gpu':
        raise RuntimeError('GPU required')
    refill_rounds = getattr(args, 'refill_rounds', 1)
    n = args.grid
    dx = 3./n
    axes = (np.arange(n)+.5)*dx-1.5
    points = jnp.asarray(np.stack(np.meshgrid(axes, axes, axes, indexing='ij'), -1).reshape(-1, 3), jnp.float32)
    shape = (n,)*3
    fluid = np.asarray(jax.vmap(geometry.contains)(points)).reshape(shape)
    _, advect = make_transport(points, np.repeat(axes[0], 3), np.repeat(dx, 3), args.dt, method='rk3')
    velocity = GridVelocity(np.repeat(axes[0], 3), np.repeat(dx, 3))
    boundary_volume = getattr(args, 'boundary_volume', 10)
    boundary_pseudo = getattr(args, 'boundary_pseudo', 10)
    solver = ExteriorPressureProjector(geometry, velocity,
        path_length=4, volume_samples=args.volume, boundary_volume_samples=boundary_volume,
        pseudo_samples=args.pseudo, boundary_pseudo_samples=boundary_pseudo,
        regularization=1e-5, query_batch_size=args.query_batch)
    field_values = np.asarray(jax.vmap(lambda x: geometry.surface.signed_value(geometry.params, x))(points)).reshape(shape)
    clearance = np.asarray(jax.jit(lambda x: jax.lax.map(geometry.wall_clearance, x,
                                                       batch_size=2048))(points)).reshape(shape)
    near = fluid & (clearance < args.near_band_cells*dx) & (args.near_paths > 0)
    far = fluid & ~near
    near_solver = ExteriorPressureProjector(geometry, velocity,
        path_length=4, volume_samples=args.volume, boundary_volume_samples=boundary_volume,
        pseudo_samples=args.pseudo, boundary_pseudo_samples=boundary_pseudo,
        regularization=1e-5, query_batch_size=args.near_query_batch)
    print(f'{n}^3: {far.sum()} bulk / {near.sum()} near-wall queries; '
          f'{args.paths} / {args.near_paths} boundary paths', flush=True)
    source = jnp.all(jnp.abs(points-jnp.array([0., -1.25, 0.], dtype=points.dtype)) < .125, -1).reshape(shape)
    mask = jnp.asarray(fluid)
    diffusion_steps = max(1, int(np.ceil(6*.001*args.dt/(dx*dx))))
    @jax.jit
    def transport_force(u, fields, dt):
        f = advect(fields, u, dt)
        concentration = jnp.where(source, jnp.minimum(f[..., 0]+dt*5, 1.), f[..., 0])
        temperature = jnp.where(source, f[..., 1]+(-jnp.expm1(-dt))*(1-f[..., 1]), f[..., 1])
        v = advect(u, u, dt).at[..., 1].add(-dt*(concentration-5*temperature))
        def diffuse(_, t):
            padded = jnp.pad(t, 1)
            lap = sum(jax.lax.slice_in_dim(padded, offset, offset+n, axis=axis)[
                tuple(slice(None) if k == axis else slice(1, -1) for k in range(3))]
                for axis in range(3) for offset in (0, 2))-6*t
            return jnp.where(mask, t+(.001*dt/diffusion_steps/dx**2)*lap, 0.)
        temperature = jax.lax.fori_loop(0, diffusion_steps, diffuse, temperature)
        return jnp.where(mask[..., None], v, 0.), jnp.where(mask[..., None], jnp.stack([concentration, temperature], -1), 0.)
    path = volume_path(args.name)
    restart_path = path.with_name('restart.npz')
    start_step = 0
    u = jnp.zeros(shape+(3,), jnp.float32)
    fields = jnp.zeros(shape+(2,), jnp.float32)
    frames, speeds, timings, frame_times = [], [], [], []
    target_time = args.steps*args.dt
    simulated_time = 0.
    if args.resume:
        io_started = time.perf_counter()
        saved, history = _resume_data(path)
        if saved['last_velocity'].shape != u.shape:
            raise ValueError('resume grid differs')
        for label, value in [('n_samples', args.paths), ('volume_samples', args.volume), ('pseudo_samples', args.pseudo)]:
            if int(saved[label]) != value:
                raise ValueError(f'resume {label} differs')
        for label, value in [('boundary_volume_samples', boundary_volume),
                             ('boundary_pseudo_samples', boundary_pseudo)]:
            if int(saved.get(label, 10)) != value:
                raise ValueError(f'resume {label} differs')
        if float(saved['dt']) != args.dt:
            raise ValueError('resume dt differs')
        for label, value in obstacle_metadata.items():
            if label not in saved or not np.array_equal(saved[label], value):
                raise ValueError(f'resume obstacle {label} differs')
        if 'seed' in saved and int(saved['seed']) != args.seed:
            raise ValueError('resume seed differs')
        u, fields = jnp.asarray(saved['last_velocity']), jnp.asarray(saved['last_fields'])
        frames, speeds, timings, frame_times = (list(history[key]) for key in
            ('smoke', 'speed', 'step_seconds', 'frame_times'))
        start_step = int(saved['completed_steps'])
        simulated_time = float(saved['simulated_time'])
        jax.block_until_ready((u, fields))
        io_seconds += time.perf_counter()-io_started
    read_seconds = io_seconds
    metadata = dict(lower=np.repeat(axes[0], 3), upper=np.repeat(axes[-1], 3),
        dt=args.dt, target_time=target_time, adaptive_dt=args.adaptive_dt,
        requested_steps=args.steps, n_samples=args.paths,
        volume_samples=args.volume, pseudo_samples=args.pseudo, seed=args.seed,
        boundary_volume_samples=boundary_volume, boundary_pseudo_samples=boundary_pseudo,
        near_paths=args.near_paths, near_band_cells=args.near_band_cells,
        query_batch=args.query_batch, near_query_batch=args.near_query_batch,
        **obstacle_metadata, advection='RK3',
        temperature_diffusion='explicit grid Laplacian, kappa=.001')

    def state_arrays():
        return dict(last_velocity=np.asarray(u), last_fields=np.asarray(fields),
            simulated_time=simulated_time, completed_steps=step,
            complete=simulated_time >= target_time-1e-7,
            **metadata)

    def save_restart():
        nonlocal io_seconds, restart_seconds, restart_writes
        io_started = time.perf_counter()
        _write_npz(restart_path, compressed=False, **state_arrays())
        elapsed = time.perf_counter()-io_started
        restart_seconds += elapsed
        io_seconds += elapsed
        restart_writes += 1

    step = start_step
    setup_seconds = time.perf_counter()-started-io_seconds
    compute_seconds = 0.
    while simulated_time < target_time-1e-7:
        t = time.perf_counter()
        dt = min(args.dt, target_time-simulated_time)
        if args.adaptive_dt:
            previous_max = float(jnp.max(jnp.linalg.norm(u, axis=-1)))
            dt = min(dt, args.target_cfl*dx/max(previous_max, 1e-20))
        key = jax.random.fold_in(jax.random.PRNGKey(args.seed), step)
        for attempt in range(4):
            intermediate, next_fields = transport_force(u, fields, jnp.asarray(dt, u.dtype))
            result = solver.project({'velocity': intermediate}, points[far.ravel()],
                key, n_samples=args.paths, max_active_states=args.active,
                steps_between_refill=4, jit_buck_size=32, n_compactions=2,
                refill_rounds=refill_rounds)
            flat_u = jnp.zeros((n**3, 3), jnp.float32).at[far.ravel()].set(result.velocity)
            if near.any():
                near_result = near_solver.project({'velocity': intermediate}, points[near.ravel()],
                    key, n_samples=args.near_paths, max_active_states=args.active,
                    steps_between_refill=4, jit_buck_size=32, n_compactions=2,
                    refill_rounds=refill_rounds)
                flat_u = flat_u.at[near.ravel()].set(near_result.velocity)
            next_u = flat_u.reshape(shape+(3,))
            speed = np.linalg.norm(np.asarray(next_u), axis=-1)
            cfl = float(speed.max()*dt/dx)
            before_energy = float(jnp.mean(jnp.sum(intermediate[fluid]**2, -1)))
            after_energy = float(jnp.mean(jnp.sum(next_u[fluid]**2, -1)))
            gain = after_energy/max(before_energy, 1e-20)
            if not np.isfinite(speed).all() or gain > 1.25:
                raise RuntimeError(f'projection failed at step {step+1}: energy gain={gain}')
            if cfl <= 2.:
                break
            if not args.adaptive_dt:
                raise RuntimeError(f'Courant limit at step {step+1}: CFL={cfl}')
            dt *= .5
            print(f'retry step {step+1}: reducing dt to {dt:.5g} (CFL={cfl:.3g})', flush=True)
        else:
            raise RuntimeError('Courant limit remains after four timestep retries')
        u, fields = next_u, next_fields
        simulated_time += dt
        frame_times.append(simulated_time)
        smoke = np.asarray(fields[..., 0])
        mass = smoke.sum()
        center_y = float((smoke*np.asarray(points[:, 1]).reshape(shape)).sum()/max(mass, 1e-20))
        frames.append(smoke.astype(np.float16)); speeds.append(speed.astype(np.float16))
        timings.append(time.perf_counter()-t)
        compute_seconds += timings[-1]
        print(f'step {step+1}: t={simulated_time:.3f}/{target_time:.3f}, '
              f'dt={dt:.4f}, {timings[-1]:.3f}s, max speed {speed.max():.4g}, '
              f'CFL {cfl:.3g}, energy gain {gain:.4f}, smoke y {center_y:.3f}', flush=True)
        step += 1
        if step % 5 == 0:
            save_restart()
    if not frame_times or frame_times[-1] < simulated_time-1e-7:
        # A restart can already be at the requested end time.
        frames.append(np.asarray(fields[..., 0], np.float16))
        speeds.append(np.asarray(jnp.linalg.norm(u, axis=-1), np.float16))
        frame_times.append(simulated_time)
    if step % 5 != 0 or step == start_step:
        save_restart()
    io_started = time.perf_counter()
    _write_npz(path, compressed=True, **state_arrays(), smoke=np.asarray(frames),
        speed=np.asarray(speeds), sdf=-field_values, first_frame_time=frame_times[0],
        frame_times=np.asarray(frame_times), step_seconds=np.asarray(timings))
    history_seconds = time.perf_counter()-io_started
    io_seconds += history_seconds
    total_seconds = time.perf_counter()-started
    report = {'start_step': start_step, 'completed_steps': step, 'new_steps': step-start_step,
        'compute_seconds': compute_seconds, 'io_seconds': io_seconds, 'read_seconds': read_seconds,
        'restart_write_seconds': restart_seconds, 'history_write_seconds': history_seconds,
        'restart_writes': restart_writes, 'history_writes': 1, 'setup_seconds': setup_seconds,
        'other_seconds': total_seconds-compute_seconds-io_seconds-setup_seconds,
        'total_seconds': total_seconds}
    path.with_name('timing.json').write_text(json.dumps(report, indent=2)+'\n')
    print('TIMING', json.dumps(report), flush=True)
    print(f'saved {path}', flush=True)


def main():
    """Buoyant smoke around an exterior sphere or torus; outer box is a pseudo-boundary."""
    import argparse
    from examples.fluids.vorticity import render3d
    from examples.fluids.pressure_prototype.boundary_sampling import ExteriorSphereBoundary, ExteriorTorusBoundary

    parser = argparse.ArgumentParser(description=main.__doc__)
    scenes = parser.add_subparsers(dest='scene', required=True)
    for scene in ('sphere', 'torus'):
        p = scenes.add_parser(scene)
        p.add_argument('--render', action='store_true', help='render saved volume instead of evolving')
        p.add_argument('--adaptive-dt', action=argparse.BooleanOptionalAction, default=scene == 'torus')
        p.add_argument('--target-cfl', type=float, default=1.5)
        p.add_argument('--resume', action='store_true')
        p.add_argument('--name', default=f'{scene}_buoyancy_smoke')
        p.add_argument('--grid', type=int, default=128 if scene == 'sphere' else 64)
        p.add_argument('--steps', type=int, default=100 if scene == 'sphere' else 134)
        p.add_argument('--dt', type=float, default=.1 if scene == 'sphere' else .075)
        p.add_argument('--paths', type=int, default=100000 if scene == 'sphere' else 65536)
        p.add_argument('--volume', type=int, default=100000 if scene == 'sphere' else 4096)
        p.add_argument('--pseudo', type=int, default=100000 if scene == 'sphere' else 4096)
        p.add_argument('--boundary-volume', type=int, default=10)
        p.add_argument('--boundary-pseudo', type=int, default=10)
        p.add_argument('--active', type=int, default=1024 if scene == 'sphere' else 2048)
        p.add_argument('--refill-rounds', type=int, default=4)
        p.add_argument('--query-batch', type=int, default=4 if scene == 'sphere' else 1024)
        p.add_argument('--near-paths', type=int, default=0 if scene == 'sphere' else 262144)
        p.add_argument('--near-band-cells', type=float, default=0. if scene == 'sphere' else 2.)
        p.add_argument('--near-query-batch', type=int, default=256)
        p.add_argument('--seed', type=int, default=0)
        p.add_argument('--render-frames', type=int, default=240 if scene == 'sphere' else 300)
    args = parser.parse_args()
    if (min(args.grid, args.steps, args.paths, args.volume, args.pseudo, args.active,
            args.boundary_volume, args.boundary_pseudo, args.query_batch,
            args.refill_rounds, args.near_query_batch, args.render_frames) < 1
            or args.volume % 2 or args.boundary_volume % 2 or args.seed < 0
            or not np.isfinite([args.dt, args.target_cfl, args.near_band_cells]).all()
            or args.dt <= 0 or args.target_cfl <= 0 or args.near_band_cells < 0 or args.near_paths < 0):
        parser.error('positive sizes/dt, even volume budgets and nonnegative seed/near-band required')
    center = np.array([0., -.75, 0.], np.float32)
    match args.scene:
        case 'sphere':
            geometry = lambda: ExteriorSphereBoundary(center, .14, [-1.5]*3, [1.5]*3)
            metadata = dict(sphere_center=center, sphere_radius=.14)
            camera, title = (3.5, .5, 5.), 'Smoke around an analytic sphere'
        case 'torus':
            jax.config.update('jax_enable_x64', True)
            geometry = lambda: ExteriorTorusBoundary(center, .30, .12, [-1.5]*3, [1.5]*3)
            metadata = dict(torus_center=center, torus_major=.30, torus_minor=.12)
            camera, title = (3., 4.2, 4.2), 'Smoke around an analytic torus'
    if args.render:
        render3d(args.name, title=title, render_frames=args.render_frames, camera=camera, rotate_camera=False)
    else:
        evolve(args, geometry=geometry(), obstacle_metadata=metadata)


if __name__ == '__main__':
    main()
