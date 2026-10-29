"""Passive smoke around an analytic torus with Dirichlet potential on torus and box.

python -m examples.fluids.smoke --grid 64 --steps 240 --continuous-smoke --advection rk4
Use --render with the same --name to render a saved simulation.
"""
import argparse

import jax
import jax.numpy as jnp
import numpy as np
from examples.fluids.vorticity import ring_vorticity, run3d
from examples.fluids.grid import sample_vector_grid, sample_vector_grid_nested
from examples.fluids.dirichlet import antithetic_source_samples, make_dirichlet_gradient_solve
from surfaces.implicit.lipschitz import sdfImplicit
from surfaces.implicit.primitives import bbox_sdf, torus_sdf
from examples.fluids.transport import make_transport

MAJOR, MINOR = .65, .18
COLUMN_RADIUS, COLUMN_BOTTOM, COLUMN_TOP = .85, -1.35, -.55
COLUMN_RING_HEIGHT, COLUMN_RING_CORE, COLUMN_CIRCULATION = -.75, .20, 1.5
_DIAGNOSTIC_NAMES = ('symmetry_rms', 'near_torus_normal_rms', 'aperture_flux',
                     'box_layer_speed_rms', 'red_mass', 'blue_mass')
_COLUMN_NAMES = _DIAGNOSTIC_NAMES + ('smoke_center_z', 'smoke_upward_speed',
    'smoke_above_torus', 'smoke_max_speed', 'smoke_dense_fraction')


def dirichlet_torus_geometry():
    """Negative in the fluid, positive in the torus or outside the box."""
    surface = sdfImplicit(fn=lambda p, x: jnp.maximum(
        -torus_sdf(p['torus'], x), bbox_sdf(p['box'], x)), dim=3)
    params = {'torus': dict(center=jnp.zeros(3), major=MAJOR, minor=MINOR),
              'box': dict(min_corner=-jnp.ones(3)*1.5, max_corner=jnp.ones(3)*1.5)}
    return surface, params


def column_vorticity(points: np.ndarray) -> np.ndarray:
    """One broad +z ring transports the filled column toward the obstacle."""
    return ring_vorticity(points, (0., 0., COLUMN_RING_HEIGHT), (0., 0., 1.),
                          COLUMN_RADIUS, COLUMN_RING_CORE, COLUMN_CIRCULATION)


def column_dye(points: np.ndarray) -> np.ndarray:
    """Unit-density cylinder with a one-voxel-scale smooth edge, below z=0."""
    radial = np.linalg.norm(points[:, :2], axis=-1)
    clearance = np.minimum.reduce((COLUMN_RADIUS-radial,
                                  points[:, 2]-COLUMN_BOTTOM, COLUMN_TOP-points[:, 2]))
    edge = np.clip(clearance/.06, 0., 1.)
    density = edge*edge*(3.-2.*edge)
    return np.stack((density, np.zeros_like(density)), axis=-1)


@jax.jit
def _column_diagnostic_values(velocity, dye, lower, spacing):
    density = dye.sum(axis=-1)
    z = lower[2]+jnp.arange(density.shape[2])*spacing[2]
    mass = jnp.maximum(density.sum(), 1e-12)
    values = jnp.stack((jnp.sum(density*z[None, None, :])/mass,
        jnp.sum(density*velocity[..., 2])/mass,
        jnp.sum(jnp.where(z[None, None, :] > MINOR, density, 0.))/mass,
        jnp.max(jnp.where(density > .1, jnp.linalg.norm(velocity, axis=-1), 0.)),
        jnp.count_nonzero(density > .5)/density.size))
    return jnp.concatenate((_diagnostic_values(velocity, dye, lower, spacing), values))


def column_diagnostics(velocity: np.ndarray, dye: jax.Array, lower: np.ndarray,
                       spacing: np.ndarray) -> dict[str, float]:
    values = np.asarray(_column_diagnostic_values(velocity, dye, lower, spacing))
    return dict(zip(_COLUMN_NAMES, map(float, values), strict=True))



def torus_probes(clearance: float = .03, n: int = 16) -> tuple[np.ndarray, np.ndarray]:
    phi, theta = np.meshgrid(np.arange(n) * 2 * np.pi / n,
                             np.arange(n) * 2 * np.pi / n, indexing="ij")
    normals = np.stack([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi),
                        np.sin(theta)], -1).reshape(-1, 3)
    centers = MAJOR * np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], -1).reshape(-1, 3)
    points = centers+MINOR*normals
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    return points+clearance*normals, normals


def aperture_probes(n: int = 64) -> np.ndarray:
    """Equal-area deterministic disk quadrature, avoiding the Dirichlet rim."""
    radius = (MAJOR - MINOR) * np.sqrt((np.arange(n) + .5) / n)
    phi = np.arange(n) * np.pi * (3 - np.sqrt(5))
    return np.stack([radius * np.cos(phi), radius * np.sin(phi), np.zeros(n)], -1)


@jax.jit
def _diagnostic_values(velocity, dye, lower, spacing):
    def sample(points):
        return jax.vmap(lambda x: sample_vector_grid(velocity, x, lower, spacing))(
            jnp.asarray(points, velocity.dtype))
    probes, normals = torus_probes()
    disk, disk_normals = aperture_probes(), np.tile([0., 0., 1.], (64, 1))
    reflected = velocity[:, :, ::-1] * jnp.array([1., 1., -1.], velocity.dtype)
    box = jnp.concatenate([velocity[0].ravel(), velocity[-1].ravel(), velocity[:, 0].ravel(),
        velocity[:, -1].ravel(), velocity[:, :, 0].ravel(), velocity[:, :, -1].ravel()])
    return jnp.stack((jnp.sqrt(jnp.mean((velocity-reflected)**2)),
        jnp.sqrt(jnp.mean(jnp.sum(sample(probes)*jnp.asarray(normals), axis=-1)**2)),
        jnp.pi*(MAJOR-MINOR)**2*jnp.sum(sample(disk)*jnp.asarray(disk_normals), axis=-1).mean(),
        jnp.sqrt(jnp.mean(box**2)), jnp.sum(dye[..., 0])*jnp.prod(jnp.asarray(spacing)),
        jnp.sum(dye[..., 1])*jnp.prod(jnp.asarray(spacing))))


@jax.jit
def prepare_local_source(params):
    """Break aligned grid pitches for local gathers; retain clamped values.

    Edge replication preserves the existing interpolant, including exterior
    queries. Only the local integral uses this layout; walker fields stay dense.
    Four extra rows/columns give cheaper generated address arithmetic on the
    profiled 64 cubed scene; this is a layout choice, not extra fluid cells.
    """
    field = jnp.pad(params['omega'], ((0, 0), (0, 4), (0, 4), (0, 0)), mode='edge')
    return dict(params, omega=field)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='torus_smoke', help='output directory name')
    parser.add_argument('--render', action='store_true', help='render the saved run without simulating')
    parser.add_argument('--render-frames', type=int, default=240)
    parser.add_argument('--frame-stride', type=int, default=1,
                        help='save every k-th visualization frame; always save the last frame')
    parser.add_argument('--grid', type=int, default=32)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--walks', type=int, default=128)
    parser.add_argument('--source-samples', type=int, default=1024)
    parser.add_argument('--max-steps', type=int, default=8)
    parser.add_argument('--max-active-states', type=int, default=65536)
    parser.add_argument('--dt', type=float, default=.02)
    parser.add_argument('--advection', choices=('euler', 'rk4'), default='euler')
    parser.add_argument('--continuous-smoke', action='store_true',
                        help='replenish the torus smoke cylinder after each advection; no velocity forcing')
    parser.add_argument('--flow-scale', type=float, default=1.,
                        help='scale the torus initial vortex-ring circulation and velocity')
    args = parser.parse_args()
    if not np.isfinite(args.flow_scale) or args.flow_scale <= 0:
        parser.error('--flow-scale must be positive and finite')
    if (args.grid < 4 or min(args.steps,
                            args.max_steps, args.max_active_states, args.render_frames, args.frame_stride) < 1
            or not np.isfinite(args.dt) or args.dt <= 0):
        parser.error('grid >= 4, positive counts, and a positive finite dt required')
    try:
        source_samples = antithetic_source_samples(args.source_samples, args.walks)
    except ValueError as error:
        parser.error(str(error))

    surface, params = dirichlet_torus_geometry()
    extent, dye0 = 1.5, column_dye

    def omega0(points):
        return args.flow_scale * column_vorticity(points)

    # The original expression is faster at 64 cubed; nested interpolation
    # avoids XLA's split at 128 cubed. Select once, before tracing either solve.
    local_sampler = sample_vector_grid_nested if args.grid >= 128 else sample_vector_grid

    def local_source(params, x):
        return local_sampler(params['omega'], x,
            (-extent+extent/args.grid,)*3, (2*extent/args.grid,)*3)

    def gradient_factory(*arguments, **options):
        return make_dirichlet_gradient_solve(*arguments,
            gradient_source_samples=source_samples,
            prepare_local_source=prepare_local_source, local_source=local_source, **options)

    print(f'Pointwise WoS: {args.walks} walks/site on the full source each step; '
          f'cap: {args.max_steps}', flush=True)
    metadata = dict(velocity_estimator='pointwise_wos', temporal_control_variate=False)
    metadata.update(scene='smoke-column', twisted_torus_center=np.zeros(3),
                    twisted_torus_major=MAJOR, twisted_torus_minor=MINOR,
                    twisted_torus_twist=0., twisted_torus_coordinate_order=np.array([0, 2, 1]),
                    render_view_up=np.array([0., 0., 1.]),
                    initial_flow_scale=args.flow_scale, source_samples=args.source_samples,
                    obstacle_boundary='dirichlet', outer_boundary='dirichlet')
    axis = -extent+(np.arange(args.grid)+.5)*(2*extent/args.grid)
    points = jnp.asarray(np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3),
                         jnp.float32)
    signed = jax.vmap(lambda x: surface.signed_value(params, x))
    fluid = signed(points) < 0
    _, diagnostic_transport = make_transport(points, (axis[0],)*3,
        (2*extent/args.grid,)*3, args.dt, method=args.advection)

    @jax.jit
    def crossings(velocity, dye):
        # Observe the selected method's departure chords; never modify transport.
        departures = diagnostic_transport(velocity, velocity, return_details=True).departures
        offset = departures-points
        crossed = jax.lax.fori_loop(0, 17,
            lambda i, hit: hit | (fluid & (signed(points+(i/16.)*offset) >= 0)),
            jnp.zeros(len(points), bool))
        occupied = dye.sum(-1).reshape(-1) > .01
        box_crossed = fluid & jnp.any(jnp.abs(points+offset) >= extent, axis=-1)
        return jnp.stack((crossed.sum(), (crossed & occupied).sum(), box_crossed.sum(),
                          (box_crossed & occupied).sum()))

    def measure(velocity, dye, lower, spacing):
        stats = column_diagnostics(velocity, dye, lower, spacing)
        all_paths, smoke_paths, box_paths, box_smoke = np.asarray(crossings(jnp.asarray(velocity), dye))
        stats.update(sampled_crossings=int(all_paths), occupied_path_crossings=int(smoke_paths),
                     box_crossings=int(box_paths), occupied_box_crossings=int(box_smoke))
        return stats

    run3d(args.name, surface, params,
          (-extent,)*3, (extent,)*3, args.grid, omega0,
          title='Dirichlet WoS: smoke around a torus',
          steps=args.steps, dt=args.dt,
          n_walks=args.walks, max_steps=args.max_steps, eps=1e-4,
          make_gradient_solve=gradient_factory,
          cell_centered=True, max_active_states=args.max_active_states,
          velocity_stride=1, advection_method=args.advection, dye0=dye0,
          continuous_dye=args.continuous_smoke, smoke=True,
          diagnostics_fn=measure, rotate_camera=False, camera=(3., -4., 1.8),
          n_compactions=7, compaction_mode='geometric', refill_rounds=16,
          obstacle_metadata=metadata, render_frames=args.render_frames,
          frame_stride=args.frame_stride,
          stages=lambda _: (not args.render, args.render))


if __name__ == '__main__':
    main()
