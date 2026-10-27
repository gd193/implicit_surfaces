"""Shared 2D Monte Carlo fluid (vorticity / stream function, after Rioux-Lavoie
et al. 2022) for the fluid examples: an inflow profile (Jet or BoundaryLayer)
enters at the left edge x = lo[0].

Each step solves -Lap psi = omega with the example's Dirichlet psi (constant on
every wall, so walls are streamlines) and gets u = (d psi/dy, -d psi/dx) at
every grid point from WoSGradientIntegrator. omega lives on the grid, enters
the walk as the source term through params (no retrace per step), and is
advected semi-Lagrangian (RK2 backtrace). The inlet column keeps the inflow
vorticity; grid points outside the domain stay at 0.

no_slip=True adds a wall vorticity source (Chorin-style vortex sheet): in a
band of `wall_cells` cells along the walls, omega += -alpha u_t / band each
step, with u_t the smoothed tangential slip. wall_fn(params, x) picks the
no-slip walls (default: the whole domain boundary; leave out inlet/outlet). It cancels slip, so walls shed
vorticity, which the inviscid model alone never does.

Outputs in assets/renders/wos/<name>/: volume.npz (omega and norm(u) per step),
bc.png, frame.png and a 20 s fluid.mp4 (600 steps at 30 fps).
"""
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.ndimage import map_coordinates

from examples.wos.common import load_volume, out_dir, plot_bc, volume_path, volume_stages
from sphereWalker.integrator import WoSGradientIntegrator
from surfaces.implicit.lipschitz import lipschitzImplicit


class Jet(NamedTuple):
    """Inflow u_x(y) = U + J sech^2(y / delta) for |y| <= halfwidth."""
    U: float = .3
    J: float = 1.2
    delta: float = .07
    halfwidth: float = 1.

    def psi(self, y):
        """Stream function of the inflow, constant beyond the inlet walls."""
        y = jnp.clip(y, -self.halfwidth, self.halfwidth)
        return self.U * y + self.J * self.delta * jnp.tanh(y / self.delta)

    def omega(self, y):
        s = 1. / jnp.cosh(y / self.delta)
        return jnp.where(jnp.abs(y) <= self.halfwidth,
                         2. * self.J / self.delta * s * s * jnp.tanh(y / self.delta), 0.)


    label = "inlet: jet u_x = U + J sech^2(y/d)"


class BoundaryLayer(NamedTuple):
    """Inflow u_x(y) = U tanh((halfwidth - |y - center|) / delta): 0 at the
    inlet walls y = center +- halfwidth, U in the core, so the wall layers
    carry vorticity of opposite sign."""
    U: float = 1.
    delta: float = .08
    halfwidth: float = 1.
    center: float = 0.

    def psi(self, y):
        """Stream function of the inflow, constant beyond the inlet walls."""
        y = jnp.clip(y - self.center, -self.halfwidth, self.halfwidth)
        cosh = lambda t: jnp.log(jnp.cosh(t))
        return jnp.sign(y) * self.U * self.delta * (cosh(self.halfwidth / self.delta)
                                                    - cosh((self.halfwidth - jnp.abs(y)) / self.delta))

    def omega(self, y):
        y = y - self.center
        s = 1. / jnp.cosh((self.halfwidth - jnp.abs(y)) / self.delta)
        return jnp.where(jnp.abs(y) <= self.halfwidth, jnp.sign(y) * self.U / self.delta * s * s, 0.)


    label = "inlet: wall layers u_x = U tanh((h - |y|)/d)"


def make_transport(points, solid, inlet, lo, hi, nx, ny, dt, *, fluid_fn=None, inflow_only=False):
    """(sample, advect) on an (ny, nx) grid. sample: bilinear lookup clamped at
    the edges. advect: semi-Lagrangian step with an RK2 backtrace; the inlet
    column is reset to `inlet` (it lies on the domain edge, so after masking).
    With inflow_only=True, prescribe inlet data only where the left velocity
    points into the domain. Outflow and stagnation retain the transported field.
    Optional fluid_fn(x) clips a segment with a solid endpoint to its last fluid
    point by bisection. This is a grid transport guard, not a certified ray query:
    a thin obstacle between two fluid endpoints can still be missed.
    """
    def sample(grid, x):
        col = (x[0] - lo[0]) / (hi[0] - lo[0]) * (nx - 1)
        row = (x[1] - lo[1]) / (hi[1] - lo[1]) * (ny - 1)
        return map_coordinates(grid, [row, col], order=1, mode="nearest")

    @jax.jit
    def advect(omega, vel):
        def clip_segment(start, end):
            if fluid_fn is None:
                return end

            def bisect(_, bracket):
                low, high = bracket
                middle = .5*(low+high)
                inside = fluid_fn(start + middle*(end-start))
                return jnp.where(inside, middle, low), jnp.where(inside, high, middle)

            low, _ = jax.lax.fori_loop(0, 12, bisect, (0., 1.))
            fraction = jnp.where(fluid_fn(end), 1., low)
            return start + fraction*(end-start)

        def back(x):
            u = jnp.stack([sample(vel[..., 0], x), sample(vel[..., 1], x)])
            raw_mid = x - .5 * dt * u
            mid = clip_segment(x, raw_mid)
            foot = x - dt * jnp.stack([sample(vel[..., 0], mid), sample(vel[..., 1], mid)])
            foot = clip_segment(mid, foot)
            if fluid_fn is not None:
                foot = jnp.where(fluid_fn(raw_mid), foot, mid)
            return foot
        new = jax.vmap(lambda x: sample(omega, back(x)))(points).reshape(ny, nx)
        new = jnp.where(solid, 0., new)
        incoming = (vel[:, 0, 0] > 0) if inflow_only else jnp.ones(ny, dtype=bool)
        left = jnp.where(incoming, inlet, new[:, 0])
        return new.at[:, 0].set(jnp.where(solid[:, 0], 0., left))

    return sample, advect


def open_boundaries(vel, inlet_velocity, solid):
    """Fill the unsolved inlet and outlet columns of an (ny, nx, 2) velocity
    grid: the known inflow at x = lo[0], zero-gradient outflow at x = hi[0]."""
    vel = vel.at[:, 0].set(inlet_velocity)
    return vel.at[:, -1].set(jnp.where(solid[:, -1, None], 0., vel[:, -2]))


def run(name, domain, geometry, psi_boundary, inflow, lo, hi, *, nx, ny, title, info,
        steps=600, dt=.012, n_walks=64, eps=1e-3, no_slip=False, wall_fn=None, wall_cells=3,
        alpha=.5):
    """domain(params, x): 1-Lipschitz field, negative in the fluid.
    psi_boundary(x): Dirichlet psi, use inflow.psi(y) on the inlet and outlet."""
    xs, ys = np.linspace(lo[0], hi[0], nx), np.linspace(lo[1], hi[1], ny)
    X, Y = np.meshgrid(xs, ys)  # image[row=y, col=x]
    points = jnp.asarray(np.stack([X.ravel(), Y.ravel()], 1), jnp.float32)
    sdf = np.asarray(jax.vmap(lambda p: domain(geometry, p))(points)).reshape(ny, nx)
    # Box faces (inlet, outlet) give sdf ~ +1e-6 from safe_norm: still fluid.
    fluid = sdf <= 1e-4
    solid = jnp.asarray(~fluid)
    cell = (hi[0] - lo[0]) / (nx - 1)
    band = wall_cells * cell
    wall = domain if wall_fn is None else wall_fn
    wall_sdf = np.asarray(jax.vmap(lambda p: wall(geometry, p))(points)).reshape(ny, nx)
    in_band = fluid & (wall_sdf > -band)
    # Tangent t = (n_y, -n_x) of the inward normal n = -grad sdf: walls below
    # the fluid get omega < 0 for flow in +x, as in a no-slip boundary layer.
    normal = -np.asarray(jax.vmap(jax.grad(lambda p: wall(geometry, p)))(points)).reshape(ny, nx, 2)
    tangent = np.stack([normal[..., 1], -normal[..., 0]], -1)
    solve_mask = sdf.ravel() < -2 * eps  # gradient walks must start off the boundary
    inlet = inflow.omega(jnp.asarray(ys))
    # On x = lo[0] psi = inflow.psi(y), so u = d psi/dy; v = -d psi/dx is taken as 0.
    inlet_velocity = jnp.stack([jax.vmap(jax.grad(inflow.psi))(jnp.asarray(ys)),
                                jnp.zeros(ny)], -1)
    sample, advect = make_transport(points, solid, inlet, lo, hi, nx, ny, dt)

    def simulate():
        surface = lipschitzImplicit(lipschitz_bound=1., fn=domain, level_set=0., dim=2)
        solver = WoSGradientIntegrator(surface.empty_sphere, psi_boundary,
                                       lambda params, x: sample(params["omega"], x), dim=2,
                                       eps=eps, max_steps=4000, source_takes_params=True,
                                       control_fn=lambda x: inflow.psi(x[1]))
        pts = points[solve_mask]
        options = dict(init_trust_region_size=1., n_samples=n_walks, steps_between_refill=16,
                       max_active_states=min(pts.shape[0] * n_walks, 1 << 20), jit_buck_size=1024)
        omega = jnp.where(solid, 0., jnp.broadcast_to(inlet[:, None], (ny, nx))).at[:, 0].set(inlet)
        omegas, speeds = [], []
        start = time.perf_counter()
        for step in range(steps):
            details = solver.integrate_batch({**geometry, "omega": omega}, pts,
                                             jax.random.PRNGKey(step), return_details=True,
                                             **options)
            # Rare walks over max_steps are dropped: average the completed ones.
            ok = jnp.maximum(details.successful_samples, 1)[:, None, None]
            grad = (details.contribution_sum / ok)[:, 0]
            vel = np.zeros((ny * nx, 2), np.float32)
            vel[solve_mask] = np.stack([grad[:, 1], -grad[:, 0]], 1)
            vel = open_boundaries(jnp.asarray(vel.reshape(ny, nx, 2)), inlet_velocity, solid)
            omegas.append(np.asarray(omega, np.float16))
            speeds.append(np.asarray(jnp.linalg.norm(vel, axis=-1), np.float16))
            omega = advect(omega, vel)
            if no_slip:
                from scipy.ndimage import gaussian_filter
                slip = gaussian_filter(np.sum(np.asarray(vel) * tangent, -1), 2.)
                omega = omega - jnp.asarray(np.where(in_band, alpha * slip / band, 0.), omega.dtype)
            if step % 50 == 0:
                print(f"step {step}/{steps}  {time.perf_counter() - start:.0f}s  "
                      f"failed walks {int(jnp.sum(n_walks - details.successful_samples))}")
        seconds = time.perf_counter() - start
        np.savez_compressed(volume_path(name), omega=np.stack(omegas), speed=np.stack(speeds),
                            seconds=seconds, n_walks=n_walks, dt=dt)
        print(f"saved {volume_path(name)}  ({seconds:.0f}s)")

    def render():
        import imageio.v2 as imageio
        import matplotlib.pyplot as plt
        from scipy.ndimage import gaussian_filter

        data = load_volume(name)
        omega = data["omega"].astype(np.float32)
        # Per-point gradient noise grows like 1/distance near walls; smooth for display.
        speed = gaussian_filter(data["speed"].astype(np.float32), (0, 1.5, 1.5))
        mask = ~fluid
        # Fixed over the video: 99th percentile of the fluid values.
        lim = float(np.percentile(np.abs(omega[:, fluid]), 99))
        top = float(np.percentile(speed[:, fluid], 99))
        extent = (lo[0], hi[0], lo[1], hi[1])
        panel_h = 7.2 * (hi[1] - lo[1]) / (hi[0] - lo[0]) + .6
        fig, axes = plt.subplots(2, 1, figsize=(9, 2 * panel_h + .6), constrained_layout=True)
        images = []
        for ax, field, cmap, clim, label in (
                (axes[0], omega, "RdBu_r", (-lim, lim), "vorticity"),
                (axes[1], speed, "magma", (0., top), "speed norm(u), smoothed 1.5 cells")):
            img = ax.imshow(np.ma.masked_where(mask, field[0]), origin="lower", extent=extent,
                            cmap=cmap, vmin=clim[0], vmax=clim[1])
            ax.contour(X, Y, sdf, levels=[0.], colors="k", linewidths=1.)
            ax.set_title(label)
            ax.set_aspect("equal")
            fig.colorbar(img, ax=ax, shrink=.8)
            images.append(img)
        suptitle = fig.suptitle("")
        path = out_dir(name)

        def draw(i):
            for img, field in zip(images, (omega, speed)):
                img.set_data(np.ma.masked_where(mask, field[i]))
            suptitle.set_text(f"Monte Carlo fluid: {title}, {int(data['n_walks'])} walks/pt/step, "
                              f"t = {i * float(data['dt']):.2f}")

        draw(len(omega) * 2 // 3)
        fig.savefig(path / "frame.png", dpi=120)
        with imageio.get_writer(path / "fluid.mp4", fps=30, macro_block_size=1) as writer:
            for i in range(len(omega)):
                draw(i)
                fig.canvas.draw()
                rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
                writer.append_data(rgb[:rgb.shape[0] // 2 * 2, :rgb.shape[1] // 2 * 2])  # yuv420p
        plt.close(fig)
        print(f"saved {path / 'frame.png'} and fluid.mp4 ({len(omega)} frames)")

        plot_bc(name, domain, psi_boundary, extent, params=geometry, cmap="coolwarm",
                info=["Euler, vorticity form: omega_t + u . grad omega = 0",
                      "-Lap psi = omega, u = (d psi/dy, -d psi/dx) by gradient WoS",
                      *info, inflow.label])

    solve_now, render_now = volume_stages(name)
    if solve_now:
        simulate()
    if render_now:
        render()


def main():
    from surfaces.implicit.primitives import bbox_sdf, star2d_sdf, subtraction

    LO, HI = (-1.5, -1.), (2.5, 1.)
    jet = Jet(U=.3, J=1.2, delta=.07, halfwidth=1.)
    geometry = {"box": {"min_corner": jnp.array(LO), "max_corner": jnp.array(HI)},
                "star": {"center": jnp.array([-.4, 0.]), "radius": jnp.asarray(.35),
                         "m": jnp.asarray(3.)}}


    def domain(params, x):
        return subtraction(bbox_sdf(params["box"], x), star2d_sdf(params["star"], x))


    def psi_boundary(x):
        """0 on the star, the inflow stream function on the channel walls."""
        on_star = jnp.abs(star2d_sdf(geometry["star"], x)) < jnp.abs(bbox_sdf(geometry["box"], x))
        return jnp.where(on_star, 0., jet.psi(x[1]))


    run("fluid_2d", domain, geometry, psi_boundary, jet, LO, HI, nx=320, ny=160, title="jet past a star",
        info=["Dirichlet psi = U y + J d tanh(y/d) on the channel, 0 on the star"])


if __name__ == '__main__':
    main()
