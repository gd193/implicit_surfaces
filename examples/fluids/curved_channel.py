"""Driven cavity mixing through Dirichlet circle-wave walls and Neumann ends.

  python -m examples.fluids.curved_channel [--solve | --render]

The top/bottom wave streamfunctions are +Q/2 and -Q/2. Their difference
sets throughflow; the bounding box has homogeneous Neumann streamfunction.
The baseline is a cached harmonic WoSt gradient solve. Subsequent solves use
zero wall data and the evolving vorticity source. Alternating localized body
forcing drives vortices at the cavity mouths; dye enters only from the inlet.
No prescribed inlet velocity,
no viscosity, and no approximate no-slip wall-vorticity source are used.
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/wost-circle-wave-matplotlib")

from examples.wos.common import out_dir
from examples.fluids.channel import make_transport

# isort: split

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

from sphereWalker.star_integrator import WoStGradientIntegrator, WoStIntegrator
from surfaces.implicit.lipschitz import lipschitzImplicit, sdfImplicit
from surfaces.implicit.primitives import bbox_sdf, circle_wave2d_sdf

RADIUS, THICKNESS, TOP = .2, .8, .55
OPEN = np.pi * 5. / 6. * THICKNESS
PERIOD = 4. * RADIUS * np.sin(OPEN)
TIP = TOP + 2. * RADIUS * np.cos(OPEN) - RADIUS
INLET = (-(TOP + RADIUS), TIP)
LO, HI = (0., -.8), (6 * PERIOD, .8)
geometry = {"box": {"min_corner": jnp.array(LO), "max_corner": jnp.array(HI)},
            "wave": {"center": jnp.array([0., TOP]), "thickness": jnp.asarray(THICKNESS),
                     "radius": jnp.asarray(RADIUS)}}


def top_wall(params, x):
    return circle_wave2d_sdf(params["wave"], x)


def bottom_wall(params, x):
    return circle_wave2d_sdf(params["wave"], jnp.stack([x[0] - PERIOD / 2, -x[1]]))


def walls(params, x):
    return jnp.maximum(top_wall(params, x), bottom_wall(params, x))


def domain(params, x):
    return jnp.maximum(bbox_sdf(params["box"], x), walls(params, x))


def psi_boundary(x, flux=.8):
    """Only wave walls terminate walks; box edges never supply Dirichlet data."""
    top = jnp.abs(top_wall(geometry, x))
    bottom = jnp.abs(bottom_wall(geometry, x))
    return jnp.where(top < bottom, flux / 2, -flux / 2)


def pilot_psi(x, flux):
    """Cheap pointwise control variate, not an assumed harmonic solution."""
    top = jnp.maximum(-top_wall(geometry, x), 0)
    bottom = jnp.maximum(-bottom_wall(geometry, x), 0)
    return flux * (bottom / jnp.maximum(top + bottom, 1e-8) - .5)


def cavity_forcing_modes(X: np.ndarray, Y: np.ndarray, fluid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spatial sine/cosine modes of an externally applied vorticity forcing.

    Gaussian rotors straddle the cavity mouths. Neighboring rotors reverse at
    different times, disrupting closed steady streamlines. This is curl(body
    force), not wall motion or a modification of the solved velocity.
    """
    cosine, sine = np.zeros_like(X), np.zeros_like(X)
    for side, centers in ((1, (np.arange(6)+.5)*PERIOD), (-1, np.arange(1, 6)*PERIOD)):
        for index, center in enumerate(centers):
            core = np.exp(-((X-center)**2 + (Y-side*.32)**2)/(2*.13**2))
            phase = index*np.pi*.65 + (0 if side == 1 else np.pi)
            cosine += np.cos(phase)*core
            sine += np.sin(phase)*core
    # No forcing in solids or on the open-end columns.
    cosine, sine = cosine*fluid, sine*fluid
    cosine[:, [0, -1]], sine[:, [0, -1]] = 0, 0
    return cosine, sine


def checked_solve(solver, params, points, walks, seed):
    details = solver.integrate_batch(
        params, points, jax.random.PRNGKey(seed), 1., walks,
        neumann_params=geometry["box"], return_details=True,
        # Walks average ~12 steps here; refill blocks of ~half that measured fastest.
        steps_between_refill=8, max_active_states=65536, jit_buck_size=256,
    )
    counts = np.asarray(details.termination_counts).sum(0)
    if not bool(jnp.all(details.complete)):
        raise RuntimeError(f"Incomplete walks: {counts}")
    value = np.asarray(details.contribution_sum[:, 0] / walks)
    if not np.isfinite(value).all():
        raise RuntimeError("Nonfinite WoSt result")
    return value, counts


def checked_gradient(solver, params, points, walks, seed):
    grad, counts = checked_solve(solver, params, points, walks, seed)
    return np.stack([grad[:, 1], -grad[:, 0]], -1), counts


def solve(args, output):
    xs, ys = np.linspace(LO[0], HI[0], args.nx), np.linspace(LO[1], HI[1], args.ny)
    X, Y = np.meshgrid(xs, ys)
    points = jnp.asarray(np.stack([X.ravel(), Y.ravel()], -1), jnp.float32)
    wall_sdf = np.asarray(jax.jit(jax.vmap(lambda x: walls(geometry, x)))(points)).reshape(X.shape)
    fluid = wall_sdf < 0
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    cell = max(dx, dy)
    # Gradient starts need an interior ball. Extend one grid cell to curved walls.
    solve_mask = wall_sdf < -cell
    solve_mask[:, [0, -1]] = False
    if not solve_mask.any():
        raise ValueError("Grid has no interior solve points")
    near = distance_transform_edt(~solve_mask, return_distances=False, return_indices=True)
    normal = np.asarray(jax.jit(jax.vmap(jax.grad(lambda x: walls(geometry, x))))(points))
    normal = normal.reshape(args.ny, args.nx, 2).copy()
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-12)
    wall_weight = np.clip(1 + wall_sdf / cell, 0, 1)[..., None]

    def reconstruct(values):
        velocity = np.zeros((args.ny, args.nx, 2), np.float32)
        velocity[solve_mask] = values
        velocity = velocity[near[0], near[1]]
        # Tangent extension in the unresolved wall band; tangential slip is retained.
        velocity -= wall_weight * np.sum(velocity * normal, axis=-1, keepdims=True) * normal
        # Neumann psi at both ends gives zero transverse velocity. u_x is extrapolated.
        velocity[:, 0] = velocity[:, 1]
        velocity[:, -1] = velocity[:, -2]
        velocity[:, [0, -1], 1] = 0
        velocity[~fluid] = 0
        return velocity

    domain_surface = lipschitzImplicit(lipschitz_bound=1., fn=walls, level_set=0., dim=2)
    neumann = sdfImplicit(fn=bbox_sdf, dim=2)
    positions = points[np.asarray(solve_mask).ravel()]
    # First balls near the Neumann ends are capped by the end distance, so their
    # (d/R)-weighted offset variance is largest there. Pilot only those columns.
    columns = np.arange(args.nx)
    pilot_mask = solve_mask & ((columns <= args.pilot_columns)
                               | (columns >= args.nx - 1 - args.pilot_columns))[None]
    pilot_positions = points[pilot_mask.ravel()]
    zero = lambda x: jnp.asarray(0., x.dtype)
    baseline_solver = WoStGradientIntegrator(
        domain_surface.empty_sphere, lambda x: psi_boundary(x, args.flux), zero, 2,
        neumann_surface=neumann, control_fn=lambda x: pilot_psi(x, args.flux),
        eps=args.eps, max_steps=4000,
    )
    start = time.perf_counter()
    dynamic_walks = (args.steps*(len(positions)*args.walks + len(pilot_positions)*args.pilot_walks)
                     if args.forcing_strength else 0)
    walk_budget = len(positions)*args.baseline_walks + dynamic_walks
    print(f"Backend: {jax.default_backend()} ({jax.devices()[0]}); "
          f"requested walk budget: {walk_budget:,}. Full defaults are intended for GPU execution.", flush=True)
    baseline_values, base_counts = checked_gradient(
        baseline_solver, geometry, positions, args.baseline_walks, args.seed)
    baseline = reconstruct(baseline_values)
    fluxes = np.trapezoid(np.where(fluid, baseline[..., 0], 0), ys, axis=0)
    print(f"Baseline: flux mean={fluxes.mean():.4f}, target={args.flux:.4f}, "
          f"range=[{fluxes.min():.4f}, {fluxes.max():.4f}], "
          f"{time.perf_counter()-start:.1f}s", flush=True)

    solid = jnp.asarray(~fluid)
    fluid_fn = lambda x: walls(geometry, x) < 0
    sample, advect = make_transport(points, solid, jnp.zeros(args.ny), LO, HI,
                                    args.nx, args.ny, args.dt, fluid_fn=fluid_fn, inflow_only=True)
    source = lambda params, x: sample(params["omega"], x)
    solver = WoStGradientIntegrator(
        domain_surface.empty_sphere, zero, source, 2, neumann_surface=neumann,
        source_takes_params=True, eps=args.eps, max_steps=4000,
        control_fn=lambda params, x: sample(params["pilot"], x), control_takes_params=True,
    )
    pilot_solver = WoStIntegrator(
        domain_surface.empty_sphere, zero, source, 2, neumann_surface=neumann,
        source_takes_params=True, eps=args.eps, max_steps=4000,
    )
    pilot = np.zeros((args.ny, args.nx), np.float32)
    omega = jnp.zeros((args.ny, args.nx), jnp.float32)
    dye = jnp.zeros((2, args.ny, args.nx), jnp.float32)
    force_cos, force_sin = (jnp.asarray(mode, jnp.float32) for mode in cavity_forcing_modes(X, Y, fluid))
    incoming_flux = np.cumsum(np.maximum(baseline[:, 0, 0], 0) * dy)
    if incoming_flux[-1] <= 0:
        raise RuntimeError("No positive inflow for the requested wall flux")
    inlet_y = np.interp([.30, .65], incoming_flux/incoming_flux[-1], ys)
    dyes_in = [jnp.asarray(np.exp(-.5*((ys-y)/.05)**2) * fluid[:, 0], jnp.float32) for y in inlet_y]
    dye = dye.at[:, :, 0].set(jnp.stack(dyes_in)*(baseline[:, 0, 0] > 0)[None])
    dye_advects = [make_transport(points, solid, inlet, LO, HI, args.nx, args.ny,
                                  args.dt, fluid_fn=fluid_fn, inflow_only=True)[1] for inlet in dyes_in]
    counts = base_counts.astype(np.int64)
    frames_omega, frames_dye, frames_velocity, times, cfles, circulation = [], [], [], [], [], []
    simulation_start = time.perf_counter()
    for step in range(args.steps+1):
        if step == 0 or args.forcing_strength == 0:
            # Zero initial/source vorticity gives exactly zero perturbation.
            values, count = np.zeros_like(baseline_values), np.zeros_like(base_counts)
        else:
            pilot_count = np.zeros_like(base_counts)
            if args.pilot_walks and len(pilot_positions):
                # Independent stream from the same omega; fixed before the gradient
                # solve, so the control stays zero-mean. Elsewhere the pilot is 0.
                scalar, pilot_count = checked_solve(
                    pilot_solver, {**geometry, "omega": omega}, pilot_positions,
                    args.pilot_walks, args.seed+args.steps+1+step)
                pilot[pilot_mask] = scalar
            values, count = checked_gradient(
                solver, {**geometry, "omega": omega, "pilot": jnp.asarray(pilot)},
                positions, args.walks, args.seed+1+step)
            count = count + pilot_count
        velocity = reconstruct(baseline_values + values)
        if step % args.save_every == 0 or step == args.steps:
            frames_omega.append(np.asarray(omega))
            frames_dye.append(np.asarray(dye))
            frames_velocity.append(velocity)
            times.append(step*args.dt)
        cfles.append(float(np.max(np.abs(velocity[..., 0])*args.dt/dx + np.abs(velocity[..., 1])*args.dt/dy)))
        circulation.append(float(np.sum(np.asarray(omega))*dx*dy))
        counts += count
        if step % 20 == 0:
            print(f"step {step}/{args.steps}, t={step*args.dt:.2f}, "
                  f"peak |omega|={float(jnp.max(jnp.abs(omega))):.3f}, "
                  f"CFL={cfles[-1]:.2f}, elapsed={time.perf_counter()-simulation_start:.1f}s", flush=True)
        if step < args.steps:
            omega = advect(omega, jnp.asarray(velocity))
            phase = 2*np.pi*(step+.5)*args.dt/args.forcing_period
            omega += args.dt*args.forcing_strength*(np.cos(phase)*force_cos - np.sin(phase)*force_sin)
            dye = jnp.stack([fn(field, jnp.asarray(velocity)) for fn, field in zip(dye_advects, dye)])
            # Prescribe clean fluid on any right-edge backflow; otherwise let material leave.
            incoming = jnp.asarray(velocity[:, -1, 0] < 0)
            omega = omega.at[:, -1].set(jnp.where(incoming, 0., omega[:, -1]))
            dye = dye.at[:, :, -1].set(jnp.where(incoming[None], 0., dye[:, :, -1]))
            if not bool(jnp.all(jnp.isfinite(omega))) or not bool(jnp.all(jnp.isfinite(dye))):
                raise RuntimeError("Nonfinite transported field")
    np.savez_compressed(output / "volume.npz", omega=frames_omega, dye=frames_dye,
                        velocity=frames_velocity, times=times, xs=xs, ys=ys, sdf=wall_sdf,
                        baseline=baseline, baseline_flux=fluxes, circulation=circulation, cfl=cfles)
    deep_cavity = fluid & (np.abs(Y) > .4) & (X > PERIOD/4) & (X < HI[0]-PERIOD/4)
    total_dye = np.asarray(frames_dye).sum(axis=1)
    stats = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
             "backend": jax.default_backend(), "device": str(jax.devices()[0]),
             "dye_inlet_y": inlet_y.tolist(), "fluid_grid_points": int(fluid.sum()), "gradient_points": int(solve_mask.sum()),
             "pilot_points": int(pilot_mask.sum()),
             "termination_counts": counts.tolist(),
             "baseline_flux_mean": float(fluxes.mean()),
             "baseline_flux_relative_rmse": float(np.sqrt(np.mean((fluxes-args.flux)**2))/args.flux),
             "max_cfl": max(cfles), "seconds": time.perf_counter()-start,
             "max_solid_vorticity": float(np.max(np.abs(np.asarray(frames_omega)[:, ~fluid]))),
             "max_solid_dye": float(np.max(np.abs(np.asarray(frames_dye)[:, :, ~fluid]))),
             "final_deep_cavity_mean_dye": float(total_dye[-1, deep_cavity].mean()),
             "final_deep_cavity_fraction_above_0.05": float(np.mean(total_dye[-1, deep_cavity] > .05)),
             "approximations": "Externally forced inviscid slip-wall model; cached noisy harmonic background; "
             "one-cell velocity extension/projection; bilinear RK2 transport with clipped wall "
             "backtraces; independent scalar pilot in end columns; numerical diffusion and "
             "Monte Carlo divergence noise remain."}
    (output / "metrics.json").write_text(json.dumps(stats, indent=2)+"\n")
    print(json.dumps(stats, indent=2), flush=True)


def dye_rgb(dye):
    """Cyan and coral passive tracers on a dark background; density mapped consistently."""
    a, b = 1-np.exp(-3*dye[0]), 1-np.exp(-3*dye[1])
    background = np.array([.035, .055, .09])
    cyan, coral = np.array([.10, .85, .95]), np.array([1., .30, .45])
    rgb = background + a[..., None]*(cyan-background) + b[..., None]*(coral-background)
    return np.clip(rgb, 0, 1)


def render(output):
    import imageio.v2 as imageio

    data = np.load(output / "volume.npz")
    config = json.loads((output / "metrics.json").read_text())["config"]
    driven = config.get("forcing_strength", 0) > 0
    xs, ys, sdf = data["xs"], data["ys"], data["sdf"]
    omega, dye, times = data["omega"], data["dye"], data["times"]
    X, Y = np.meshgrid(xs, ys)
    extent = (xs[0], xs[-1], ys[0], ys[-1])
    solid = sdf >= 0
    lim = max(float(np.max(np.abs(omega))), 1e-6)

    def decorate(ax):
        ax.contourf(X, Y, sdf, levels=[0., max(float(sdf.max())+1., 1.)], colors=["#d7dbe0"])
        ax.contour(X, Y, sdf, levels=[0.], colors=["#687689"], linewidths=.7)
        for x in (xs[0], xs[-1]):
            ax.axvline(x, color="#efae45", lw=3)
        ax.set(xlim=(xs[0], xs[-1]), ylim=(ys[0], ys[-1]), aspect="equal", xlabel="x", ylabel="y")

    selected = np.linspace(0, len(times)-1, 4).astype(int)
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), constrained_layout=True)
    for ax, index in zip(axes, selected):
        ax.imshow(dye_rgb(dye[index]), origin="lower", extent=extent, interpolation="bilinear")
        decorate(ax)
        ax.set_title(f"t = {times[index]:.2f}", loc="left")
    label = "Driven cavity mixing" if driven else "Unforced throughflow"
    fig.suptitle(f"{label} · Dirichlet wave walls · Neumann box ends (gold)")
    fig.savefig(output / "snapshots.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    image = axes[0].imshow(dye_rgb(dye[0]), origin="lower", extent=extent, interpolation="bilinear")
    vort = axes[1].imshow(np.ma.masked_where(solid, omega[0]), origin="lower", extent=extent,
                          cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="bilinear")
    for ax in axes:
        decorate(ax)
    axes[0].set_title("Cyan / coral passive dye", loc="left")
    forcing_label = "alternating body forcing at cavity mouths" if driven else "unforced control"
    axes[1].set_title(f"Vorticity · {forcing_label}", loc="left")
    fig.colorbar(vort, ax=axes[1], shrink=.8, label="vorticity")
    title = fig.suptitle("")
    gif = []
    with imageio.get_writer(output / "fluid.mp4", fps=24, macro_block_size=1) as writer:
        for i, t in enumerate(times):
            image.set_data(dye_rgb(dye[i]))
            vort.set_data(np.ma.masked_where(solid, omega[i]))
            title.set_text(f"Circle-wave channel · t={t:.2f} · Neumann ends / Dirichlet walls")
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            frame = frame[:frame.shape[0]//2*2, :frame.shape[1]//2*2]
            writer.append_data(frame)
            if i % 2 == 0:
                gif.append(frame[::2, ::2])
    imageio.mimsave(output / "fluid.gif", gif, duration=83, loop=0)
    plt.close(fig)

    # Steady throughflow diagnostic: both end profiles are computed, not prescribed.
    base = data["baseline"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True,
                             gridspec_kw={"width_ratios": [2.5, 1]})
    speed = np.ma.masked_where(solid, np.linalg.norm(base, axis=-1))
    im = axes[0].imshow(speed, origin="lower", extent=extent, cmap="viridis")
    axes[0].streamplot(xs, ys, np.ma.masked_where(solid, base[..., 0]),
                       np.ma.masked_where(solid, base[..., 1]), color="white", density=1., linewidth=.5)
    decorate(axes[0])
    axes[0].set_title("Cached harmonic throughflow · WoSt gradients")
    fig.colorbar(im, ax=axes[0], shrink=.8, label="speed")
    axes[1].plot(xs, data["baseline_flux"], label="Integrated grid velocity")
    axes[1].axhline(config["flux"], ls="--", color="#d88720", label="Wall streamfunction difference")
    axes[1].set(xlabel="x", ylabel="flow rate", title="Flux diagnostic")
    axes[1].legend(fontsize=8)
    fig.savefig(output / "baseline.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--solve", action="store_true", help="solve and render")
    stage.add_argument("--render", action="store_true", help="render saved arrays only")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--nx", type=int, default=173)
    parser.add_argument("--ny", type=int, default=67)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--dt", type=float, default=.012)
    parser.add_argument("--walks", type=int, default=512)
    parser.add_argument("--baseline-walks", type=int, default=32768)
    parser.add_argument("--pilot-walks", type=int, default=64, help="independent scalar walks for the control; 0 disables")
    parser.add_argument("--pilot-columns", type=int, default=3, help="solved columns per end that get a pilot control")
    parser.add_argument("--flux", type=float, default=.35)
    parser.add_argument("--forcing-strength", type=float, default=40., help="vorticity injection per unit time; 0 disables stirring")
    parser.add_argument("--forcing-period", type=float, default=3.6)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    if min(args.nx, args.ny) < 9 or min(args.steps, args.walks, args.baseline_walks, args.save_every) < 1:
        parser.error("grid dimensions must be >=9 and step/sample counts positive")
    if not all(np.isfinite(v) and v > 0 for v in (args.dt, args.flux, args.eps, args.forcing_period)):
        parser.error("dt, flux, eps and forcing-period must be finite and positive")
    if not np.isfinite(args.forcing_strength) or args.forcing_strength < 0:
        parser.error("forcing-strength must be finite and nonnegative")
    if args.pilot_walks < 0 or args.pilot_columns < 0:
        parser.error("pilot-walks and pilot-columns must be nonnegative")
    output = args.output or out_dir("fluid_circle_wave")
    output.mkdir(parents=True, exist_ok=True)
    if not args.render:
        solve(args, output)
    render(output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
