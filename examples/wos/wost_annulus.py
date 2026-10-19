"""Plotted 2D WoSt validation against an analytical mixed-boundary solution.

Run from any directory, e.g.:
  JAX_PLATFORMS=cpu python examples/wos/wost_annulus.py --output /path/to/output

Domain: a < r < b.  -Delta u = 1, u(a) = 0, u'(b) = 0.
Truth: u(r) = (a^2-r^2)/4 + (b^2/2) log(r/a).
Thus u'(r) = -r/2+b^2/(2r), and u''+u'/r = -1.
The field image is solved independently at every valid grid point. Radial
uncertainty is computed from independent batch means, not angular averages.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/wost-matplotlib")

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

from sphereWalker.star_integrator import WoStIntegrator
from surfaces.implicit.lipschitz import sdfImplicit
from surfaces.implicit.primitives import sphere_sdf


def truth(r, a=0.3, b=1.0):
    return (a * a - r * r) / 4 + 0.5 * b * b * np.log(r / a)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "assets/renders/wost/annulus"
    )
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--map-walks", type=int, default=512)
    parser.add_argument("--profile-points", type=int, default=25)
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--batch-walks", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261019)
    args = parser.parse_args()
    if (
        min(args.grid, args.profile_points, args.batches) < 2
        or min(args.map_walks, args.batch_walks) < 1
    ):
        parser.error(
            "grid, profile-points, batches must be >=2; walk counts must be positive"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    dtype = jnp.float32
    inner = sdfImplicit(fn=sphere_sdf, dim=2)
    outer = sdfImplicit(fn=sphere_sdf, dim=2)
    a, b, eps = 0.3, 1.0, 2e-4
    params = {"center": jnp.zeros(2, dtype), "radius": jnp.asarray(a, dtype)}
    neumann = {"center": jnp.zeros(2, dtype), "radius": jnp.asarray(b, dtype)}
    solver = WoStIntegrator(
        inner.empty_sphere,
        lambda x: jnp.zeros((), x.dtype),
        lambda x: jnp.ones((), x.dtype),
        2,
        neumann_surface=outer,
        eps=eps,
        max_steps=3000,
    )

    def solve(points, walks, key):
        start = time.perf_counter()
        diagnostics = solver.integrate_batch(
            params,
            jnp.asarray(points, dtype),
            key,
            1.0,
            walks,
            neumann_params=neumann,
            steps_between_refill=32,
            max_active_states=32768,
            jit_buck_size=128,
            return_details=True,
        )
        sums = np.asarray(jax.block_until_ready(diagnostics.contribution_sum))[:, 0]
        counts = np.asarray(diagnostics.termination_counts).sum(axis=0)
        if not np.all(np.asarray(diagnostics.complete)):
            raise RuntimeError(
                f"Incomplete walks; termination counts={counts.tolist()}"
            )
        return sums / walks, counts, time.perf_counter() - start

    map_key, profile_key = jax.random.split(jax.random.PRNGKey(args.seed))
    edges = np.linspace(-b, b, args.grid + 1)
    centers = (edges[1:] + edges[:-1]) / 2
    X, Y = np.meshgrid(centers, centers)
    radii = np.hypot(X, Y)
    mask = (radii > a) & (radii < b)
    print(f"Solving 2D map: {mask.sum()} points x {args.map_walks} walks", flush=True)
    values, map_counts, map_seconds = solve(
        np.stack((X[mask], Y[mask]), axis=1), args.map_walks, map_key
    )
    estimate = np.full(X.shape, np.nan)
    estimate[mask] = values
    exact_map = np.full(X.shape, np.nan)
    exact_map[mask] = truth(radii[mask], a, b)
    error = estimate - exact_map

    r = np.linspace(a, b, args.profile_points)
    profile_positions = np.stack(
        (np.repeat(r, args.batches), np.zeros(len(r) * args.batches)), axis=1
    )
    print(
        f"Solving radial profile: {len(r)} radii x {args.batches} independent batches x {args.batch_walks} walks",
        flush=True,
    )
    batches, profile_counts, profile_seconds = solve(
        profile_positions, args.batch_walks, profile_key
    )
    batches = batches.reshape(len(r), args.batches)
    mean = batches.mean(axis=1)
    se = batches.std(axis=1, ddof=1) / np.sqrt(args.batches)
    half_ci = student_t.ppf(0.975, args.batches - 1) * se
    exact_profile = truth(r, a, b)
    delta = mean - exact_profile
    nonzero = se > 0
    standardized = delta[nonzero] / se[nonzero]
    family_threshold = student_t.ppf(1 - 0.05 / (2 * nonzero.sum()), args.batches - 1)
    stats = {
        "pde": "-Delta u=1; u(a)=0; outward du/dn(b)=0",
        "inner_radius": a,
        "outer_radius": b,
        "epsilon": eps,
        "dtype": "float32",
        "devices": [str(d) for d in jax.devices()],
        "seed": args.seed,
        "grid": args.grid,
        "map_points": int(mask.sum()),
        "map_walks_per_point": args.map_walks,
        "profile_points": len(r),
        "independent_batches_per_radius": args.batches,
        "walks_per_batch": args.batch_walks,
        "map_rmse": float(np.sqrt(np.mean(error[mask] ** 2))),
        "map_max_absolute_error": float(np.max(np.abs(error[mask]))),
        "profile_rmse": float(np.sqrt(np.mean(delta**2))),
        "profile_max_absolute_error": float(np.max(np.abs(delta))),
        "profile_pointwise_95_percent_coverage": int(
            np.sum(np.abs(delta) <= half_ci + 1e-12)
        ),
        "max_absolute_standardized_profile_error": float(np.max(np.abs(standardized))),
        "simultaneous_95_percent_bonferroni_threshold": float(family_threshold),
        "profile_within_simultaneous_sampling_intervals": bool(
            np.all(np.abs(standardized) <= family_threshold)
        ),
        "termination_order": ["active", "dirichlet_hit", "max_steps", "invalid"],
        "map_termination_counts": map_counts.tolist(),
        "profile_termination_counts": profile_counts.tolist(),
        "map_seconds_including_compilation": map_seconds,
        "profile_seconds_including_compilation": profile_seconds,
        "uncertainty": "Student-t 95% pointwise CIs across independent batch means; excludes epsilon/floating-point bias",
    }
    np.savez_compressed(
        args.output / "annulus_data.npz",
        X=X,
        Y=Y,
        mask=mask,
        estimate=estimate,
        truth=exact_map,
        error=error,
        profile_r=r,
        profile_batches=batches,
        profile_mean=mean,
        profile_truth=exact_profile,
        profile_se=se,
        profile_ci95=half_ci,
    )
    (args.output / "metrics.json").write_text(json.dumps(stats, indent=2) + "\n")

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.7), constrained_layout=True)
    vmax = max(float(np.nanmax(estimate)), float(truth(b, a, b)))
    for ax, field, title in zip(
        axs[0],
        (estimate, exact_map),
        (f"WoSt estimate · {args.map_walks:,} walks / point", "Analytical solution"),
    ):
        image = ax.pcolormesh(
            edges,
            edges,
            field,
            cmap="viridis",
            vmin=0,
            vmax=vmax,
            shading="flat",
            rasterized=True,
        )
        ax.add_patch(
            plt.Circle((0, 0), a, fill=False, edgecolor="#333333", linewidth=1.4)
        )
        ax.add_patch(
            plt.Circle(
                (0, 0),
                b,
                fill=False,
                edgecolor="#333333",
                linewidth=1.4,
                linestyle="--",
            )
        )
        ax.set(aspect="equal", xlabel="x", ylabel="y", title=title)
    fig.colorbar(image, ax=axs[0].tolist(), label="u(x, y)", shrink=0.85)
    axs[0, 0].text(0, 0, "Dirichlet\nu = 0", ha="center", va="center")
    axs[0, 1].text(0, 0, "Inner hole", ha="center", va="center")
    ax = axs[1, 0]
    dense_r = np.linspace(a, b, 400)
    ax.plot(
        dense_r,
        truth(dense_r, a, b),
        color="#222222",
        label="Analytical truth",
        linewidth=1.7,
    )
    ax.errorbar(
        r,
        mean,
        yerr=half_ci,
        fmt="o",
        color="#0072B2",
        markersize=3.5,
        capsize=2,
        label="WoSt ± 95% CI",
    )
    ax.set(
        xlabel="Radius r",
        ylabel="u(r)",
        title=f"Radial profile · {args.batches * args.batch_walks:,} walks / radius",
    )
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.18)
    ax = axs[1, 1]
    ax.axhline(0, color="#222222", linewidth=1)
    ax.errorbar(
        r, delta, yerr=half_ci, fmt="o", color="#0072B2", markersize=3.5, capsize=2
    )
    ax.set(
        xlabel="Radius r",
        ylabel="WoSt − analytical truth",
        title=f"Profile error · RMSE {stats['profile_rmse']:.4f}",
    )
    ax.grid(alpha=0.18)
    fig.suptitle(
        "Walk on Stars: 2D annulus with a reflecting outer circle\n"
        r"$-\Delta u=1,\quad u(0.3)=0,\quad \partial_r u(1)=0$",
        fontsize=15,
    )
    fig.supxlabel(
        "Error bars: 95% pointwise sampling intervals from independent batch means; numerical bias excluded.",
        fontsize=9,
    )
    fig.savefig(args.output / "annulus_comparison.png", dpi=180)
    fig.savefig(args.output / "annulus_comparison.pdf")
    plt.close(fig)
    print(json.dumps(stats, indent=2), flush=True)
    print(f"Figure: {args.output / 'annulus_comparison.png'}", flush=True)


if __name__ == "__main__":
    main()
