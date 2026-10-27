# Walk on Stars

`WoStIntegrator` solves `-Δu = f` with an implicit Dirichlet boundary and a
homogeneous Neumann enclosure (`du/dn_out = 0`). It is a separate integrator
using the existing WoS batch/refill/compaction machinery. There is no new
boundary hierarchy or domain object.

## Construction

```python
from sphereWalker.star_integrator import WoStIntegrator
from surfaces.implicit.lipschitz import sdfImplicit
from surfaces.implicit.primitives import bbox_sdf, sphere_sdf

inner = sdfImplicit(fn=sphere_sdf, dim=3)
outer = sdfImplicit(fn=bbox_sdf, dim=3)
solver = WoStIntegrator(
    empty_sphere_fn=inner.empty_sphere,
    boundary_condition_fn=g,          # g(x), scalar or fixed-size vector
    source_term_fn=f,                 # f(x), scalar; -Δu = f
    dim=3,
    neumann_surface=outer,
    eps=1e-3,
    max_steps=2000,
)
result = solver.integrate_batch(
    inner_params, points, key,
    init_trust_region_size=1., n_samples=4096,
    neumann_params=outer_params,
)
```

The analytic query functions live beside `sphere_sdf` and `bbox_sdf` in
`primitives/sdfnd.py`. `sdfImplicit` dispatches to them for these two functions;
other functions (including wrapped or composed primitives) raise
`NotImplementedError` when a star or intersection query is called.

`sphere_sdf` parameters are `center` and `radius`; `bbox_sdf` parameters
are `min_corner` and `max_corner`, matching the existing primitive conventions.
Both remain SDF surfaces supporting ordinary empty-sphere and ray-step queries.
Their analytic star queries apply to the **interior** as the reflecting domain;
they do not implement the exterior-obstacle case.

The initial points must lie inside the enclosure and on the chosen side of the
Dirichlet surface (outside the inner object in the configuration above). The
Dirichlet object should be contained in the enclosure. Global containment and
Dirichlet-side membership are caller preconditions, not automatically proved.

Neumann geometry parameters are dynamic, supplied at each solve. The original
`params` still reaches `empty_sphere_fn` and, with `source_takes_params=True`,
`source_term_fn(params, x)` unchanged. Internally `WoStParameters` combines the
two parameter groups. Warmup accepts the same `neumann_params` keyword.

`max_radius` optionally caps each step, defaulting to infinity. The actual
Dirichlet clearance and the returned star radius must be finite. There is no
minimum-radius clamp that silently overrides geometric bounds. A zero
clearance (an undecided bounded query) keeps the walker in place with the
next trust region, as in WoS; such retries count toward `max_steps`.

## Optional surface protocol

The existing `ImplicitSurface` base defines two optional operations, both
defaulting to `NotImplementedError`. `sdfImplicit` implements them for
`bbox_sdf` and `sphere_sdf`. Other primitive functions raise when queried;
Lipschitz, bounded, and neural implicit surfaces remain unsupported as
Neumann geometry unless they override these operations. They can still
supply the Dirichlet empty-sphere query.

```python
star_radius(params, position, radius_cap, *, on_boundary=False, normal=None)
    -> StarRadius(radius, status)

intersect_ray(params, origin, direction, max_distance, *, on_boundary=False)
    -> SurfaceHit(distance, position, normal, status)

inward_direction(params, position, direction, normal)
    -> unit_direction
```

`inward_direction` has a smooth-boundary default that folds a uniform sphere
sample into the inward hemisphere. The integrator owns random sampling; this
hook only transforms the sampled direction. Nonsmooth geometry overrides it to
preserve uniform angular sampling over its admissible directions.

Both results live in `surfaces.queries`; status uses `utils.types.Termination`:

- Radius `EXIT`: a valid positive radius no greater than the requested cap.
- Intersection `HIT`: first positive hit within the segment, with unit outward
  geometric normal at smooth contacts. A box edge/corner returns a deterministic
  representative face normal; direction folding uses all active faces, not this
  representative normal alone.
- Intersection `EXIT`: the entire segment is clear; distance is the requested
  maximum and position is its endpoint. Normal is zero and unused.
- `INVALID` or `MAX_STEPS`: query failed or is unresolved, never a clear segment.

These are pure single-point queries, batched with `vmap`; integer/status fields
are int32 and floats follow the position dtype. Ray directions must be unit
length. The starting contact is excluded, but zero-length outward/tangent
starts fail explicitly. `on_boundary=True` asserts an actual boundary contact;
it is not a request to project an arbitrary point onto the surface.

The surface also supplies its existing `signed_value` and `normal_at` operations
for recognizing exact wall points, checked at every step: boundary starts,
tangent gradient first-ball samples, and clear rays whose endpoint rounds onto
the wall. Checking only starts and reported hits left about 1 in 10^8 steps
invalid, which failed a 181M-walk solve. Reported hits retain their returned
normal. The box normal uses an unrolled comparison rather than `jnp.argmin`,
which lowered to a separate GPU reduction kernel. `ray_step` is unchanged: it is a local marching increment,
whereas `intersect_ray` is a complete bounded query.

For convex box/sphere interiors, the radius query returns the cap after checking
containment. Analytic intersections clip rays to the enclosure. Box hits snap
every simultaneously intersected coordinate to its wall; sphere hits normalize
back onto the sphere, within floating-point rounding. Exact box edges/corners
and tied hits are supported. The box derives active faces from exact contact
coordinates, then flips each outward component independently. This gives a
uniform quadrant/octant sample at edges/corners and does not move the walker.
Inactive components retain their signs, even arbitrarily near another face.
The representative normal is metadata, not a unique normal at a corner; it
must not be used to define nonzero flux at nonsmooth contacts.

## Estimator and state

`StarWalkerState` adds `on_neumann` and `normal` to the ordinary walker state.
Each step obtains Dirichlet clearance, restricts the star radius, samples a
uniform direction, folds it inward at a Neumann contact, then traces the bounded
ray. A Neumann hit remains `ACTIVE`; an unobstructed ray lands on the sphere.
Only Dirichlet termination produces `BOUNDARY_HIT` success.

The source sample reuses the same direction. Its normalized radial density is
`4 t log(1/t)` in 2D and `6 t (1-t)` in 3D. The Green integral is `R²/(2d)`.
Samples beyond the clipped endpoint contribute zero; `f` is not evaluated
outside the star. Crucially, `R` remains the original star radius, not the ray's
clipped length. Uniform inward angular sampling cancels the boundary angular
factor: one half on a face, one quarter at a box edge/2D corner, and one eighth
at a 3D box corner. The source normalization therefore remains `R²/(2d)`.
This supports a scalar source and scalar or vector Dirichlet values
(with the same scalar source contribution to each component, as in WoS).

Nonzero Neumann data is **not implemented**. Passing `neumann_condition_fn`
raises `NotImplementedError`; adding it requires independent boundary sampling
with a known surface-measure PDF and its boundary-integral estimator. Existing
`BoundaryConditions.dirichlet_fn` can still provide `g`; Neumann data is not
passed through that Dirichlet adapter.

The shared diagnostics preserve failure counts rather than averaging only
successful samples. `return_details=True` exposes these counts; ordinary solves
raise if any walks are incomplete. Internal direct `step` calls take
`WoStParameters(problem_params, neumann_params)`.

### Batch driver

- **Refill interval:** walkers run in blocks of `steps_between_refill` steps
  between refills. A walker that finishes inside a block still costs a full
  step per remaining block step. Use about half the mean walk length,
  `return_details=True` reports it as `nSteps`. Measured fastest (GPU, 65,536
  lanes):
  - 8 for the 2D fluid examples (12–14 steps per walk);
  - 16–32 for the 3D box/sphere case (44 steps per walk).
- **Dead lanes:** once no tasks remain, lanes without a sample are marked done
  and `INVALID`. Phase transitions (differential WoS) cannot revive these
  lanes. There is no device-side skip for a block of only dead lanes. On GPU
  a `lax.cond` copies its predicate to the host each round, and skipping one
  flush block per solve did not measurably help.
- **Gradient WoSt blocks:** a block runs one masked step. After it, if no live
  walker still lacks its first ball, the rest of the block runs
  continuation-only steps. This skips the first-ball work, and the results
  equal the masked steps exactly.
- **Host syncs:** the host reads the refill count and the live count once per
  round, as one two-element array (one device-to-host copy). The gradient
  block's `lax.cond` adds one more predicate copy per round.

## Numerical limits and verification

### Plotted analytical example

Run `examples/wos/wost_annulus.py` to compare an actual 2D WoSt field with
the closed-form solution on `a < r < b`:

```text
-Delta u = 1,   u(a) = 0,   u'(b) = 0
u(r) = (a² - r²)/4 + (b²/2) log(r/a),   a=0.3, b=1
```

```sh
python examples/wos/wost_annulus.py --output assets/renders/wost/annulus
```

The example saves a PNG/PDF comparison, raw arrays in `annulus_data.npz`, and
`metrics.json` with errors, sample counts, and termination diagnostics. The
field uses 512 walks per grid point by default. A separate radial profile uses
32 independent batches of 512 walks per radius; its error bars are pointwise
95% Student-t intervals over batch means. These show sampling uncertainty, not
the finite-epsilon or floating-point bias. All walks must complete before a
plot is produced.

## Gradients

`WoStGradientIntegrator` lives beside `WoStIntegrator` in `star_integrator.py`
and accepts the same arguments plus optional `control_fn(x)`. It returns
`(N, value_components, dim)` spatial gradients. It shares the first-ball
mean-value gradient weights with `WoSGradientIntegrator`, then uses the
ordinary WoSt step for the value continuation. The first ball is capped by
both Dirichlet and Neumann empty-sphere distances, not silhouette distance.
Initial points on either boundary are invalid for this estimator. An undecided
zero-radius query updates the trust region and counts toward `max_steps`, but
does not consume the first-ball estimate or subtract its control value.
`control_fn(x)` optionally subtracts a local value from the continued value
estimate. With `control_takes_params=True`, the callback receives `(params, x)`
and its output must broadcast to the boundary value shape. This allows a
dynamic pilot without capturing a mutable field in a compiled closure. The
pilot must be fixed independently of the current gradient samples. It reduces
solution-offset variance; random source/path variance can still grow strongly
as the first radius shrinks.

## Circle-wave walls with open Neumann ends

The existing `examples/fluids/curved_channel.py` now uses constant Dirichlet
streamfunction `+Q/2` on the top wave and `-Q/2` on the bottom wave. Only the
wave field supplies Dirichlet clearance; `bbox_sdf` supplies the Neumann
geometry. In this channel the horizontal box edges are outside the fluid,
so the active Neumann boundary consists of the left/right openings. These
impose zero transverse velocity, with horizontal velocity determined by the
solution rather than a prescribed inlet profile.

```sh
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false python -m examples.fluids.curved_channel --solve
python -m examples.fluids.curved_channel --render
```

A cached harmonic WoSt gradient solve establishes the baseline flow. A
pointwise wall-distance control variate reduces its sampling noise. Dynamic
WoSt solves use zero Dirichlet values and the current grid vorticity, and their
velocity is added to the baseline. Before each gradient solve, a separate
64-walk scalar WoSt solve of the same vorticity supplies a streamfunction pilot
at the solved nodes in the `--pilot-columns` (default 3) columns next to each
Neumann end, where first balls are capped by the end distance and variance is
largest. Its random stream is separate from all gradient solves; the pilot is
zero elsewhere, and grid sampling supplies `control_fn(params, x)`. Set
`--pilot-walks 0` to disable it.
The pilot does not change the PDE or remove all source/path variance.
Localized, time-dependent body forcing
drives vortices at the upper and lower cavity mouths. Its curl enters the
vorticity equation as `d omega/dt + u.dot(grad omega) = forcing`; it does not
replace the WoSt velocity or change boundary data. Gaussian forcing patches
reverse with a 3.6-unit period and staggered phases along the channel.
Slower throughflow (`Q=0.35`) allows exchange between the channel and cavities.
Cyan/coral dyes enter only through two inlet streaks, placed at quantiles of
the computed incoming flux. Initially the cavities contain no dye and the
vorticity is zero. Incoming vorticity is zero. Left-end dye/vorticity data are
applied only where horizontal velocity is positive; outflow and stagnation
retain the transported values. Right-end backflow receives clean fluid.
The ends still intersect partial cavities and allow recirculation through
those openings. There is no no-slip wall-vorticity
source or viscosity in this example. Use `--forcing-strength 0` for an
otherwise matched steady-throughflow control; its zero vorticity perturbation
is evaluated exactly without dynamic walks.

The shared `make_transport` helper accepts optional `fluid_fn(x)` to clip
each RK2 half-segment to its last fluid point using 12 bisections when its
endpoint is solid. It no longer returns the whole trace to its starting point.
This permits travel up to wall contact but is not tangential sliding or an
exact characteristic boundary treatment. Thin solids between fluid endpoints
can still be missed. `inflow_only=True` enables the left-end condition above;
other examples retain the prescribed-inlet default. Within one unresolved grid cell of the wave walls,
velocity is extrapolated from solved points and its normal component is
attenuated toward the wall. The box-end horizontal velocity is extrapolated
from the adjacent column and transverse velocity is zero. These operations
are grid boundary approximations, not certified surface tracing. Independent
Monte Carlo velocity samples retain divergence noise; no projection or
velocity smoothing is applied. Bilinear transport also diffuses dye/vorticity.

The default run uses 173x67 nodes, 512 walks per solved point per dynamic step,
32768 baseline walks, 64 scalar pilot walks in three end columns, and 900 steps at dt=0.012. The
cached baseline remains noisy: quadrupling its former 8192-walk budget halves
Monte Carlo standard deviation in the usual independent-sampling scaling,
but does not remove fixed realization error or grid divergence. All query
points still use independent random samples; shared-random-number coupling
is not implemented and would need a measured spatial-difference comparison.
Full defaults are intended for a GPU; the backend and walk budget are printed
before the solve. Options `--nx`, `--ny`, `--walks`,
`--baseline-walks`, `--pilot-walks`, `--pilot-columns`, `--steps`, `--dt`, `--flux`, `--forcing-strength`,
`--forcing-period`, `--output`
control the example. `--solve` solves and renders; `--render` uses saved arrays.
Artifacts include `fluid.mp4`, `fluid.gif`, dye snapshots, baseline speed/flux
plots, raw `volume.npz`, and `metrics.json`. Incomplete walks cause an error;
solid-cell dye/vorticity and cross-sectional flow-rate error are reported.
Deep-cavity dye diagnostics use fluid nodes with `abs(y)>0.4`, excluding
the partial cavities at the open ends. They report mean total dye and the
fraction above concentration 0.05; these measure penetration, not a claim
of physically resolved molecular mixing.

### Solver checks

Dirichlet termination retains the current WoS test `abs(residual) <= eps`.
For an exact SDF this is distance-based; for a general scaled implicit field it
is a field-value tolerance, not a certified distance tolerance. Conservative
empty-sphere radii alone never establish boundary proximity. Finite epsilon,
floating-point geometry, and finite walk budgets remain numerical limitations.

`scripts/test_star_walker.py` checks analytic rays in 2D/3D, boundary starts,
invalid contacts, dynamic parameters, batching/compaction, output dtypes,
explicit failure propagation, radial sampling moments, constant/vector data,
and manufactured Poisson solutions with homogeneous reflecting walls. The
statistical regressions use fixed seeds and sample-derived standard errors.

```sh
JAX_PLATFORMS=cpu python -m unittest scripts.test_star_walker -v
JAX_ENABLE_X64=true JAX_PLATFORMS=cpu python -m unittest scripts.test_star_walker -v
```

Reference: Sawhney et al., [Walk on Stars](https://www.cs.cmu.edu/~kmcrane/Projects/WalkOnStars/WalkOnStars.pdf),
Algorithm 1 and Sections 4.4–4.6. This implementation does not include the
paper's general silhouette hierarchy, concavity minimum-radius approximation,
nonzero Neumann estimator, or differential WoSt.
