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
minimum-radius clamp that silently overrides geometric bounds.

## Optional surface protocol

The existing `Surface` base defines two optional operations, both defaulting
to `NotImplementedError`. `WoStIntegrator` checks for their implementation at
construction. Generic `sdfImplicit(fn=...)`, Lipschitz, bounded, and neural
implicit surfaces remain unsupported as Neumann geometry. They can still
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
for recognizing exact initial boundary starts. Subsequent contacts retain the
returned normal. `ray_step` is unchanged: it is a local marching increment,
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

## Numerical limits and verification

### Plotted analytical example

Run `examples/wos/wost_annulus.py` to compare an actual 2D WoSt field with
the closed-form solution on `a < r < b`:

```text
-Delta u = 1,   u(a) = 0,   u'(b) = 0
u(r) = (a² - r²)/4 + (b²/2) log(r/a),   a=0.3, b=1
```

```sh
JAX_PLATFORMS=cpu python examples/wos/wost_annulus.py --output assets/renders/wost/annulus
```

The example saves a PNG/PDF comparison, raw arrays in `annulus_data.npz`, and
`metrics.json` with errors, sample counts, and termination diagnostics. The
field uses 512 walks per grid point by default. A separate radial profile uses
32 independent batches of 512 walks per radius; its error bars are pointwise
95% Student-t intervals over batch means. These show sampling uncertainty, not
the finite-epsilon or floating-point bias. All walks must complete before a
plot is produced.

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
