# Geometry queries and traversal

Geometry, traversal, and shading have separate responsibilities:

```
RayQuery ── RayMarcher ── SimpleIntegrator (camera + normal shading)
                └─────── ImplicitScene ── MaterialIntegrator (scattering)
         └─ SurfaceSampler (multi-hit sampling)
EmptySphereQuery ── WoSIntegrator / DifferentialWoSIntegrator
```

RayMarcher, SurfaceSampler, and MaterialIntegrator use
`utils.batch_driver.run_bounded_batch` to track original indices, refill finished
slots from pending work, and flush results. Their state types and step functions
remain independent. The driver knows nothing about hits, signs, epsilon,
advancement, or scattering. WoS retains its reduction and phase-transition driver.

All three, plus SimpleIntegrator and ImplicitScene, accept `max_active_states`
(default 4096, positive integer). Only that many live states are initialized.
Completed slots are written to the original output indices and refilled while
other tasks continue; this is not a sequence of fully drained chunks. Once the
queue is empty, `n_compactions` controls shrinking the tail. A value of zero
turns off tail shrinking, not the active-state bound or refilling.

The limit covers marching queries, multi-hit state, live material paths,
normal/shading queries, and the sampler's candidate-generation batch. Per-path
random streams are keyed by original task index, so changing capacity does not
change paths. Sampling ray generation still uses its existing batch-based RNG.
Input rays, output images, and padded per-ray hit results remain proportional to
the total requested workload; `max_active_states` is a working-state limit, not
a cap on total GPU memory. JAX preallocation is configured separately.

Rendering no longer needs script-level image tiling. The orbit CLI exposes
`--max-active-states 4096`. Diagnostic `return_buffers`/`debug` returns only the
remaining bounded state buffer and its original indices, alongside full outputs.

## Geometry contract

`surfaces.queries.RayQuery` adapts a backend `ray_step` and a configured
trust-region update function into a named `RayStep(residual, distance,
next_trust_region, gradient)`. The optional gradient path is selected at
construction, outside JIT. `EmptySphereQuery` returns named
`EmptySphere(residual, radius, next_trust_region)` results without changing the
backend's update policy.

- Residuals are **f(x) - level_set**. Consumers compare against zero.
- Distances/radii are world-space, nonnegative, and apply on either side of the
  surface. A negative residual is not itself a hit.
- Ray queries receive unit directions. RayMarcher normalizes incoming rays;
  SurfaceSampler's RayBuffer requires unit directions and stores world-space
  exit distances measured from its origins.
- Zero distance may mean an undecided bound: refine the trust region, do not
  force a positive step or automatically report a hit.
- A `RayQuery` owns interpretation of the backend's third tuple item. Existing
  backend implementations retain their tuple API. Algorithms see the next
  trust region instead of interpreting that flag themselves. Without an explicit
  update function, the adapter uses the backend's `update_ray_trust_region`
  method when present, otherwise the existing shared update. SDF/Lipschitz and
  dense backends preserve their unused ray trust region.
- `query.certified` describes whether the backend claims safe distances. The
  dense fixed-step reference backend sets it to False. This metadata is not a
  floating-point certification or a claim of complete root enumeration.

## First hit and rendering

```python
from rayMarcher.traversal import RayMarcher
from rayMarcher.scene import ImplicitScene
from rayMarcher.marcher import SimpleIntegrator

marcher = RayMarcher.from_surface(surface, max_steps=256, epsilon=1e-4, max_active_states=4096)
hit = marcher.first_hit(params, origins, directions)

scene = ImplicitScene([surface_a, surface_b], max_steps=256)
hit = scene.closest_hit([params_a, params_b], origins, directions)

renderer = SimpleIntegrator.from_surface(camera, background, surface)
image, query_counts = renderer.render(params, key)
```

`MaterialIntegrator(camera, materials, background, surfaces, ...)` constructs an
ImplicitScene and uses the same RayMarcher. Each row of materials corresponds
to one surface. `renderer.intersect(...)` delegates to `scene.closest_hit(...)`.
No renderer is constructed just to perform geometry queries.

The existing callback constructor of SimpleIntegrator and `trace_rays` alias
remain available. Its old `level_set` argument is ignored: the supplied callback
must already return the shifted residual. Prefer `from_surface` for new code.

## Sampling through the interior

```python
from surfaceSampler.sampler import SurfaceSampler

sampler = SurfaceSampler.from_surface(surface, max_step=512, max_hits=16)
samples = sampler.sample_surface_points(
    params, key, box_min, box_max, num_rays=1024, return_details=True)
points = samples.points
complete_rays = samples.rays.complete
```

Sampling records entry into the epsilon band, advances past the hit, then
continues regardless of the residual sign. `in_hit_band` means proximity to the
zero set, not interior membership. This preserves both entering and exiting
hits and supports rays whose origins are inside objects.

For SDF/Lipschitz surfaces the continuation uses `epsilon / L` inside the band.
Otherwise supply `lipschitz_bound=L` valid throughout the sampling region, or an
explicit callback:

```
continuation(params, position, unit_direction, ray_step_result, epsilon)
    -> minimum world-space advance inside the hit band
```

The sampler takes max(query distance, continuation distance) only inside the
band. Outside it the original query distance is preserved, including zero.
An explicit continuation is a sampling approximation and needs an appropriate
scale for the field; it is not a guaranteed empty-space step. There is no silent
L=1 fallback. Keep the bound current if surface parameters change.

`sampler.trace_rays(params, RayBuffer(...))` bypasses random ray generation and
returns padded multi-hit output with per-ray statuses. `sample_surface_points`
flattens the recorded hits. It raises on incomplete traversal by default;
`return_details=True` exposes the partial results and reasons, while
`allow_incomplete=True` explicitly accepts just the partial point array.
The existing `debug=True` path returns the internal buffers for diagnostics.

## Completion and limits

`utils.types.Termination` distinguishes HIT, EXIT, MAX_STEPS, HIT_CAPACITY,
INVALID, and INACTIVE. ACTIVE is internal. EXIT means the configured interval
was exhausted; it says nothing about the surface outside that interval.
MAX_STEPS counts actual query evaluations, including terminal evaluations.
The exit endpoint is evaluated if the query budget permits it.

Multi-hit traversal stops at storage capacity and reports HIT_CAPACITY rather
than silently dropping subsequent hits. Reaching the interval endpoint with
its final hit stored is complete. Incomplete rays can bias a sampled point set.
Even completed finite-tolerance sampling is not a proof that all mathematical
roots were resolved; nearby crossings can merge within the epsilon band.

SceneHit preserves each surface's status in `surface_status`. If `complete`
is False, its position is only the nearest known hit: an unfinished surface
query might contain a closer one. The image-rendering convenience methods retain
best-effort rendering (background on misses); inspect geometry results when
completion guarantees matter.

## Walk-on-spheres completion

WoS and differential WoS distinguish `ACTIVE`, `BOUNDARY_HIT`, `MAX_STEPS`,
and `INVALID` using `sphereWalker.types.WalkerTermination`. Only successful
samples enter the accumulated contribution. A differential sample succeeds
only after both phases hit the boundary; the step budget is shared across phases.
A boundary hit at the final allowed step is accepted.

`integrate_batch(...)` still returns `WalkerOutput` when all requested samples
succeed. It raises `RuntimeError` if any actual input position has an exhausted
or invalid sample. To diagnose an incomplete run:

```python
from sphereWalker.types import WalkerTermination

details = walker.integrate_batch(
    params, positions, key, init_trust_region_size=0.5,
    n_samples=128, return_details=True)
sums = details.contribution_sum  # same array/PyTree structure as the estimate
successes = details.successful_samples
exhausted = details.termination_counts[:, WalkerTermination.MAX_STEPS]
invalid = details.termination_counts[:, WalkerTermination.INVALID]
complete_positions = details.complete
```

These are sums of completed sample contributions, not estimates averaged over
successful samples. Discarding failed walks or treating their contributions as
zero can bias a solution. `details.nSteps` averages work over all requested
samples, including failures. Increase the total `max_steps` budget and rerun
when necessary; `steps_between_refill` is a scheduling interval, not a reason to
accept unfinished walks. Completed state is frozen until reduction or phase
transition. Differential phase one restarts adaptation with the caller's
`init_trust_region_size` at the offset point. The geometry backend determines
the safe sphere and next trust region; the walker does not override that policy. This change does not add resumable checkpoints for interrupted walks.

WoS position buckets pad storage only: the queued task count is always
`number_of_input_positions * n_samples`. Padding does not launch extra walks.
Empty `(0, dim)` batches return empty arrays or PyTrees with the usual output
structure; compaction warm-up is a no-op for an empty batch. Output structure
is inferred by abstract tracing, without executing geometry queries or evaluating
boundary values at a fabricated point. Floating walker state uses the input
position dtype, including explicit float32 geometry when x64 is enabled.
The sample count, active-buffer cap, refill interval, and bucket size must be
positive integers; compaction count must be nonnegative.

## Validation

Six tests in two files cover the public contracts and integration paths:

- `scripts/test_traversal.py`: first hits and clipping; adaptive queries and
  limits and bounded refill; multi-hit sampling through interiors with explicit completion status.
- `scripts/test_renderer.py`: material paths and nearest-surface selection;
  both renderers and camera updates; sphere walking with a constant boundary.

From the repository root:

```
JAX_PLATFORMS=cpu python -m unittest discover -s scripts -p 'test_renderer.py'
JAX_PLATFORMS=cpu python -m unittest discover -s scripts -p 'test_traversal.py'
```

Repeat with `JAX_ENABLE_X64=true` to check the same contracts in x64 mode.
Cases use small analytical scenes and fixed seeds, without image snapshots.
