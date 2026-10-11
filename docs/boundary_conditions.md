# Boundary conditions

`boundaryConditions/` says which boundary condition applies where, and with
what data. It is independent of the implicit geometry: the surface answers
geometric queries, a partition labels boundary patches, and boundary data gives
each patch its coefficients.

## Contracts

`BoundaryPartition` assigns every point one int32 region id:

- `classify_point(params, x)` returns an id in `[0, n_regions)`, never `MIXED`.
- `classify_box(params, lower, upper)` returns an id only if every point of the
  closed box has it, otherwise `MIXED = -1`. `MIXED` means a uniform label was
  not established, not that the box contains several regions. Certification
  holds in exact arithmetic; implementations widen boxes by rounding error.

`BoundaryData` gives coefficients of `a u + b du/dn_out = c`, with `n_out` the
outward unit normal supplied by the geometry:

| Kind | Condition | `(a, b, c)` |
|---|---|---|
| `DIRICHLET` | `u = g` | `(1, 0, g)` |
| `NEUMANN` | `du/dn_out = h` | `(0, 1, h)` |

`a` and `b` are scalars: a region's kind applies to every component. `c` has
the data's `value_shape`, fixed at construction: `()` for a scalar field,
`(m,)` for an `m`-component vector field. Returned values must match it
exactly; scalars are not broadcast.

Neumann data is a normal derivative, not physical flux `-k du/dn`. Robin
(`(mu, 1, h)`) is reserved, not supported. `evaluate(params, region_id, x,
normal)` must be differentiable in `x` near the boundary, because differential
WoS takes the normal derivative of Dirichlet data. The region is looked up once
at the boundary point and held fixed under that derivative.

`FunctionBoundaryData(kinds, fn)` covers constant and varying data with one
function `fn(params, region_id, x, normal) -> c`:

```python
FunctionBoundaryData(kinds, lambda p, r, x, n: p['values'][r])        # gather
FunctionBoundaryData.from_region_fns(kinds, [f0, f1])                  # lax.switch
```

A gather costs the same for any region count; `from_region_fns` traces every
branch.

`BoundaryConditions(partition, data)` checks region counts and provides
`dirichlet_fn(partition_params, data_params, normal_fn=None)` for the current
integrators' `boundary_condition_fn`. Walks ending in a Neumann region return
NaN and are counted `INVALID`. Without `normal_fn` the normal is NaN, so data
that reads it fails explicitly. Gradients with respect to data params are not
estimated.

`WoSIntegrator` accepts vector-valued `g` with a scalar or zero source term;
the estimate has shape `(N, m)`. A vector source term is not supported: the
walker value has one component. `DifferentialWoSIntegrator` runs with scalar
`g`. With vector `g` it is correct only for scalar parameter leaves: the
`V_n (g - u)` product broadcasts each leaf against the `m` components, which
fails or, when a leaf has `m` entries, silently mixes them.

## Dtypes

Integers (ids, kinds, tree indices, loop carries) are int32 in both precision
modes. Floats follow the query: `jnp.result_type(x, float32)`, so explicit f32
geometry stays f32 under x64. Stored floats use the widest available dtype and
are cast to the query dtype. Split-plane ownership is deterministic for one
dtype; points within rounding of a plane may be owned differently in f32 and
f64.

## Trees

`TreePartition` holds shared storage (`TreeArrays`: int32 `children` and
`uniform`, root bounds, subclass `split` data) and the traversal loops. Points
outside the half-open root `[lower, upper)` get `default_region`. Homogeneous
subtrees are collapsed, so internal nodes are `MIXED` and leaves carry labels.
Partitions are pytrees and can be passed as jit arguments.

A new tree implements four hooks and a builder:

- `_encode_point(x) -> (inside, key)` and `_point_child(node, key, depth)` for
  point descent;
- `_encode_box(lower, upper) -> (inside, outside, key)`, with the key describing
  the box clipped to the root, and `_root_context()`;
- `_box_children(node, context, key, depth) -> (overlap_mask, child_contexts)`;
- `from_grid(labels, lower, upper, default_region)`.

Box queries visit every node meeting the box depth-first and stop at the first
differing leaf. A straddling box is certified only if its in-root part has the
default label. After `max_box_visits` nodes (default 1024) the answer is
`MIXED`.

`VoxelTreePartition` splits at midpoints into `2**dim` children. Traversal uses
int32 cell coordinates on the finest grid, so a point on a split plane belongs
to the upper child. `from_grid` takes an `(n,)*dim` grid with `n` a power of
two; depth is limited to 30, and to about 22 without x64.

## Validation

`scripts/test_boundary.py` checks point lookup against `HalfSpacePartition`,
split-plane and outside-root ownership, conservative boxes against sampled
points, collapsing, a custom tree subclass, pytree passing, data kinds and
dtypes, scalar and vector values, and the WoS adapter on a split disk
(`u(0) = 1/2`, and `u(p) = p` for vector data `g(x) = x`).

```
JAX_PLATFORMS=cpu python -m unittest scripts/test_boundary.py
JAX_ENABLE_X64=true JAX_PLATFORMS=cpu python -m unittest scripts/test_boundary.py
```
