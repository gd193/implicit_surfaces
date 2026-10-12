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
`uniform`, root bounds, subclass `split` data) and the descent loop. Points
outside the half-open root `[lower, upper)` get `default_region`. Homogeneous
subtrees are collapsed, so internal nodes are `MIXED` and leaves carry labels.
Partitions are pytrees and can be passed as jit arguments.

Queries are written for vmapped batches on the GPU: loops have a fixed trip
count (`max_depth`), there are no per-lane stacks or scatters, and lanes do not
diverge. A stack-based search that certifies more boxes belongs in a dedicated
kernel (Pallas or Warp) later.

A new tree implements three hooks and a builder:

- `_encode_point(x) -> (inside, key)` and `_point_child(node, key, depth)`,
  used by the shared `_descend(key, limit)`;
- `_box_label(lower, upper) -> (inside, outside, label)`, where `label`
  certifies the box clipped to the root. The shared `classify_box` handles
  invalid boxes, boxes outside the root (default region), and straddling boxes
  (certified only if the in-root part has the default label);
- `from_grid(labels, lower, upper, default_region, n_regions)`. Extra builder
  options must be keyword arguments with defaults.

`VoxelTreePartition` splits at midpoints into `2**dim` children. Traversal uses
int32 cell coordinates on the finest grid, so a point on a split plane belongs
to the upper child. Depth is limited to 30, and to about 22 without x64.

- **Box query (fixed cost).** Choose the deepest level where the box covers at
  most two cells per axis, descend to those `<= 2**dim` cells, and certify only
  if all descents end at leaves with one label: at most `2**dim * max_depth`
  lookups. A region boundary inside a covered cell gives `MIXED` even when the
  box misses it.
- **Construction (JAX only).** `from_grid` takes an `(n,)*dim` grid with `n` a
  power of two and runs where the labels live. It builds a label pyramid by
  block min/max, numbers nodes level by level with prefix sums (children of a
  node are contiguous, in slot order), and scatters rows into exact-size
  arrays. Host transfers: the label range and the node count. Peak memory is
  roughly 2-3 times the dense int32 grid (pyramid plus id grids; not measured).

On an RTX 5060 laptop GPU (3D ball, f32, after compilation): a `256^3` grid
builds in about 4 ms into 166k nodes; 1M point queries take about 1 ms and
1M box queries about 13 ms, 92% of them certified.

## Validation

`scripts/test_boundary.py` checks point lookup against `HalfSpacePartition`,
split-plane and outside-root ownership, conservative boxes against sampled
points, construction on random grids (valid storage, minimal node count,
exact lookup), GPU build and query when available, a custom tree subclass,
pytree passing, data kinds and
dtypes, scalar and vector values, and the WoS adapter on a split disk
(`u(0) = 1/2`, and `u(p) = p` for vector data `g(x) = x`).

```
JAX_PLATFORMS=cpu python -m unittest scripts/test_boundary.py
JAX_ENABLE_X64=true JAX_PLATFORMS=cpu python -m unittest scripts/test_boundary.py
JAX_PLATFORMS=cuda,cpu python -m unittest scripts/test_boundary.py   # GPU test
```
