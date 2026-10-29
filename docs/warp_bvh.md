# GPU BVH queries with Warp and JAX

`bvh.sah.warp` builds static binary SAH trees on the CPU and queries them on
the GPU. `bvh.snch.warp` adds normal cones, silhouette queries and boundary
sampling. Geometry stays shared under JIT and nested `vmap`.

## Setup

Install alongside JAX, NumPy and trimesh:

```sh
python -m pip install -r requirements-warp.txt
```

Validated with JAX/JAXLIB 0.10.2 and Warp 1.18.0. Warp is optional; imports
and backend discovery do not initialize CUDA. Set `CUDA_VISIBLE_DEVICES`
before starting Python. Use `JAX_PLATFORMS=cpu` for CPU execution and
`XLA_PYTHON_CLIENT_PREALLOCATE=false` to disable JAX's GPU memory reservation.

## Usage

```python
import jax
import jax.numpy as jnp

from surfaces.base import TriMesh
from surfaces.explicit.primitives import TriMeshPrimitives
from surfaces.explicit.accelerators import BVHAccelerator, Config

primitives = TriMeshPrimitives(TriMesh(
    vertices=jnp.asarray(vertices, dtype=jnp.float32),
    faces=jnp.asarray(faces, dtype=jnp.int32),
))
tree = BVHAccelerator.build(
    primitives, "bvh.snch.warp", Config(leaf_size_threshold=4),
)
closest = jax.jit(tree.backend.closest_point_with_status)
point, distance, face, status = closest(tree.payload, points)
sign, status = jax.jit(tree.backend.classify_point_with_status)(tree.payload, points)

silhouette = jax.jit(tree.backend.closest_silhouette_with_status)
point, distance, edge, status = silhouette(tree.payload, points, max_distance=0.5)

robin = tree.with_robin_coefficients(1.0)
radius, status = jax.jit(robin.backend.robin_radius_with_status)(
    robin.payload, points, 0.5,
)
```

Vectors have shape `(3,)` or `(..., 3)`; scalar bounds broadcast to query
shape. Ray origins and directions must have matching shapes. Bound methods
such as `tree.closest_point(points)` are also available.

## Static query factories

Select primitives and metrics with ordinary Python arguments before JAX tracing:

```python
from surfaces.explicit.accelerators.bvh.warp import (
    Segments, Euclidean, build_tree, make_nearest, make_ray_query,
)

primitive = Segments(2)
tree, data = build_tree(primitive, primitive.data(starts, ends))
nearest = make_nearest(primitive, Euclidean())
point, distance, segment, status = jax.jit(nearest)(tree, data, points)
all_hits = make_ray_query(primitive, "all")
times, segments, count, status = all_hits(tree, data, origins, directions, max_hits=8)
```

Float32 builtins: `Points(2|3)`, `Segments(2|3)`, `Triangles()` and
`SilhouetteEdges()`. Euclidean nearest queries support all four. `Angular()`
supports 2D/3D points and segments; queries end in `(2, dimension)` with viewpoint
and direction as rows, and distances/caps are radians. For view-dependent edges, supply
`SilhouetteEdges().data(a,b,n0,n1)` and `predicate=Silhouette()`.

Implement `PrimitiveSpec` and `MetricSpec.bind(primitive)` to extend the factory.
Primitive specifications provide an array-only Warp struct, device load/bounds
functions, and host bounds/packing. Host bounds use the primitive dimension;
device bounds use `vec3`, with zero z in 2D. Metrics return `NearestOps`: query type/shape,
validity, box bound, primitive minimizer, cap/score conversions and culling slack.
Silhouette metrics also supply an explicit `viewpoint` callback.
`NearestOps.group_leaves` selects static traversal scheduling; Euclidean point
queries enable it when bounds are cached. Other queries retain flat traversal.
Callbacks are `@wp.func` or `@wp.func_native`. Bounds and minimizers must use the
same score units; bounds must be conservative. Optional `node_bound_with_data`
receives `(query, tree, data, node)` for additional node arrays.

Specifications are immutable and hashable; geometry and queries remain dynamic
array pytrees. Leaf query kernels are cached per specialization and exposed as `.kernel`.
Star radius composes silhouette distance with clipping.
Depth-first queries share one walker and node layout. Static callbacks supply
primitive/metric math; backend methods use the same factories. Equivalent policies
reuse the exact kernel. Boundary sampling has one CDF descent implementation.

Ray modes are `first`, `any`, `count` and `all`. Triangles and 2D segments support
rays; points and 3D segments have no ray operation. First/any accept nonnegative
intervals; count/all also support signed lines. Use `exclude=True` when creating
first/any queries that need face exclusion. Count/all always support exclusion.

### Indexed boundaries

Boundary queries use indexed segments in 2D or indexed triangles in 3D:

```python
from surfaces.explicit.accelerators.bvh.warp import (
    Segments, build_boundary, make_classify, make_silhouette,
    prepare_boundary_sample, make_boundary_sample, prepare_robin, make_robin_radius,
)

primitive = Segments(2)
tree, data, boundary = build_boundary(primitive, primitive.indexed(vertices, segments))
sign, status = make_classify(primitive)(tree, data, boundary, points)
point, distance, feature, status = make_silhouette(primitive)(tree, data, boundary, points)
sampling = prepare_boundary_sample(tree, boundary)
sample = make_boundary_sample(primitive)(tree, data, sampling, points, radii, randoms)
robin = prepare_robin(tree, boundary, minimum=1.0)
radius, status = make_robin_radius(primitive)(tree, data, robin, points, radii)
```

`build_boundary(..., silhouettes=False)` skips optional feature/cone construction.
2D silhouettes are vertices; 3D silhouettes are edges. Indexed connectivity is
explicit; loose endpoints are never welded. Original primitive/feature IDs persist.
Classification requires closed, non-degenerate connectivity and detects exact
contact separately from parity crossings. A geometric tangent hit need not cross.

Custom `BoundaryAdapter` objects supply topology, feature construction, visibility,
leaf predicates and classification operations. `MeasureOps` supplies local samples,
normals and densities, plus optional selection weights; use the same policy with
`prepare_boundary_sample` and `make_boundary_sample`. `RegionOps` separates
conservative overlap from exact sample membership. PDFs include path, primitive
selection and local density factors; rejected samples remain null events.

`RobinOps` supplies conservative primitive/node/feature bounds, with optional typed
metadata and data-aware callbacks. Built-in Euclidean bounds use different 2D and
3D formulas. Other metrics require explicit valid radius policies.

`make_star_radius` caps Euclidean silhouette distance for Neumann boundaries.
`make_overlap` tests primitive AABBs. `ClassificationOps` supplies closure validity
and per-primitive miss/crossing/contact/failure events. Custom multi-hit primitives
use `RayOps(state_type=..., next_hit=...)`; its zero-initialized state enumerates
all eligible hits once, returning updated state, parameter and hit/done/failure
flag. Callbacks receive a prepared `Ray` with origin, direction and projection
frame; setup runs once per query.

## Query contracts

| Method | Result |
|---|---|
| `classify_point_with_status(points)` | Sign: -1 inside, 0 boundary, +1 outside; status |
| `closest_point_with_status(points)` | Point, distance, original face ID, status |
| `first_hit_with_status(origins, directions, max_t, min_t=0)` | Ray parameter, original face ID, hit, status |
| `any_hit_with_status(origins, directions, max_t, min_t=0)` | Occlusion, status |
| `count_hits_with_status(origins, directions, max_t=inf, min_t=0, exclude_face=-1)` | Count, status |
| `all_hits_with_status(origins, directions, max_t=inf, max_hits=K, min_t=0, exclude_face=-1)` | Unsorted parameters, original face IDs, total count, status |
| `aabb_overlap_with_status(lower, upper)` | Triangle-AABB broad-phase overlap, status |
| `closest_silhouette_with_status(points, max_distance=inf)` | Point, distance, original edge ID, status |
| `star_radius_with_status(points, cap)` | Minimum of cap and silhouette distance, status |
| `robin_radius_with_status(points, cap)` | Robin star-radius bound, status |
| `sample_boundary(points, radii, randoms)` | Point, normal, area PDF, original face ID, status |

The triangle backend silhouette, star-radius, Robin and sampling methods require SNCH.
Factories also support the indexed 2D boundary path and explicit custom policies.
Status **1** means a valid result, **0** means a valid
miss/empty geometry, and **-1** means invalid input or failure. Equal computed
distances or ray parameters choose the smaller original ID. Closest point
reports failure when squared distances exceed float32 range.

Classification uses orientation-independent even-odd parity on closed indexed
meshes. Each edge must occur twice; open/nonmanifold edges and degenerate faces
return NaN/status -1. Empty meshes return +1/status 0. Nested shells alternate
inside/outside; self-intersections are unchecked. Half-open edge ownership and
float64 fallback handle ambiguous float32 predicates. Boundary tests have no
tolerance band. `TriMeshSurface.signed_value` combines classification with
closest-point distance.

Ray intervals `[min_t, max_t]` and silhouette caps are closed. Positive
`min_t` excludes starting contact; infinite `max_t` is supported. Ray misses
return `t=max_t`, face `-1`, and `hit=False`. Directions need not be unit
length; the ray parameter is a distance only for unit directions. Silhouette
queries include tangencies and open edges; misses return distance `inf` and
edge `-1`.

Count/all-hit intervals may be signed: `min_t=-inf` queries a full line.
`exclude_face` skips one original face. Half-open edges give consistent ownership
of shared edges; coplanar faces are ignored. `all_hits` requires static positive
`max_hits` and pads with `inf`/face -1. Status -2 retains an unsorted prefix and
the exact total count. Invalid queries return count/status -1. `count_hits`
allocates no hit buffer.

Star and Robin radius queries return status 1 for a valid cap even without a
smaller bound. They provide geometric bounds; a full explicit-mesh WoSt solver
is not implemented. Boundary sampling consumes three uniforms in `[0,1)`.
Samples outside the sphere are null events with PDF/status 0 and no retry;
retain them in estimators. Accepted PDFs include branch probabilities.

`with_robin_coefficients(minimum, maximum=None)` shares geometry and accepts
nonnegative scalar or per-face bounds in original face order. Omitting `maximum`
sets constant face coefficients. Host updates round intervals outward to float32.
Zero recovers Neumann; infinity gives closest-surface distance. Finite coefficients
also restrict faces without silhouettes ([Walkin' Robin, Algorithms 2–4](https://imaging.cs.cmu.edu/walk_on_stars_robin/index_files/paper.pdf)).
The estimator evaluates actual coefficients and boundary data.

## Construction and extension

Builders require finite referenced float32 vertices and valid triangle
indices. SAH accepts degenerate triangles; SNCH rejects them and edges with
more than two incident faces. Winding determines normals. Build separate
absorbing/reflecting subset trees for mixed boundary conditions.

Topology, leaf ranges and packed indices are validated. Leaf size is an SAH
threshold; retained leaves can exceed it. Tree depth must fit
`max_stack_depth`, at most 64. Rebuild after geometry changes. Geometry refit and
automatic differentiation are not supported.

Warp's built-in `wp.jax_kernel` handles FFI, compilation and ownership. The
[batching adapter](../surfaces/explicit/accelerators/bvh/backends/_warp_jax.py)
flattens query axes without tiling geometry. Factories bind typed geometry and optional metadata through the same adapter;
existing triangle backend methods delegate to them. SNCH cones enclose whole edges.
Packed float4 nodes hold bounds and topology for cached nearest, ray, overlap and
classification queries.
Closest point caches node bounds in eight-byte stack entries; silhouette and Robin
use four-byte indices/tags and recompute cheap bounds when popping. These choices
specialize by query semantics, independently of primitives and metrics. FFI arrays are contiguous.
Rebuild the tree when changing bounds or topology.

Adapted ray helpers retain Warp's
[Apache-2.0 license](../surfaces/explicit/accelerators/bvh/backends/licenses/warp-LICENSE.txt).

Native C++ lives in [headers](../surfaces/explicit/accelerators/bvh/backends/native/query.h);
Python keeps typed one-line bindings. The query factory attaches include paths
and every header as a Warp build dependency. Header edits trigger recompilation
in a new process; reapply module options when editing within a running process.
Primitive, dimension and metric specialization still happens during JIT compilation.

Format and lint the headers with their local Clang configurations:

```sh
native_dir=surfaces/explicit/accelerators/bvh/backends/native
warp_native=$(python -c 'from pathlib import Path; import warp; print(Path(warp.__file__).parent / "native")')
clang-format --dry-run --Werror "$native_dir"/*.h
clang-tidy "$native_dir/query.h" -- -x c++ -std=c++17 \
    -Wno-pragma-once-outside-header -isystem "$warp_native"
```

## Validation and timing

```sh
python -m pytest -q tests

python -m scripts.bench.spatial_trees.warp_bvh_device \
    --backend bvh.snch.warp --queries closest_silhouette \
    --mesh torus --point-distribution shell --radius .5
```

The shared [case catalog](../scripts/bench/spatial_trees/_warp_cases.py) covers all
37 supported builtin primitive/metric/query combinations. Parametrized tests check
independent geometry oracles, equivalent-policy kernel identity and benchmark input
packing. Unsupported combinations fail explicitly; custom policies have focused tests.

```sh
python -m pytest -q tests/spatial_trees/test_warp_matrix.py
JAX_PLATFORMS=cpu python -m pytest -q tests/spatial_trees/test_warp_matrix.py
python -m scripts.bench.spatial_trees.warp_query_matrix --counts 8192 --primitives 1024
```

[Matrix timings](benchmarks/warp_query_matrix.json) cover tree kernels only. Star
radius shares silhouette traversal; its JAX cap composition is excluded. Timings stay
outside the test suite; no hardware-dependent latency thresholds run in ordinary tests.

[FCPW/cuBQL comparisons](benchmarks/query_references.md) cover 2D segment queries
and 2D/3D point nearest. The [reference harness](../scripts/bench/spatial_trees/references/README.md)
saves generated CUDA C++/PTX, samples, source/input hashes and kernel resources.

The benchmark times warmed CUDA graphs with CUDA events, excluding build,
compilation, transfers, readback and CPU dispatch. `--native-warp` adds Warp mesh
references. Results go to
`assets/benchmark_results/`.

Recorded [ray](benchmarks/warp_bvh_device.json),
[factory](benchmarks/warp_factories.json),
[generalization](benchmarks/warp_generalization.json),
[crossing](benchmarks/ray_crossings.json),
[closest-point](benchmarks/closest_point.json),
[silhouette](benchmarks/closest_silhouette.json),
[Robin](benchmarks/robin_radius.json) and
[classification](benchmarks/inside_classification.json) comparisons retain samples,
timing scope and source hashes. WoSX’s `insideDomain` uses closest point and
normal orientation, allowing sign to reuse closest-point work; it uses no parity
test in the measured GPU path.
