# GPU BVH queries with Warp and JAX

Build a binary BVH from a primitive provider and its data, then bind the queries
the provider supports. Points, segments, triangles and user-defined primitives
share this API. A provider supplies bounding boxes, packs its data in leaf order,
and implements the geometric operations it needs. The bound queries accept JAX
arrays and support JIT, nested vmap and direct calls from Warp kernels.

## Build and query

```python
from sengi.bvh.warp import Points, Triangles, Euclidean, build_tree, make_nearest

primitive = Points(3)
source = primitive.data(coordinates)
# An indexed mesh uses the same build and query calls:
# primitive = Triangles()
# source = primitive.indexed(vertices, faces)

tree, geometry = build_tree(primitive, source, builder="sah")
nearest = make_nearest(primitive, Euclidean())
point, distance, original_id, status = nearest(tree, geometry, query_points)
```

Choose `builder="lbvh"` to partition by Morton order. Both builders prepare the
same binary tree representation. Reuse the same bound query with either tree
and its matching packed geometry. Use `build_boundary` when a query needs
connectivity or feature metadata. The [factory examples](#static-query-factories)
cover other queries and [indexed boundaries](#indexed-boundaries).

## Triangle mesh compatibility API and host builders

`BVHAccelerator.build` retains the named triangle mesh adapters
`bvh.sah.warp` and `bvh.snch.warp`. SNCH adds normal cones, silhouette
queries and boundary sampling. Without Warp, host construction remains
available and query discovery returns no backends.

The mesh adapter stores `tree`, `geometry`, `boundary`, `sampling` and
`coefficients` in its `WarpBVHPayload`. Existing named array reads such as
`payload.prim_ids` and `payload.mesh_vertices` still work. Code that
constructs, unpacks or updates the old flat tuple by field must migrate to
the nested fields. Rebuild serialized flat payloads from the source mesh.
The named SNCH adapter retains its historical split, edge IDs and cone
construction for callers that consume silhouette and sampling results.

The removed query registrations are `bvh.lbvh.jax`, `bvh.sah.jax`,
`bvh.snch.jax`, `kd.left_balanced.jax`, `kd.sah.jax` and
`octree.morton_explicit.jax`. Their host construction algorithms remain
available independently of query discovery:

```python
from sengi import Config
from sengi.bvh import build_lbvh, build_sah, build_snch
from sengi.kd import build_left_balanced, build_sah as build_kd_sah
from sengi.octree import build_morton_explicit

payload = build_lbvh(primitives, Config(leaf_size_threshold=8))
```

These functions retain their construction algorithms, primitive ordering,
payload layouts and build configuration behavior. They return build payloads
with JAX arrays and do not register query backends. The kd and octree query
accelerator wrappers have been removed; their builders and payload types remain.
The legacy accelerator `distance_bounds(box)` API is removed. Surface-level
`box_bounds` remains available through the conservative signed-distance Lipschitz
enclosure. BVH traversal bounds, query caps, and numerical validity checks remain
part of the Warp queries.
Construction and implicit-surface spatial trees are separate from this query
cleanup.

## Setup

Create the geometry environment from the lab conda channel and conda-forge:

```sh
mamba env create -f environment-warp.yml
mamba activate implicit-surfaces-warp
```

Outside the lab, build the pinned [Sengi package](https://github.com/gd193/sengi/tree/56d0256f6dac1a297527dabbf18fc53cca15f348) using its conda recipe and replace the lab channel in `environment-warp.yml` with the resulting local channel URL.

Validated with JAX/JAXLIB 0.10.2 and Warp 1.18.0. Warp is optional; imports
and backend discovery do not initialize CUDA. Set `CUDA_VISIBLE_DEVICES`
before starting Python. Use `JAX_PLATFORMS=cpu` for CPU execution and
`XLA_PYTHON_CLIENT_PREALLOCATE=false` to disable JAX's GPU memory reservation.

## Triangle mesh adapter example

```python
import jax
import jax.numpy as jnp

from surfaces.base import TriMesh
from sengi.primitives import TriMeshPrimitives
from sengi import BVHAccelerator, Config

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
from sengi.bvh.warp import (
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
`NearestOps.group_leaves` supplies the default traversal schedule; Euclidean point and
triangle queries use grouped descent. For Euclidean triangles, nearest recomputes
deferred bounds and kNN caches them. The built-in Euclidean triangle-nearest
kernel loads the three vertices for each visited candidate, computes its
bounding box, and projects onto surviving triangles in float32. It keeps the
closest point with the winning squared distance and original ID. If multiple
triangles attain the same float32 score, any of them may be returned.
The nearest query accepts a first hit exactly at its distance cap.

An external provider can use ordinary nearest operations, as the
[planar quad example](../tests/spatial_trees/test_warp_external_quad.py) does.
For a specialized single-nearest evaluator, the metric can supply
`NearestOps.bind_fused`. At factory creation it returns either `None` or
`FusedNearestOps` with a slot evaluator, a loaded-item evaluator for
predicates, and a conservative node bound in the same score units. The
Euclidean policy also accepts this specialization from a primitive through
`bind_fused_euclidean`. k-nearest and radius queries keep their own scoring
and tie rules.

The shared triangle payload also supports kNN and radius queries. Their
built-in Euclidean scorers use a longest-edge coordinate frame prepared by
`Triangles.indexed` in float64 and refine degenerate or near-contact cases
from the original vertices. User-supplied evaluators receive the triangle item.

The two leaf-ordered vertex arrays `b` and `c` occupy 24 bytes per triangle.
The `frame` array adds 64 bytes per triangle and uses 16 uint32 words to carry
mixed-precision coefficients even when JAX float64 arrays are disabled.
Manual `TrianglesArrays3` construction requires these three fields.
Rebuild from the indexed vertices after changing the mesh coordinates.
`make_nearest` and `make_knn` expose three independent
compile-time controls:

- `cache_bounds=True` stores the deferred node's lower-bound score in an eight-byte
  stack entry. `False` stores a four-byte node/tag entry and recomputes the score on pop.
- `packed_nodes=True` reads packed float4 bounds and topology. `False` reads the
  separate bounds, child indices and leaf-range arrays. Both layouts are already in
  the built payload; this selects access during traversal, without rebuilding.
- `group_leaves=None` uses the policy's default schedule. `True` groups descent
  before leaf processing; `False` uses the flat traversal loop. Neither choice
  depends on caching or node access. `None` and its resolved boolean share a kernel.

These arguments apply to metrics and user callbacks alike. For example,
`make_knn(primitive, 16, callbacks, cache_bounds=False, packed_nodes=True,
group_leaves=False)` recomputes bounds while retaining packed reads and flat
scheduling. Changing one option leaves the other two alone. Older `cache_bounds=False` measurements also selected
indexed access and flat scheduling; reproduce that combination by setting all
three arguments explicitly. Silhouette queries explicitly use indexed access
because their feature metadata replaces the leaf ranges.

Treat these controls as workload-specific tuning options. Their performance
interacts with primitive size and overlap, primitive data size, evaluator and
bound cost, tree and leaf configuration, query distribution, `k`, and GPU
architecture. Point benchmarks do not establish optimal settings for other
primitives. Defaults remain unchanged; choose overrides from measurements of
the intended workload.

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
from sengi.bvh.warp import (
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
leaf predicates and classification operations. Each adapter provides
`prepare(primitive, data)` and `pack_core(core, data, tree, geometry)`. Preparation
returns its own host core schema; packing preserves that schema and reorders
per-primitive arrays using `tree.prim_ids`. Global arrays stay in their original
order. `data` is the original geometry and `geometry` is the packed geometry, so
an adapter can reuse shared arrays. Feature construction receives the unpacked
core and returns ready-to-use metadata, including any needed array conversions.
Generic construction preserves both custom schemas without replacing fields.
`MeasureOps` supplies local samples,
normals and densities, plus optional selection weights; use the same policy with
`prepare_boundary_sample` and `make_boundary_sample`. A custom core without
`primitive_measure` supplies `MeasureOps.weights(core)` in packed primitive order.
`RegionOps` separates
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
miss/empty geometry, and **-1** means invalid input or failure. Closest point
may return any face tied at the minimum computed distance and reports failure
when squared distances exceed float32 range. Equal ray parameters and equal
silhouette distances choose the smaller original ID.

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

Generic builders consume finite, ordered primitive bounds representable in
float32. The named mesh adapters also validate referenced vertices and
triangle indices. The SAH mesh adapter accepts degenerate triangles. The
SNCH mesh adapter rejects triangles with doubled area at or below `1e-30`
and edges with more than two incident faces. Winding determines normals.
Build separate absorbing/reflecting subset trees for mixed boundary conditions.

Topology, leaf ranges and packed indices are validated. Leaf size is an SAH
threshold; retained leaves can exceed it. Tree depth must fit
`max_stack_depth`, at most 64. Rebuild after geometry changes. Geometry refit and
automatic differentiation are not supported.

Warp's built-in `wp.jax_kernel` handles FFI, compilation and ownership. The
batching adapter (installed at `sengi/bvh/backends/warp_queries/ffi.py`)
flattens query axes without tiling geometry. Factories bind typed geometry and optional metadata through the same adapter;
existing triangle backend methods delegate to them. SNCH cones enclose whole edges.
Packed float4 nodes hold bounds and topology for cached nearest, ray, overlap and
classification queries.
Nearest queries can cache node bounds in eight-byte stack entries. The default
Euclidean triangle-nearest path, silhouette and Robin queries use four-byte
indices/tags and recompute bounds on pop. FFI arrays are contiguous.
Rebuild the tree when changing bounds or topology.

Adapted ray helpers retain Warp's
Apache-2.0 license (installed at `sengi/bvh/backends/licenses/warp-LICENSE.txt`).

Native C++ lives in headers (installed at `sengi/bvh/backends/native/query.h`);
Python keeps typed one-line bindings. The query factory attaches include paths
and every header as a Warp build dependency. Header edits trigger recompilation
in a new process; reapply module options when editing within a running process.
Primitive, dimension and metric specialization still happens during JIT compilation.

Format and lint the headers with their local Clang configurations:

```sh
native_dir=$(python -c "from importlib.resources import files; print(files('sengi') / 'bvh/backends/native')")
warp_native=$(python -c 'from pathlib import Path; import warp; print(Path(warp.__file__).parent / "native")')
clang-format --dry-run --Werror "$native_dir"/*.h
clang-tidy "$native_dir/query.h" -- -x c++ -std=c++17 \
    -Wno-pragma-once-outside-header -isystem "$warp_native"
```

## Validation and timing

```sh
python -m pytest -q tests
# GPU validation also needs the CPU device for explicit CPU-interop checks:
JAX_PLATFORMS=cuda,cpu python -m pytest -q tests/spatial_trees
# Host builders remain testable independently of query registration:
JAX_PLATFORMS=cpu python -m pytest -q tests/spatial_trees/test_build_invariants.py

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

Recorded benchmark reports are historical provenance and retain their original
source hashes. Current harnesses still hash retained `sah_jax.py` and
`snch_jax.py` construction code where used; those filenames do not indicate
pure-JAX query execution. Optimized Warp matrix, device and external-reference
benchmark entry points remain available.

## Nearest queries with user callbacks

For a custom score, pass both callbacks instead of a metric:

```python
nearest = make_nearest(primitive, evaluate=evaluate, lower_bound=lower_bound)
point, score, original_id, status = jax.jit(nearest)(
    tree, data, queries, max_distance=score_cap,
)
```

`evaluate(query, item)` returns `(closest_point, score)`, minimizing the score
on that item. `lower_bound(query, lo, hi)` returns a conservative lower bound
on every item's score in the box. Both must be Warp device functions decorated
with `@wp.func` (or compatible Warp native functions), not arbitrary Python or
JAX functions. The query uses `primitive.point_type` and has trailing shape
`(primitive.dimension,)`. Box endpoints are always `wp.vec3`; 2D boxes have zero
z. The returned closest point uses the primitive's point type.

Candidate scores must be finite and nonnegative. Bounds and candidate scores
must use the same units. An overestimated bound can prune the true winner and
silently give a wrong result; zero is a safe bound for nonnegative scores but
reduces pruning. Bounds must cover the entire primitive, not only its center.
Choose formulas that remain conservative under floating-point rounding; callback
mode adds no Euclidean rounding slack. Negative conservative bounds are allowed,
although evaluated candidate scores must remain nonnegative.

Callback mode returns **raw scores**, and `max_distance` is a raw score cap:
there is no square root or squaring conversion. A squared Euclidean callback
returns 9 for a separation of 3; use a cap of 9 to include it. The cap is closed,
so an exact equality is accepted. Existing `make_nearest(primitive, Euclidean())`
continues to return Euclidean distances and accept distance caps.
Both callbacks are required, and neither may be combined with `metric`.

The result retains the normal nearest-query status convention: 1 for a hit,
0 for a miss or empty geometry, and -1 for invalid input/failure. Misses return
score infinity and ID -1. In callback mode, equal computed scores choose the smaller original ID.
Negative or NaN caps and nonfinite queries are invalid; an infinite cap is allowed.
A nearest query also returns status -1 if its selected closest point contains
NaN or infinity. This applies to custom result policies before their result
callback runs.

The complete example below implements two extensions using only public imports:

- A weighted squared point score with weights `(0.01, 16)`. At the origin,
  `(5, 0)` wins with score `0.25`, while ordinary Euclidean distance prefers
  `(0, 1)`. The box callback applies the same weights to coordinate gaps.
- A custom solid-ball primitive with center/radius arrays, host bounds/packing,
  a Warp item loader, and device bounds. Its evaluator projects onto the ball
  (returning the query for interior points); its bound uses squared distance
  to the enclosing box. No `closest_euclidean` hook or metric binding is needed.

Save the following example as `nearest_callbacks_example.py` and run from the
repository root with the project's Python environment:

```sh
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' XLA_PYTHON_CLIENT_PREALLOCATE=false \
  WARP_CACHE_PATH=/tmp/nearest-callback-example-warp \
  python nearest_callbacks_example.py
```

Create the callbacks and factory once, outside JIT. The factory statically
specializes the existing walker on callback identity and primitive; repeated
calls with the same specialization reuse its kernel. Changing either callback
creates a separate specialization. The example uses `@wp.func(module="unique")`
for functions created inside a Python factory so each closure keeps its own
weights and Warp identity. Geometry and queries remain dynamic, and JIT and
nested `vmap` work as for built-in metrics.

```python
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp

from sengi.bvh.warp import (
    NearestCallbacks, Points, PrimitiveSpec, WarpFunction, WarpType,
    build_tree, make_nearest, warp_func, warp_struct,
)

def weighted_callbacks(dimension, weights):
    Vec = wp.vec2 if dimension == 2 else wp.vec3
    weights = Vec(*weights)

    @wp.func(module="unique")
    def evaluate(q: Vec, item: Any):
        d = q-item.a
        score = float(0.0)
        for axis in range(wp.static(dimension)):
            score += d[axis]*d[axis]*wp.static(weights)[axis]
        return item.a, score

    @wp.func(module="unique")
    def lower_bound(q: Vec, lo: wp.vec3, hi: wp.vec3):
        score = float(0.0)
        for axis in range(wp.static(dimension)):
            d = wp.max(wp.max(lo[axis]-q[axis], 0.0), q[axis]-hi[axis])
            score += d*d*wp.static(weights)[axis]
        return score

    return evaluate, lower_bound


@warp_struct
class BallData:
    centers: wp.array(dtype=wp.vec3)
    radii: wp.array(dtype=wp.float32)


@warp_struct
class Ball:
    center: wp.vec3
    radius: float


@warp_func
def ball_load(data: BallData, slot: int) -> Ball:
    item = Ball()
    item.center = data.centers[slot]
    item.radius = data.radii[slot]
    return item


@warp_func
def ball_bounds(item: Ball) -> tuple[wp.vec3, wp.vec3]:
    return item.center-wp.vec3(item.radius), item.center+wp.vec3(item.radius)


@warp_func
def ball_evaluate(query: wp.vec3, item: Ball) -> tuple[wp.vec3, float]:
    delta = query-item.center
    length = wp.length(delta)
    point = query
    if length > item.radius:
        point = item.center+delta*(item.radius/length)
    return point, wp.length_sq(query-point)


@warp_func
def ball_lower_bound(query: wp.vec3, lo: wp.vec3, hi: wp.vec3) -> float:
    delta = wp.max(wp.max(lo-query, wp.vec3(0.0)), query-hi)
    return wp.length_sq(delta)


class BallArrays(NamedTuple):
    centers: jax.Array
    radii: jax.Array


@dataclass(frozen=True)
class Balls:
    dimension: int = 3
    point_type: WarpType[wp.vec3] = wp.vec3
    data_type: WarpType[BallData] = BallData
    load: WarpFunction[[BallData, int], Ball] = ball_load
    bounds_of_item: WarpFunction[[Ball], tuple[wp.vec3, wp.vec3]] = ball_bounds

    def bounds(self, data: BallArrays) -> tuple[np.ndarray, np.ndarray]:
        centers, radii = np.asarray(data.centers), np.asarray(data.radii)
        return centers-radii[:, None], centers+radii[:, None]

    def pack(self, data: BallArrays, ids: np.ndarray | jax.Array) -> BallArrays:
        return BallArrays(data.centers[ids], data.radii[ids])


def main():
    primitive = Points(2)
    evaluate, lower_bound = weighted_callbacks(2, (.01, 16.))
    tree, data = build_tree(primitive, primitive.data([[0., 1.], [5., 0.]]))
    nearest = jax.jit(make_nearest(primitive, evaluate=evaluate, lower_bound=lower_bound))
    point, score, index, status = nearest(tree, data, jnp.zeros(2), max_distance=.25)
    np.testing.assert_allclose(point, [5., 0.])
    np.testing.assert_allclose(score, .25)
    assert int(index) == 1 and int(status) == 1
    print("Weighted point:", point, "raw score:", score)

    primitive: PrimitiveSpec[BallArrays, BallData, Ball, wp.vec3] = Balls()
    raw = BallArrays(jnp.array([[0., 0., 0.], [4., 0., 0.]]), jnp.array([1., .5]))
    tree, data = build_tree(primitive, raw)
    callbacks: NearestCallbacks[Ball, wp.vec3] = NearestCallbacks(ball_evaluate, ball_lower_bound)
    query = make_nearest(primitive, callbacks)
    assert query.kernel is make_nearest(
        primitive, evaluate=ball_evaluate, lower_bound=ball_lower_bound).kernel
    nearest = jax.jit(query)
    point, score, index, status = nearest(tree, data, jnp.array([2., 0., 0.]), max_distance=1.)
    np.testing.assert_allclose(point, [1., 0., 0.])
    np.testing.assert_allclose(score, 1.)
    assert int(index) == 0 and int(status) == 1
    print("Solid ball:", point, "raw squared distance:", score)


if __name__ == "__main__":
    main()
```

### Typed extension contract

The public `PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT]` protocol
connects host bounds/packing, the device payload, the loaded item, and query
points. An external class can implement it structurally, as `Balls` does above;
`Primitive[...]` also provides an immutable descriptor accepting `host_bounds`
and `host_pack` functions. Neither path requires registration or inheritance.
`NearestCallbacks[ItemT, PointT]` uses the same `bind(primitive)` contract as
`Euclidean` and `Angular`. Callback keyword arguments normalize to this policy,
so both spellings share the same cached query and kernel.

Use `@warp_struct` for concrete device structs and `@warp_func` on concretely annotated device functions to retain their
signature in Python checking. It returns the original `wp.Function`, with no
runtime wrapper. For an existing `@wp.func` or native function, annotate its
callable signature before passing it through `as_warp_function`; that bridge
checks Warp identity and preserves it. Avoid `Any` in an extension's item and
callback annotations: it would remove the connections the checker verifies.
Dynamic JAX array dimensions and dtypes still require runtime validation.

Focused strict checks cover the shared contracts and representative external
extensions, including expected failures for mismatched items, queries, scores,
bounds, and primitive/policy combinations. They do not change repository-wide
typing settings. With Pyright installed outside the shared Python environment:

```bash
PYRIGHT=/path/to/pyright \
PYRIGHT_PYTHONPATH="$(command -v python)" \
  python tests/typing/check_warp_api.py
```

The runner first requires zero diagnostics for valid fixtures, then requires
exact diagnostic rules on every marked negative line. Negative examples use
ordinary invalid expressions, without `type: ignore` or expected-error
suppression.

Built-in item structs are public runtime Warp types, so a typed callback can
annotate them directly. For indexed triangles, use `TriangleItem3.a`, `b`,
and `c` for vertex positions. The `e1` and `e2` fields are float32 edge
vectors; adding them back to `a` can round away part of a vertex coordinate.
For example, this evaluator compiles with `Points(3)`:

```python
from sengi.bvh.warp import PointItem3, NearestCallbacks, warp_func

@warp_func
def constant_score(query: wp.vec3, item: PointItem3) -> tuple[wp.vec3, float]:
    return item.a, 1.0

@warp_func
def constant_bound(query: wp.vec3, lo: wp.vec3, hi: wp.vec3) -> float:
    return 1.0

nearest = make_nearest(Points(3), NearestCallbacks(constant_score, constant_bound))
```

This score intentionally treats all points as tied; smallest original ID wins.
The compiled, strictly checked Euclidean-score example lives in
`tests/typing/warp_api/real_extension.py`, used directly by
`tests/spatial_trees/test_warp_typed_api.py`. Its small native math helper has a
narrow `.pyi` signature because upstream Warp vector arithmetic typing is
incomplete. The extension itself has no `Any` callback or item annotations.
Array fields use a `TYPE_CHECKING` indexing protocol with a concrete return type
and a runtime `wp.array(dtype=...)` annotation; the data/item classes and device
functions are the same objects exercised by the CPU tests.

### Typed policies across query families

Every query factory carries the primitive's host payload type into its returned
callable. Device callbacks retain separate geometry, item, point, state, and
metadata types. `RayOps[DeviceData, State]` supports both `intersect` and
`next_hit`; the latter returns `(State, float, int)`. All four ray modes use the
same contract. A primitive declares its default single-event callback through
`intersect`. Pass `ops=RayOps(...)` to the query factory for an explicit or
stateful policy. Equivalent default and explicit policies share a cached kernel.
`ClassificationOps[Point, DeviceData, DeviceMetadata]` connects
both metadata callbacks. When the host metadata record differs from the device
struct, pass `metadata_host_type=HostMetadata` to `make_classify`. That marker
only supplies host typing; it shares the same cached specialization and kernel.

`MeasureOps[Item, Point, HostCore]` connects sampling and host weights;
`RegionOps[Query, Point]` connects validity, conservative box overlap, and exact
sample membership. `RobinOps[Query, DeviceData, Item, DeviceCore, DeviceFeature]`
also types its optional data-aware callbacks and score conversions. Explicit
Robin host schemas can be supplied with `core_host_type` and
`feature_host_type`. A metric can implement the `RobinMetric.bind_robin(primitive)`
capability to supply its own conservative `RobinOps`; `Euclidean` uses this
same route. A metric without this capability must supply an explicit policy.
Boundary adapters declare sampling and Robin factories alongside classification
and silhouette operations. Star radius composes silhouette with ordinary JAX operations.
Overlap remains conservative primitive-AABB overlap.

The executable examples in `tests/typing/warp_api/query_extension.py` are used
unchanged by strict Pyright fixtures and CPU Warp compilation tests. For example,
this runs the external struct-state ray policy through the public API:

```python
import jax
import jax.numpy as jnp
from sengi.bvh.warp import RayOps, build_tree, make_ray_query
from tests.typing.warp_api.real_extension import ExternalArrays, ExternalPoints
from tests.typing.warp_api.query_extension import Cursor, next_hit

primitive = ExternalPoints()
tree, payload = build_tree(primitive, ExternalArrays(jnp.array([[0., 0., 0.]])))
query = make_ray_query(primitive, "all", ops=RayOps(state_type=Cursor, next_hit=next_hit))
times, ids, count, status = jax.jit(query, static_argnames=("max_hits",))(
    tree, payload, jnp.array([[-1., 0., 0.]]), jnp.array([[1., 0., 0.]]), max_hits=4)
assert int(count[0]) == 2
```

The deliberately small external point policy emits two events per primitive;
it illustrates iterator state without relying on a built-in intersection
implementation. The same module includes concrete classification, sampling,
region, Robin, silhouette-adapter, and built-in `PointsData3` ray callbacks.
Run all of these examples on CPU with:

```bash
JAX_PLATFORMS=cpu python -m pytest tests/spatial_trees/test_warp_query_contracts.py -q
```

Built-in device payloads (`PointsData3`, `SegmentsData2`, `TrianglesData3`, and
others) and their host `*Arrays` records are public. A built-in primitive and an
external primitive use identical callback contracts; a callback for the wrong
payload is a type error. Optional capability availability is still validated
when constructing a query. Typed callback compatibility does not imply that an
optional capability is present.

## Generic k-nearest queries

`make_knn(primitive, k, metric=None, *, evaluate=None, lower_bound=None,
stack_capacity=64, block_dim=64, cache_bounds=True, packed_nodes=True,
group_leaves=None, predicate=None, return_points=False)` specializes
one fused traversal kernel for a positive static **Python integer** `k` (booleans
are rejected). Construct the factory outside JIT. `k` can exceed the number of
primitives; each primitive contributes at most one candidate.

```python
import jax
import jax.numpy as jnp
from sengi.bvh.warp import Points, Euclidean, build_tree, make_knn

primitive = Points(2)
tree, geometry = build_tree(primitive, primitive.data([[0., 0.], [2., 0.], [-2., 0.]]))
knn = jax.jit(make_knn(primitive, 4, Euclidean()))
distances, ids, count, status = knn(tree, geometry, jnp.array([[1., 0.]]), max_distance=1.)
# distances = [[1., 1., inf, inf]], ids = [[0, 1, -1, -1]]
# count = [2], status = [1]
```

A query has shape `batch + (primitive.dimension,)` for ordinary point policies;
custom policies retain their own query shape. Results are `(values, ids, count,
status)`, with shapes `batch + (k,)`, `batch + (k,)`, `batch`, and `batch`.
Candidates sort by internal score, then **original primitive ID**, independently
of BVH packing. Unfilled entries contain positive infinity and ID -1. `count`
is the number returned, at most `k`. Status is 1 when any candidate is found,
0 for no candidates (including an empty tree), and -1 for invalid input,
numerical failure, or stack exhaustion. Failure clears the entire row to padding
and sets count to zero. Nonfinite queries, negative/NaN caps, and evaluated scores
that are negative or nonfinite are invalid; positive-infinite caps are allowed.
Caps broadcast over query batch dimensions and include exact equality.

An explicit metric or **both** callbacks are required; there is no implicit
Euclidean metric. `NearestPolicy` and `NearestCallbacks` are exactly the policies
used by `make_nearest`. Explicit metrics apply their `to_score` cap conversion
and `to_distance` result conversion. Callback shorthand uses raw score units for
both outputs and caps. For example, a squared-distance callback needs cap 9 to
include separation 3. The evaluator returns `(point, score)` for each primitive.
By default, kNN ignores the point and ranks the score. A predicate can reject candidates.

Set the static factory option `return_points=True` to replace IDs with each
neighbor's closest-point coordinates:

```python
knn_points = jax.jit(make_knn(primitive, 4, Euclidean(), return_points=True))
distances, points, count, status = knn_points(tree, geometry, jnp.array([[1., 0.]]), max_distance=1.)
# points = [[[0., 0.], [2., 0.], [0., 0.], [0., 0.]]]
# count = [2]; the last two points are padding.
```

The coordinate array has shape `batch + (k, primitive.dimension)`; other outputs
are unchanged. Coordinates come from the evaluator, so segments, triangles and
custom primitives return their projected closest points rather than primitive
centers. Ordering still uses score and original primitive ID, including ties.
Missing entries contain zero coordinates; use `count` and `status` to distinguish
padding from a valid point at the origin. A nonfinite selected coordinate fails
and clears the whole row to infinity scores, zero coordinates, count 0 and
status -1.

Above k=64, candidates use a four-child max heap; smaller lists use unrolled
sorted storage. This selection is internal and applies to both output modes.

Point output retains one extra int32 packed primitive slot per candidate, then
reevaluates the selected primitives inside the same fused kernel. Callbacks must
be deterministic and free of side effects: the evaluator runs again for each
returned neighbor. This avoids moving coordinate vectors through candidate
insertion and sorting, at the cost of up to `k` extra evaluator calls and larger
output arrays. The default ID output has no slot storage or reevaluation.

For example, reuse the `weighted_callbacks` functions defined above with either
factory; changing the callbacks creates a new compiled specialization:

```python
evaluate, lower_bound = weighted_callbacks(2, (.01, 16.))
knn = jax.jit(make_knn(primitive, 4, evaluate=evaluate, lower_bound=lower_bound))
scores, ids, count, status = knn(tree, geometry, jnp.array([[1., 0.]]), max_distance=.01)
# scores = [[.01, .01, inf, inf]], ids = [[0, 1, -1, -1]]
```

The callback requirements in the preceding section apply unchanged: callbacks
are Warp device functions; the bound must conservatively bound every primitive
in its box, in the same units as the evaluated score. An overestimated bound can
silently prune a valid neighbor. Zero is safe for nonnegative scores but prunes
less. Callback shorthand and `NearestCallbacks(evaluate, lower_bound)` share the
same cached kernel. JIT and nested `vmap` preserve the single fused custom call;
selection and ordering happen inside that traversal, without repeated nearest
queries or a separate sorting kernel.

Candidate storage uses an unrolled register list for k up to 64 and an in-place
packed heap above that. Larger k can increase register pressure and local-memory
traffic. `cache_bounds=False` reduces traversal storage by recomputing deferred
bounds; whether that helps depends on the callback cost and workload. See the
[GPU comparison report](benchmarks/warp_knn.md) for measured settings and limits.


## Static kernel composition

The query implementation lives in `bvh/backends/warp_queries`. Python factories
resolve options and capabilities into concrete Warp functions before compilation.
Immutable host bundles carry those functions and their device state types. The
bundles themselves never enter device execution. The public `bvh.warp` factories
and the contracts for external primitives and metric callbacks remain the entry
points; no registration or inheritance is required for extensions.

Responsibilities are separated as follows:

- `contracts.py`, `domain.py`, and `metrics.py` define typed extension contracts
  and the existing scoring, ray, and classification policies.
- `primitives/` contains builtin item/data types, host preparation, item loading,
  and geometry operations. Item loaders are selected in Python by primitive kind.
- `io/arrays.py` owns native array addressing; `io/nodes.py` selects indexed or
  packed node access. `io/results.py` selects kNN ID or coordinate emission,
  including output validation and failure clearing.
- `traversal/callbacks.py` adapts query callbacks; `traversal/deferred.py` owns
  deferred stack entries and restoring nodes; `traversal/scheduling.py` selects
  flat or grouped-leaf traversal. Packed ray entries, cached node bounds, and
  recomputed node bounds are distinct deferred-state policies.
- `selection/candidates.py` binds kNN candidate operations and their native ABI.
  Candidate storage remains an unrolled list through k=64 and a heap above it.
- `queries/` assembles each query's state, operations, traversal, and output.
  Ray modes and optional Robin operations resolve during Python construction.
- `launch.py` supplies the thin array kernel entry point; `ffi.py` handles JAX
  batching. `native.py` centralizes native registration and build dependencies.
  The native C++ headers remain in `backends/native/`.

Node access (`packed_nodes`), deferred bound storage (`cache_bounds`), and leaf
scheduling (`group_leaves`) are independent choices for nearest/kNN. The
internal packed ray/overlap traversal contract also changes what is stored on
the stack; it is not another spelling of packed node access. Resolution checks
unsupported combinations during construction. Local `wp.static` decisions for
fixed dimensions, loop unrolling, and small numerical variants are intentional.
The cached deferred-pop loop stays in each scheduler: moving its early exits
into a helper changed CUDA reconvergence and caused a measured slowdown. The
selected stack operations and node metadata accessor remain separate components.

Bound factory queries provide a host-only `describe()` method. For example:

```python
query = make_knn(Points(), 16, Euclidean(), packed_nodes=False)
print(query.describe())
# components: node_access=indexed, scheduling=grouped,
# deferred=cached_bound, selection=unrolled, output=ids
```

This does not compile or launch a kernel. Wrapper queries such as silhouette
expose their underlying bound query through `.query`. Descriptions identify the
chosen components; they are not performance predictions or device cache keys.
Factories cache normalized selections and concrete callback identities. Host
schema markers remain excluded from kernel specialization.

Changes to an access, traversal, selection, or output component must preserve
fused execution and be checked against matched query results and compiled GPU
resources. Extracting a function can affect CUDA inlining and register lifetimes,
so modular source alone does not guarantee identical performance. Benchmark
provenance includes the nested package's sources as well as native headers.
Host tree construction algorithms and payload layouts are independent of this
query composition layer.

## Calling queries from a Warp kernel

Every query factory exposes a `.device` Warp function. Select the primitive,
metric/callbacks, traversal options, and **static `k`** on the host, then capture
that function in your own `@wp.func` or `@wp.kernel`. The function performs one
query per invocation; it never launches another kernel or invokes JAX.
Nearest, kNN, classification, overlap, sampling, and Robin reuse the same device
implementation as their host query object. Ray adapters expose only the buffers
used by the selected mode. Silhouette prepares the feature-tree view on device;
star radius composes silhouette and clipping on device.

Register the query dependencies with `configure_query_kernel(kernel, *queries)`
before compiling or launching the caller's kernel. Include every query used,
including indirect calls through other Warp functions. Registration merges native
include paths, header cache dependencies, ABI declarations, and required unroll
limits into the caller's module. It retains existing native preambles and other
compiler settings. Multiple query types and static `k` values can share a kernel;
repeating registration is safe. Use `enable_backward=False`: these operations do
not provide differentiation rules.

This complete example consumes a nearest result inside the caller's kernel:

```python
import numpy as np
import warp as wp
from sengi.bvh.warp import (
    DeviceTree, Euclidean, Points, build_tree, configure_query_kernel, make_nearest,
)

primitive = Points(3)
tree, geometry = build_tree(primitive, primitive.data([[0., 0., 0.], [3., 0., 0.]]))
query = make_nearest(primitive, Euclidean())
nearest = query.device
Geometry = primitive.data_type

@wp.kernel(module="unique", enable_backward=False)
def evaluate(tree: DeviceTree, geometry: Geometry,
             points: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    i = wp.tid()
    closest, distance, primitive_id, status = wp.static(nearest)(points[i], wp.inf, tree, geometry)
    if status == 1:
        out[i] = distance + wp.length(closest)
    else:
        out[i] = wp.inf

configure_query_kernel(evaluate, query)
device = wp.get_preferred_device()

# Upload the already packed host arrays. Allocation/copying happens outside the kernel.
def upload(schema, arrays):
    payload = schema()
    for name, field in schema.vars.items():
        setattr(payload, name, wp.array(np.asarray(getattr(arrays, name)),
                                       dtype=field.type.dtype, device=device))
    return payload

points = wp.array([[1., 0., 0.]], dtype=wp.vec3, device=device)
out = wp.empty(1, dtype=float, device=device)
wp.launch(evaluate, dim=1,
          inputs=[upload(DeviceTree, tree), upload(Geometry, geometry), points],
          outputs=[out], device=device)
np.testing.assert_array_equal(out.numpy(), [1.])
```

Existing Warp arrays can be assigned directly to the payload fields. For shared
JAX arrays, use `wp.from_jax(array, dtype=field.type.dtype)` and manage ownership
and stream synchronization according to Warp's interop contract. The query API
does not upload arrays or synchronize runtimes inside a device call.

Device signatures use scalar query values and device structs, with explicit caps:

| Factory | `.device(...)` arguments | Return |
|---|---|---|
| `make_nearest` | `query, cap, tree, geometry` | point, value, original ID, status |
| `make_knn` | `query, cap, tree, geometry, values, neighbors, row` | count, status |
| `make_overlap` | `lower, upper, tree, geometry` | overlap as int32, status |
| `make_classify` | `point, tree, geometry, core` | sign, status |
| `make_silhouette` | `query, cap, tree, feature_metadata` | point, value, original feature ID, status |
| `make_star_radius` | `point, cap, tree, feature_metadata` | radius, status |
| `make_robin_radius` | `query, cap, tree, geometry, core, feature_metadata, coefficients` | radius, status |
| `make_boundary_sample` | `query, radius, random_vec3, tree, geometry, sampling_metadata` | point, normal, PDF, original ID, status |

kNN writes `values` and `neighbors`, both **dense contiguous `(rows, k)` Warp
arrays with exactly the specialized `k` columns**. Values have float32 dtype;
neighbors have int32 dtype for IDs or the primitive's vector dtype for
`return_points=True`. Coordinate output therefore has NumPy shape `(rows, k, D)`.
`k` is never a device argument. Count, padding, caps, custom score units, tie
ordering, and failure clearing follow the host API's contract.

Ray device functions write caller-provided buffers and return no value:

```text
first/all:
    device(origin, direction, min_t, max_t, excluded_id, tree, geometry,
           times, ids, counts, statuses, row)
any/count:
    device(origin, direction, min_t, max_t, excluded_id, tree, geometry,
           counts, statuses, row)
```

For `first`, all buffers are one-dimensional; `counts[row]` is the 0/1 hit flag.
For `all`, `times` and `ids` have matching `(rows, max_hits)` shapes; counts and
statuses remain one-dimensional. For `any`, counts contain 0/1. Time buffers use
float32 and ID/count/status buffers use int32. Pass `excluded_id=-1` for no
exclusion; other IDs take effect only when the specialization enables exclusion.
All-hit overflow still reports the total count and status -2.

The caller supplies valid device structs, matching metadata, dense contiguous
arrays on the same device, in-range row indices, and nonoverlapping writable rows
for concurrent invocations. These are device-call preconditions, not per-query
host checks. The query retains its numerical validation and status reporting.
No intermediate output arrays are needed for scalar-return queries; kNN and ray
modes use the supplied buffers inside the caller's kernel.

Typed `.device` signatures distinguish host payloads from device structs and
one-dimensional arrays from `DeviceArray2D`. Buffer extents and some dynamically
resolved metadata/output variants remain runtime Warp contracts, as with the
existing host factories.

## Radius queries and public traversal policies

`make_radius` visits every qualifying primitive under a **closed** threshold.
`mode="any"` stops at the first match; `mode="count"` counts every match.
Both return `(value, status)` through JAX, support JIT/vmap, and expose
`.device(query, cap, tree, geometry)` for composition in a caller's kernel.
Status is `1` for matches, `0` for none, and `-1` for failure. Failed queries
return zero. Counts use int32 and report failure rather than wrap on overflow.

```python
from sengi.bvh.warp import (
    Points, Euclidean, build_tree, make_radius,
)

primitive = Points()
tree, geometry = build_tree(primitive, primitive.data([[0., 0., 0.], [1., 0., 0.]]))
count = make_radius(primitive, Euclidean(), mode="count")
values, status = count(tree, geometry, [[0., 0., 0.]], radius=1.0)
# values=[2], status=[1]: equality with the threshold is included.
```

A metric converts the cap using its existing score policy. With the
`evaluate=...`, `lower_bound=...` shorthand, caps and evaluator scores have
identical units. Predicates, custom primitives, and `cache_bounds`,
`packed_nodes`, and `group_leaves` select the same traversal components as nearest queries.
Radius defaults to `cache_bounds=False`, retaining the smaller deferred-node stack;
callers can select cached bounds explicitly. Radius queries
are independent of the static `k` used by kNN; they do not allocate neighbor lists.

`RadiusAccumulator(state_type, initialize, update, finish)` supplies a custom
reduction for `make_radius(..., mode="accumulate", accumulator=...)`. Its
`RadiusReduction` is **device-only**. Callbacks are statically compiled:

- `initialize(query, score_cap) -> State` initializes caller-owned data.
- `update(query, item, point, score, original_id, state: WarpRef[State]) -> None`
  processes a qualifying primitive; it can set `state.stop` to finish early.
- `finish(state) -> Result` returns a Warp scalar, vector, or struct.

State must declare `limit: float`, `failed: bool`, `stop: bool`, and
`matched: bool` in addition to its own fields. The factory sets the score limit
and match flag. Keep the limit unchanged for a fixed-radius reduction. On
invalid input, `finish` receives a zero-initialized state and status is `-1`;
on traversal failure, it receives the partial state and status is `-1`.
Results with failure status must not be treated as complete reductions.
No host representation is inferred for arbitrary custom result structs.

For more general queries, `make_traversal(Context, State, bounds, visit, ...)`
exposes the existing walker as `.device(tree, context, state)`. A caller-declared
`@warp_struct` state requires `limit: float`, `failed: bool`, and `stop: bool`.
The `make_traversal_state(name, extra_fields)` helper can create one dynamically;
declaring the struct explicitly gives static checkers its field types.

```text
bounds(context, tree, node, lower_vec3, upper_vec3, state: WarpRef[State])
    -> conservative_score, eligible, tag
visit(context, tree, node, begin_slot, end_slot, tag, state: WarpRef[State])
    -> None
```

The leaf range contains packed primitive slots; `tree.prim_ids[slot]` gives
original IDs. Initialize `state.limit` in the same units as the node scores.
Callbacks may lower it, set `stop`, or flag `failed`. Bounds must remain
conservative under float32 rounding. NaN bounds and stack overflow flag failure.
With `cache_bounds=False`, supply
`resume(context, tree, node, lower, upper) -> score` to recompute deferred bounds.
Cached traversal requires no resume callback. Layout, ordering, grouping, and
stack capacity are static factory options.

Use `WarpRef[State]` with `@warp_func` for typed mutable-reference parameters.
It is the actual `wp.ref[State]` annotation at runtime and the underlying State
for type checking; it introduces no callable wrapper or device work. Register
traversals and reductions with `configure_query_kernel(kernel, *queries)` before
compiling caller-owned kernels, just like the other device-callable queries.

## Exact box intersection and region classification

`make_overlap` remains a primitive-AABB broad-phase query. For geometric
intersection, pair a primitive with a `RegionIntersectionPolicy`:

```python
from sengi.bvh.warp import (
    Points, build_tree, make_region, point_region_policy,
)

primitive = Points(3)
tree, geometry = build_tree(primitive, primitive.data([[0., 0., 0.], [1., 0., 0.]]))
query = make_region(primitive, point_region_policy(3), mode="count")
count, status = query(tree, geometry, [[0., 0., 0.]], [[1., 0., 0.]])
# count=[2], status=[1]
```

Builtin policies are `point_region_policy(2|3)`, `segment_region_policy(2|3)`,
and `triangle_region_policy()`. An extension supplies
`RegionIntersectionPolicy(intersects)`, where the typed Warp callback is
`intersects(geometry, packed_slot, lower_vec3, upper_vec3) -> bool`.
Node bounds cull candidates; the callback performs the full primitive test.
The slot indexes packed geometry, not the original primitive ID. Callbacks
can load original indexed vertices or custom geometry without reconstructing
them from rounded derived values. Builtin triangle tests use the original
indexed vertices, consistent with the tree bounds. Builtin segment and triangle tests
use float64 intermediates to avoid float32 overflow/underflow at extreme scales.
Boxes are closed: face/edge/vertex contact and zero-width boxes can intersect.

`mode="any"` returns a boolean and stops at the first exact intersection.
`mode="count"` returns an int32 count. Optional positive static
`count_limit` saturates the count and permits early exit; without a limit all
eligible items are examined. Invalid boxes or traversal failure return `(0,-1)`;
valid queries return status `1`, including queries with zero intersections.
The `.device(lower, upper, tree, geometry)` result uses int32 for both modes.

`make_region_classify(primitive, region_policy, classification_policy, ...)`
combines geometric overlap with an existing `ClassificationOps` policy.
Its host call is `(tree, geometry, boundary_or_core, lower, upper)`, and its
device call adds the device metadata after geometry. The classification is
`+1` inside, `-1` outside, or `0` intersecting the surface. Valid classification
has status `1`; invalid boxes or metadata and failed traversal have status `-1`.
The policy must describe a closed surface, including when a box touches it.
If no surface intersects the box, point classification at its lower corner
classifies the entire connected box. An empty valid surface is outside.

Supply `metadata_host_type=YourArraySchema` when host classification metadata
has a different type from the device struct. This is a typing marker and does
not create another device specialization. Region queries use the public typed
traversal mechanism and compile into one fused call, including classification.

## Selected-result policies

Pass `result_policy=ResultPolicy(Result, build)` to `make_nearest` or
`make_knn` to construct a caller-defined Warp struct for each selected neighbor.
The typed callback is:

```text
build(query, item, original_id: int, selected_point, reported_value: float)
    -> Result
```

`reported_value` uses the metric's output conversion; callback-only metrics
return raw score units. The item comes from its packed primitive slot, while
`original_id` remains the input ID. Result construction runs only after selection
(and sorting for kNN), once per selected valid neighbor. kNN reloads selected
items and reevaluates their closest points, as coordinate-output kNN already
does. Callbacks must therefore be deterministic and must not modify geometry.

```python
import warp as wp
from sengi.bvh.warp import (
    PointItem3, Points, Euclidean, ResultPolicy, make_knn, warp_func, warp_struct,
)

@warp_struct
class Hit:
    original_id: int
    position: wp.vec3
    distance: float

@warp_func
def build_hit(query: wp.vec3, item: PointItem3, index: int,
              point: wp.vec3, value: float) -> Hit:
    hit = Hit()
    hit.original_id = index
    hit.position = point
    hit.distance = value
    return hit

query = make_knn(Points(), 8, Euclidean(), result_policy=ResultPolicy(Hit, build_hit))
# Inside a caller-owned Warp kernel:
# count, status = query.device(point, cap, tree, geometry, values, hits, row)
# values: float32 Warp array (rows, 8); hits: Hit Warp array (rows, 8).
```

`IdPoint()` is a builtin policy producing a struct with `id` and `point` fields.
Its actual Warp struct descriptor is `query.result_policy.result_type`, suitable
for allocating `wp.empty((rows, k), dtype=...)`. `IdPointResult[PointT]` describes
these fields to static type checkers.

`SelectedNearestQuery.device(query, cap, tree, geometry)` returns `(result,status)`.
`SelectedKnnQuery.device(query, cap, tree, geometry, values, results, row)` writes
dense contiguous `(rows,k)` buffers and returns `(count,status)`. `k` remains
static. Misses and unused entries have a zero-initialized result struct; kNN
values are `inf`. On failure kNN clears the complete row and returns `(0,-1)`.
Use count/status to distinguish valid ID zero from empty struct padding.

Custom struct queries are device-only and cannot be called as JAX host queries.
Their types deliberately have no host `__call__`; users can write individual
result fields to ordinary arrays in their own fused kernel. Register native
dependencies with `configure_query_kernel` before launch. Default nearest/kNN
host and device outputs remain unchanged. `return_points=True` and
`result_policy` are mutually exclusive for kNN.
