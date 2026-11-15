# GPU BVH queries with Warp and JAX

`bvh.sah.warp` builds static binary SAH trees on the host and executes queries
with Warp on CPU or GPU. `bvh.snch.warp` adds normal cones, silhouette queries
and boundary sampling. Geometry stays shared under JIT and nested `vmap`.

## Query scope and retained builders

The supported explicit query backends are `bvh.sah.warp` and `bvh.snch.warp`.
Use `BVHAccelerator`, `Config` and the static factories in
`surfaces.explicit.accelerators.bvh.warp` as shown below. JAX arrays, pytrees,
FFI, `jax.jit` and nested `jax.vmap` remain supported, including Warp execution
on CPU. There is no pure-JAX explicit tree traversal fallback. Without Warp,
query discovery returns no backends.

The removed query registrations are `bvh.lbvh.jax`, `bvh.sah.jax`,
`bvh.snch.jax`, `kd.left_balanced.jax`, `kd.sah.jax` and
`octree.morton_explicit.jax`. Their host construction algorithms remain
available independently of query discovery:

```python
from surfaces.explicit.accelerators import Config
from surfaces.explicit.accelerators.bvh import build_lbvh, build_sah, build_snch
from surfaces.explicit.accelerators.kd import build_left_balanced, build_sah as build_kd_sah
from surfaces.explicit.accelerators.octree import build_morton_explicit

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
`NearestOps.group_leaves` supplies the default traversal schedule; Euclidean point
queries use grouped descent. `make_nearest` and `make_knn` expose three independent
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
scheduling. Changing one option leaves the other two alone. Defaults retain the
previous default kernels. Older `cache_bounds=False` measurements also selected
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
score infinity and ID -1. Equal computed scores choose the smaller original ID.
Negative or NaN caps and nonfinite queries are invalid; an infinite cap is allowed.

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

from surfaces.explicit.accelerators.bvh.warp import (
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
annotate them directly. For example, this evaluator compiles with `Points(3)`:

```python
from surfaces.explicit.accelerators.bvh.warp import PointItem3, NearestCallbacks, warp_func

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
from surfaces.explicit.accelerators.bvh.warp import RayOps, build_tree, make_ray_query
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
group_leaves=None, predicate=None)` specializes
one fused traversal kernel for a positive static **Python integer** `k` (booleans
are rejected). Construct the factory outside JIT. `k` can exceed the number of
primitives; each primitive contributes at most one candidate.

```python
import jax
import jax.numpy as jnp
from surfaces.explicit.accelerators.bvh.warp import Points, Euclidean, build_tree, make_knn

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
include separation 3. The evaluator returns `(point, score)` for each primitive;
kNN ignores the point and ranks the score. A predicate can reject candidates.

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
