# Triangle queries with Warp and JAX

`bvh.sah.warp` and `bvh.snch.warp` build static binary SAH trees on the CPU
and run batched queries on the GPU. SNCH adds normal cones, silhouette edges
and boundary sampling for reflecting boundaries. Geometry stays explicit and
shared across queries, including under nested `vmap`.

The repository also retains the JAX BVH, kd-tree and octree research backends.

## Setup

Install into an environment with JAX, NumPy and trimesh:

```sh
python -m pip install -r requirements-warp.txt
```

Validated versions: JAX/JAXLIB 0.10.2 and Warp 1.17.0. Tests require pytest.
Warp is optional; discovery and JAX-only imports do not initialize CUDA.
Set `CUDA_VISIBLE_DEVICES` to the assigned GPU before starting Python.
`JAX_PLATFORMS=cpu` forces CPU execution;
`XLA_PYTHON_CLIENT_PREALLOCATE=false` disables JAX's GPU memory reservation.

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
    primitives, "bvh.snch.warp",
    Config(leaf_size_threshold=4, max_stack_depth=64),
)
closest = jax.jit(tree.backend.closest_point_with_status)
point, distance, face, status = closest(tree.payload, points)

first_hit = jax.jit(tree.backend.first_hit_with_status)
t, face, hit, status = first_hit(
    tree.payload, origins, directions, max_t, min_t=0.0,
)
```

Queries accept `(3,)` or `(..., 3)` vectors. Ray origins/directions must have
matching shapes; interval bounds broadcast to that shape. Bound methods such
as `tree.closest_point(points)` and `tree.first_hit(...)` are also available.

## Contracts

| Operation | Result |
|---|---|
| `closest_point_with_status(points)` | Point, distance, original face ID, status |
| `first_hit_with_status(origins, directions, max_t, min_t=0)` | Ray parameter, original face ID, hit flag, status |
| `any_hit_with_status(origins, directions, max_t, min_t=0)` | Occlusion flag, status |
| `aabb_overlap_with_status(lower, upper)` | Triangle-AABB broad-phase overlap flag, status |
| `closest_silhouette_with_status(points, max_distance=inf)` | Point, distance, original edge ID, status |
| `star_radius_with_status(points, cap)` | Minimum of cap and silhouette distance, status |
| `sample_boundary(points, radii, randoms)` | Point, normal, area PDF, original face ID, status |

The last three require SNCH. Star radius is the scalar geometric bound;
Robin coefficients and a complete WoSt solver are not implemented here.

Status **1** means found, **0** means valid miss/empty geometry, and **-1**
means invalid input or query failure. Closest point reports numerical failure
when squared distances exceed float32 range. Equal computed distances or ray
parameters choose the smaller original face ID.

Ray intervals are closed `[min_t,max_t]`; positive `min_t` excludes starting
contact. A miss returns `t=max_t`, face `-1`, and `hit=False`. Infinite
`max_t` is supported. Directions need not be unit length; ray parameters
measure distance only for unit directions. Geometry/directions must remain
within a usable float32 range.

Star radius returns status 1 when the cap is valid, including when no closer
silhouette exists. The sampler consumes three uniforms in `[0,1)`. Samples
outside the sphere are null events with PDF/status 0 and no retry. Accepted
PDFs include the branch probabilities: retain null trials in estimators.

## Geometry and runtime

Builders require finite referenced float32 vertices and valid triangle indices.
SAH supports degenerate triangles; SNCH rejects degeneracies and edges with
more than two incident faces. Winding determines normals. For mixed boundary
conditions, build separate absorbing/reflecting subset trees.

Construction validates topology, leaf ranges and packed indices. Leaf size
is an SAH threshold; a retained leaf can exceed it. Actual tree depth must
fit `max_stack_depth`, at most 64. Rebuild after geometry changes; refit and
automatic differentiation are not implemented.

Warp's built-in `wp.jax_kernel` handles JAX FFI, compilation and ownership.
The [batching adapter](../surfaces/explicit/accelerators/bvh/backends/_warp_jax.py)
flattens query axes without tiling geometry. The
[query table](../surfaces/explicit/accelerators/bvh/backends/_warp_specs.py)
defines input order, output counts and launch sizes. Add a kernel, its table
entry and a backend method to extend the interface.

Ray traversal shares a native C++ body for first/any hit. Packed float4 node
metadata contains opaque integer bits; do not transform it numerically.
Closest point caches deferred nodes and squared box bounds. Both use
uninitialized 64-entry local stacks (512 bytes/thread). Closest point uses
Warp's triangle helper for regular triangles and projection/segment fallbacks
for slender or degenerate triangles. Its ID/tie path runs only for competitive
candidates. Launch blocks are 64 threads for closest point and 128 for rays.
SNCH cone data is separate from ray/closest-point traversal.

Adapted ray helpers retain NVIDIA Warp's
[Apache-2.0 license](../surfaces/explicit/accelerators/bvh/backends/licenses/warp-LICENSE.txt).

## Validation and device timing

```sh
python -m pytest -q tests/spatial_trees/test_warp_bvh.py \
    tests/spatial_trees/test_backend_discovery.py \
    tests/spatial_trees/test_warp_payload.py

python -m scripts.bench.spatial_trees.warp_bvh_device \
    --queries closest_point --point-distribution shell --native-warp

python -m scripts.bench.spatial_trees.warp_bvh_device \
    --queries first_hit any_hit --native-warp
```

Tests cover independent analytic/NumPy oracles, ties, thin/degenerate geometry,
invalid inputs, ray endpoints, bounded silhouettes, sampler PDFs/nulls, CPU,
JIT and nested `vmap` with shared geometry.

The benchmark times warmed CUDA graphs with CUDA events, excluding build,
compilation, input transfers, readback and CPU dispatch. JSON Lines are saved
under `assets/benchmark_results/`. Native Warp distance/status checks allow
its different tie and sliver policies; point-distance consistency is checked
for every closest-point result.

The [closest-point report](benchmarks/closest_point.json) retains raw samples,
fixture hashes, source versions and validation. A40, 20,480 triangles,
65,536 unsorted queries, leaf threshold 4:

| Points | Ours | Native Warp | cuBQL | FCPW |
|---|---:|---:|---:|---:|
| Gaussian | 6.835 ms | 9.832 ms | 10.562 ms | 7.428 ms |
| Shell (near surface) | 0.182 ms | 0.211 ms | 0.185 ms | 0.189 ms |
| Interior | 6.531 ms | 8.670 ms | 10.215 ms | 6.704 ms |

Ours/native Warp use 64-thread blocks. External references use their own trees
and best tested settings: cuBQL SAH/64 threads; FCPW 64 threads, or 128 for
interior points. FCPW timestamps include runtime parameter uploads. Near-surface
points have radius 0.95–1.05; interior points fill the radius-0.95 ball.
These query timings on one mesh do not establish arbitrary-mesh rankings.

[Ray measurements](benchmarks/warp_bvh_device.json) record 0.116 ms first hit
and 0.099 ms any hit on the same-sized fixture with 128-thread blocks.
