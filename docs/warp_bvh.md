# Triangle queries with Warp and JAX

`bvh.sah.warp` builds static binary SAH trees on the CPU and queries them on
the GPU. `bvh.snch.warp` adds normal cones, silhouette queries and boundary
sampling. Geometry stays shared under JIT and nested `vmap`.

## Setup

Install alongside JAX, NumPy and trimesh:

```sh
python -m pip install -r requirements-warp.txt
```

Validated with JAX/JAXLIB 0.10.2 and Warp 1.17.0. Warp is optional; imports
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

silhouette = jax.jit(tree.backend.closest_silhouette_with_status)
point, distance, edge, status = silhouette(tree.payload, points, max_distance=0.5)
```

Vectors have shape `(3,)` or `(..., 3)`; scalar bounds broadcast to query
shape. Ray origins and directions must have matching shapes. Bound methods
such as `tree.closest_point(points)` are also available.

## Query contracts

| Method | Result |
|---|---|
| `closest_point_with_status(points)` | Point, distance, original face ID, status |
| `first_hit_with_status(origins, directions, max_t, min_t=0)` | Ray parameter, original face ID, hit, status |
| `any_hit_with_status(origins, directions, max_t, min_t=0)` | Occlusion, status |
| `aabb_overlap_with_status(lower, upper)` | Triangle-AABB broad-phase overlap, status |
| `closest_silhouette_with_status(points, max_distance=inf)` | Point, distance, original edge ID, status |
| `star_radius_with_status(points, cap)` | Minimum of cap and silhouette distance, status |
| `sample_boundary(points, radii, randoms)` | Point, normal, area PDF, original face ID, status |

The last three require SNCH. Status **1** means found, **0** means valid
miss/empty geometry, and **-1** means invalid input or failure. Equal computed
distances or ray parameters choose the smaller original ID. Closest point
reports failure when squared distances exceed float32 range.

Ray intervals `[min_t, max_t]` and silhouette caps are closed. Positive
`min_t` excludes starting contact; infinite `max_t` is supported. Ray misses
return `t=max_t`, face `-1`, and `hit=False`. Directions need not be unit
length; the ray parameter is a distance only for unit directions. Silhouette
queries include tangencies and open edges; misses return distance `inf` and
edge `-1`.

Star radius returns status 1 for a valid cap even without a closer silhouette.
It provides a geometric bound; Robin coefficients and a full WoSt solver are
not implemented. Boundary sampling consumes three uniforms in `[0,1)`.
Samples outside the sphere are null events with PDF/status 0 and no retry;
retain them in estimators. Accepted PDFs include branch probabilities.

## Construction and extension

Builders require finite referenced float32 vertices and valid triangle
indices. SAH accepts degenerate triangles; SNCH rejects them and edges with
more than two incident faces. Winding determines normals. Build separate
absorbing/reflecting subset trees for mixed boundary conditions.

Topology, leaf ranges and packed indices are validated. Leaf size is an SAH
threshold; retained leaves can exceed it. Tree depth must fit
`max_stack_depth`, at most 64. Rebuild after geometry changes. Refit and
automatic differentiation are not supported.

Warp's built-in `wp.jax_kernel` handles FFI, compilation and ownership. The
[batching adapter](../surfaces/explicit/accelerators/bvh/backends/_warp_jax.py)
flattens query axes without tiling geometry. Add a kernel, a
[query specification](../surfaces/explicit/accelerators/bvh/backends/_warp_specs.py)
and a backend method to extend the interface. SNCH cones enclose whole edges;
ray and closest-point queries use separate traversal data.

Adapted ray helpers retain Warp's
[Apache-2.0 license](../surfaces/explicit/accelerators/bvh/backends/licenses/warp-LICENSE.txt).

## Validation and timing

```sh
python -m pytest -q tests/spatial_trees/test_warp_bvh.py \
    tests/spatial_trees/test_backend_discovery.py \
    tests/spatial_trees/test_warp_payload.py

python -m scripts.bench.spatial_trees.warp_bvh_device \
    --backend bvh.snch.warp --queries closest_silhouette \
    --mesh torus --point-distribution shell --radius .5
```

The benchmark times warmed CUDA graphs with CUDA events, excluding build,
compilation, transfers, readback and CPU dispatch. Results go to
`assets/benchmark_results/`.

Recorded [ray](benchmarks/warp_bvh_device.json),
[closest-point](benchmarks/closest_point.json) and
[silhouette](benchmarks/closest_silhouette.json) comparisons retain samples,
timing scope and measured-source hashes.
