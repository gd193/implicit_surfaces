# Explicit triangle queries with Warp and JAX

`bvh.sah.warp` and `bvh.snch.warp` build a static binary SAH hierarchy on the
CPU and run batched GPU queries through JAX. SNCH adds normal-cone and
silhouette-edge data for reflecting-boundary geometry. Each query uses one
GPU thread; every payload field is an explicit JAX array shared across the
batch, including under nested `vmap`.

## Implemented backends

| Backend | Implementation |
|---|---|
| `bvh.sah.warp` | CPU SAH build, Warp closest-point, ray and overlap queries |
| `bvh.snch.warp` | Same BVH queries plus normal cones, silhouettes and boundary sampling |
| `bvh.lbvh.jax`, `bvh.sah.jax`, `bvh.snch.jax` | Recovered JAX BVH implementations; also supply the shared CPU builders |
| `kd.left_balanced.jax`, `kd.sah.jax` | Recovered JAX kd-tree research |
| `octree.morton_explicit.jax` | Recovered JAX octree research |

Warp currently supports only BVH. The six Warp kernels implement closest
point, first ray hit, any ray hit, AABB overlap, closest silhouette and boundary
sampling. Star radius combines a bounded silhouette query with a radius cap.
The ray kernels have been optimized; the other kernels have correctness tests
and remain candidates for further performance work.

## Environment

Install the optional accelerator into an environment that already has JAX,
NumPy and trimesh:

```sh
python -m pip install -r requirements-warp.txt
```

The validated versions are JAX/JAXLIB 0.10.2 and Warp 1.17.0. Install the JAX
CPU/CUDA build appropriate for your machine separately. Tests additionally
need pytest. Warp remains optional: JAX backends can be imported and used
without it, and package discovery does not initialize Warp or CUDA.

Set `CUDA_VISIBLE_DEVICES` to your assigned GPU before starting Python. Set
`JAX_PLATFORMS=cpu` to force CPU execution. Tests and benchmarks otherwise use
JAX's first available device. `XLA_PYTHON_CLIENT_PREALLOCATE=false` avoids
reserving most of the GPU memory at startup.

## Build and query

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

# Keep geometry explicit when compiling or reusing a query.
closest = jax.jit(tree.backend.closest_point_with_status)
point, distance, face, status = closest(tree.payload, points)

first_hit = jax.jit(tree.backend.first_hit_with_status)
t, face, hit, status = first_hit(
    tree.payload, origins, directions, max_t, min_t=0.0,
)
```

Queries accept a single `(3,)` point/vector or arrays with arbitrary leading
batch dimensions `(..., 3)`. Ray origins and directions must have matching
batch shapes; interval bounds broadcast to that shape. Geometry buffers stay
unbatched under `vmap`. Bound convenience methods such as `tree.first_hit`
and `tree.closest_point` are also available.

Both builders require finite referenced float32 geometry and valid triangle
indices. SAH supports degenerate triangles; SNCH rejects degenerate triangles
and edges incident to more than two faces. Construction validates topology,
leaf ranges, root placement and packed index limits before GPU execution.
The leaf-size setting is an SAH threshold, not a strict maximum: its cost
rule can retain a larger leaf. Actual depth must fit `max_stack_depth` and
the kernels' 64-slot stack capacity.

For mixed boundary conditions, construct separate trees for the absorbing
and reflecting subsets. Subset membership is a caller responsibility. Use
consistent face winding for the intended domain normals.

## Query contracts

| Operation | Result |
|---|---|
| `closest_point_with_status(points)` | Closest point, distance, original face ID, status |
| `first_hit_with_status(origins, directions, max_t, min_t=0)` | Ray parameter, original face ID, hit flag, status |
| `any_hit_with_status(origins, directions, max_t, min_t=0)` | Occluded flag and status; exits at the first eligible triangle |
| `aabb_overlap_with_status(lower, upper)` | Triangle-AABB broad-phase overlap flag and status |
| `closest_silhouette_with_status(points, max_distance=inf)` | Silhouette point, distance, original edge ID, status |
| `star_radius_with_status(points, cap)` | Minimum of cap and silhouette distance, plus validity status |
| `sample_boundary(points, radii, randoms)` | Surface point, oriented normal, area PDF, original face ID, status |

The last three operations require SNCH. Silhouette caps prune the traversal;
`star_radius` uses this bounded search. This is the scalar geometric bound,
not the coefficient-dependent Robin extension or a complete WoSt solver.

For ordinary checked queries, status **1** means found, **0** means a valid
miss or empty geometry, and **-1** means invalid input or traversal failure.
A ray miss returns `t=max_t`, face ID `-1`, and `hit=False`. Ray intervals are
closed `[min_t,max_t]`; equal-parameter hits choose the smaller original face
ID. A positive `min_t` excludes starting contact. Directions need not be unit
length, but the parameter measures distance only for unit directions.
Positive infinite `max_t` is supported. Float32 arithmetic still requires
sensible geometry and direction scales; this is not an arbitrary-precision
intersection implementation.

A valid star-radius query has status 1 even when there is no closer silhouette
and the caller's cap is returned. The sampler consumes three uniform variates
in `[0,1)`. Samples outside the requested sphere are null events with PDF 0
and status 0, without retries. Its accepted PDF is the unconditional surface
proposal density including branch probabilities; retain null trials in an
estimator rather than renormalizing accepted samples.

## Runtime and ownership

All queries use Warp's built-in `wp.jax_kernel` integration for JAX FFI,
module compilation, device selection, and lifetime management. There is no
project-specific FFI launcher, compiler cache, or runtime mode switch.

The shared [query specifications](../surfaces/explicit/accelerators/bvh/backends/_warp_specs.py)
define payload argument order, output counts, batching counts and launch sizes.
Output types come from the Warp kernel signatures. The small
[batching adapter](../surfaces/explicit/accelerators/bvh/backends/_warp_jax.py)
flattens mapped query dimensions while keeping the tree arrays shared;
Warp's default `broadcast_all` rule would copy the tree per mapped query.
To add an operation, write its Warp kernel, add a query specification, and
call the shared `query(name, payload, *inputs)` helper from the backend.
The FFI and batching code needs no operation-specific branches.

The payload builders provide contiguous row-major buffers. Packed ray-node
metadata is stored as integer bit patterns in float4 arrays and must be treated as opaque, without numeric transforms.
Rebuild after geometry changes; refit and automatic differentiation are not
implemented. Implicit-tree bridge interfaces remain explicitly deferred.

The [ray traversal](../surfaces/explicit/accelerators/bvh/backends/_warp_ray.py)
shares a native C++ body between first-hit and any-hit modes. It carries
packed child metadata on a 64-entry uint64 stack (512 bytes allocated per
thread), writing slots only when pushed. Packed ray bounds add 32 bytes per
tree node. SNCH cone buffers are separate and are not loaded by ray queries.
The adapted Warp helpers retain their
[Apache-2.0 license](../surfaces/explicit/accelerators/bvh/backends/licenses/warp-LICENSE.txt).

## Validation and device-only measurements

```sh
python -m pytest -q tests/spatial_trees/test_warp_bvh.py \
    tests/spatial_trees/test_backend_discovery.py \
    tests/spatial_trees/test_warp_payload.py
```

The numerical tests use independent NumPy/analytic oracles, including closed
endpoints, cross-leaf ties, axis-aligned and subnormal directions, degenerate
and thin triangles, bounded silhouettes, PDFs/null events, and exact int32
IDs above 2**24. They also cover explicit payload reuse under JIT/nested vmap,
CPU execution and optional-Warp discovery.

Use the portable benchmark for optimization work:

```sh
python -m scripts.bench.spatial_trees.warp_bvh_device \
    --queries first_hit any_hit --counts 65536 --leaf-size 4 --native-warp

python -m scripts.bench.spatial_trees.warp_bvh_device \
    --backend bvh.snch.warp --queries closest_silhouette sample_boundary \
    --radius 0.5
```

It times warmed CUDA graphs with CUDA events, excluding construction,
compilation, JAX/FFI dispatch and host synchronization. Cases run in randomized
round-robin order. JSON Lines under `assets/benchmark_results/` include raw
samples, GPU properties, versions and source/query hashes. Native ray outputs
are checked before timing; its triangle tolerances and strict upper endpoint
differ from the custom contract. The separate `warp_bvh.py` runner measures
synchronized host/JAX calls and construction costs.

The measured A40 fixture has 20,480 triangles, 65,536 rays, leaf threshold 4
and 128-thread blocks for both implementations:

| Query | Previous custom | Native Warp mesh | Packed custom |
|---|---:|---:|---:|
| First hit | 0.235 ms | 0.199 ms | 0.116 ms |
| Any hit | 0.213 ms | 0.178 ms | 0.099 ms |

[Recorded measurements](benchmarks/warp_bvh_device.json) include individual
event samples, resource properties and source hashes for this revision.

Warp's first-hit traversal on the custom tree took 0.181 ms, controlling
topology. The main removed cost was two hidden 64-slot stack-clearing loops
in the previous Warp-vector traversal: 128 unnecessary local stores per ray.
A same-traversal C++ experiment took 0.141 ms without clearing and 0.209 ms
with both clearing loops restored. Packed metadata then reduced further
loads. These results concern ray kernels on this fixture; closest-point,
silhouette and whole-solver competitiveness require separate measurements.
