# JAX field mesh extraction

## Mesh geometry

Import `Mesh`, `SegmentMesh`, `TriMesh`, and `QuadMesh` from `surfaces.meshes`. Each stores
`vertices` of shape `(V, dim)` and zero-based integer `elements` of shape `(N, K)`.
The coordinate dimension can be 2 or 3 for any topology. For example, a planar
quad has four indices and two coordinates per vertex.

```python
import numpy as np
from surfaces.meshes import QuadMesh
from meshExtractor import write_obj

square = QuadMesh(np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
                  np.array([[0, 1, 2, 3]]))
write_obj("square.obj", square)
```

OBJ export writes segments as lines and preserves polygon winding and arity.
It embeds 2D positions at z=0. The extraction algorithms below operate on 3D
fields and produce triangle meshes.

## Adaptive marching cubes

```python
import jax.numpy as jnp
from meshExtractor import make_marching_cubes
from surfaces.implicit.lipschitz import lipschitzImplicit

surface = lipschitzImplicit(fn=lambda radius, x: jnp.linalg.norm(x) - radius,
                            level_set=0., lipschitz_bound=1., dim=3)
extract = make_marching_cubes(surface, ((-1.,) * 3, (1.,) * 3),
                              tolerance=.12, max_depth=6)
result = extract(jnp.float32(.6))
mesh, unresolved_boxes = result.mesh, result.unresolved_boxes
```

The extractor samples `surface.signed_value(params, point)` in a rectangular
3D domain. A strict `classify_box` result prunes an entire subtree.
Autobound can classify a group of child boxes using one parent enclosure.
Point-only surfaces can provide just `signed_value`. Their same-sign boxes
remain unresolved without a certificate.

`tolerance` bounds two-sided geometric distance in world units within each
accepted leaf. The coarse acceptance test looks for a patch spanning an
axis-aligned footprint. Opposite, strictly signed caps enclose the level set
in a slab of at most the requested thickness. Continuity guarantees a crossing
over every point in its footprint. The mesher constrains shared edge crossings
to the slab. For other crossing patterns, a nonempty patch in a box whose
diagonal fits the tolerance satisfies the same distance bound. Small features
within that distance can merge.

The traversal subdivides inconclusive boxes up to `max_depth`. Each remaining
unresolved box contributes a pair of world-space corners to `unresolved_boxes`
with shape `(N, 2, 3)`. These regions can contain omitted surface geometry.
Shared face subdivisions, edge crossings and an identical negative-corner
pairing rule join contour loops across different leaf sizes. The indexed
triangles have outward winding for a negative interior. Domain boundaries
and unresolved regions can have open edges.

Surface queries run in JAX batches. The octree, contour loops and mesh assembly
run on the host. Reuse the factory with different parameters outside JAX
transformations. To export the accepted triangles to OBJ, run

```sh
python scripts/extract_mesh.py --extractor mc --surface torus --max-depth 6 --tolerance .15 --out torus.obj
```

## Adaptive dual contouring

Use the same surface, bounds and world-space tolerance as marching cubes.

```python
from meshExtractor import make_dual_contouring

extract = make_dual_contouring(surface, ((-1.,) * 3, (1.,) * 3),
                               tolerance=.12, max_depth=6)
mesh, unresolved_boxes = extract(jnp.float32(.6))
```

The eager extractor returns an indexed `TriMesh`. It refines shared crossing
edges by bracketed bisection and interpolation. It uses
`surface.spatial_derivative(params, point)` for Hermite normals, or JAX
autodiff of `signed_value` on surfaces that supply only point values.
Each active cell fits a box-constrained least-squares vertex. Rank-deficient
fits favor the crossing centroid. Multiple contour loops in one cell trigger
subdivision or an unresolved box at the depth limit.

Triangles join the incident cell vertices around a crossing edge. Four cells
form a triangulated quad and three form a transition triangle. Along the
domain boundary, the extractor uses shared face-segment midpoints and edge
crossings to extend the mesh to the box. It also uses these points beside
certified coarse slabs to keep their triangles inside the enclosure.
The octree acceptance, unresolved-box format, and host execution model match
marching cubes. Query batching runs on the JAX device.

## Fixed-capacity dual contouring

For GPU-resident dual contouring, choose static capacity budgets and compile
the outer computation:

```python
import jax
from meshExtractor import make_dual_contouring_resident, DC_COMPILER_OPTIONS

extract = make_dual_contouring_resident(
    field, bounds, resolution=256,
    capacity=(max_cells, max_edges, max_triangles),
    batch_size=65536,
)
extract = jax.jit(extract, compiler_options=DC_COMPILER_OPTIONS)
buffers = extract(params)
mesh = buffers.compact()  # host reads counts and checks overflow
```

The raw callable returns `DualContouringBuffers` and composes inside a larger JAX computation. Apply the graph
options only to its outermost JIT. The captured pipeline keeps counts on the
GPU; `batch_size` independently bounds field and gradient batches.

`buffers.vertices` and `buffers.faces` have capacity-sized leading dimensions.
`buffers.counts` is int32 `[C, E, T, status, F, K, U]`: active cells, crossing edges,
triangles, overflow bits, peak frontier, raw corner records and unique corners.
When `status == 0`, the first C vertices and T faces are valid; tails are
unspecified. Overflow bits 1/2/4/8/16 identify C/E/T/F/K shortages. All payload
is invalid on overflow; unknown downstream counts are -1.

Sparse resident sampling additionally requires `sparse_capacity=(F, K)`.
F bounds the frontier including expanded children; K bounds eight records per
retained leaf before corner deduplication. On frontier overflow, the reported
requirement is a lower bound at the first failed level. Dense F/K/U counts are 0.
Budgets are explicit. Calling `buffers.compact()` raises `CapacityOverflow`
on shortage and exposes the count record as `.counts`.

The benchmark supports the adaptive pipeline with `--backend dual-adaptive`.
Use `--backend dual-dense` or `dual-sparse` with explicit capacities for
the fixed-grid pipeline. The latter also requires sparse frontier and corner
capacities. Add `--resident` to time padded output before compaction.
