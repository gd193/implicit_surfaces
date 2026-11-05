# JAX field mesh extraction

```python
from meshExtractor.dual import make_dual_contouring

extract = make_dual_contouring(
    field, ((-1., -1., -1.), (1., 1., 1.)), resolution=128,
    lipschitz_bound=1.,  # omit for dense sampling
)
vertices, faces = extract(params)
```

`make_dual_contouring` and `meshExtractor.warp.make_marching_cubes` take
`field(params, point) -> scalar float32` and return eager
extractors producing JAX float32 vertices `(V, 3)` and int32 triangles `(F, 3)`.
Parameters remain dynamic; invoke extraction outside `jit`, `grad`, and `vmap`
because mesh sizes depend on field values. Resolution counts cells and accepts
one integer or three integers; bounds may be rectangular. `level=0.` selects the
isovalue, and `batch_size=65536` bounds field and gradient query batches.

Dual contouring evaluates gradients at unique crossing edges, using
`jax.grad(field, argnums=1)` by default. Supply `gradient(params, point)` returning
a float32 `(3,)` vector when an analytic gradient is available. `root_steps=0`
uses linear edge interpolation; positive integers add safeguarded bracketed
refinement passes. Gradients always come from the original field, including at
domain boundaries. A cell-constrained Hermite QEF preserves sharp features;
faces join cell vertices around crossing edges and split each quad into two
triangles. The domain boundary remains open. One vertex per active cell cannot
represent every ambiguous configuration; nonmanifold meshes and degenerate
triangles at exact grid ties are possible. Extraction is not differentiable.

A supplied positive, conservative Lipschitz bound enables sparse octree pruning.
This retains the same finest uniform grid as dense extraction, rather than
joining cells at mixed refinement levels. An underestimated bound can omit
surface components. Rectangular and non-power-of-two requests retain their
physical grid spacing: only pruning queries outside the domain use a clamped
extension. Sparse extraction stores retained cells and unique corners, without a
full-volume allocation. The implementation pins Warp 1.18 private helpers.

For GPU-resident dual contouring, choose static capacity budgets and compile
the outer computation:

```python
import jax
from meshExtractor.dual import make_dual_contouring_resident, DC_COMPILER_OPTIONS

extract = make_dual_contouring_resident(
    field, bounds, resolution=256,
    capacity=(max_cells, max_edges, max_triangles),
    batch_size=65536,
)
extract = jax.jit(extract, compiler_options=DC_COMPILER_OPTIONS)
mesh = extract(params)
```

The raw callable returns `ResidentMesh` and composes inside a larger JAX computation. Apply the graph
options only to its outermost JIT. The captured pipeline keeps counts on the
GPU; `batch_size` independently bounds field and gradient batches.

`mesh.vertices` and `mesh.faces` have capacity-sized leading dimensions.
`mesh.counts` is int32 `[C, E, T, status, F, K, U]`: active cells, crossing edges,
triangles, overflow bits, peak frontier, raw corner records and unique corners.
When `status == 0`, the first C vertices and T faces are valid; tails are
unspecified. Overflow bits 1/2/4/8/16 identify C/E/T/F/K shortages. All payload
is invalid on overflow; unknown downstream counts are -1.

Sparse resident sampling additionally requires `sparse_capacity=(F, K)`.
F bounds the frontier including expanded children; K bounds eight records per
retained leaf before corner deduplication. On frontier overflow, the reported
requirement is a lower bound at the first failed level. Dense F/K/U counts are 0.
Budgets are explicit and never grow silently. Passing the same capacities to
`make_dual_contouring` returns exact-sized arrays with one count readback and
raises `CapacityOverflow` on shortage; without capacities it retains dynamic
allocation. The exception exposes the count record as `.counts`.

The benchmark module `scripts.bench.mesh_extraction.benchmark` accepts
`dual-dense` and `dual-sparse` alongside the marching-cubes backends. Pipeline
measurements include sampling, gradients, QEF, and topology. Extraction
measurements start from resident unique corners and values but still include
Hermite gradient evaluation, QEF, topology, and field queries for enabled root
refinement. Triangle counts are compared within each algorithm family.

Use `--capacity C E T --resident` to benchmark the resident dual pipeline; add
`--sparse-capacity F K` for `dual-sparse`. Omit `--resident` to include exact-size
compaction. Conditional CUDA graph bodies are missing from CUPTI's kernel trace
on this stack, so these runs report synchronized wall time and mark full device
activity timing unavailable.
