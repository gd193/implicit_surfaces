# Primitive-generic BVH migration plan

Design baseline: commit `9bdc06f`. Points, segments and triangles should use one public build and query path. A future quad provider should be able to implement bounds, packing and nearest-point evaluation without changing the BVH builder or walker. Each built tree initially indexes a homogeneous collection. Mixed-primitive scenes are a separate design decision.

## Implementation

The [API guide](warp_bvh.md) describes the implemented binary BVH. Generic
callers choose SAH or Morton/LBVH when building. Both produce the same device
representation and use the same query factories. The [quad integration
test](../tests/spatial_trees/test_warp_external_quad.py) exercises an external
primitive through JAX and a caller-owned Warp kernel.

The two named mesh adapters now store the generic tree, packed geometry and
boundary data. Their names and surface methods remain available. The named
SNCH adapter retains its historical topology, feature numbering, cone
preparation and degeneracy threshold. This preserves observable silhouette
IDs and seeded sampling results. The public host builders and the named
backend discovery interface also remain available. The old flat mesh
payload's positional and serialization formats require migration.

## Ownership and dispatch

```text
Primitive data ── bounds ──► builder ──► typed index + primitive permutation
               └─ pack using permutation ──► leaf-ordered geometry

Query + primitive operations + index representation [+ prepared boundary data]
                                  │
                          bind before tracing
                                  ▼
                specialized traversal and leaf evaluation
                                  │
                     result in original primitive IDs
```

- **Primitive** supplies conservative bounds, packing and the geometry operations it supports. Surface normals and ray intersections are optional. The triangle projection fast path belongs to the triangle/metric implementation.
- **Builder** chooses partitions. Its output defines dimension, node and leaf encoding, child addressing, permutation and depth constraints. Two builders producing the same representation can share a traversal even when their trees have different shapes. A different arity or incompatible leaf encoding requires compatible node access and traversal. Start with the existing binary representation.
- **Query** defines inputs, bounds, ordering, termination and result semantics. At construction, bind the query to the index representation and required primitive operations. Resolve unsupported combinations here, outside JAX tracing and GPU loops. Keep this dispatch static and small while there is only one executable index family. Packed versus indexed node loads can remain compile-time choices on the same binary tree.
- **Boundary and surface** own connectivity, normals, closedness and feature or coefficient data. Bind optional metadata to the geometry and tree ordering used to prepare it. SNCH feature ranges and cone bounds need compatible preparation. A triangle surface may require triangle topology while its spatial index stays generic.

## Initial inventory

| Existing entry point | Consumer or dependency | Intended disposition |
|---|---|---|
| [Generic tree construction](../surfaces/explicit/accelerators/bvh/backends/_warp_tree.py) and [query factories](../surfaces/explicit/accelerators/bvh/warp.py) | Direct Warp kernels, batched JAX callers, benchmarks and factory tests | Use as the starting implementation. Expose the current builder choice explicitly when a second compatible implementation needs it. |
| [Mesh accelerator](../surfaces/explicit/accelerators/base.py), [backend protocol](../surfaces/explicit/accelerators/protocol.py) and [BVH forwarding methods](../surfaces/explicit/accelerators/bvh/accelerator.py) | Surface methods, mesh benchmarks and backend tests | Migrate used behavior to queries. Remove duplicated capability flags and forwarding methods as their callers move. |
| [SAH backend](../surfaces/explicit/accelerators/bvh/backends/sah_warp.py) and [SNCH backend](../surfaces/explicit/accelerators/bvh/backends/snch_warp.py) | Named backend discovery, mesh payload creation, geometric and boundary queries | Separate build, geometry packing and boundary preparation. Inventory users of registered names and payload fields before changing compatibility. |
| [TriMeshSurface](../surfaces/explicit/trimesh.py) and [host primitives](../surfaces/explicit/primitives.py) | Signed distance, surface normal and vertex-velocity callers | Keep surface semantics here. Consolidate duplicated geometry storage where original indexing and numerical behavior permit. |
| Independent LBVH, kd-tree and octree host builders | Construction APIs and tests | Inventory separately. A host builder does not establish that its output supports the current Warp walker. |

## Three caller examples

These examples describe the existing public factory API. A caller prepares geometry once and reuses the resulting index and packed arrays.

```python
def prepare_nearest(primitive, source):
    tree, geometry = build_tree(primitive, source)
    query = make_nearest(primitive, Euclidean())
    return tree, geometry, query

p = Points(3)
tree, geometry, nearest = prepare_nearest(p, p.data(coordinates))
points, distances, ids, status = nearest(tree, geometry, queries)

t = Triangles()
tree, geometry, nearest = prepare_nearest(t, t.indexed(vertices, faces))
points, distances, ids, status = nearest(tree, geometry, queries)

s = Segments(2)
tree, geometry, boundary = build_boundary(s, s.indexed(vertices_2d, edges))
sign, status = make_classify(s)(tree, geometry, boundary, queries_2d)
```

The examples assume the present binary index. With multiple executable representations, supply the static representation to the factory at binding time. Use the same specialized operation from batched JAX and caller-owned Warp kernels.

## Migration sequence and acceptance

The legacy SAH and SNCH methods already call the generic query factories. The first deletion concerns the duplicate build and storage path. Query-family work consists mainly of parity checks.

1. **Capture the baseline at the clean commit.** Audit public exports, backend-name strings, examples, notebooks and direct payload-field users. Start with `test_warp_bvh.py`, `test_warp_inside.py`, `test_warp_robin.py`, `test_warp_boundary_queries.py` and `warp_bvh_device.py`. Capture outputs, reference inputs and warmed kernel timings for small box, large box and bunny. Keep build, preparation and JIT timings separate. Use the same device, inputs and build settings on both sides.
2. **First PR — replace the triangle accelerator’s storage.** Keep the two named build entry points and surface methods callable. Convert the mesh to indexed triangle data once and store the generic tree, leaf-ordered geometry and boundary data for all existing methods. Use the existing boundary preparation, with silhouette features enabled for SNCH. Prepare sampling and Robin data as needed and preserve immutable coefficient updates. Keep `TriMeshSurface` as a consumer. Move SAH callers to the stored arrays. Remove each reconstruction helper after its last SNCH method and its direct benchmark imports have moved. Run the external primitive test below in this PR. Retire the mixed payload builders when their last caller moves. Keep the shared tree validator and packed-node encoder. An exported legacy type or public host builder needs an explicit compatibility decision before deletion.
3. **Gate the storage switch on behavioral parity.** For unchanged float32 SAH fixtures require bitwise geometric query outputs, including original IDs and statuses. Test empty inputs, caps and ties, hit-buffer overflow, JIT/vmap, direct device calls and surface normals/velocity lookup. SNCH currently uses its own split and feature/cone construction. Capture feature IDs, nonmanifold rejection and the legacy doubled-area threshold of 1e-30. Preserve these on the named compatibility path. Compare bounds conservatively, check geometric results against independent oracles, validate PDFs and sampling distributions where tree-order changes alter individual samples, and check warmed query latency. If parity requires separate work, land the SAH storage switch first and retain the SNCH path until it passes. Reuse the generic feature builder wherever these contracts agree.
4. **Remove triangle identity dispatch in nearest.** Replace `_builtin_triangle` with optional statically bound geometric operations for the fused candidate evaluation and its matching conservative node bound. Preserve both slot and loaded-item evaluation for predicates and selected results. Keep metric compatibility, cap rounding, ties, and the existing compiler specialization. Treat grouping and bound-caching choices as tuning hints. Run the external primitive test again after moving the fast path. Keep the FP32 triangle-nearest evaluator and its documented freedom on exact ties. k-nearest retains its own scorer and original-ID tie ordering, including for k=1. Compare the triangle fast path with the clean baseline.
5. **Prove representation-based dispatch.** Give the generic build function a small builder argument, with SAH as default. Exercise the existing Morton/LBVH construction on supported inputs. Validate root, bounds, children, leaf coverage, original-ID permutation and stack depth before packing geometry. Equal Python field names alone do not prove compatibility. Handle empty inputs before invoking LBVH and reject trees exceeding the configured traversal depth. Run the same bound query kernel on both binary builds for each primitive type, checking results against an oracle and respecting each query’s tie contract. Keep the independent kd-tree and octree host builders outside this migration.
6. **Trim the old surface.** Check remaining imports and derive capabilities from the bound primitive operations and prepared boundary data. Retire duplicate validation, capability forwarding, storage and registry machinery as callers move. Preserve the public host SNCH build API independently of whether the mesh query adapter still uses its implementation.

## Verification gates

| Contract | Concrete check |
|---|---|
| External primitive | A test-only provider imports the public API and defines indexed, convex planar quads with nearest evaluation. Use a fixture with a verified nonidentity leaf permutation. Compare interior, edge, corner, cap and miss results to an independent oracle on CPU and GPU. Exercise nested JAX batching and a caller-owned Warp kernel. Binding a ray query must reject the missing operation before launch. |
| Caller compatibility | Preserve legacy return arities, status-bearing variants and one-argument batching such as `jax.vmap(accel.closest_point)`. Generic callers batch query coordinates while sharing tree and geometry. Check original-face normals and barycentric velocity lookup with a reordered mesh. Packed normals can differ in both order and numerical construction. |
| Optional runtime and typing | Keep host-only builders importable and usable with Warp unavailable. Run the [backend discovery test](../tests/spatial_trees/test_backend_discovery.py) and [public typing checks](../tests/typing/check_warp_api.py) when changing imports, factory signatures or stubs. Static provider descriptors must be hashable and reusable across batches. |
| Representation and boundary data | The current device tree carries both packed and unpacked arrays. Empty checks read packed arrays even for some indexed walks. Silhouette traversal substitutes feature ranges in the unpacked fields, and sampling reads unpacked nodes directly. Keep both views until these consumers have explicit equivalents. Prepare metadata for the same permutation and device as its tree. Test empty features, tiny triangles and shared-edge IDs on SNCH. |
| Performance reproduction | Use the tracked [mesh benchmark](../scripts/bench/spatial_trees/warp_bvh_device.py) and [query matrix](../scripts/bench/spatial_trees/warp_query_matrix.py). The previously profiled 768- and 12,288-triangle boxes and 69,451-triangle bunny are an additional lab workload. Its harness and samples live under `/system/user/publicwork/volkmann/implicit-surface-project/results/bvh-profile-3348eb4-rtx3090/`. Record a fresh comparison from an isolated copy of the design baseline. |

## Decisions before implementation

- Which externally used imports, backend names or serialized payloads need a migration window? Repository references establish local usage only.
- For the first PR, prefer existing factories and array pytrees. Introduce a built-index wrapper only if a specific caller needs it to carry static layout identity or prevent mismatched tree and geometry.
- The extension test uses a convex planar quad. A production quad API can choose its own surface definition, including a bilinear patch, independently of BVH traversal.

Keep new construction algorithms, the experimental slow-query fallback and layout tuning as independent projects. This migration can be evaluated on the current builders and the existing performance baseline.
