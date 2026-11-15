# GPU query references

FCPW: 2D closest point, first hit, occlusion and Euclidean silhouette.
cuBQL: 2D/3D point nearest. Each uses upstream query code and its own tree.
cuBQL uses spatial median in 2D; its SAH builder supports only float3.

Project queries use the supported Warp factories through JAX FFI. Host tree
construction remains separate; retained builder sources with `_jax.py` names
are included in source hashes, but no legacy pure-JAX tree traversal is timed.
Existing reports and generated reference provenance remain historical records.

Build against an existing FCPW CUDA/Slang-RHI build and cuBQL checkout:

```sh
python -m scripts.bench.spatial_trees.build_references \
  --fcpw /path/to/fcpw --fcpw-build /path/to/fcpw-build \
  --cubql /path/to/cuBQL --cuda /path/to/cuda --output /path/to/bin
CUDA_VISIBLE_DEVICES=0 python -m scripts.bench.spatial_trees.warp_reference_queries \
  --binaries /path/to/bin --output /path/to/results
```

The FCPW build expects `source/deps/slang-rhi` and `_deps/slang-src` under
its build directory. The queue shim targets the recorded Slang-RHI version;
it accesses private CUDA stream/heap state to isolate one upstream kernel.

Timings exclude build, compilation, uploads, parameter copies and host dispatch.
CUDA graphs repeat the resident kernel; block sizes and query buffers match.
FCPW initializes miss outputs before timing. Its graph output is checked against
ordinary RHI dispatch. Nearest checks allow distance ties while checking primitive
membership and distance. Silhouette IDs are mapped through FCPW's own feature data.

Results include samples, input/source/binary hashes, tree sizes and register/local
memory usage. Generated Warp CUDA C++ and PTX are saved alongside outputs. Use
`--reuse-reference /path/to/results.json` to reuse reference timings and outputs
only when input and executable hashes match. Generated files stay outside git.

Reuse also checks GPU UUID, shaders/compiler, timing settings and output hashes.
`--fixtures star cluster2 cluster3` adds nonconvex and clustered held-out data.
`--leaf-size`, `--reference-leaf-size` (cuBQL), `--block-dim` and `--queries`
expose tuning; settings are recorded.
For parity claims compare competitive reference block sizes, not just one matched
launch. Reference trees remain independent; FCPW/cuBQL use their own predicates.

## Independent k-nearest point references

The [initial A40 comparison](../../../../docs/benchmarks/warp_knn.md) records
validated results and the remaining larger-k gap against cudaKDTree.

`build_knn_references.py` pins clean official upstream checkouts to cuBQL
`d2e62a39b83c1b8ca72c266703fdc0dfdcb67952` and cudaKDTree
`37ce4c377170d13670632143b976192d5b69ed03`. Sources:
https://github.com/ingowald/cuBQL and https://github.com/ingowald/cudaKDTree.
No upstream code is patched. Each dimension/k/list combination is compiled
separately; architecture, compiler, commands, source hashes and binary hashes
are recorded in `build.json`. All generated files belong outside the checkout.

```sh
ART=/path/to/local/knn-artifacts
CUDA_ROOT=/path/to/cuda
git clone https://github.com/ingowald/cuBQL "$ART/cuBQL"
git -C "$ART/cuBQL" checkout d2e62a39b83c1b8ca72c266703fdc0dfdcb67952
git clone https://github.com/ingowald/cudaKDTree "$ART/cudaKDTree"
git -C "$ART/cudaKDTree" checkout 37ce4c377170d13670632143b976192d5b69ed03
python -m scripts.bench.spatial_trees.build_knn_references \
  --cubql "$ART/cuBQL" --cudakdtree "$ART/cudaKDTree" \
  --cuda "$CUDA_ROOT" --output "$ART/knn-bin" --arch 86
CUDA_VISIBLE_DEVICES=0 python -m scripts.bench.spatial_trees.warp_knn_reference \
  --binaries "$ART/knn-bin" --output "$ART/knn-smoke" \
  --dimensions 2 3 --k 4 --sizes 1024 --queries 1024 --fixtures uniform
```

The full matrix covers uniform, clustered and sphere/circle surface float32
points in 2D/3D, k=1/4/8/16/32/64/128, three N scales and two Q scales.
`--implementations`, `--dimensions`, `--k`, `--sizes`, `--queries`,
`--fixtures`, `--block-dim`, `--leaf-size`, `--no-cache-bounds`,
`--node-access packed|indexed`, and `--scheduling auto|flat|grouped` filter/tune runs.
Caching, node access and scheduling vary independently. The default is cached,
packed, and the policy's schedule (grouped for these point queries). Reproduce
the old combined uncached mode with `--no-cache-bounds --node-access indexed
--scheduling flat`; changing only `--no-cache-bounds` now isolates caching. Timings use
resident CUDA graphs and events after warmup, excluding construction, JIT,
transfers and host invocation. All paths emit squared scores, original IDs,
count and status. Warp uses the generic callback path with Euclidean closest
and box bound callbacks. cuBQL uses its pointData `findKNN` and independent
BVH (2D spatial median, 3D SAH); cudaKDTree uses its default independent
thrust builder and CCT traversal with payload IDs surviving reordering.
Build defaults include separate `_stack` variants; select them using
`--implementations cudakdtree_fixed_stack cudakdtree_heap_stack`.
`--traversals cct` limits builds to the default CCT variants.
Both FixedCandidateList and HeapCandidateList are measured. Already sorted cuBQL and fixed-list outputs are emitted directly. Heap entries
are sorted in place with O(k log k) heapsort inside the timed kernel, without
a duplicate candidate array; ID remapping is also timed.

External ties may select any equally close subset, rather than Warp's smallest
original IDs. External wrappers preserve upstream score ordering and permit arbitrary
ordering and selection among ties. The NumPy float64 sampled oracle checks cardinality,
distances, membership, uniqueness and padding with common tolerances; incorrect
results raise and leave a FAILED row in `results.json`. Warp additionally checks
emitted tie order. `--validate-samples` controls oracle coverage. Fixtures are
seeded and saved as common row-major little-endian float32 binary arrays.
Outputs are separate `.scores` float32 and `.ids`, `.count`, `.status` int32
arrays; raw per-repeat milliseconds and provenance are retained.

Main measurements are uncapped. `--cap-squared` enables a diagnostic comparison
against closed-cap NumPy semantics and deliberately fails incompatibilities.
At the pinned cuBQL revision, `insert_linear` checks `>= sqrDistMax` only when
its list is already full, accepting out-of-cap candidates while filling;
A host-only probe against the actual pinned header confirms cap=1 accepts a
score=4 candidate into a partially filled list. Exact-boundary behavior also
depends on upstream traversal. Newer revisions
may reject `>=` before insertion, giving open-cap semantics. Never silently
patch or post-filter the pinned query to claim cap equivalence. cudaKDTree's
radius-based candidate lists also require boundary compatibility checks.
External wrappers require nonempty fixtures and cannot expose upstream traversal
failure status; synthesized status is 1 for any result and 0 otherwise. Empty,
invalid and numerical-failure contracts remain project implementation tests.
