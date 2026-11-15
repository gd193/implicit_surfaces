# Generic Warp k-nearest: initial GPU comparison

Generic kNN is implemented and correct on the tested workloads, but does not yet match cudaKDTree across all k. It accepts user Warp evaluator and conservative box-bound callbacks, compiled into one fused traversal/selection kernel. No Euclidean calculation is hardcoded in the query. The point benchmark explicitly supplies squared Euclidean callbacks.

NVIDIA A40, float32 points and queries, independent trees. Medians use nine warmed CUDA-graph samples, with 30 launches per sample for small cases and 10 otherwise. Build, compilation, transfers, and host invocation are excluded. Selection, output ordering, ID remapping and stores are included. All paths emit scores, original IDs, counts, and status; external status is synthesized.

Raw JSON samples, settings and provenance are generated artifacts kept outside
the repository. [Build/run instructions](../../scripts/bench/spatial_trees/references/README.md)
describe how to produce these reports; this document retains the results summary.

## Matched settings

Block size 64, explicit BVH leaf threshold 4, `cache_bounds=True`. The leaf setting is a benchmark choice, not the builder default. cuBQL uses its own 2D spatial-median / 3D SAH BVH; cudaKDTree uses its own thrust builder and CCT traversal. No upstream source changes. These are matched settings, not a claim that every library is optimally tuned.

The matrix covers 1,024 points / 1,024 queries and 16,384 points / 32,768 queries at k=1,4,8,16,32,64,128, plus 65,536 points / 32,768 queries at k=1,16,64. Every scale includes 2D/3D uniform, clustered and circle/sphere-surface fixtures. All 408 default-setting implementation/case results passed validation.

For 16,384 points and 32,768 queries, the table gives the range across the three fixtures. A ratio below 1 favors Warp.

| Dimension | k | Warp / cuBQL | Warp / cudaKDTree fixed |
|---|---:|---:|---:|
| 2D | 1 | 0.89–1.04 | 0.50–0.89 |
| 2D | 4 | 0.78–0.85 | 0.69–1.46 |
| 2D | 8 | 0.58–0.61 | 0.69–1.46 |
| 2D | 16 | 0.41–0.43 | 0.84–1.68 |
| 2D | 32 | 0.22–0.23 | 0.87–1.89 |
| 2D | 64 | 0.11–0.12 | 1.00–2.39 |
| 2D | 128 | 0.17–0.53 | 3.97–5.23 |
| 3D | 1 | 1.01–1.03 | 0.47–0.74 |
| 3D | 4 | 0.86–0.95 | 0.71–1.28 |
| 3D | 8 | 0.67–0.79 | 0.84–1.50 |
| 3D | 16 | 0.48–0.55 | 1.04–1.62 |
| 3D | 32 | 0.26–0.32 | 1.45–2.10 |
| 3D | 64 | 0.14–0.17 | 2.09–3.64 |
| 3D | 128 | 0.12–0.17 | 2.82–3.78 |

Warp is near cuBQL at k=1 and faster for k≥4 in this medium-size matrix. cudaKDTree fixed is often faster, especially at larger k. At k=128, Warp is approximately 2.8–5.2× slower than cudaKDTree fixed across these six fixtures. Small/large cases and the cudaKDTree heap variant remain in the JSON; performance varies by distribution and scale.

## Bound caching and larger k

The production default remains `cache_bounds=True`: arbitrary user bounds may be expensive. The historical `cache_bounds=False` comparison used indexed node access and flat scheduling as well as recomputed bounds. These are now separate options: reproduce that combination with `cache_bounds=False, packed_nodes=False, group_leaves=False`. The same block/leaf sizes and all three baselines validate 48 additional 3D cases. Its measured gain cannot be attributed to bound caching alone.

For 16,384 3D points and 32,768 queries, times below are milliseconds; the uncached Warp and reference columns were measured together. Cached Warp is from the separate default run.

| Fixture | k | Warp cached | Warp uncached | cuBQL | cudaKDTree fixed |
|---|---:|---:|---:|---:|---:|
| uniform | 16 | 0.621 | 0.378 | 1.117 | 0.384 |
| uniform | 64 | 4.581 | 2.309 | 27.688 | 1.252 |
| uniform | 128 | 16.867 | 17.426 | 134.000 | 4.426 |
| clustered | 16 | 0.633 | 0.399 | 1.162 | 0.567 |
| clustered | 64 | 4.182 | 2.171 | 27.477 | 2.011 |
| clustered | 128 | 16.401 | 17.574 | 133.653 | 5.940 |
| surface | 16 | 0.258 | 0.267 | 0.537 | 0.245 |
| surface | 64 | 2.051 | 1.566 | 15.397 | 0.970 |
| surface | 128 | 13.017 | 14.801 | 76.552 | 3.489 |

The exploratory Warp-only sweep additionally records leaves 1/4/8, blocks 32/64/128/256, and both cache modes. Its fastest settings are not substituted into the matched-setting table. Candidate storage uses an unrolled list through k=64 and a packed heap above that. Tested alternatives that unrolled k=128 were substantially slower and were rejected. Larger-k storage/traversal remains an optimization target; no universal speed-parity claim is made.

## Validation and scope

- Full spatial-tree GPU suite: 431 passed. After the final kNN output-store adjustment: all 36 kNN tests passed on GPU and CPU.
- Strict typing: zero valid-fixture diagnostics and all 49 expected negative diagnostics.
- Independent weighted L1/L∞ stress tests: 192 specializations/cases × 27 queries; native candidate storage: 600 randomized/boundary probes.
- One fused custom call verified with JIT and nested vmap; custom primitives, raw scores, predicates, ties, caps, padding, invalid conversions and stack failures covered.
- All 39 audited existing query kernels have identical normalized GPU machine code and bitwise-equal results against the committed baseline. Existing queries and builders were not modified.
- All 456 benchmark results passed global shape/count/status/padding/order checks and 64 sampled query rows per case against a float64 brute-force oracle. Non-Euclidean callback performance has no external apples-to-apples baseline here.

Upstream pins: cuBQL `d2e62a39b83c1b8ca72c266703fdc0dfdcb67952`, cudaKDTree `37ce4c377170d13670632143b976192d5b69ed03`. The pinned cuBQL implementation has incompatible finite-cap insertion semantics, so main comparisons are uncapped. External libraries may choose any equal-score subset; Warp deterministically favors lower original IDs. See the reference README for those compatibility limits.

## Reproduction

Build the pinned references using the linked instructions, then run:

```sh
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda,cpu python -m scripts.bench.spatial_trees.warp_knn_reference \
  --binaries "$ART/knn-bin" --output "$ART/medium" \
  --dimensions 2 3 --sizes 16384 --queries 32768 --launches 10 --repeats 9
```

Use `--sizes 1024 --queries 1024 --launches 30` for the small matrix; `--sizes 65536 --k 1 16 64` for the larger matrix. Add `--no-cache-bounds --node-access indexed --scheduling flat --dimensions 3 --k 4 16 64 128` to reproduce the historical uncached combination. The fixture seed is 619. Each report records input hashes, software versions, source and binary hashes, settings, GPU UUIDs, raw samples and validation outcomes.
