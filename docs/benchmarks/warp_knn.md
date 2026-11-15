# Generic Warp k-nearest: GPU comparisons

The original comparison below predates the large-k heap refinement described
later in this document. Its tables retain the original measurements.

Performance reports include pinned cudaKDTree fixed-list and heap timings for
each supported matched workload, alongside our before/after measurements.
Include cuBQL when available. If an external library does not support the
primitive, metric, or output contract being measured, state that explicitly;
an internal speedup alone does not establish competitiveness with that library.

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

## Lower-k unrolled candidate comparison

A fresh 3D comparison measured k=1,4,8,16,32,64 on the A40 with 16,384
points and 32,768 queries. Both Warp configurations use the same unrolled
candidate list, cached bounds, grouped leaves, block 64, and SAH leaf threshold
4. The indexed configuration changes only `packed_nodes=False`. It is one
exposed option, not a search for the best settings for each workload. References
use the pins below and CCT traversal for both cudaKDTree variants.

Nine warmed graph samples contained ten launches each. Implementations and
Warp configurations ran sequentially; fixture hashes match. All 90 cases
passed global output validation and 64 sampled float64 oracle rows each.
Reference binaries were reused after verifying their archived wrapper source
hashes and identical C++ token sequences against the current formatting-only
wrapper changes. Building, compilation, transfers, and host dispatch are
excluded; sorted scores, original IDs, counts, and status are included.

Uniform-point times in milliseconds:

| k | Ours default | Ours indexed | cudaKDTree fixed | cudaKDTree heap | cuBQL |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.086 | 0.082 | 0.117 | 0.117 | 0.086 |
| 4 | 0.208 | 0.140 | 0.163 | 0.233 | 0.218 |
| 8 | 0.350 | 0.191 | 0.234 | 0.462 | 0.441 |
| 16 | 0.620 | 0.279 | 0.384 | 1.170 | 1.115 |
| 32 | 1.487 | 0.523 | 0.711 | 2.828 | 4.685 |
| 64 | 4.580 | 1.498 | 1.263 | 7.167 | 27.522 |

Indexed Warp beats cudaKDTree fixed at every tested k through 32 on all three
fixtures (uniform, clustered, and sphere surface). At k=64, its time relative
to fixed is 1.186 on uniform, 0.689 on clustered, and 1.061 on surface points.
The default packed-node configuration is slower than fixed on all three
fixtures at k=32 and 64. Indexed access does not always improve our own time:
at k=1 on surface points, default takes 0.0410 ms and indexed 0.0486 ms,
versus cudaKDTree fixed 0.0926 ms, heap 0.0925 ms, and cuBQL 0.0408 ms.

At k=64, compiled register counts are 168 for default Warp, 165 for indexed
Warp, 157 for cudaKDTree fixed, and 40 for both heap and cuBQL. Allocated local
storage is respectively 512, 512, 480, 992, and 1024 bytes per thread. These
counts include traversal storage and do not measure dynamic spills or traffic.
The substantial indexed-access gain with unchanged candidate selection shows
that unrolling alone does not explain the full-query performance difference.
These measurements do not isolate the remaining costs or establish results
for larger primitives, custom metrics, coordinate output, or other scales.

Reproduce with the reference harness using
`--dimensions 3 --sizes 16384 --queries 32768 --k 1 4 8 16 32 64 --launches 10 --repeats 9`.
Then rerun ours with `--implementations ours --node-access indexed --scheduling grouped`
and a separate output directory. Raw reports and the interactive comparison
remain outside Git.

## Large-k heap refinement

Candidate storage above k=64 now uses a four-child max heap. During sifting,
the selected child's packed score/ID value stays in a local variable rather
than being loaded again through the selected index. The register list through
k=64, storage layouts, traversal options, callback interface, and output
ordering are unchanged. Coordinate output moves its packed primitive slot
with the selected candidate as before.

The refined implementation and pinned references were measured on the same
A40 with 16,384 points, 32,768 queries, block 64, and uncapped squared
Euclidean scores plus original IDs. Ours uses default traversal options and
SAH leaf threshold 4; cudaKDTree uses CCT traversal. Each library builds its
own tree. Seven warmed CUDA-graph samples each contained three launches;
implementations ran separately, not interleaved. Building, compilation,
transfers, and host dispatch were excluded; output sorting and ID mapping
were included. All 24 current comparison cases passed float64 validation.

Times are milliseconds. “Ours before” comes from the earlier paired run
against `f30ccaf` on the same GPU; input hashes match this reference run.
Ratios above 1 favor cudaKDTree.

| Point distribution | k | Ours before | Ours now | cudaKDTree fixed | cudaKDTree heap | cuBQL | Ours / best KD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Uniform | 128 | 16.459 | 12.938 | 4.575 | 18.437 | 134.408 | 2.83 |
| Clustered | 128 | 16.056 | 12.728 | 6.084 | 18.252 | 132.466 | 2.09 |
| Surface | 128 | 12.678 | 9.355 | 3.620 | 16.220 | 75.292 | 2.58 |
| Uniform | 256 | 45.541 | 31.540 | 146.013 | 50.576 | 648.162 | 0.62 |
| Clustered | 256 | 47.738 | 32.868 | 178.256 | 50.952 | 599.096 | 0.65 |
| Surface | 256 | 41.667 | 27.959 | 111.390 | 47.807 | 430.773 | 0.58 |

cudaKDTree fixed remains faster at k=128 in these fixtures. At k=256, its heap
variant is the faster cudaKDTree baseline, and ours is faster than both
variants. This is a workload-specific comparison, not a universal ranking.

The original randomized interleaved before/after matrix also included weighted
squared callbacks (weights 1, 2, 4),
segments, segment coordinates, and surface-point coordinates. Their query
times fell 22.0–28.7%. All 14 pairs produced identical output arrays and passed
sampled float64 oracles. External baseline: **not measured** for these cases;
the pinned reference wrappers currently benchmark Euclidean point scores and
IDs, not these custom metrics, extended primitives, or coordinate outputs.
Registers and allocated local memory remained the same in each pair;
these measurements do not establish dynamic memory traffic
or achieved occupancy. Separate smaller matrices covered k=64/65/96/128/256
and uncached/indexed/flat traversal: all 50 pairs matched, with k=64 neutral.

Diagnostic removal of final sorting reduced full-query time only 1.3–5.6%.
That experiment changes compiler behavior and output order, so its timing
difference is not an isolated sorting duration. Wider heaps without retaining
the selected value, alternative heap layouts, and attempted front caches
did not improve the measured workloads and were not retained. The retained
change passed 99 CPU tests, 99 GPU tests, and 4,320 native randomized/boundary
probes with address and undefined-behavior sanitizers, plus focused source
review. Leak checking was disabled for the native probe because the execution
environment uses ptrace.

These gains are scoped to the tested A40 workloads. The k=128/256 table above
includes fresh external baselines; the earlier broad initial-comparison tables
retain their historical results.
Raw samples, frozen baseline sources, compiler hashes, and experimental
harnesses remain generated artifacts outside Git.

## Original comparison validation and scope

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

For the large-k refinement comparison, build the same pinned sources with
`--dimensions 3 --k 128 256 --traversals cct`, then run the comparison harness
with `--dimensions 3 --sizes 16384 --queries 32768 --k 128 256 --launches 3 --repeats 7`.
The default implementation list includes ours, cuBQL, and both cudaKDTree
candidate-list variants.

## Static composition validation

The query code now uses Python-selected Warp components for node access, stack
policy, scheduling, primitive operations, candidate selection, and output.
Public factories, payload layouts, builders, and native algorithms are unchanged
relative to the pre-refactor workspace, including its pending heap refinement.

On the A40, 95 randomized interleaved before/after cases produced bitwise-identical
outputs. Each case used 21 warmed graph samples of ten launches. The matrix covers
37 builtin query configurations, nearest traversal options, and kNN k=4,16,64,65,128,256
with points, segments, weighted squared callbacks, and ID/coordinate outputs.
Median after/before time across cases was 0.997; individual ratios ranged from
0.661 to 1.023. The worst observed increase was 2.3%, so this is evidence of close
performance on this matrix, not a guarantee of identical timing for every workload.
The larger regressions initially caused by extracting cached-pop control flow
were fixed by retaining that small loop in each scheduler.

Allocated local memory was unchanged in all 95 pairs. Registers were unchanged
in 90 pairs: triangle silhouette/star changed 47 to 48, triangle Robin 56 to 60,
and two flat recomputed segment configurations 42 to 44 and 44 to 45. These are
compiler allocation counts, not measurements of dynamic spills. One fused custom
call remains covered by the runtime tests. Full CPU and GPU suites each passed
536 tests; strict typing passed with zero valid-case diagnostics and all 52
expected negative diagnostics.

Fresh reference timings use cudaKDTree `37ce4c377170d13670632143b976192d5b69ed03`
and cuBQL `d2e62a39b83c1b8ca72c266703fdc0dfdcb67952`,
CCT for both cudaKDTree variants, and matching fixture hashes. Each library builds
its own tree. Ours uses SAH leaf threshold 4, block 64, cached bounds, grouped
leaves, and either packed or indexed node access. These are sequential warmed
graph runs: nine samples of ten launches through k=64, seven samples of three
launches at k=128/256. All 75 final implementation/fixture cases passed validation.
Compilation, building, transfers, and host dispatch are excluded. Sorted scores,
original IDs, counts, and status are included.

Uniform 3D points, N=16,384, Q=32,768; milliseconds:

| k | Ours packed | Ours indexed | cudaKDTree fixed | cudaKDTree heap | cuBQL |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.207 | 0.140 | 0.163 | 0.233 | 0.218 |
| 16 | 0.621 | 0.279 | 0.383 | 1.169 | 1.118 |
| 64 | 4.584 | 1.499 | 1.254 | 7.278 | 27.679 |
| 128 | 12.957 | 12.100 | 4.670 | 18.474 | 135.703 |
| 256 | 31.869 | 33.163 | 145.307 | 50.252 | 645.921 |

These reference wrappers do not provide matched custom-metric, segment, coordinate,
or other-query comparisons; external performance for those cases is **not measured**.
Their before/after checks assess the refactor only. Raw timings, compiler resource
records, source hashes, test logs, and the frozen baseline remain outside Git under
`/local00/bioinf/volkmann/warp-composition-jmiq7gho/`.

Reproduce the point comparison with `warp_knn_reference.py`, the pinned binaries,
`--dimensions 3 --sizes 16384 --queries 32768 --k 4 16 64 --launches 10 --repeats 9`.
For k=128/256 use `--k 128 256 --launches 3 --repeats 7`. Run ours again with
`--implementations ours --node-access indexed --scheduling grouped` into a separate
output directory.
