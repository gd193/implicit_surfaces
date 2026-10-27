# GPU query comparisons

**Initial tuning snapshot.** The [current cuBQL comparison](cubql_diagnosis.md)
reaches point-nearest parity on all six measured fixtures. NVIDIA A40, GPU 0,
resident CUDA kernels only. Medians use
30 launches per graph and 15 samples. Build, JIT, copies and host dispatch are
excluded. [Harness](../../scripts/bench/spatial_trees/references/README.md),
[samples, settings and provenance](query_references.json).

Times below are **ours / reference, in µs**. Small: 1,024 primitives and 8,192
queries. Large: 8,192 primitives and 65,536 queries.

| Query | Small | Large |
|---|---:|---:|
| Square first hit / FCPW | 18.30 / 20.51 | 44.03 / 42.77 |
| Square any hit / FCPW | 11.64 / 14.13 | 20.99 / 35.12 |
| Circle first hit / FCPW | 29.01 / 29.12 | 89.84 / 91.37 |
| Circle any hit / FCPW | 19.87 / 26.01 | 58.13 / 85.44 |
| 2D point nearest / cuBQL | 26.59 / 17.85 | 66.39 / 52.39 |
| 3D point nearest / cuBQL | 35.29 / 29.93 | 103.19 / 98.88 |

These use explicit leaf/block tuning, not changed production defaults. FCPW
retains leaf threshold 4; cuBQL sweeps leaves 2/4/8/16 and blocks 64/128/256.
Point references report the best sweep configuration. Trees are independent;
cuBQL uses spatial median in 2D and SAH in 3D. Default configurations and held-out
nonconvex/clustered cases are also recorded. On the held-out star, first/any hit
measure 52.60/36.59 µs versus FCPW's 57.28/52.63 µs.

Packed nearest-node loads and simpler metadata reduce traversal overhead.
Segment predicates use conservative FP32 certificates with the original FP64
fallback. Exact axis factors allow precise FP32 division where rounding is
preserved; PTX confirms `div.rn.f32`. No fast-math change or second query walker
was added. Earlier repairs fixed 2D SAH costs and eliminated the padded Z slab.

Conservative box intervals fix a one-ULP false miss exposed by small leaves.
The initial matrix showed triangle closest-point 15% faster and segment
count/all-hit 2–2.5× faster than the preceding implementation; the
[current 37-query matrix](warp_query_matrix.json) includes the leaf-range fix.
The box fix costs 4–5% on triangle first/any. Those queries retain their existing
FP32 primitive predicate; count/all use robust half-open crossings.

Matched tree replay finds similar node/primitive work for 2D point nearest;
its remaining gap is not explained by tree quality. Layout, callback-fusion,
stack and guard-removal experiments did not earn adoption. With 65,536 points
and 1,048,576 queries, matched tuned leaves give **1.541 / 1.019 ms in 2D** and
**2.492 / 1.756 ms in 3D**. Both libraries prefer block 64 here. The remaining
nearest gap is 42–51% at this scale.

Validation: 451 full-suite GPU tests plus eight generalized slab checks
(455 distinct tests), 122 distinct focused CPU tests, and independent numerical
stress checks. References call upstream query implementations:
[cuBQL](https://github.com/NVIDIA/cuBQL/blob/20f9db19cdb160e2b83d8f05dfdb0437fe11fe24/cuBQL/queries/pointData/findClosest.h),
[FCPW](https://github.com/rohan-sawhney/fcpw/blob/e498ca81fe01ae20ea3482485c90379307008de8/include/fcpw/gpu/geometry.slang).
The conservative box treatment follows [Ize's revised analysis](https://jcgt.org/published/0002/02/02/).
