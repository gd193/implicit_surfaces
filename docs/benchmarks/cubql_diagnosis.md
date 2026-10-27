# cuBQL nearest-query diagnosis

cuBQL is [Apache-2.0 source](https://github.com/NVIDIA/cuBQL), pinned here to
`20f9db19cdb160e2b83d8f05dfdb0437fe11fe24`. These controls use GPU 0 (A40),
resident buffers, block 64, 30 launches per CUDA graph and 15 interleaved samples.
Build, compilation, transfers and host dispatch are excluded.
[Measurements and provenance](cubql_diagnosis.json).

**Warp compilation does not explain the gap.** The actual upstream query has
identical query instructions under NVCC 13.2, NVRTC 12.9 and NVRTC 13.2.
Timings differ by less than 0.5%; the Warp launch wrapper adds 0.5–1.8%.
All 24 compiler/wrapper configurations produce bitwise-identical outputs.
NVRTC header preparation removes host-only utilities; an NVCC control verifies
the prepared headers against upstream.

**Fixed: nearest traversal reloaded leaf ranges already present in packed nodes.**
The shared walker now uses those ranges directly: two net production lines,
no new policy or traversal. Uncached feature-range views retain their behavior.
Before/after device times on our trees:

| Fixture | 2D | 3D |
|---|---:|---:|
| 1,024 points / 8,192 queries | 29.29 → 25.40 µs | 41.20 → 35.57 µs |
| 8,192 points / 65,536 queries | 88.71 → 84.00 µs | 115.64 → 107.90 µs |
| 65,536 points / 1,048,576 queries | 1.543 → 1.480 ms | 2.486 → 2.331 ms |

Small/large leaves use threshold 4; mass uses 16 in 2D and 8 in 3D.
All before/after outputs match bitwise, including replay on cuBQL topology.
The full 37-query matrix shows Euclidean nearest gains of 7–14% across points,
segments, triangles and edges. Other queries stay within 0.8% of the prior run.

**Remaining cost is partly candidate work.** Keeping cuBQL traversal unchanged,
adding dynamic caps and candidate tie handling costs little. Adding our per-item
bound/cull, score arithmetic and numeric checks raises 2D time by 17–24% and 3D
by 11–17%. Retaining the winning point has little effect. Our 2D traversal remains
11–15% slower than that richer callback on the same tree; our 3D traversal is
faster in these small/large controls. These are diagnostic callbacks: upstream
strict pruning remains, so full closed-cap/tie semantics are not equivalent,
despite stages 3/4 matching production bitwise on ordinary fixtures.
This is not a parity claim.

Disabling leaf-loop unrolling helps small 2D but hurts the mass fixture.
Compact 2D bounds halve requested bytes in bound-load instructions and lower
registers, but gain only 0.9–4.2% after the range fix. Neither was adopted. The mass comparison
still trails the earlier tuned cuBQL medians (1.019 ms in 2D, 1.756 ms in 3D);
the small/large callback controls do not fully explain that scale.

Validation: 455 GPU tests pass; final range timings and matrix use current code.
Nsight Compute counters are unavailable (`ERR_NVGPUCTRPERM`); conclusions use
controlled substitutions, device timings and compiled C++/PTX/SASS inspection.
