# cuBQL nearest-query comparison

**Our kernel uses 2–14% less device time on all six fixtures below.** NVIDIA A40,
GPU 0; 30 resident launches per CUDA graph, 15 samples. Build, JIT, transfers and
host dispatch are excluded. [Samples, source hashes and tuning](cubql_diagnosis.json).

| Fixture | Ours (µs) | cuBQL (µs) | Ours leaf/block | cuBQL leaf/block |
|---|---:|---:|---:|---:|
| Small 2D | 16.69 | 17.78 | 8/256 | 4/256 |
| Small 3D | 29.29 | 29.90 | 16/256 | 8/256 |
| Large 2D | 46.39 | 51.47 | 32/64 | 16/256 |
| Large 3D | 89.36 | 99.94 | 16/64 | 8/256 |
| Mass 2D | 891.32 | 1,037.07 | 32/64 | 16/64 |
| Mass 3D | 1,455.48 | 1,632.80 | 32/256 | 16/64 |

Small: 1,024 points / 8,192 queries; large: 8,192 / 65,536;
mass: 65,536 / 1,048,576. These are explicit tuned settings, not changed defaults.
Both libraries retain independent trees; cuBQL uses spatial median in 2D and SAH
in 3D. Larger leaves were checked for both libraries. These measurements cover
these fixtures and GPU, not every distribution or architecture.

The main gap was traversal scheduling. On the same trees, primitive visits matched,
but the old walker often executed leaf work with only 3 active lanes. Grouping
cheap point-query descents restores CUDA reconvergence before leaf processing.
Other callbacks retain their flat schedule. Child tests, stack encoding and
root/leaf/pop handling remain shared; specialization happens at compile time.

Clamping box coordinates before subtraction removes redundant arithmetic and
unifies the 2D/3D bound. Packed leaf ranges are reused directly. Caps, ties,
overflow handling and precision are unchanged. Removing numeric checks did not
justify adoption. Compiler controls found identical upstream query instructions
under NVCC/NVRTC; Warp launch wrapping contributed only 0.5–1.8%.

Validation: 457 GPU-suite tests and 45 focused CPU tests pass. All four outputs match immutable goldens bitwise; full membership and
distance checks plus sampled FP64 global-nearest oracles pass. The clamp formula
matches 400,000 CPU/GPU bound evaluations bitwise. The final 37-query matrix
keeps non-point timings within 1.1%, except triangle overlap at +3.6% (0.65 µs).
At the default leaf size 4, point-nearest changes by −22% in 2D and +3.8% in 3D;
the tuned settings above give the competitive results.

cuBQL is [Apache-2.0 source](https://github.com/NVIDIA/cuBQL), pinned to
`20f9db19cdb160e2b83d8f05dfdb0437fe11fe24`. Nsight hardware counters are unavailable;
the diagnosis uses controlled kernels, visit counts and C++/PTX/SASS inspection.
