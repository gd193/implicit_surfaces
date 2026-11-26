# JAX mesh extraction

Requires `environment-warp.yml` (Warp 1.18.0) and CUDA JAX; tested with
JAX/JAXLIB 0.10.2. Adaptive dual contouring uses SciPy for the bounded least-squares fit. The reference tests also require scikit-image.

The adaptive marching-cubes and dual-contouring pipelines use the same field,
isovalue and world-space tolerance. For a resolution of N cells per axis,
the harness uses the cell diagonal (2√3/N) as the tolerance and caps octree
depth at ⌈log₂ N⌉. Both report unresolved boxes alongside triangle counts
and field residuals. See [mesh extraction](../mesh_extraction.md).

Dense extraction imports Warp's kernels and tables. It scans in place and reuses
dead vertex-prefix storage as the edge map; geometry and output order match
stock Warp exactly. Private APIs require the explicit version pin. Revalidate
the buffer-alias proof before upgrading. Sampling retains JAX buffers through
their last asynchronous Warp use.

## Harness

Run from the repository root with artifacts on local storage:

```sh
python -m scripts.bench.mesh_extraction.benchmark \
  --gpu 0 --field sphere torus neural --resolutions 16 32 \
  --backend mc-adaptive dual-adaptive \
  --phase pipeline --output /path/to/local/mesh-run
```

Each case runs in a fresh process. Results include source hashes, versions,
GPU identity, raw samples, allocation reports and validation; traces stay beside
them. Failed cases or source changes fail the run. Output directories must be new.

| Backend | Pipeline | Extraction only |
|---|---|---|
| `warp-dense` | Bounded JAX sampling + stock Warp MC | Resident scalar grid + stock Warp MC |
| `compact-dense` | Same sampler + compact MC | Resident scalar grid + compact MC |
| `warp-sparse` | Warp Lipschitz octree + JAX sampling + sparse MC | Resident cells/corner values + sparse MC |
| `mc-adaptive` | Certified adaptive octree and marching-cubes triangulation | Pipeline only |
| `dual-adaptive` | Certified adaptive octree and Hermite triangulation | Pipeline only |
| `dual-dense` | Fixed-capacity dense dual contouring | Pipeline only |
| `dual-sparse` | Fixed-capacity sparse dual contouring | Pipeline only |

All backends use the same fields and isovalue. The neural fixture is a seeded
residual MLP. Export and validation run outside the timed region.

- **Device activity:** summed CUPTI kernel, copy and memset durations after
  warmup. Components are also recorded separately. Host gaps are excluded;
  summed activity is not critical-path latency.
- **Wall time:** separate synchronized, unprofiled calls including allocation,
  dispatch and count readbacks. Compilation is warmed out.
- **Warm memory:** JAX live-allocation peak per sample from XPlane allocation
  events, including resident input. Warp records the peak of newly tracked
  allocations across the profiling pass, including outputs and tiny CPU count
  buffers; pre-existing Warp pool usage is separate. These peaks have different
  scopes and must not be summed as a simultaneous GPU peak. Cold allocator
  high-water marks are retained separately and include compilation/preparation.
- **Validation:** indices, finite coordinates, bounds and field residuals;
  small parametrized tests also compare independent classic MC geometry,
  connectivity, empty outputs, rectangular grids and changing parameters.

## Independent CUDA reference

The benchmark-only adapter calls pinned upstream VTK-m MarchingCells and Flying
Edges. CUDA 13 compatibility fixes come from pinned Viskores, its maintained
successor; extraction kernels are unchanged. Both are BSD-style licensed; source
and licenses remain in the external build directory. The fixed-grid comparisons use Warp's Apache-2.0 kernels.

```sh
python -m scripts.bench.mesh_extraction.references.build_reference \
  --output /path/to/local/vtkm
python -m scripts.bench.mesh_extraction.references.profile_reference \
  --library /path/to/local/vtkm/native-build/libvtkm_mc_reference.so \
  --field sphere torus gyroid --resolutions 128 256 \
  --output /path/to/local/reference-run
```

Reference timing covers extraction from identical resident FP32 values in each
library's native contiguous layout. Layout preparation/upload is excluded and
recorded separately. Edge IDs are captured only during untimed validation.
Flying Edges matches lattice vertices and triangle counts but uses alternate
polygon diagonals; all 256 isolated cell cases have matching polygon boundaries.
Generic MarchingCells differs at exact-level corners, with a lattice-edge
mismatch on gyroid256; it is not an exact-geometry parity reference.

## A40 results

Median device activity in ms, five warmed samples, resident extraction:

| Field / cells | Stock Warp | Compact | Flying Edges |
|---|---:|---:|---:|
| Sphere / 128³ | 0.616 | 0.479 | 1.662 |
| Sphere / 256³ | 4.688 | 3.629 | 6.089 |
| Torus / 128³ | 0.614 | 0.476 | 1.401 |
| Torus / 256³ | 4.671 | 3.617 | 5.427 |
| Gyroid / 128³ | 0.717 | 0.578 | 4.294 |
| Gyroid / 256³ | 5.255 | 4.186 | 14.798 |

Compact extraction reduces device activity by 19–23% on these cases. At 256³,
warmed extraction allocation peaks fall from 612.6 to 275.1 MB (sphere) and
623.6 to 306.2 MB (gyroid), excluding resident input and including output.
These memory comparisons are against stock Warp; reference-library memory
parity has not been measured. The independent timing used the frozen compact
helper; the integrated helper adds validation and safe JAX sampling ownership.

Final integrated pipelines, including field sampling (same device metric):

| Field / cells | Stock dense | Compact dense | Sparse |
|---|---:|---:|---:|
| Sphere / 128³ | 0.774 | 0.646 | 0.350 |
| Sphere / 256³ | 5.869 | 4.908 | 1.092 |
| Residual MLP / 128³ | 3.090 | 2.977 | 0.599 |
| Residual MLP / 256³ | 22.632 | 21.774 | 1.839 |

Sparse MLP sampling improves from 1.045 to 0.599 ms at 128³ and 2.314 to
1.839 ms at 256³ against the pre-optimization paired baseline. Small evaluations
avoid full-batch padding; larger evaluations fuse padding, mapping and slicing.
Dense sampling remains bounded to two live scalar batches. A whole-grid JAX
sampling experiment improved dispatch time but increased memory and was rejected.

[Optimization samples and provenance](mesh_extraction_optimization.json) retain
the raw measurements. The [initial baseline](mesh_extraction_baseline.json)
predates these changes. Full reports and traces remain at the artifact paths
recorded in the JSON; historical cold-memory figures include compilation.
