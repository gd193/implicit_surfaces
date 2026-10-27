# GPU query references

FCPW: 2D closest point, first hit, occlusion and Euclidean silhouette.
cuBQL: 2D/3D point nearest. Each uses upstream query code and its own tree.
cuBQL uses spatial median in 2D; its SAH builder supports only float3.

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
