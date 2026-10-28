"""Resident-grid CUPTI comparison; sampling and layout preparation excluded."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from .geometry_check import compare_with_edges, warp_lattice_edges
from .vtkm_reference import VTKmReference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", default="0", help="physical GPU index or UUID")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--field", nargs="+", default=["sphere", "torus", "neural"])
    p.add_argument("--resolutions", nargs="+", type=int, default=[32, 64, 128])
    p.add_argument(
        "--backends",
        nargs="+",
        choices=["warp", "warp_compact", "vtkm_mc", "vtkm_fe"],
        default=["warp", "warp_compact", "vtkm_mc", "vtkm_fe"],
    )
    p.add_argument("--library", type=Path, help="built native reference library")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    args = p.parse_args()
    if min(*args.resolutions, args.repeats, args.warmup) < 1:
        p.error("resolutions, repeats and warmup must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import warp as wp
    from warp.geometry import IsoSurfaceMarchingCubes
    from meshExtractor.warp import _FieldEvaluator
    from scripts.bench.mesh_extraction._cases import fixture, LOWER, UPPER
    from scripts.bench.mesh_extraction.benchmark import trace_samples, source_digest
    from .build_reference import source_hashes

    wp.config.kernel_cache_dir = str(args.output.resolve().parent / "warp-cache")
    device = jax.devices()[0]
    if device.platform != "gpu":
        raise RuntimeError("CUDA required")
    wp.init()
    wp.set_device("cuda:0")
    from meshExtractor.warp import _extract_dense as compact_extract

    reference = VTKmReference(args.library)
    initial_sources = source_hashes()
    repository_digest = source_digest()
    build_manifest_path = reference.library_path.parent.parent / "build_manifest.json"
    build_manifest = json.loads(build_manifest_path.read_text())

    library_digest = hashlib.sha256(reference.library_path.read_bytes()).hexdigest()
    if (
        build_manifest.get("source_sha256") != initial_sources
        or build_manifest.get("library_sha256") != library_digest
    ):
        raise RuntimeError(
            "reference source/library hashes differ from build manifest; rebuild first"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    metadata = {
        "jax_version": jax.__version__,
        "warp_version": wp.__version__,
        "gpu": wp.get_device().name,
        "vtkm_commit": build_manifest["commit"],
        "float_precision": "FP32",
        "indices": "int32",
        "reference_source_sha256": initial_sources,
        "reference_library_sha256": library_digest,
        "repository_source_sha256": repository_digest,
        "build_manifest": build_manifest,
        "bounds": [LOWER, UPPER],
        "scope": "resident native contiguous extraction, allocation included; layout preparation/upload excluded",
    }
    for name in args.field:
        scene = fixture(name)
        for cells in args.resolutions:
            values = _FieldEvaluator(scene, 65536).dense(
                scene.params, (cells + 1,) * 3, LOWER, UPPER
            )
            wp.synchronize_device()
            preparation_start = time.perf_counter()
            host_values = values.numpy()
            native = wp.array(
                np.ascontiguousarray(host_values.transpose(2, 1, 0)),
                dtype=wp.float32,
                device="cuda:0",
            )
            wp.synchronize_device()
            preparation_ms = (time.perf_counter() - preparation_start) * 1000
            baseline = IsoSurfaceMarchingCubes.extract(values, 0.0, lower=LOWER, upper=UPPER)
            baseline_host = tuple(x.numpy().reshape(-1, 3) for x in baseline)
            baseline_edges = warp_lattice_edges(host_values)
            del baseline
            for backend in args.backends:

                def invoke(capture_edges=False):
                    if backend.startswith("warp"):
                        extract = (
                            compact_extract
                            if backend == "warp_compact"
                            else IsoSurfaceMarchingCubes.extract
                        )
                        mesh = extract(values, 0.0, lower=LOWER, upper=UPPER)
                    else:
                        mesh = reference.extract(
                            native,
                            lower=LOWER,
                            upper=UPPER,
                            synchronize_input=False,
                            capture_edges=capture_edges,
                            backend="flying_edges" if backend == "vtkm_fe" else "marching_cells",
                        )
                    wp.synchronize_device()
                    return mesh

                def release(mesh):
                    if not backend.startswith("warp"):
                        mesh.close()

                for _ in range(args.warmup):
                    mesh = invoke()
                    release(mesh)
                    del mesh
                wall = []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    mesh = invoke()
                    wall.append((time.perf_counter() - start) * 1000)
                    release(mesh)
                    del mesh
                trace_dir = args.output / f"{name}_{cells}_{backend}"
                with jax.profiler.trace(str(trace_dir)):
                    for i in range(args.repeats):
                        with jax.profiler.TraceAnnotation("mc_sample", sample=i, backend=backend):
                            mesh = invoke()
                        release(mesh)
                        del mesh
                (trace_file,) = trace_dir.rglob("*.trace.json.gz")
                samples, kernels = trace_samples(trace_file)
                if len(samples) != args.repeats:
                    raise RuntimeError("incomplete CUPTI trace")
                mesh = invoke(capture_edges=True)
                output = (
                    tuple(x.numpy().reshape(-1, 3) for x in mesh)
                    if backend.startswith("warp")
                    else mesh.numpy()
                )
                output_edges = baseline_edges if backend.startswith("warp") else mesh.edges()
                comparison = compare_with_edges(
                    *baseline_host, baseline_edges, *output, output_edges
                )
                release(mesh)
                del mesh
                if not comparison["finite_valid_indices"]:
                    raise RuntimeError("reference produced invalid mesh")
                if backend.startswith("warp") and (
                    not comparison.get("triangles_match_oriented")
                    or comparison.get("max_vertex_distance") != 0.0
                ):
                    raise RuntimeError("compact output differs from stock Warp")
                record = {
                    "field": name,
                    "cells_per_axis": cells,
                    "backend": backend,
                    "native_layout_preparation_wall_ms_excluded": preparation_ms,
                    "comparison_to_stock_warp": comparison,
                    "wall_ms": wall,
                    "device_samples": samples,
                    "median_wall_ms": float(np.median(wall)),
                    "median_kernel_ms": float(np.median([x["kernel_ms"] for x in samples])),
                    "median_memcpy_ms": float(np.median([x["memcpy_ms"] for x in samples])),
                    "median_memset_ms": float(np.median([x["memset_ms"] for x in samples])),
                    "median_device_activity_ms": float(
                        np.median(
                            [
                                sum(x[k] for k in ("kernel_ms", "memcpy_ms", "memset_ms"))
                                for x in samples
                            ]
                        )
                    ),
                    "top_kernel_total_us": kernels,
                    "trace": str(trace_file),
                }
                records.append(record)
                (args.output / "results.json").write_text(
                    json.dumps(
                        {"metadata": metadata, "records": records, "complete": False},
                        indent=2,
                    )
                    + "\n"
                )
                print(json.dumps(record), flush=True)

    if source_hashes() != initial_sources or source_digest() != repository_digest:
        raise RuntimeError("benchmark sources changed during capture")
    if hashlib.sha256(reference.library_path.read_bytes()).hexdigest() != library_digest:
        raise RuntimeError("reference library changed during capture")
    (args.output / "results.json").write_text(
        json.dumps({"metadata": metadata, "records": records, "complete": True}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
