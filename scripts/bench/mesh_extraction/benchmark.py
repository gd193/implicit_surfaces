"""Isolated mesh benchmarks: wall time, CUPTI kernel activity, allocator peaks."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]


def trace_samples(path, *, allow_no_kernels=False):
    with gzip.open(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    gpu = {e["pid"] for e in events if e.get("name") == "process_name"
           and "/device:GPU:" in e["args"]["name"]}
    samples, kernels = [], Counter()
    for marker in sorted((e for e in events if e.get("name") == "mc_sample"), key=lambda e: e["ts"]):
        activities = [e for e in events if e.get("pid") in gpu and e.get("ph") == "X"
                      and marker["ts"] <= e["ts"] < marker["ts"] + marker["dur"]]
        compute = [e for e in activities if "kernel_details" in e.get("args", {})]
        if not compute and not allow_no_kernels:
            raise RuntimeError("CUPTI recorded no kernels; device timing is unavailable")
        for e in compute:
            kernels[e["name"]] += e["dur"]
        samples.append({"kernel_ms": sum(e["dur"] for e in compute) / 1000,
                        "memcpy_ms": sum(e["dur"] for e in activities if e["name"].startswith("Memcpy")) / 1000,
                        "memset_ms": sum(e["dur"] for e in activities if e["name"].startswith("Memset")) / 1000,
                        "kernels": len(compute)})
    if not samples:
        raise RuntimeError("trace contains no benchmark samples")
    return samples, dict(kernels.most_common(12))


def warm_jax_peaks(path):
    from jaxlib._profile_data import ProfileData

    profile = ProfileData.from_file(str(path))
    markers, allocations = [], []
    for plane in profile.planes:
        for line in plane.lines:
            for event in line.events:
                if event.name == "mc_sample":
                    markers.append((event.start_ns, event.end_ns, dict(event.stats)["jax_live_bytes"]))
                elif event.name in ("MemoryAllocation", "MemoryDeallocation"):
                    stats = dict(event.stats)
                    if stats.get("allocator_name", "").startswith("GPU_"):
                        allocations.append((event.start_ns, stats["bytes_allocated"]))
    if not markers:
        raise RuntimeError("no warmed memory markers")
    return [max([initial] + [size for time, size in allocations if start <= time <= end])
            for start, end, initial in sorted(markers)]


def worker(args):
    # Imports follow device/allocator configuration in the parent process.
    import jax
    import numpy as np
    import warp as wp
    from ._cases import fixture, host_mesh, make_runner, validate_mesh

    device = jax.devices()[0]
    if device.platform != "gpu":
        raise RuntimeError("mesh benchmarks require CUDA")
    wp.init()
    wp.set_device(wp.device_from_jax(device))
    case_dir = args.output
    case_dir.mkdir(parents=True, exist_ok=True)
    scene = fixture(args.field[0], args.level)
    with wp.ScopedMemoryTracker("mesh", print=False) as tracker:
        run, details = make_runner(args.backend[0], scene, args.resolutions[0], args.phase[0], args.batch_size)
        mesh = run()
        jax.block_until_ready(mesh) if args.backend[0] == "legacy" else wp.synchronize_device()
        stream = io.StringIO()
        tracker.report(file=stream, max_items=0)
        memory = {"scope": "fresh process: input preparation and first extraction, including compilation",
                  "jax": device.memory_stats(),
                  "warp_pool_peak_used_bytes": wp.get_mempool_used_mem_high(),
                  "warp_allocations": stream.getvalue()}
        del mesh

    def invoke():
        mesh = run()
        if args.backend[0] == "legacy":
            jax.block_until_ready(mesh)
        wp.synchronize_device()
        return mesh

    for _ in range(args.warmup):
        mesh = invoke()
        del mesh
    wall = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        mesh = invoke()
        wall.append((time.perf_counter() - start) * 1000)
        del mesh

    trace_dir = case_dir / "trace"
    wp.synchronize_device()
    with wp.ScopedMemoryTracker("warm", print=False) as tracker:
        tracker.clear()
        warp_resident = wp.get_mempool_used_mem_current()
        with jax.profiler.trace(str(trace_dir)):
            for sample in range(args.repeats):
                live = device.memory_stats()["bytes_in_use"]
                with jax.profiler.TraceAnnotation("mc_sample", sample=sample, jax_live_bytes=live):
                    mesh = invoke()
                    del mesh
        warm_warp_peak = wp._src.context.runtime.core.wp_alloc_tracker_get_peak_bytes()
    xplane, = trace_dir.rglob("*.xplane.pb")
    memory["warm"] = {"jax_peak_bytes_per_sample": warm_jax_peaks(xplane),
                      "warp_resident_pool_bytes": warp_resident,
                      "warp_new_allocations_peak_bytes_all_devices": warm_warp_peak}
    trace_file, = trace_dir.rglob("*.trace.json.gz")
    mesh = invoke()
    vertices, faces = host_mesh(mesh)
    validation = validate_mesh(vertices, faces, scene, args.resolutions[0])
    no_op = args.phase[0] == "extraction" and (
        (args.backend[0] == "legacy" and not len(faces)) or
        (args.backend[0] == "warp-sparse" and details["cells"] == 0))
    device_samples, kernels = trace_samples(trace_file, allow_no_kernels=no_op)
    if len(device_samples) != args.repeats:
        raise RuntimeError("incomplete device trace")
    result = {"backend": args.backend[0], "field": args.field[0], "cells_per_axis": args.resolutions[0],
              "phase": args.phase[0], "level": args.level, "batch_size": args.batch_size,
              "fixture_seed": 817, "bounds": [-1., 1.], "dtype": "float32", "warmup": args.warmup,
              "jax_version": jax.__version__, "warp_version": wp.__version__, "gpu": wp.get_device().name,
              "cuda_toolkit": wp.get_cuda_toolkit_version(), "driver": wp.get_cuda_driver_version(),
              "input": details, "mesh": validation, "memory": memory,
              "wall_ms": wall, "device_samples": device_samples,
              "median_wall_ms": float(np.median(wall)),
              "median_kernel_ms": float(np.median([s["kernel_ms"] for s in device_samples])),
              "median_device_activity_ms": float(np.median([
                  s["kernel_ms"] + s["memcpy_ms"] + s["memset_ms"] for s in device_samples])),
              "top_kernel_total_us": kernels, "trace": str(trace_file), "status": "ok"}
    (case_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def source_digest():
    digest = hashlib.sha256()
    for folder in (ROOT / "meshExtractor", ROOT / "surfaces/implicit", Path(__file__).parent):
        for path in sorted(folder.rglob("*.py")):
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    from ._cases import BACKENDS, FIELDS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--field", nargs="+", choices=FIELDS, default=["sphere", "torus", "neural"])
    parser.add_argument("--resolutions", nargs="+", type=int, default=[32, 64, 128], help="cells per axis")
    parser.add_argument("--phase", nargs="+", choices=("pipeline", "extraction"), default=["pipeline"])
    parser.add_argument("--level", type=float, default=0.)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--gpu", default="0", help="physical GPU index or UUID")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(*args.resolutions, args.batch_size, args.repeats) < 1 or args.warmup < 1:
        parser.error("resolutions, batch size, repeats and warmup must be positive")
    args.output = args.output.resolve()
    if args.worker:
        worker(args)
        return
    if args.output.exists():
        parser.error("output directory already exists; choose a fresh run directory")
    args.output.mkdir(parents=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, XLA_PYTHON_CLIENT_PREALLOCATE="false",
               PYTHONDONTWRITEBYTECODE="1")
    env.setdefault("WARP_CACHE_PATH", str(args.output.parent / "warp_cache"))
    manifest = {"source_sha256": source_digest(), "started_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "gpu_info": subprocess.check_output(["nvidia-smi", "--id", args.gpu,
                    "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"], text=True).strip(),
                "gpu_selector": args.gpu, "results": [], "complete": False}
    failed = False
    triangle_counts = {}
    for name in args.field:
        for resolution in args.resolutions:
            for phase in args.phase:
                for backend in args.backend:
                    label = f"{name}_{resolution}_{phase}_{backend}"
                    folder = args.output / label
                    folder.mkdir()
                    command = [sys.executable, "-m", "scripts.bench.mesh_extraction.benchmark", "--worker",
                               "--backend", backend, "--field", name, "--resolutions", str(resolution),
                               "--phase", phase, "--level", str(args.level), "--batch-size", str(args.batch_size),
                               "--repeats", str(args.repeats), "--warmup", str(args.warmup), "--output", str(folder)]
                    with (folder / "run.log").open("w") as log:
                        process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    if process.returncode:
                        failed = True
                        result = {"case": label, "status": "failed", "exit_code": process.returncode,
                                  "log": str(folder / "run.log")}
                        print(f"{label}: FAILED ({folder / 'run.log'})", flush=True)
                    else:
                        result = json.loads((folder / "result.json").read_text())
                        previous = triangle_counts.setdefault((name, resolution), result["mesh"]["triangles"])
                        if previous != result["mesh"]["triangles"]:
                            failed = True
                            result.update(status="failed", error=f"triangle count differs from peer: {previous}")
                        print(f"{label}: device activity {result['median_device_activity_ms']:.3f} ms, "
                              f"wall {result['median_wall_ms']:.3f} ms, "
                              f"{result['mesh']['triangles']} triangles", flush=True)
                    manifest["results"].append(result)
                    (args.output / "results.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if source_digest() != manifest["source_sha256"]:
        failed = True
        manifest["error"] = "benchmark sources changed during the run"
    manifest["complete"] = not failed
    (args.output / "results.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
