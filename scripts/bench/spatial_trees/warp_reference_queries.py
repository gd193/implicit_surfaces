"""Matched FCPW segment and cuBQL point queries; resident CUDA kernels only."""
import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import jax
import numpy as np
import warp as wp

from scripts.bench.spatial_trees._warp_cases import PRIMITIVES, QueryCase, Scene, arguments, scene
from scripts.bench.spatial_trees._warp_timing import _array_digest, _make_case, _measure
from scripts.bench.spatial_trees.warp_query_matrix import ROOT, kernel_inputs
from surfaces.explicit.accelerators.bvh.warp import build_boundary, build_tree
from surfaces.explicit.accelerators.protocol import Config


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(name, size, leaf_size=None):
    config = Config(leaf_size_threshold=leaf_size) if leaf_size is not None else None
    if name.startswith("cluster"):
        primitive = PRIMITIVES["points"+name[-1]]
        rng = np.random.default_rng(619)
        centers = rng.normal(size=(7, primitive.dimension))
        points = centers[np.arange(size)%7]+rng.normal(size=(size, primitive.dimension))*np.geomspace(.005, .3, primitive.dimension)
        raw = primitive.data(points)
        return Scene(*build_tree(primitive, raw, config), None, raw)
    if name not in ("circle", "star"):
        return scene("segments2" if name == "square" else name, size, leaf_size or 4)
    angle = np.linspace(0., 2*np.pi, size, endpoint=False)
    vertices = np.column_stack((np.cos(angle), np.sin(angle))).astype(np.float32)
    if name == "star":
        vertices *= (1+.35*np.cos(7*angle))[:, None]
    raw = PRIMITIVES["segments2"].indexed(vertices, np.column_stack((np.arange(size), np.roll(np.arange(size), -1))))
    return Scene(*build_boundary(PRIMITIVES["segments2"], raw, config), raw)


def segment_projection(raw, points, index):
    assert np.all((index >= 0) & (index < len(raw.a)))
    a, b = (np.asarray(v, np.float64)[index] for v in (raw.a, raw.b))
    edge = b-a
    t = np.clip(np.sum((points-a)*edge, axis=1)/np.sum(edge*edge, axis=1), 0., 1.)
    return a+t[:, None]*edge


def validate(spec, actual, reference, geometry, queries, directions, features=None):
    hit = reference["id"] >= 0
    np.testing.assert_array_equal(actual[-1], hit)
    if spec.query in ("nearest", "silhouette"):
        np.testing.assert_allclose(actual[1], np.where(hit, reference["d"], np.inf), atol=1e-5, rtol=3e-5)
        for points, ids, distances, external in ((actual[0], actual[2], actual[1], False),
                                                (reference["p"], reference["id"], reference["d"], True)):
            p, index = points[hit].astype(np.float64), ids[hit]
            if spec.primitive.startswith("points"):
                assert np.all((index >= 0) & (index < len(geometry.raw.a)))
                expected = np.asarray(geometry.raw.a)[index]
            elif spec.query == "silhouette":
                vertices = features if external else geometry.boundary.silhouette.feature_a[:, :2]
                assert np.all((index >= 0) & (index < len(vertices)))
                expected = np.asarray(vertices)[index]
            else:
                expected = segment_projection(geometry.raw, p, index)
            np.testing.assert_allclose(p, expected, atol=1e-5, rtol=3e-5)
            np.testing.assert_allclose(np.linalg.norm(p-np.asarray(queries, np.float64)[hit], axis=1), distances[hit], atol=1e-5, rtol=3e-5)
    elif spec.query == "first":
        np.testing.assert_array_equal(actual[2], hit)
        np.testing.assert_allclose(actual[0], np.where(hit, reference["d"], 10.), atol=1e-5, rtol=3e-5)
        p = np.asarray(queries, np.float64)[hit]+actual[0][hit, None]*np.asarray(directions, np.float64)[hit]
        np.testing.assert_allclose(p, segment_projection(geometry.raw, p, actual[1][hit]), atol=1e-5, rtol=3e-5)
    else:
        np.testing.assert_array_equal(actual[0], hit)


def main():
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binaries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-reference", type=Path)
    parser.add_argument("--fixtures", nargs="+", choices=("square", "circle", "star", "points2", "points3", "cluster2", "cluster3"), default=["square", "circle", "points2", "points3"])
    parser.add_argument("--counts", type=int, default=8192)
    parser.add_argument("--primitives", type=int, default=1024)
    parser.add_argument("--launches", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--leaf-size", type=int)
    parser.add_argument("--reference-leaf-size", type=int, help="cuBQL leaf threshold; FCPW keeps its default")
    parser.add_argument("--block-dim", type=int, choices=(64, 128, 256))
    parser.add_argument("--queries", nargs="+", choices=("nearest", "first", "any", "silhouette"),
                        default=["nearest", "first", "any", "silhouette"])
    args = parser.parse_args()
    if min(args.counts, args.primitives, args.launches, args.repeats) < 1:
        parser.error("counts must be positive")
    if any(v is not None and v < 1 for v in (args.leaf_size, args.reference_leaf_size)):
        parser.error("leaf sizes must be positive")
    if jax.devices()[0].platform != "gpu":
        parser.error("CUDA GPU required")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    build = json.loads((args.binaries/"build.json").read_text())
    environment = os.environ.copy()
    cuda = Path(build["cuda_root"])
    environment.setdefault("CUDA_PATH", str(cuda/"targets/x86_64-linux" if (cuda/"targets/x86_64-linux").exists() else cuda))
    environment["LD_LIBRARY_PATH"] = str(cuda/"lib")+":"+environment.get("LD_LIBRARY_PATH", "")
    device = wp.device_from_jax(jax.devices()[0])
    runtime_paths = [*sorted((Path(build["references"]["fcpw"]["path"])/"include/fcpw/gpu").rglob("*.slang")),
                     *(Path(p) for p in build["commands"][0] if p.endswith(".so"))]
    fingerprint = dict(gpu_uuid=device.uuid, launches=args.launches, repeats=args.repeats,
                       runtime_sha256={str(p): digest(p) for p in runtime_paths})
    previous_report = None if args.reuse_reference is None else json.loads(args.reuse_reference.read_text())
    if previous_report is not None and previous_report.get("reference_fingerprint") != fingerprint:
        parser.error("reference GPU, runtime or timing configuration differs; measure fresh references")
    reference_rows = {} if previous_report is None else {r["case"]: r for r in previous_report["results"]}
    sources = [*sorted((ROOT/"surfaces/explicit/accelerators/bvh/backends").glob("*warp*.py")),
               ROOT/"surfaces/explicit/accelerators/bvh/backends/sah_jax.py", Path(__file__),
               *sorted(Path(__file__).parent.glob("_warp*.py")), Path(__file__).with_name("warp_query_matrix.py"),
               Path(__file__).with_name("build_references.py"),
               *sorted(Path(__file__).with_name("references").glob("*"))]
    sources = {str(p.relative_to(ROOT)): digest(p) for p in sources if p.is_file()}
    report = dict(gpu=device.name, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        versions={p: importlib.metadata.version(p) for p in ("warp-lang", "jax", "jaxlib", "numpy")},
        scope="Warmed CUDA graphs, kernel-only events. Build, compile, transfers, dispatch and Slang parameter uploads excluded. Independent trees: SAH except cuBQL 2D spatial median (its SAH supports float3 only). Matching block sizes and inputs. FCPW miss outputs initialized before timing.",
        source_sha256=sources, build=build, reference_fingerprint=fingerprint, launches=args.launches, repeats=args.repeats,
        reference_timings_reused_from=str(args.reuse_reference) if args.reuse_reference else None,
        leaf_size=args.leaf_size, reference_leaf_size=args.reference_leaf_size, block_dim=args.block_dim, results=[])
    for name in args.fixtures:
        geometry = fixture(name, args.primitives, args.leaf_size)
        is_points = name.startswith(("points", "cluster"))
        primitive = "points"+name[-1] if is_points else "segments2"
        dimension = PRIMITIVES[primitive].dimension
        raw = geometry.raw
        modes = ("nearest",) if is_points else ("nearest", "first", "any", "silhouette")
        for mode in modes:
            if mode not in args.queries:
                continue
            folder = args.output/name/mode
            folder.mkdir(parents=True, exist_ok=True)
            np.asarray(raw.a if is_points else raw.vertices, np.float32).tofile(folder/"vertices.f32")
            if not is_points:
                np.asarray(raw.indices, np.int32).tofile(folder/"indices.i32")
            spec = QueryCase(primitive, mode, "euclidean" if mode in ("nearest", "silhouette") else "none")
            label = name+"/"+mode
            values, _ = arguments(spec, geometry, args.counts)
            points = values[3] if mode == "silhouette" else values[2]
            directions = values[3] if mode in ("first", "any") else np.zeros_like(points)
            np.asarray(points, np.float32).tofile(folder/"points.f32")
            np.asarray(directions, np.float32).tofile(folder/"directions.f32")
            call, inputs, shapes = kernel_inputs(spec, geometry, args.counts)
            if args.block_dim is not None:
                call = replace(call, block_dim=args.block_dim)
            timed = _make_case(mode, "ours", call.kernel, inputs, args.counts, device, call.block_dim, shapes)
            _measure([timed], args.counts, device, args.launches, args.repeats)
            actual = tuple(a.numpy() for a in timed.outputs)
            np.savez_compressed(folder/(mode+"_ours.npz"), *actual)
            identity = dict(geometry_sha256=_array_digest(raw), query_sha256=_array_digest(inputs[:call.num_query_args]))
            prefix = folder/mode
            binary = args.binaries/(f"cubql_points{dimension}" if is_points else "fcpw_queries")
            if label in reference_rows:
                previous = reference_rows[label]
                assert all(previous[k] == v for k, v in identity.items()), label
                assert previous["reference_binary_sha256"] == digest(binary), label
                assert previous["block_dim"] == call.block_dim, label
                assert previous["reference"]["properties"].get("leaf_threshold", 4) == ((args.reference_leaf_size or 4) if is_points else 4), label
                reference = previous["reference"]
            else:
                command = ([str(binary), str(folder)] if is_points else
                           [str(binary), build["references"]["fcpw"]["path"], str(folder), mode])
                command += list(map(str, (call.block_dim, args.launches, args.repeats, prefix)))
                if is_points and args.reference_leaf_size is not None:
                    command.append(str(args.reference_leaf_size))
                with prefix.with_suffix(".log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment, check=True)
                samples = np.loadtxt(prefix.with_suffix(".ms"), ndmin=1).tolist()
                reference = dict(samples_ms=samples, p50_ms=float(np.median(samples)),
                                 output=str(prefix.with_suffix(".bin")), properties=json.loads(prefix.with_suffix(".json").read_text()),
                                 output_sha256={suffix: digest(prefix.with_suffix(suffix))
                                                for suffix in ((".bin", ".features") if mode == "silhouette" else (".bin",))})
            assert all(digest(Path(reference["output"]).with_suffix(suffix)) == value
                       for suffix, value in reference["output_sha256"].items()), label
            dtype = np.dtype([("p", "<f4", (dimension,)), ("d", "<f4"), ("id", "<i4")])
            expected = np.fromfile(reference["output"], dtype=dtype)
            assert len(expected) == args.counts
            features = np.fromfile(Path(reference["output"]).with_suffix(".features"), np.float32).reshape(-1, 2) if mode == "silhouette" else None
            validate(spec, actual, expected, geometry, points, directions, features)
            artifacts = folder/(mode+"_compiled")
            artifacts.mkdir(exist_ok=True)
            cache = Path(wp.config.kernel_cache_dir)/call.kernel.module.get_module_identifier(call.block_dim)
            compiled = {}
            for path in cache.iterdir():
                if path.suffix in (".cu", ".ptx"):
                    shutil.copy2(path, artifacts/path.name)
                    compiled[path.name] = digest(path)
            ptx_paths = [artifacts/name for name in compiled if name.endswith(".ptx")]
            assert len(ptx_paths) == 1
            ptx = ptx_paths[0].read_text().split(".entry", 1)[1]
            leaves = np.asarray(geometry.tree.node_leaf_count)
            ours = dict(samples_ms=timed.samples_ms, p50_ms=float(np.median(timed.samples_ms)),
                        properties=wp.get_cuda_kernel_properties(call.kernel, device=device, block_dim=call.block_dim))
            row = dict(case=label, n_queries=args.counts, n_primitives=len(geometry.tree.prim_ids), block_dim=call.block_dim,
                       **identity, reference_name="cuBQL" if is_points else "FCPW", reference_binary_sha256=digest(binary),
                       ours=ours, reference=reference, ours_over_reference=ours["p50_ms"]/reference["p50_ms"],
                       tree=dict(nodes=len(leaves), max_leaf=int(leaves.max())), compiled_sha256=compiled,
                       static_ptx=dict(calls=len(re.findall(r"\bcall(?:\.|\s)", ptx)),
                                       f64_divisions=len(re.findall(r"\bdiv\.rn\.f64", ptx)),
                                       f64_ops=len(re.findall(r"^\s*[\w.]+\.f64\b", ptx, re.M))))
            report["results"].append(row)
            print(f"{label:22} ours={ours['p50_ms']:.6f} reference={reference['p50_ms']:.6f} ratio={row['ours_over_reference']:.3f}", flush=True)
            assert all(digest(ROOT/p) == h for p, h in sources.items())
            (args.output/"results.json").write_text(json.dumps(report, indent=2)+"\n")
    assert all(digest(Path(p)) == h for p, h in fingerprint["runtime_sha256"].items())


if __name__ == "__main__":
    main()
