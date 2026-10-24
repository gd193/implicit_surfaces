"""CUDA-event kernel timings; build, compile and host dispatch excluded.
Run: python -m scripts.bench.spatial_trees.warp_bvh_device --native-warp"""
from __future__ import annotations

# ruff: noqa: E402

import argparse
from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp

from scripts.bench.spatial_trees._warp_fixtures import _mesh, _queries
from surfaces.explicit.accelerators import BVHAccelerator
from surfaces.explicit.accelerators.protocol import Config
from surfaces.explicit.accelerators.bvh.backends._warp_geometry import finite3
from surfaces.explicit.accelerators.bvh.backends._warp_specs import (
    QUERY_SPECS, query_kernel, query_spec,
)


@wp.func
def _valid_ray(o: wp.vec3, d: wp.vec3, lo: float, hi: float) -> bool:
    return (finite3(o) and finite3(d) and lo == lo and hi == hi
            and not wp.isinf(lo) and lo >= 0.0 and hi >= lo
            and wp.dot(d, d) > 1.e-30)


@wp.func
def _native_ray_result(o: wp.vec3, d: wp.vec3, lo: float, hi: float,
                       mesh_id: wp.uint64) -> tuple[float, int, int, int]:
    t = hi
    face = int(-1)
    hit = int(0)
    status = int(0)
    if (not finite3(o) or not finite3(d) or lo != lo or hi != hi or wp.isinf(lo)
            or lo < 0.0 or hi < lo or wp.dot(d, d) <= 1.e-30):
        status = -1
    else:
        q = wp.mesh_query_ray(mesh_id, o+lo*d, d, hi-lo)
        if q.result:
            t = q.t+lo
            face = q.face
            hit = 1
            status = 1
    return t, face, hit, status


@wp.kernel
def _native_first_hit(
    origins: wp.array(dtype=wp.vec3), directions: wp.array(dtype=wp.vec3),
    min_t: wp.array(dtype=float), max_t: wp.array(dtype=float), mesh_id: wp.uint64,
    out_t: wp.array(dtype=float), out_id: wp.array(dtype=int),
    out_hit: wp.array(dtype=int), out_status: wp.array(dtype=int),
):
    i = wp.tid()
    t, face, hit, status = _native_ray_result(
        origins[i], directions[i], min_t[i], max_t[i], mesh_id)
    out_t[i] = t
    out_id[i] = face
    out_hit[i] = hit
    out_status[i] = status


@wp.func_native(r"""
    return (wp::mesh_query_ray_count_intersections(mesh_id, point, wp::vec3(1.0f, 0.0f, 0.0f)) & 1)
        ? -1.0f : 1.0f;
""")
def _native_parity(mesh_id: wp.uint64, point: wp.vec3) -> float:
    ...


@wp.kernel
def _native_classify_point(
    points: wp.array(dtype=wp.vec3), mesh_id: wp.uint64,
    out_sign: wp.array(dtype=float), out_status: wp.array(dtype=int),
):
    i = wp.tid()
    out_sign[i] = wp.nan
    out_status[i] = -1
    if finite3(points[i]):
        out_sign[i] = _native_parity(mesh_id, points[i])
        out_status[i] = 1


@wp.kernel
def _native_closest_point(
    points: wp.array(dtype=wp.vec3), mesh_id: wp.uint64,
    out_point: wp.array(dtype=wp.vec3), out_distance: wp.array(dtype=float),
    out_id: wp.array(dtype=int), out_status: wp.array(dtype=int),
):
    i = wp.tid()
    p = points[i]
    out_point[i] = wp.vec3(0.0)
    out_distance[i] = wp.inf
    out_id[i] = -1
    out_status[i] = 0
    if not finite3(p):
        out_status[i] = -1
    else:
        q = wp.mesh_query_point_no_sign(mesh_id, p, wp.inf)
        if q.result:
            point = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
            out_point[i] = point
            out_distance[i] = wp.length(point-p)
            out_id[i] = q.face
            out_status[i] = 1


@wp.kernel
def _native_any_hit(
    origins: wp.array(dtype=wp.vec3), directions: wp.array(dtype=wp.vec3),
    min_t: wp.array(dtype=float), max_t: wp.array(dtype=float), mesh_id: wp.uint64,
    out_hit: wp.array(dtype=int), out_status: wp.array(dtype=int),
):
    i = wp.tid()
    o, d, lo, hi = origins[i], directions[i], min_t[i], max_t[i]
    out_hit[i] = 0
    out_status[i] = 0
    if not _valid_ray(o, d, lo, hi):
        out_status[i] = -1
    elif wp.mesh_query_ray_anyhit(mesh_id, o+lo*d, d, hi-lo):
        out_hit[i] = 1
        out_status[i] = 1


def _array_digest(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.asarray(value)
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass
class _Case:
    query: str
    implementation: str
    kernel: Any
    inputs: list[Any]
    outputs: list[Any]
    block_dim: int
    graph: Any = None
    samples_ms: list[float] = field(default_factory=list)


def _make_case(query, implementation, kernel, inputs, n, device, block_dim):
    spec = query_spec(query+"_kernel")
    outputs = [wp.empty(n, dtype=parameter.type.dtype, device=device)
               for parameter in kernel.adj.args[-spec.num_outputs:]]
    converted = [wp.from_jax(value, dtype=parameter.type.dtype)
                 if isinstance(value, jax.Array) else value
                 for value, parameter in zip(inputs, kernel.adj.args, strict=False)]
    return _Case(query, implementation, kernel, converted, outputs, block_dim)


def _measure(cases, n, device, launches, repeats):
    # Keep input owners alive for captured graphs.
    def launch(case):
        wp.launch(case.kernel, n, inputs=case.inputs, outputs=case.outputs,
                  device=device, block_dim=case.block_dim)

    with wp.ScopedDevice(device):
        reference = {}
        for case in cases:
            for _ in range(5):
                launch(case)
            got = tuple(array.numpy() for array in case.outputs)
            if case.query == "closest_point":
                found = got[3] == 1
                distance = np.linalg.norm(got[0][found]-case.inputs[0].numpy()[found], axis=1)
                np.testing.assert_allclose(got[1][found], distance, atol=3e-5, rtol=3e-5)
                assert np.all(got[2][found] >= 0)
            if case.implementation == "custom":
                reference[case.query] = got
            elif case.query == "first_hit":
                np.testing.assert_allclose(got[0], reference[case.query][0], atol=3e-5, rtol=3e-5)
                for actual, expected in zip(got[1:], reference[case.query][1:], strict=True):
                    np.testing.assert_array_equal(actual, expected)
            elif case.query == "closest_point":
                # Native Warp uses different tie and sliver rules.
                np.testing.assert_allclose(got[1], reference[case.query][1], atol=3e-5, rtol=3e-5)
                np.testing.assert_array_equal(got[3], reference[case.query][3])
            else:
                for actual, expected in zip(got, reference[case.query], strict=True):
                    np.testing.assert_array_equal(actual, expected)
            with wp.ScopedCapture(device=device) as capture:
                for _ in range(launches):
                    launch(case)
            case.graph = capture.graph
            wp.capture_launch(case.graph)
            wp.synchronize_device(device)

        start, stop = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
        for repeat in range(repeats):
            for index in np.random.default_rng(404+repeat).permutation(len(cases)):
                case = cases[index]
                wp.record_event(start)
                wp.capture_launch(case.graph)
                wp.record_event(stop)
                case.samples_ms.append(wp.get_event_elapsed_time(start, stop)/launches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("bvh.sah.warp", "bvh.snch.warp"), default="bvh.sah.warp")
    parser.add_argument("--queries", nargs="+", default=["first_hit", "any_hit"],
                        choices=[name.removesuffix("_kernel") for name in QUERY_SPECS])
    parser.add_argument("--counts", nargs="+", type=int, default=[65536])
    parser.add_argument("--mesh", choices=("sphere", "torus", "open_torus"), default="sphere")
    parser.add_argument("--subdivisions", type=int, default=5, help="Icosphere subdivisions.")
    parser.add_argument("--point-distribution", choices=("gaussian", "shell", "interior", "center"), default="gaussian")
    parser.add_argument("--leaf-size", type=int, default=4)
    parser.add_argument("--radius", type=float, default=0.5,
                        help="Cap for silhouette/Robin queries; radius for overlap/sampling.")
    parser.add_argument("--robin-coefficient", type=float, default=1.)
    parser.add_argument("--block-dim", type=int, help="Override each query's default for both implementations.")
    parser.add_argument("--launches", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--native-warp", action="store_true", help="Check and time native mesh query references.")
    parser.add_argument("--output", type=Path, default=ROOT/"assets/benchmark_results/warp_bvh_device.jsonl")
    args = parser.parse_args()
    if min(*args.counts, args.leaf_size, args.launches, args.repeats) < 1:
        parser.error("query counts, leaf size, launches and repeats must be positive")
    if max(args.counts) > np.iinfo(np.int32).max or args.subdivisions < 0:
        parser.error("query counts must fit int32 and subdivisions must be nonnegative")
    if args.block_dim is not None and not 1 <= args.block_dim <= 1024:
        parser.error("block dimension must be in [1, 1024]")
    if not np.isfinite(args.radius) or args.radius < 0:
        parser.error("radius must be finite and nonnegative")
    if np.isnan(args.robin_coefficient) or args.robin_coefficient < 0:
        parser.error("Robin coefficient must be nonnegative and not NaN")
    if args.backend != "bvh.snch.warp" and any(
            q in args.queries for q in ("closest_silhouette", "sample_boundary", "robin_radius")):
        parser.error("silhouette, sampling and Robin queries require --backend bvh.snch.warp")
    device = jax.devices()[0]
    if device.platform != "gpu":
        parser.error("CUDA-event measurements require a CUDA JAX device")
    wp.init()
    warp_device = wp.device_from_jax(device)
    backend_dir = ROOT/"surfaces/explicit/accelerators/bvh/backends"
    source_paths = sorted(backend_dir.glob("*warp*.py"))
    source_paths += [backend_dir/"snch_jax.py", Path(__file__), Path(__file__).with_name("_warp_fixtures.py")]
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    versions = {name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "warp-lang", "trimesh", "numpy")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with jax.default_device(device), args.output.open("a", encoding="utf-8") as output:
        primitives, ntri = _mesh(args.subdivisions, args.mesh)
        tree = BVHAccelerator.build(primitives, args.backend, Config(leaf_size_threshold=args.leaf_size))
        if "robin_radius" in args.queries:
            tree = tree.with_robin_coefficients(args.robin_coefficient)
        mesh = None
        if args.native_warp:
            mesh = wp.Mesh(points=wp.from_jax(primitives.mesh.vertices, dtype=wp.vec3),
                           indices=wp.from_jax(primitives.mesh.faces.reshape(-1), dtype=wp.int32),
                           bvh_constructor="sah", bvh_leaf_size=args.leaf_size)
        geometry_hash = _array_digest((primitives.mesh.vertices, primitives.mesh.faces))
        for n in args.counts:
            points, origins, directions, max_t = _queries(n, point_distribution=args.point_distribution)
            min_t = jnp.zeros_like(max_t)
            radii = jnp.full((n,), args.radius, dtype=jnp.float32)
            randoms = jnp.asarray(np.random.default_rng(808+n).random((n, 3), dtype=np.float32))
            query_inputs = {
                "classify_point": (points,), "closest_point": (points,), "first_hit": (origins, directions, min_t, max_t),
                "any_hit": (origins, directions, min_t, max_t),
                "aabb_overlap": (points-args.radius, points+args.radius),
                "closest_silhouette": (points, radii), "sample_boundary": (points, radii, randoms),
                "robin_radius": (points, radii),
            }
            cases = []
            for query in args.queries:
                name = query+"_kernel"
                spec = query_spec(name)
                block_dim = args.block_dim or spec.block_dim
                inputs = [*query_inputs[query], *spec.payload_arguments(tree.payload)]
                cases.append(_make_case(query, "custom", query_kernel(name), inputs, n, warp_device, block_dim))
                native_kernel = {"classify_point": _native_classify_point,
                                 "closest_point": _native_closest_point,
                                 "first_hit": _native_first_hit, "any_hit": _native_any_hit}.get(query)
                if mesh is not None and native_kernel is not None:
                    cases.append(_make_case(query, "native_mesh", native_kernel,
                                            [*query_inputs[query], mesh.id], n, warp_device, block_dim))
            _measure(cases, n, warp_device, args.launches, args.repeats)
            for case in cases:
                record = dict(
                    query=case.query, implementation=case.implementation,
                    backend=args.backend if case.implementation == "custom" else "warp.Mesh(sah)",
                    n_queries=n, n_triangles=ntri, leaf_size=args.leaf_size, block_dim=case.block_dim,
                    mesh=args.mesh,
                    point_distribution=args.point_distribution,
                    radius=args.radius, gpu=warp_device.name, versions=versions,
                    geometry_sha256=geometry_hash, query_sha256=_array_digest(query_inputs[case.query]),
                    source_sha256=sources, metric="cuda_event_graph_kernel_ms",
                    launches_per_graph=args.launches, repeats=args.repeats,
                    device_p50_ms=float(np.median(case.samples_ms)), device_samples_ms=case.samples_ms,
                    properties=wp.get_cuda_kernel_properties(case.kernel, device=warp_device, block_dim=case.block_dim),
                    result_count=int(np.count_nonzero(case.outputs[-1].numpy() == 1)),
                    invalid_count=int(np.count_nonzero(case.outputs[-1].numpy() < 0)),
                )
                if case.query == "classify_point":
                    record["reference_scope"] = ("classification only; native Warp intersection-count "
                                                 "helper, fixed +X ray; no closest-point work")
                    native_header = Path(wp.__file__).parent/"native/mesh.h"
                    record["native_mesh_header_sha256"] = hashlib.sha256(
                        native_header.read_bytes()).hexdigest()
                if case.query == "robin_radius":
                    record["robin_coefficient"] = ("inf" if np.isinf(args.robin_coefficient)
                                                    else args.robin_coefficient)
                output.write(json.dumps(record)+"\n")
                output.flush()
                print(f"{case.query:20s} {case.implementation:12s} {n:7d} queries "
                      f"{record['device_p50_ms']:.6f} ms", flush=True)


if __name__ == "__main__":
    main()
