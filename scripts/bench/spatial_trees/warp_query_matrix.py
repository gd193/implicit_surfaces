"""Device-only timings for every builtin primitive/metric/query combination."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp

from scripts.bench.spatial_trees._warp_cases import CASES, RAYS, arguments, query_factory, scene
from scripts.bench.spatial_trees._warp_timing import _array_digest, _make_case, _measure

from surfaces.explicit.accelerators.bvh.backends._warp_factory import struct_arrays

ROOT = Path(__file__).resolve().parents[3]


def kernel_inputs(case, geometry, n):
    call = query_factory(case)
    args, kwargs = arguments(case, geometry, n)
    tree, data, *values = args
    metadata, shapes = (), None
    query = case.query
    if query in RAYS:
        minimum = kwargs.get("min_t", jnp.zeros(n, jnp.float32))
        values = [*values, minimum, kwargs["max_t"]]
        if call.exclusion:
            values.append(jnp.full((n,), -1, jnp.int32))
        if query == "all":
            shapes = ((n, kwargs["max_hits"]), (n, kwargs["max_hits"]), (n,), (n,))
    elif query in ("silhouette", "star"):
        boundary, *values = values
        call = call.query if query == "silhouette" else call.silhouette.query
        data = boundary.silhouette
        tree = tree._replace(node_leaf_offset=data.node_feature_offset,
                             node_leaf_count=data.node_feature_count, prim_ids=data.feature_refs)
    elif query in ("classify", "sample", "robin"):
        meta, *values = values
        metadata = ((meta.core,) if query == "classify" else
                    (meta,) if query == "sample" else (meta.core, meta.silhouette, meta.coefficients))
    flat = [*values, *tree, *data, *(a for group, schema in zip(metadata, call.group_types, strict=True)
                                    for a in struct_arrays(group, schema))]
    return call, flat, shapes


def main():
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=int, default=8192)
    parser.add_argument("--primitives", type=int, default=1024, help="Target fixture size; actual counts are recorded.")
    parser.add_argument("--launches", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--output", type=Path, default=ROOT/"assets/benchmark_results/warp_query_matrix.json")
    args = parser.parse_args()
    if min(args.counts, args.primitives, args.launches, args.repeats) < 1:
        parser.error("counts, primitives, launches and repeats must be positive")
    if jax.devices()[0].platform != "gpu":
        parser.error("CUDA-event timings require a GPU")
    device = wp.device_from_jax(jax.devices()[0])
    paths = [*sorted((ROOT/"surfaces/explicit/accelerators/bvh/backends").glob("*warp*.py")),
             ROOT/"surfaces/explicit/accelerators/bvh/backends/sah_jax.py",
             Path(__file__), Path(__file__).with_name("_warp_cases.py"), Path(__file__).with_name("_warp_timing.py")]
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report = dict(gpu=device.name, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), versions={p: importlib.metadata.version(p) for p in ("warp-lang", "jax", "jaxlib", "numpy", "trimesh")},
                  metric="cuda_event_graph_kernel_ms", launches_per_graph=args.launches, samples=args.repeats, warmup_launches=5,
                  fixture="Random points/segments; subdivided unit square/cube boundaries; Gaussian queries. Caps .75, except unbounded full-boundary sampling.",
                  scope="Resident tree kernel only; build, compile, transfers and dispatch excluded. Star reuses silhouette traversal; JAX cap composition excluded.",
                  source_sha256=sources, results=[])
    for spec in CASES:
        geometry = scene(spec.primitive, args.primitives)
        call, inputs, shapes = kernel_inputs(spec, geometry, args.counts)
        case = _make_case(spec.query, "custom", call.kernel, inputs, args.counts, device, call.block_dim, shapes)
        _measure([case], args.counts, device, args.launches, args.repeats)
        assert np.all(case.outputs[-1].numpy() >= 0), spec.id
        row = dict(case=spec.id, primitive=spec.primitive, query=spec.query, distance_metric=spec.metric,
                   n_queries=args.counts, n_primitives=len(geometry.tree.prim_ids), block_dim=call.block_dim,
                   geometry_sha256=_array_digest(geometry.raw), query_sha256=_array_digest(inputs[:call.num_query_args]),
                   device_p50_ms=float(np.median(case.samples_ms)), samples_ms=case.samples_ms,
                   properties=wp.get_cuda_kernel_properties(call.kernel, device=device, block_dim=call.block_dim))
        report["results"].append(row)
        print(f"{spec.id:38s} {row['device_p50_ms']:.6f} ms", flush=True)
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == digest for p, digest in sources.items())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
