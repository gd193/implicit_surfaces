"""Resident pressure gather timing, excluding packing and host transfers.

Run from the repository root, for example::

    python scripts/bench/pressure/benchmark.py --output gather.json
    python scripts/bench/pressure/benchmark.py --fixture records.npz \
        --reference-module /path/to/baseline/warp_gather.py --output comparison.json

Fixtures contain points, velocities, records (interleaved seven-field walks,
then optional pseudo records), n_walks, continuations_per_root, regularization.
Use CUDA_VISIBLE_DEVICES to select an otherwise idle GPU. Compilation, five
warmup launches, allocation and reference error measurement are outside the CUDA event interval.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import socket
import sys

import jax
import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.bench.spatial_trees._warp_timing import _array_digest, _make_case  # noqa: E402


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resident_case(module, name, dtype, regularization, hops, inputs, q, records, device):
    block = module._BLOCK_SIZE
    partitions = module._partition_count(q, records) if hasattr(module, '_partition_count') else 1
    cooperative = False
    args = (dtype, regularization, True, hops)
    if hasattr(module, '_source_lanes'):
        cooperative = dtype == np.dtype('float64')
        if cooperative:
            partitions = module._source_lanes(inputs[2].shape[0], inputs[3].shape[0], hops)
        args += (partitions,)
    elif hasattr(module, '_gather_policy'):
        block, partitions, cooperative = module._gather_policy(q, records)
        args += (partitions, cooperative)
    elif hasattr(module, '_partition_count'):
        args += (partitions,)
    kernel = module._gather_kernel(*args).kernel
    targets = block // partitions if cooperative else block
    n = ((q + targets - 1) // targets) * targets * partitions
    case = _make_case(str(q), name, kernel, inputs, n, device, block,
                      ((q if cooperative else q * partitions, 3),))
    final = case.outputs[0]
    reducer = None
    if partitions > 1 and not cooperative:
        final = wp.empty((q, 3), dtype=case.outputs[0].dtype, device=device)
        reducer = module._reduce_kernel(dtype, partitions).kernel

    def launch():
        wp.launch(kernel, n, inputs=inputs, outputs=case.outputs,
                  device=device, block_dim=block)
        if reducer is not None:
            wp.launch(reducer, q, inputs=case.outputs, outputs=[final], device=device)

    with wp.ScopedDevice(device):
        for _ in range(5):
            launch()
        with wp.ScopedCapture(device=device) as capture:
            for _ in range(3):
                launch()
        case.graph = capture.graph
        wp.capture_launch(case.graph)
        wp.synchronize_device(device)
    scratch = q * partitions * 3 * dtype.itemsize if reducer is not None else 0
    return case, final, partitions, scratch


def _reference(points, velocities, records, regularization):
    """Double precision reference with bounded query-by-source workspace."""
    source = records.astype(np.float64)
    values = []
    for x, u in zip(points, velocities, strict=True):
        r = source[:, :3] - x.astype(np.float64)
        r2 = np.sum(r*r, axis=1)
        denominator = np.maximum(4*np.pi*np.sqrt(r2)*np.maximum(r2, regularization),
                                 np.finfo(points.dtype).tiny)
        weight = source[:, 3] + np.sum(source[:, 4:]*u.astype(np.float64), axis=1)
        values.append(np.sum((r/denominator[:, None])*weight[:, None], axis=0))
    return np.asarray(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--reference-module', type=Path)
    parser.add_argument('--queries', type=int, nargs='+', default=[64, 256, 2048])
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.queries) < 1:
        parser.error('queries must be positive')
    jax.config.update('jax_enable_x64', True)
    dtype = np.dtype(args.dtype)
    if args.fixture:
        with np.load(args.fixture) as fixture:
            points, velocities, records = (fixture[key].astype(dtype)
                for key in ('points', 'velocities', 'records'))
            walks = int(fixture['n_walks'])
            hops = int(fixture['continuations_per_root'])
            regularization = float(fixture['regularization'])
        if max(args.queries) > len(points):
            parser.error('fixture has fewer points than requested queries')
    else:
        rng = np.random.default_rng(51)
        points = rng.uniform(-.8, .8, (max(args.queries), 3)).astype(dtype)
        velocities = rng.normal(size=points.shape).astype(dtype)
        walks, hops, regularization = 1024, 6, 0.
        records = rng.normal(size=(walks*(hops+1), 7)).astype(dtype)
        records[:, :3] += np.asarray([2., 0., 0.], dtype=dtype)
        records.reshape(walks, hops+1, 7)[:, 1:, 4:] = 0.
    if (points.ndim != 2 or points.shape[1] != 3 or velocities.shape != points.shape
            or records.ndim != 2 or records.shape[1] != 7):
        parser.error('expected (Q,3) points/velocities and (M,7) records')
    if not np.isfinite(regularization) or regularization < 0:
        parser.error('regularization must be finite and nonnegative')
    if walks < 1 or not 0 <= hops <= 128 or len(records) < walks*(hops+1):
        parser.error('invalid walk layout or continuation count')
    layout = records[:walks*(hops+1)].reshape(walks, hops+1, 7)
    if np.any(layout[:, 1:, 4:] != 0):
        parser.error('continuations must have zero velocity coefficients')
    affine = np.pad(np.concatenate((layout[:, 0], records[walks*(hops+1):])),
                    ((0, 0), (0, 1)))
    scalar = layout[:, 1:, :4].reshape(-1, 4).copy()
    path = ROOT/'examples/fluids/pressure_prototype/warp_gather.py'
    modules = [('current', path)]
    if args.reference_module:
        modules.insert(0, ('reference', args.reference_module.resolve()))
    loaded = [(name, _load(path, f'pressure_bench_{name}')) for name, path in modules]
    wp.init()
    device = wp.get_device('cuda:0')
    report = {'environment': {'hostname': socket.gethostname(),
        'python': platform.python_version(), 'jax': jax.__version__, 'warp': wp.__version__,
        'device': device.name, 'uuid': device.uuid, 'architecture': device.arch,
        'driver_version': wp._src.context.runtime.driver_version,
        'toolkit_version': wp._src.context.runtime.toolkit_version},
        'source_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for name, path in modules},
        'inputs_sha256': _array_digest((points, velocities, records)),
        'fixture': str(args.fixture) if args.fixture else None,
        'dtype': dtype.name, 'records': len(records), 'walks': walks,
        'continuations_per_root': hops, 'regularization': regularization,
        'timing': 'resident CUDA graph including partial-sum reduction, 3 launches/replay, 7 randomized paired repeats', 'results': []}
    for q in args.queries:
        inputs = [wp.array(value, device=device)
                  for value in (points[:q], velocities[:q], affine, scalar)]
        cases = []
        for name, module in loaded:
            cases.append(_resident_case(module, name, dtype, regularization,
                                        hops, inputs, q, len(records), device))
        start, stop = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
        for repeat in range(7):
            for index in np.random.default_rng(404+repeat).permutation(len(cases)):
                case = cases[index][0]
                wp.record_event(start)
                wp.capture_launch(case.graph)
                wp.record_event(stop)
                case.samples_ms.append(wp.get_event_elapsed_time(start, stop)/3)
        outputs = [final.numpy() for _, final, _, _ in cases]
        indices = np.unique(np.linspace(0, q-1, min(q, 16), dtype=int))
        reference = _reference(points[indices], velocities[indices], records, regularization)
        for (case, _, partitions, scratch), output in zip(cases, outputs, strict=True):
            error = output[indices].astype(np.float64)-reference
            latency = float(np.median(case.samples_ms))
            report['results'].append({'queries': q, 'implementation': case.implementation,
                'median_ms': latency, 'samples_ms': case.samples_ms,
                'partitions': partitions, 'scratch_bytes': scratch,
                'interactions_per_second': q*len(records)/(latency*1.e-3),
                'reference_queries': len(reference), 'reference_indices': indices.tolist(),
                'max_error': float(np.max(np.abs(error))),
                'rms_error': float(np.sqrt(np.mean(error*error))),
                'reference_rms': float(np.sqrt(np.mean(reference*reference))),
                'bitwise_first_implementation': bool(np.array_equal(output, outputs[0]))})
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
