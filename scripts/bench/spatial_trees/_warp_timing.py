"""Resident Warp arrays, validation hooks and CUDA graph/event timing."""
from dataclasses import dataclass, field
import hashlib
from typing import Any

import jax
import numpy as np
import warp as wp


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


def _make_case(query, implementation, kernel, inputs, n, device, block_dim, output_shapes=None):
    parameters = kernel.adj.args[len(inputs):]
    shapes = output_shapes or ((n,),) * len(parameters)
    outputs = [wp.empty(shape, dtype=parameter.type.dtype, device=device)
               for shape, parameter in zip(shapes, parameters, strict=True)]
    converted = [wp.from_jax(value, dtype=parameter.type.dtype)
                 if isinstance(value, jax.Array) else value
                 for value, parameter in zip(inputs, kernel.adj.args[:len(inputs)], strict=True)]
    return _Case(query, implementation, kernel, converted, outputs, block_dim)


def _measure(cases, n, device, launches, repeats, validate=None):
    # Keep input owners alive for captured graphs.
    def launch(case):
        wp.launch(case.kernel, n, inputs=case.inputs, outputs=case.outputs,
                  device=device, block_dim=case.block_dim)

    with wp.ScopedDevice(device):
        for case in cases:
            for _ in range(5):
                launch(case)
            if validate is not None:
                validate(case, tuple(array.numpy() for array in case.outputs))
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
