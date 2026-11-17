"""Small host-side helpers for Warp query kernel tests."""

import numpy as np
import warp as wp

from surfaces.explicit.accelerators.bvh.warp import configure_query_kernel


def device_payload(schema, host, device):
    """Copy a host tree or geometry into its Warp struct schema."""
    result = schema()
    for name, var in schema.vars.items():
        setattr(result, name, wp.array(np.asarray(getattr(host, name)),
                                       dtype=var.type.dtype, device=device))
    return result


def run_kernel(kernel, dim, inputs, output_specs, *, queries=(), device):
    """Configure and launch a query kernel, returning host NumPy outputs."""
    configure_query_kernel(kernel, *queries)
    outputs = [wp.empty(shape, dtype=dtype, device=device)
               for shape, dtype in output_specs]
    wp.launch(kernel, dim, inputs=inputs, outputs=outputs, device=device)
    return tuple(output.numpy() for output in outputs)
