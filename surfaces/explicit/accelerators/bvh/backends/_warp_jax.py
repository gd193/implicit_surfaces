"""JAX FFI; vmap batches queries while sharing BVH arrays."""
from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp

from ._warp_specs import query_kernel, query_spec


def query(kernel_name: str, payload, *query_args):
    """Run a flat query batch with explicit geometry."""
    spec = query_spec(kernel_name)
    if len(query_args) != spec.num_query_args:
        raise ValueError(f"{kernel_name} expects {spec.num_query_args} query arguments")
    return _batched_kernel(kernel_name)(*query_args, *spec.payload_arguments(payload))


@lru_cache(maxsize=None)
def _batched_kernel(kernel_name: str):
    spec = query_spec(kernel_name)
    num_query_args = spec.num_query_args
    import warp as wp

    ffi_call = wp.jax_kernel(query_kernel(kernel_name),
                             num_outputs=spec.num_outputs,
                             vmap_method="expand_dims",
                             block_dim=spec.block_dim)

    @jax.custom_batching.custom_vmap
    def selective_call(*args):
        return tuple(ffi_call(*args))

    @selective_call.def_vmap
    def selective_vmap(axis_size, in_batched, *args):
        query_args = args[:num_query_args]
        payload_args = args[num_query_args:]
        payload_batched = in_batched[num_query_args:]
        if any(payload_batched):
            raise ValueError("Warp BVH payload arrays must remain unbatched under vmap")

        query_count = query_args[0].shape[1 if in_batched[0] else 0]
        expanded = []
        for arg, is_batched in zip(query_args, in_batched[:num_query_args]):
            value = arg if is_batched else jnp.broadcast_to(arg, (axis_size,) + arg.shape)
            expanded.append(value.reshape((-1,) + value.shape[2:]))

        outputs = selective_call(*expanded, *payload_args)
        reshaped = tuple(out.reshape((axis_size, query_count) + out.shape[1:])
                         for out in outputs)
        return reshaped, (True,) * len(reshaped)

    return selective_call
