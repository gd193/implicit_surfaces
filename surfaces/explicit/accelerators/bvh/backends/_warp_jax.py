"""JAX FFI; vmap batches queries while sharing BVH arrays."""
from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp

from ._warp_specs import query_kernel, query_spec


def _flatten_vec_queries(x, dimension=3):
    x = jnp.asarray(x, dtype=jnp.float32)
    if x.shape[-1:] != (dimension,):
        raise ValueError(f"expected query shape (..., {dimension}), got {x.shape}")
    batch_shape = x.shape[:-1]
    return x.reshape((-1, dimension)), batch_shape


def _restore(value, batch_shape):
    return value.reshape(batch_shape + value.shape[1:])


def _ray_args(origin, direction, minimum, maximum, dimension=3):
    origins, shape = _flatten_vec_queries(origin, dimension)
    directions, dshape = _flatten_vec_queries(direction, dimension)
    if shape != dshape:
        raise ValueError("ray_origin and ray_direction batch shapes must match")
    bounds = tuple(jnp.broadcast_to(jnp.asarray(v, jnp.float32), shape).reshape((-1,))
                   for v in (minimum, maximum))
    return (origins, directions, *bounds), shape


def _crossing_args(origin, direction, minimum, maximum, exclude_face, dimension=3):
    args, shape = _ray_args(origin, direction, minimum, maximum, dimension)
    face = jnp.asarray(exclude_face)
    if not jnp.issubdtype(face.dtype, jnp.integer):
        raise TypeError("exclude_face must contain integer face IDs")
    face32 = face.astype(jnp.int32)
    min_face = 0 if jnp.issubdtype(face.dtype, jnp.unsignedinteger) else -1
    face = jnp.where((face32 >= min_face) & (face32.astype(face.dtype) == face), face32, -2)
    return (*args, jnp.broadcast_to(face, shape).reshape((-1,))), shape


def query(kernel_name: str, payload, *query_args, output_shapes=None):
    """Run flat queries; output_shapes gives trailing dimensions per output."""
    spec = query_spec(kernel_name)
    if len(query_args) != spec.num_query_args:
        raise ValueError(f"{kernel_name} expects {spec.num_query_args} query arguments")
    return _batched_kernel(kernel_name, output_shapes)(*query_args, *spec.payload_arguments(payload))


@lru_cache(maxsize=None)
def _batched_kernel(kernel_name: str, output_shapes=None):
    spec = query_spec(kernel_name)
    return batched_kernel(query_kernel(kernel_name), spec.num_query_args,
                          spec.num_outputs, spec.block_dim, output_shapes)


@lru_cache(maxsize=None)
def batched_kernel(kernel, num_query_args, num_outputs, block_dim, output_shapes=None):
    import warp as wp

    ffi_call = wp.jax_kernel(kernel, num_outputs=num_outputs,
                              vmap_method="expand_dims", block_dim=block_dim)

    @jax.custom_batching.custom_vmap
    def selective_call(*args):
        dims = None
        if output_shapes is not None:
            dims = {arg.label: (args[0].shape[0],) + shape for arg, shape in
                    zip(kernel.adj.args[-num_outputs:], output_shapes, strict=True)}
        return tuple(ffi_call(*args, output_dims=dims))

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
