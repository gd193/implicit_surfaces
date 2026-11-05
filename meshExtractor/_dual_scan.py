"""Two-level inclusive int32 scan; all payload/scratch buffers belong to JAX."""

import jax
import jax.numpy as jnp
import warp as wp

wp.set_module_options({"enable_backward": False})
_THREADS = 256
_TILE = wp.constant(1024)


@wp.kernel
def _blocks(data: wp.array(dtype=int), totals: wp.array(dtype=int)):
    block, lane = wp.tid()
    values = wp.tile_load(data, shape=_TILE, offset=block * _TILE)
    prefix = wp.tile_scan_inclusive(values)
    total = wp.tile_extract(prefix, _TILE - 1)
    wp.tile_store(data, prefix, offset=block * _TILE)
    if lane == 0:
        totals[block] = total


@wp.kernel
def _offsets(data: wp.array(dtype=int), totals: wp.array(dtype=int)):
    i = wp.tid()
    block = i // _TILE
    if block > 0:
        data[i] += totals[block - 1]


blocks = wp.jax_kernel(
    _blocks, num_outputs=2, in_out_argnames=["data"], block_dim=_THREADS, module_preload_mode=wp.JaxModulePreloadMode.NONE
)
offsets = wp.jax_kernel(
    _offsets, num_outputs=1, in_out_argnames=["data"], block_dim=_THREADS, module_preload_mode=wp.JaxModulePreloadMode.NONE
)


def _cuda_scan(data):
    n = data.shape[0]
    if not n:
        return data
    count = (n + _TILE - 1) // _TILE
    data, totals = blocks(data, launch_dims=(count, _THREADS), output_dims={"totals": count})
    totals = jnp.cumsum(totals, dtype=jnp.int32)
    return offsets(data, totals, launch_dims=n)[0]


def scan(data):
    return jax.lax.platform_dependent(data, cpu=lambda x: jnp.cumsum(x, dtype=jnp.int32), cuda=_cuda_scan)
