"""Bounded, device-resident sparse DC input preparation; caller owns the outer JIT."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp
from jax import enable_x64
from meshExtractor._dual_resident_warp import decode_points
from meshExtractor._dual_common import _make_cell_test


class SparseInputs(NamedTuple):
    cells: jax.Array  # F x3, stable leaf prefix
    corner_map: jax.Array  # K, raw cell*8+corner -> sorted unique corner
    points: jax.Array  # K x3, unique prefix
    values: jax.Array  # K, unique prefix
    counts: jax.Array  # [-1,-1,-1,status,F_required,8*leaf_count,unique_count]


def _live_batches(fn, source, count, batch_size, dtype):
    """Evaluate scalar batches over a live prefix, masking at most one tail."""
    capacity = source.shape[0]
    B = min(batch_size, capacity)
    offsets = jnp.arange(B, dtype=jnp.int32)

    def batch(i, output):
        ids = i * B + offsets
        safe = jnp.minimum(ids, jnp.maximum(count - 1, 0))
        result = fn(source[safe])
        return output.at[ids].set(
            jnp.where(ids < count, result, 0), mode="drop", indices_are_sorted=True, unique_indices=True
        )

    return jax.lax.fori_loop(0, (count + B - 1) // B, batch, jnp.zeros(capacity, dtype=dtype))


def make_sparse_inputs(evaluate, cells_shape, lower, upper, level, bound, *, sparse_capacity):
    """Build a sparse grid under explicit frontier/raw-corner budgets."""
    F, K = sparse_capacity
    depth = (max(cells_shape) - 1).bit_length()
    stride = (1 << depth) + 1
    if stride**3 >= 2**63:
        raise ValueError("corner codes exceed int64 range")
    spacing = wp.vec3(*((np.asarray(upper) - lower) / np.asarray(cells_shape)))
    test = _make_cell_test(evaluate, cells_shape, lower, upper, level, bound)
    batch_size = evaluate.batch_size
    decode = wp.jax_kernel(decode_points, module_preload_mode=wp.JaxModulePreloadMode.NONE)

    def sample(params):
        state = (jnp.zeros((1, 3), jnp.int32), jnp.int32(1), jnp.int32(0), jnp.int32(1))
        for d in range(depth + 1):
            final = d == depth
            next_capacity = min(F, 8 ** (d if final else d + 1))

            def step(state, d=d, final=final, next_capacity=next_capacity):
                cells, live, status, required = state

                flags = _live_batches(lambda c: test(params, c, d), cells, live, batch_size, jnp.int32)
                scan = jnp.cumsum(flags, dtype=jnp.int32)
                kept = scan[-1]
                needed = kept if final else 8 * kept
                required = jnp.maximum(required, needed)

                def fit(_):
                    target = jnp.where(flags != 0, scan - 1, jnp.int32(cells.shape[0]))
                    compact = jnp.zeros_like(cells).at[target].set(cells, mode="drop")
                    if final:
                        return compact, kept, status, required
                    rows = jnp.arange(next_capacity, dtype=jnp.int32)
                    offsets = jnp.stack((rows & 1, (rows >> 1) & 1, (rows >> 2) & 1), axis=1)
                    children = 2 * compact[rows // 8] + offsets
                    children = jnp.where((rows < needed)[:, None], children, 0)
                    return children, needed, status, required

                def overflow(_):
                    return jnp.zeros((next_capacity, 3), jnp.int32), jnp.int32(0), status | jnp.int32(8), required

                return jax.lax.cond(needed <= F, fit, overflow, operand=None)

            def skip(s, next_capacity=next_capacity):
                return jnp.zeros((next_capacity, 3), jnp.int32), jnp.int32(0), s[2], s[3]

            state = jax.lax.cond((state[2] == 0) & (state[1] > 0), step, skip, state)
        cells, leaf_count, status, required = state
        cells = jnp.pad(cells, ((0, F - cells.shape[0]), (0, 0)))
        raw_count = jnp.where(status == 0, 8 * leaf_count, jnp.int32(-1))
        status |= jnp.where((status == 0) & (raw_count > K), jnp.int32(16), jnp.int32(0))

        # Keep integer topology in x64 without changing field/gradient precision.
        with enable_x64():
            ids = jnp.arange(K, dtype=jnp.int32)
            c = ids % 8
            offset = jnp.stack(((c ^ (c >> 1)) & 1, (c >> 1) & 1, c >> 2), axis=1)
            corners = cells[jnp.minimum(ids // 8, F - 1)] + offset
            q = corners.astype(jnp.int64)
            code = (q[:, 0] * stride + q[:, 1]) * stride + q[:, 2]
            sentinel = jnp.int64(np.iinfo(np.int64).max)
            code = jnp.where((status == 0) & (ids < raw_count), code, sentinel)
            # Unsigned keys enable CUB; keep the sort outside conditional graph nodes.
            code, perm = jax.lax.sort_key_val(code.astype(jnp.uint64), ids, is_stable=True)
            code = code.astype(jnp.int64)

        def dedupe(_):
            with enable_x64():
                first = (code != sentinel) & jnp.concatenate((jnp.ones(1, bool), code[1:] != code[:-1]))
                unique_scan = jnp.cumsum(first, dtype=jnp.int32)
                unique_count = unique_scan[-1]
                inverse = jnp.empty(K, jnp.int32).at[perm].set(unique_scan - 1)
                target = jnp.where(first, unique_scan - 1, jnp.int32(K))
                unique_code = jnp.zeros(K, jnp.int64).at[target].set(code, mode="drop")
            points = decode(
                unique_code, unique_count.reshape(1), stride, *lower, *tuple(spacing), launch_dims=K, output_dims=K
            )[0]
            values = _live_batches(
                lambda p: evaluate.evaluate(params, p), points, unique_count, batch_size, jnp.float32
            )
            return inverse, points, values, unique_count

        def unused(_):
            count = jnp.where(status == 0, jnp.int32(0), jnp.int32(-1))
            return jnp.zeros(K, jnp.int32), jnp.zeros((K, 3), jnp.float32), jnp.zeros(K, jnp.float32), count

        corner_map, points, values, unique_count = jax.lax.cond(
            (status == 0) & (raw_count > 0), dedupe, unused, operand=None
        )
        counts = jnp.stack((jnp.int32(-1), jnp.int32(-1), jnp.int32(-1), status, required, raw_count, unique_count))
        return SparseInputs(cells, corner_map, points, values, counts)

    return sample
