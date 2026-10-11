"""Voxel tree: 2**dim children per node, split at box midpoints.

Traversal runs in int32 cell coordinates on the finest grid,
cell = floor((x - lower) / (upper - lower) * 2**max_depth). The child slot at
a depth is read from one bit of each axis' cell index (bit i of the slot is
axis i's upper half), so a point on a split plane always belongs to the upper
child. Box queries visit children whose cell range meets the box's
outward-rounded cell range.
"""
import warnings
from typing import Self

import jax
import jax.numpy as jnp
import numpy as np

from boundaryConditions.base import MIXED
from boundaryConditions.trees.base import TreePartition

# int32 cells must hold 2**max_depth. Beyond ~22 levels f32 cannot resolve
# cells anywhere in the root, so deeper trees need x64.
_MAX_DEPTH = 30
_F32_DEPTH = 22


class VoxelTreePartition(TreePartition):

    def _cells(self, x):
        """Normalized coordinate u in cell units, and a rounding bound on it."""
        ftype = x.dtype
        lower, upper = self.arrays.lower.astype(ftype), self.arrays.upper.astype(ftype)
        width = upper - lower
        n_cells = jnp.asarray(2 ** self.max_depth, ftype)
        u = (x - lower) / width * n_cells
        # Error from casting the root bounds and from (x - lower) / width,
        # widened generously. Absolute coordinates matter for offset roots.
        error = (8 * jnp.finfo(ftype).eps * n_cells
                 * (jnp.abs(x) + jnp.abs(lower) + width) / width)
        return u, error

    def _floor_cells(self, u):
        n_cells = 2 ** self.max_depth
        u = jnp.where(jnp.isfinite(u), jnp.floor(u), -1)
        return jnp.clip(u, -1, n_cells).astype(jnp.int32)

    def _encode_point(self, x):
        u, _ = self._cells(x)
        cell = self._floor_cells(u)
        inside = jnp.all((cell >= 0) & (cell < 2 ** self.max_depth))
        return inside, cell

    def _encode_box(self, lower, upper):
        u_lo, err_lo = self._cells(lower)
        u_hi, err_hi = self._cells(upper)
        lo, hi = self._floor_cells(u_lo - err_lo), self._floor_cells(u_hi + err_hi)
        n_cells = 2 ** self.max_depth
        inside = jnp.all((lo >= 0) & (hi < n_cells))
        outside = jnp.any((hi < 0) | (lo >= n_cells))
        return inside, outside, (jnp.clip(lo, 0, n_cells - 1), jnp.clip(hi, 0, n_cells - 1))

    def _slot(self, bits):
        return jnp.sum(bits << jnp.arange(self.dim, dtype=jnp.int32), dtype=jnp.int32)

    def _bits(self, cell, depth):
        return (cell >> (jnp.int32(self.max_depth - 1) - depth)) & jnp.int32(1)

    def _point_child(self, node, key, depth):
        return self._slot(self._bits(key, depth))

    def _root_context(self):
        return jnp.zeros(self.dim, jnp.int32)  # node origin in cells

    def _box_children(self, node, context, key, depth):
        size = jnp.int32(1) << (jnp.int32(self.max_depth - 1) - depth)
        slots = jnp.arange(2 ** self.dim, dtype=jnp.int32)
        bits = (slots[:, None] >> jnp.arange(self.dim, dtype=jnp.int32)) & jnp.int32(1)
        origins = context + bits * size  # context is the node origin in cells
        lo, hi = key
        overlap = jnp.all((origins <= hi) & (origins + size - 1 >= lo), axis=1)
        return overlap, origins

    @classmethod
    def from_grid(cls, labels, lower, upper, default_region: int = 0,
                  n_regions: int | None = None) -> Self:
        """labels: (n,)*dim region ids, n a power of two, cell (i, j, ...) at
        lower + (i, j, ...) * (upper - lower) / n. Homogeneous blocks collapse."""
        labels, n_regions = cls._validate_labels(labels, default_region, n_regions)
        dim, n = labels.ndim, labels.shape[0]
        if dim < 1 or any(s != n for s in labels.shape) or n & (n - 1):
            raise ValueError("labels must be a cube grid with power-of-two side")
        max_depth = n.bit_length() - 1
        if max_depth > _MAX_DEPTH:
            raise ValueError(f"max_depth {max_depth} exceeds {_MAX_DEPTH} (int32 cells)")
        if max_depth > _F32_DEPTH and not jax.config.read("jax_enable_x64"):
            warnings.warn(f"max_depth {max_depth} exceeds f32 cell resolution; enable x64")
        lower, upper = cls._validate_box(lower, upper, dim)

        # Bottom-up: per level, a block's label if uniform, else MIXED.
        levels = [labels]
        for _ in range(max_depth):
            finer = levels[-1]
            m = finer.shape[0] // 2
            blocks = finer.reshape(sum(((m, 2) for _ in range(dim)), ()))
            blocks = blocks.transpose(tuple(range(0, 2 * dim, 2)) + tuple(range(1, 2 * dim, 2)))
            blocks = blocks.reshape((m,) * dim + (-1,))
            first = blocks[..., 0]
            uniform = np.all(blocks == first[..., None], axis=-1) & (first != MIXED)
            levels.append(np.where(uniform, first, MIXED).astype(np.int32))
        levels.reverse()  # levels[k] has side 2**k

        # Top-down, one level at a time: emit nodes and wire internal ones.
        offsets = (np.arange(2 ** dim)[:, None] >> np.arange(dim)[None, :]) & 1  # slot -> bits
        coords = np.zeros((1, dim), np.int64)
        children_parts, uniform_parts = [], []
        n_emitted = 1
        for depth in range(max_depth + 1):
            label = levels[depth][tuple(coords.T)]
            internal = label == MIXED
            children = np.full((len(coords), 2 ** dim), -1, np.int64)
            n_internal = int(internal.sum())
            children[internal] = n_emitted + np.arange(n_internal * 2 ** dim).reshape(n_internal, 2 ** dim)
            children_parts.append(children)
            uniform_parts.append(label)
            n_emitted += n_internal * 2 ** dim
            coords = (2 * coords[internal][:, None, :] + offsets[None]).reshape(-1, dim)
        arrays = cls._make_arrays(np.concatenate(children_parts), np.concatenate(uniform_parts),
                                  lower, upper)
        return cls(arrays, dim=dim, n_regions=n_regions,
                   default_region=default_region, max_depth=max_depth)
