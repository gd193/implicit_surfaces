"""Voxel tree: 2**dim children per node, split at box midpoints.

Traversal runs in int32 cell coordinates on the finest grid,
cell = floor((x - lower) / (upper - lower) * 2**max_depth). The child slot at
a depth is read from one bit of each axis' cell index (bit i of the slot is
axis i's upper half), so a point on a split plane always belongs to the upper
child.

Box queries have fixed cost: pick the deepest level where the box covers at
most two cells per axis, descend to each of those <= 2**dim cells, and certify
only if every descent ends at a leaf and all leaves agree. A region boundary
inside a covered cell gives MIXED even if the box misses it.

Construction is JAX-only and runs on the default device: a label pyramid by
block reduction, node ids from prefix sums, then one scatter. The only host
transfers are the label range and the node count.
"""
import warnings
from functools import partial
from typing import Self

import jax
import jax.numpy as jnp

from boundaryConditions.base import MIXED
from boundaryConditions.trees.base import TreePartition

# int32 cells must hold 2**max_depth. Beyond ~22 levels f32 cannot resolve
# cells anywhere in the root, so deeper trees need x64.
_MAX_DEPTH = 30
_F32_DEPTH = 22


def _slot_bits(dim):
    """(2**dim, dim) int32: bit i of slot s selects axis i's upper half."""
    slots = jnp.arange(2 ** dim, dtype=jnp.int32)
    return (slots[:, None] >> jnp.arange(dim, dtype=jnp.int32)) & jnp.int32(1)


def _pyramid(labels):
    """levels[k] has side 2**k: a block's label if uniform, else MIXED."""
    dim, levels = labels.ndim, [labels]
    while levels[-1].shape[0] > 1:
        finer = levels[-1]
        m = finer.shape[0] // 2
        blocks = finer.reshape(sum(((m, 2) for _ in range(dim)), ()))
        axes = tuple(range(1, 2 * dim, 2))
        lo, hi = blocks.min(axis=axes), blocks.max(axis=axes)
        # MIXED = -1 is the minimum, so lo >= 0 also rejects mixed children.
        levels.append(jnp.where((lo == hi) & (lo >= 0), lo, jnp.int32(MIXED)))
    return levels[::-1]


def _upsample(grid):
    for axis in range(grid.ndim):
        grid = jnp.repeat(grid, 2, axis=axis)
    return grid


def _parity_slot(side, dim):
    """Slot of each cell of a side**dim grid within its parent."""
    parity = jnp.arange(side, dtype=jnp.int32) & jnp.int32(1)
    slot = jnp.zeros((side,) * dim, jnp.int32)
    for axis in range(dim):
        shape = [1] * dim
        shape[axis] = side
        slot = slot + (parity.reshape(shape) << axis)
    return slot


@jax.jit
def _assign_ids(labels):
    """Per level: node id grid (-1 where no node) and labels; and the node count.

    Nodes are numbered level by level. Children of the internal node with
    rank r (row-major among its level's internal nodes) are contiguous at
    next_level_offset + r * 2**dim + slot.
    """
    dim = labels.ndim
    levels = _pyramid(labels)
    ids, offset, id_grids = jnp.zeros((1,) * dim, jnp.int32), jnp.int32(0), []
    for k, level in enumerate(levels):
        id_grids.append(ids)
        exists = ids >= 0
        offset = offset + jnp.sum(exists, dtype=jnp.int32)  # next level's first id
        if k + 1 == len(levels):
            break  # finest level: all leaves, no child grid
        internal = exists & (level == MIXED)
        rank = jnp.cumsum(internal.ravel(), dtype=jnp.int32).reshape(internal.shape) - 1
        base = offset + rank * jnp.int32(2 ** dim)
        ids = jnp.where(_upsample(internal),
                        _upsample(base) + _parity_slot(2 * level.shape[0], dim),
                        jnp.int32(-1))
    return id_grids, levels, offset


@partial(jax.jit, static_argnames="n_nodes")
def _scatter_nodes(id_grids, levels, n_nodes):
    dim = levels[0].ndim
    children = jnp.full((n_nodes, 2 ** dim), -1, jnp.int32)
    uniform = jnp.full((n_nodes,), MIXED, jnp.int32)
    for k, (ids, level) in enumerate(zip(id_grids, levels)):
        target = jnp.where(ids >= 0, ids, jnp.int32(n_nodes)).ravel()
        uniform = uniform.at[target].set(level.ravel(), mode="drop")
        if k + 1 < len(levels):
            # Slot-0 child id of every cell; siblings follow contiguously.
            first = id_grids[k + 1][tuple(slice(None, None, 2) for _ in range(dim))]
            rows = first.ravel()[:, None] + jnp.arange(2 ** dim, dtype=jnp.int32)
            internal = ((ids >= 0) & (level == MIXED)).ravel()
            children = children.at[target].set(
                jnp.where(internal[:, None], rows, jnp.int32(-1)), mode="drop")
    return children, uniform


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

    def _point_child(self, node, key, depth):
        bits = (key >> (jnp.int32(self.max_depth - 1) - depth)) & jnp.int32(1)
        return jnp.sum(bits << jnp.arange(self.dim, dtype=jnp.int32), dtype=jnp.int32)

    def _box_label(self, lower, upper):
        u_lo, err_lo = self._cells(lower)
        u_hi, err_hi = self._cells(upper)
        lo, hi = self._floor_cells(u_lo - err_lo), self._floor_cells(u_hi + err_hi)
        n_cells = 2 ** self.max_depth
        inside = jnp.all((lo >= 0) & (hi < n_cells))
        outside = jnp.any((hi < 0) | (lo >= n_cells))
        lo, hi = jnp.clip(lo, 0, n_cells - 1), jnp.clip(hi, 0, n_cells - 1)

        # Smallest shift s (deepest level max_depth - s) at which the box
        # covers at most two cells per axis; s = max_depth always qualifies.
        shifts = jnp.arange(self.max_depth + 1, dtype=jnp.int32)
        spans = (hi[None] >> shifts[:, None]) - (lo[None] >> shifts[:, None])
        shift = jnp.argmax(jnp.all(spans <= 1, axis=1)).astype(jnp.int32)
        level = jnp.int32(self.max_depth) - shift
        cells = jnp.where(_slot_bits(self.dim) == 1, hi >> shift, lo >> shift) << shift
        nodes, _ = jax.vmap(lambda cell: self._descend(cell, level))(cells)
        labels = self.arrays.uniform[nodes]
        certified = jnp.all(self._is_leaf(nodes)) & jnp.all(labels == labels[0])
        return inside, outside, jnp.where(certified, labels[0], jnp.int32(MIXED))

    @classmethod
    def from_grid(cls, labels, lower, upper, default_region: int = 0,
                  n_regions: int | None = None) -> Self:
        """labels: (n,)*dim region ids, n a power of two, cell (i, j, ...) at
        lower + (i, j, ...) * (upper - lower) / n. Homogeneous blocks collapse.

        Runs where `labels` lives (device arrays stay put; host arrays go to
        the default device once).
        """
        labels = jnp.asarray(labels)
        if not jnp.issubdtype(labels.dtype, jnp.integer):
            raise ValueError("labels must be integers")
        dim = labels.ndim
        n = labels.shape[0] if dim else 0
        if dim < 1 or n < 1 or any(s != n for s in labels.shape) or n & (n - 1):
            raise ValueError("labels must be a cube grid with power-of-two side")
        max_depth = n.bit_length() - 1
        if max_depth > _MAX_DEPTH:
            raise ValueError(f"max_depth {max_depth} exceeds {_MAX_DEPTH} (int32 cells)")
        if max_depth > _F32_DEPTH and not jax.config.read("jax_enable_x64"):
            warnings.warn(f"max_depth {max_depth} exceeds f32 cell resolution; enable x64")
        lower, upper = cls._validate_box(lower, upper, dim)
        lo, hi = (int(v) for v in jax.device_get((labels.min(), labels.max())))
        n_regions = cls._check_regions(lo, hi, default_region, n_regions)

        id_grids, levels, n_nodes = _assign_ids(labels.astype(jnp.int32))
        children, uniform = _scatter_nodes(id_grids, levels, int(n_nodes))
        arrays = cls._tree_arrays(children, uniform, lower, upper)
        return cls(arrays, dim=dim, n_regions=n_regions,
                   default_region=default_region, max_depth=max_depth)
