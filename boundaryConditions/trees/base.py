"""Shared storage and traversal for tree-backed partitions.

A subclass supplies its split rule and box query through hooks; descent,
outside-root handling, pytree registration, and array conversion live here.
Trees are fixed during a solve. Queries use fixed-trip-count loops with no
per-lane stacks, so vmapped batches run in lockstep without divergence.

Storage invariants:
  * children[n] is all -1 for a leaf, otherwise all valid node ids.
  * uniform[n] is the leaf's region id; internal nodes hold MIXED because
    homogeneous subtrees are collapsed into leaves at build time.
  * Node 0 is the root. Depth never exceeds max_depth.
All integer arrays and loop carries are int32 in both precision modes.
"""
import abc
from typing import Any, NamedTuple, Self, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from boundaryConditions.base import MIXED, BoundaryPartition, float_dtype, storage_float_dtype


class TreeArrays(NamedTuple):
    children: jax.Array  # (n_nodes, max_children) int32, -1 for leaves
    uniform: jax.Array   # (n_nodes,) int32 leaf label, MIXED for internal nodes
    lower: jax.Array     # (dim,) root box, storage float dtype
    upper: jax.Array     # (dim,)
    split: Any           # subclass-specific pytree, e.g. kd split planes


class TreePartition(BoundaryPartition):
    """Partition stored as a tree over the half-open root box [lower, upper).

    Points outside the root get default_region. Subclasses are registered as
    pytrees (arrays as leaves, metadata static), so a partition can be passed
    as a jit argument instead of being baked into the trace as constants.
    """

    # Static metadata; subclasses with extra hashable fields extend this.
    _aux_fields: Tuple[str, ...] = ("dim", "n_regions", "default_region", "max_depth")

    def __init__(self, arrays: TreeArrays, dim: int, n_regions: int,
                 default_region: int, max_depth: int):
        super().__init__(dim, n_regions)
        if not 0 <= default_region < n_regions:
            raise ValueError("default_region must be in [0, n_regions)")
        if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0:
            raise ValueError("max_depth must be a nonnegative integer")
        self.arrays = arrays
        self.default_region = int(default_region)
        self.max_depth = max_depth

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        jax.tree_util.register_pytree_node(cls, cls._tree_flatten, cls._tree_unflatten)

    def _tree_flatten(self):
        return (self.arrays,), tuple(getattr(self, f) for f in self._aux_fields)

    @classmethod
    def _tree_unflatten(cls, aux, leaves):
        obj = object.__new__(cls)
        obj.arrays = leaves[0]
        for name, value in zip(cls._aux_fields, aux):
            setattr(obj, name, value)
        return obj

    # ------------------------------------------------------------------ hooks
    @abc.abstractmethod
    def _encode_point(self, x: jax.Array) -> Tuple[jax.Array, Any]:
        """(inside_root: bool, key) for x already cast to its ftype."""

    @abc.abstractmethod
    def _point_child(self, node: jax.Array, key: Any, depth: jax.Array) -> jax.Array:
        """int32 child slot of the internal node containing the point.

        Called with depth < max_depth on internal nodes; results for other
        lanes are computed but discarded.
        """

    @abc.abstractmethod
    def _box_label(self, lower: jax.Array, upper: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """(inside_root, outside_root, label) for the box widened by rounding error.

        inside_root: the widened box lies in the root. outside_root: it misses
        the root entirely. label certifies the box clipped to the root (an id
        or MIXED); it is ignored when outside_root.
        """

    @classmethod
    @abc.abstractmethod
    def from_grid(cls, labels, lower, upper, default_region: int = 0,
                  n_regions: int | None = None) -> Self:
        """Build from a dense grid of region ids over the root box.

        Subclasses may add builder options only as keyword arguments with
        defaults, so every tree accepts this call.
        """

    # -------------------------------------------------------------- traversal
    def _is_leaf(self, node):
        return self.arrays.children[node, 0] < 0

    def _descend(self, key, limit):
        """Follow _point_child from the root, stopping at a leaf or at depth
        `limit`. Fixed max_depth trips; returns int32 (node, depth)."""
        def body(_, carry):
            node, depth = carry
            go = ~self._is_leaf(node) & (depth < limit)
            slot = jnp.where(go, self._point_child(node, key, depth), jnp.int32(0))
            child = self.arrays.children[node, slot]
            return jnp.where(go, child, node), jnp.where(go, depth + jnp.int32(1), depth)
        return jax.lax.fori_loop(jnp.int32(0), jnp.int32(self.max_depth), body,
                                 (jnp.int32(0), jnp.int32(0)))

    def classify_point(self, params, x):
        ftype = float_dtype(x)
        inside, key = self._encode_point(jnp.asarray(x, ftype))
        node, _ = self._descend(key, jnp.int32(self.max_depth))
        return jnp.where(inside, self.arrays.uniform[node], jnp.int32(self.default_region))

    def classify_box(self, params, lower, upper):
        ftype = float_dtype(lower, upper)
        lower, upper = jnp.asarray(lower, ftype), jnp.asarray(upper, ftype)
        valid = jnp.all(jnp.isfinite(lower) & jnp.isfinite(upper) & (lower <= upper))
        inside, outside, label = self._box_label(lower, upper)
        # A box straddling the root is certified only if its in-root part
        # carries the default label of the outside part.
        default = jnp.int32(self.default_region)
        certified = (label >= 0) & (inside | (label == default))
        result = jnp.where(outside, default, jnp.where(certified, label, jnp.int32(MIXED)))
        return jnp.where(valid, result, jnp.int32(MIXED))

    # ---------------------------------------------------------- build helpers
    @staticmethod
    def _check_regions(lo: int, hi: int, default_region: int,
                       n_regions: int | None) -> int:
        """Validate the label range [lo, hi] and return n_regions."""
        if n_regions is None:
            n_regions = max(hi, default_region) + 1
        if lo < 0 or hi >= n_regions:
            raise ValueError("labels must lie in [0, n_regions)")
        if not 0 <= default_region < n_regions:
            raise ValueError("default_region must be in [0, n_regions)")
        if n_regions > np.iinfo(np.int32).max:
            raise ValueError("too many regions for int32 ids")
        return n_regions

    @staticmethod
    def _validate_box(lower, upper, dim: int) -> Tuple[np.ndarray, np.ndarray]:
        lower, upper = np.asarray(lower, np.float64), np.asarray(upper, np.float64)
        if lower.shape != (dim,) or upper.shape != (dim,):
            raise ValueError(f"root bounds must have shape ({dim},)")
        if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper)) and np.all(lower < upper)):
            raise ValueError("root bounds must be finite with lower < upper")
        return lower, upper

    @staticmethod
    def _tree_arrays(children, uniform, lower, upper, split: Any = ()) -> TreeArrays:
        """Assemble storage without host checks; arrays may stay on device."""
        fdtype = storage_float_dtype()
        return TreeArrays(children=jnp.asarray(children, jnp.int32),
                          uniform=jnp.asarray(uniform, jnp.int32),
                          lower=jnp.asarray(lower, fdtype), upper=jnp.asarray(upper, fdtype),
                          split=split)

    @classmethod
    def _make_arrays(cls, children, uniform, lower, upper, split: Any = ()) -> TreeArrays:
        """Host-validated storage for hand-built trees."""
        children, uniform = np.asarray(children), np.asarray(uniform)
        n_nodes = len(uniform)
        if n_nodes > np.iinfo(np.int32).max:
            raise ValueError("too many nodes for int32 indices")
        leaf = children[:, 0] < 0
        if not (np.all(children[leaf] == -1) and np.all(children[~leaf] > 0)
                and np.all(children[~leaf] < n_nodes)):
            raise ValueError("children must be all -1 (leaf) or all valid node ids")
        if np.any(uniform[leaf] < 0) or np.any(uniform[~leaf] != MIXED):
            raise ValueError("leaves need a region id and internal nodes MIXED")
        return cls._tree_arrays(children, uniform, lower, upper, split)
