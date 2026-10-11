"""Shared storage and traversal for tree-backed partitions.

A subclass supplies only its split rule through four hooks; the descent loops,
outside-root handling, pytree registration, and array conversion live here.
Trees are built once on the host and are fixed during a solve.

Storage invariants (checked by `_make_arrays`):
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


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


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
    _aux_fields: Tuple[str, ...] = ("dim", "n_regions", "default_region", "max_depth",
                                    "max_box_visits")

    def __init__(self, arrays: TreeArrays, dim: int, n_regions: int,
                 default_region: int, max_depth: int, max_box_visits: int = 1024):
        super().__init__(dim, n_regions)
        if not 0 <= default_region < n_regions:
            raise ValueError("default_region must be in [0, n_regions)")
        if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0:
            raise ValueError("max_depth must be a nonnegative integer")
        self.arrays = arrays
        self.default_region = int(default_region)
        self.max_depth = max_depth
        # Box queries give up (MIXED) after this many nodes; keeps cost bounded.
        self.max_box_visits = _positive(max_box_visits, "max_box_visits")

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
    def _encode_box(self, lower: jax.Array, upper: jax.Array) -> Tuple[jax.Array, jax.Array, Any]:
        """(inside_root, outside_root, key) for the box widened by rounding error.

        inside_root: the widened box lies in the root. outside_root: it misses
        the root entirely. Both False means it straddles the root boundary; the
        key must then describe the box clipped to the root.
        """

    @abc.abstractmethod
    def _point_child(self, node: jax.Array, key: Any, depth: jax.Array) -> jax.Array:
        """int32 child slot of the internal node containing the point."""

    @abc.abstractmethod
    def _root_context(self) -> Any:
        """Pytree describing the root's extent for box queries (e.g. its origin)."""

    @abc.abstractmethod
    def _box_children(self, node: jax.Array, context: Any, key: Any,
                      depth: jax.Array) -> Tuple[jax.Array, Any]:
        """(overlap, child_contexts): (max_children,) bool mask of children that
        meet the widened box, and each child's context stacked on axis 0."""

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

    def classify_point(self, params, x):
        ftype = float_dtype(x)
        inside, key = self._encode_point(jnp.asarray(x, ftype))

        def cond(carry):
            node, depth = carry
            return ~self._is_leaf(node) & (depth < self.max_depth)

        def body(carry):
            node, depth = carry
            slot = self._point_child(node, key, depth)
            return self.arrays.children[node, slot], depth + jnp.int32(1)

        node, _ = jax.lax.while_loop(cond, body, (jnp.int32(0), jnp.int32(0)))
        return jnp.where(inside, self.arrays.uniform[node], jnp.int32(self.default_region))

    def classify_box(self, params, lower, upper):
        ftype = float_dtype(lower, upper)
        lower, upper = jnp.asarray(lower, ftype), jnp.asarray(upper, ftype)
        valid = jnp.all(jnp.isfinite(lower) & jnp.isfinite(upper) & (lower <= upper))
        inside, outside, key = self._encode_box(lower, upper)

        # Depth-first over every node meeting the box, stopping at the first
        # leaf whose label differs. A full stack never exceeds this capacity.
        n_children = self.arrays.children.shape[1]
        capacity = self.max_depth * (n_children - 1) + 1
        unset = jnp.int32(-2)
        stack = (jnp.zeros(capacity, jnp.int32), jnp.zeros(capacity, jnp.int32),
                 jax.tree.map(lambda a: jnp.broadcast_to(a, (capacity,) + jnp.shape(a)),
                              self._root_context()))

        def cond(carry):
            _, top, _, uniform, visits = carry
            return (top > 0) & uniform & (visits < self.max_box_visits)

        def body(carry):
            (nodes, depths, contexts), top, label, uniform, visits = carry
            top = top - jnp.int32(1)
            node, depth = nodes[top], depths[top]
            context = jax.tree.map(lambda a: a[top], contexts)
            leaf = self._is_leaf(node)
            leaf_label = self.arrays.uniform[node]
            uniform = uniform & (~leaf | (label == unset) | (label == leaf_label))
            label = jnp.where(leaf, leaf_label, label)
            overlap, child_contexts = self._box_children(node, context, key, depth)
            overlap = overlap & ~leaf
            pos = jnp.where(overlap, top + jnp.cumsum(overlap, dtype=jnp.int32) - 1,
                            jnp.int32(capacity))
            push = lambda buf, values: buf.at[pos].set(values, mode='drop')
            nodes = push(nodes, self.arrays.children[node])
            depths = push(depths, jnp.broadcast_to(depth + jnp.int32(1), pos.shape))
            contexts = jax.tree.map(push, contexts, child_contexts)
            top = top + jnp.sum(overlap, dtype=jnp.int32)
            return (nodes, depths, contexts), top, label, uniform, visits + jnp.int32(1)

        # A box straddling the root is traversed over its in-root part, which
        # must then carry the default label of the outside part.
        start = (stack, jnp.where(outside, jnp.int32(0), jnp.int32(1)), unset,
                 jnp.asarray(True), jnp.int32(0))
        _, top, label, uniform, _ = jax.lax.while_loop(cond, body, start)
        default = jnp.int32(self.default_region)
        certified = uniform & (top == 0) & (label >= 0) & (inside | (label == default))
        result = jnp.where(outside, default, jnp.where(certified, label, jnp.int32(MIXED)))
        return jnp.where(valid, result, jnp.int32(MIXED))

    # ---------------------------------------------------------- build helpers
    @staticmethod
    def _validate_labels(labels, default_region: int, n_regions: int | None) -> Tuple[np.ndarray, int]:
        labels = np.asarray(labels)
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("labels must be integers")
        if labels.size == 0:
            raise ValueError("labels must be nonempty")
        if n_regions is None:
            n_regions = int(max(labels.max(), default_region)) + 1
        if labels.min() < 0 or labels.max() >= n_regions:
            raise ValueError("labels must lie in [0, n_regions)")
        if not 0 <= default_region < n_regions:
            raise ValueError("default_region must be in [0, n_regions)")
        if n_regions > np.iinfo(np.int32).max:
            raise ValueError("too many regions for int32 ids")
        return labels.astype(np.int32), n_regions

    @staticmethod
    def _validate_box(lower, upper, dim: int) -> Tuple[np.ndarray, np.ndarray]:
        lower, upper = np.asarray(lower, np.float64), np.asarray(upper, np.float64)
        if lower.shape != (dim,) or upper.shape != (dim,):
            raise ValueError(f"root bounds must have shape ({dim},)")
        if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper)) and np.all(lower < upper)):
            raise ValueError("root bounds must be finite with lower < upper")
        return lower, upper

    @staticmethod
    def _make_arrays(children: np.ndarray, uniform: np.ndarray, lower: np.ndarray,
                     upper: np.ndarray, split: Any = ()) -> TreeArrays:
        n_nodes = len(uniform)
        if n_nodes > np.iinfo(np.int32).max:
            raise ValueError("too many nodes for int32 indices")
        leaf = children[:, 0] < 0
        if not (np.all(children[leaf] == -1) and np.all(children[~leaf] > 0)
                and np.all(children[~leaf] < n_nodes)):
            raise ValueError("children must be all -1 (leaf) or all valid node ids")
        if np.any(uniform[leaf] < 0) or np.any(uniform[~leaf] != MIXED):
            raise ValueError("leaves need a region id and internal nodes MIXED")
        fdtype = storage_float_dtype()
        return TreeArrays(children=jnp.asarray(children, jnp.int32),
                          uniform=jnp.asarray(uniform, jnp.int32),
                          lower=jnp.asarray(lower, fdtype), upper=jnp.asarray(upper, fdtype),
                          split=split)
