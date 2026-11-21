"""Primitive-independent BVH arrays."""
from __future__ import annotations

from collections import namedtuple
from functools import partial
from operator import index

import jax.numpy as jnp
import numpy as np
import warp as wp

from surfaces.explicit.accelerators.protocol import Config
from . import lbvh_jax, sah_jax
from ._warp_layout import _int32_array, _pack_ray_nodes, _validate_tree
from .warp_queries.contracts import warp_struct


STACK_CAPACITY = 64


@warp_struct
class DeviceTree:
    node_lower: wp.array(dtype=wp.vec3)
    node_upper: wp.array(dtype=wp.vec3)
    node_left: wp.array(dtype=wp.int32)
    node_right: wp.array(dtype=wp.int32)
    node_leaf_offset: wp.array(dtype=wp.int32)
    node_leaf_count: wp.array(dtype=wp.int32)
    prim_ids: wp.array(dtype=wp.int32)
    node_ray_lower: wp.array(dtype=wp.vec4)
    node_ray_upper: wp.array(dtype=wp.vec4)


class TreeData(namedtuple("TreeData", DeviceTree.vars)):
    __slots__ = ()

    @classmethod
    def from_payload(cls, payload):
        return cls(*(getattr(payload, name) for name in cls._fields))


def build_tree(primitive, data, config=None, *, builder="sah"):
    """Build a binary SAH or Morton LBVH and pack geometry in leaf order."""
    if builder == "sah":
        build = partial(sah_jax._build_payload, dimension=primitive.dimension)
    elif builder == "lbvh":
        build = lbvh_jax._build_payload
    else:
        raise ValueError("builder must be 'sah' or 'lbvh'")
    tree, geometry, _, _ = _build_tree(primitive, data, config, build)
    return tree, geometry


def _build_tree(primitive, data, config, build):
    """Validate and pack a binary host build, retaining data for boundary prep."""
    config = Config() if config is None else config
    if primitive.dimension not in (2, 3):
        raise ValueError("BVH bounds support 2D or 3D primitives")
    try:
        max_depth = index(config.max_stack_depth)
        leaf_size = index(config.leaf_size_threshold)
    except TypeError as exc:
        raise ValueError("leaf size and stack depth must be integers") from exc
    if not 1 <= max_depth <= STACK_CAPACITY or leaf_size < 1:
        raise ValueError("invalid leaf size or stack depth")
    raw = primitive.bounds(data)
    if any(np.iscomplexobj(v) for v in raw):
        raise ValueError("primitive bounds must be real")
    lo, hi = (np.asarray(v, np.float64) for v in raw)
    if lo.ndim != 2 or lo.shape[1] != primitive.dimension or hi.shape != lo.shape:
        raise ValueError("primitive bounds must have matching shape (N, dimension)")
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo > hi):
        raise ValueError("primitive bounds must be finite and ordered")
    with np.errstate(over="ignore", under="ignore"):
        low32, high32 = lo.astype(np.float32), hi.astype(np.float32)
        np.nextafter(low32, np.float32(-np.inf), out=low32, where=low32 > lo)
        np.nextafter(high32, np.float32(np.inf), out=high32, where=high32 < hi)
    lo, hi = low32, high32
    if not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError("primitive bounds exceed finite float32 range")
    if not len(lo):
        empty = jnp.empty((0,), jnp.int32)
        bounds = jnp.empty((0, 3), jnp.float32)
        packed = jnp.empty((0, 4), jnp.float32)
        tree = TreeData(bounds, bounds, empty, empty, empty, empty, empty, packed, packed)
        return tree, primitive.pack(data, empty), None, None
    if len(lo) >= 2**31:
        raise ValueError("primitive IDs must fit int32")
    if primitive.dimension == 2:
        lo, hi = (np.pad(v, ((0, 0), (0, 1))) for v in (lo, hi))
    built = build(lo, hi, min(leaf_size, len(lo)))
    topology = _validate_tree(built, len(lo))
    arrays, depths, _ = topology
    if depths.max() > max_depth:
        raise ValueError("tree exceeds configured stack depth")

    ids = _int32_array("prim_ids", built.prim_ids)
    if (ids.shape != (len(lo),) or np.any(ids < 0) or np.any(ids >= len(lo))
            or np.unique(ids).size != len(lo)):
        raise ValueError("tree primitive IDs must be a permutation of the input IDs")
    node_lo, node_hi = np.asarray(built.node_lower), np.asarray(built.node_upper)
    count = arrays["node_leaf_count"]
    offset = arrays["node_leaf_offset"]
    left, right = arrays["node_left"], arrays["node_right"]
    if (node_lo.shape != (len(count), 3) or node_hi.shape != node_lo.shape
            or node_lo.dtype != np.float32 or node_hi.dtype != np.float32
            or not np.isfinite(node_lo).all() or not np.isfinite(node_hi).all()
            or np.any(node_lo > node_hi)):
        raise ValueError("tree bounds must be finite, ordered float32 arrays of shape (nodes, 3)")

    leaves = np.flatnonzero(count)
    leaf_order = leaves[np.argsort(offset[leaves])]
    slot_leaves = np.repeat(leaf_order, count[leaf_order])
    if np.any(node_lo[slot_leaves] > lo[ids]) or np.any(node_hi[slot_leaves] < hi[ids]):
        raise ValueError("tree leaf bounds must enclose their primitives")
    internal = count == 0
    for child in (left[internal], right[internal]):
        if np.any(node_lo[internal] > node_lo[child]) or np.any(node_hi[internal] < node_hi[child]):
            raise ValueError("tree internal bounds must enclose their children")

    packed = _pack_ray_nodes(node_lo, node_hi, left, right, offset, count)
    tree = TreeData(*(jnp.asarray(a) for a in (node_lo, node_hi, left, right, offset, count,
                                               ids, *packed)))
    return tree, primitive.pack(data, tree.prim_ids), built, topology
