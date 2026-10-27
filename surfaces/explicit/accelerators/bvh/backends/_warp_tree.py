"""Primitive-independent BVH arrays."""
from __future__ import annotations

from collections import namedtuple

import jax.numpy as jnp
import numpy as np
import warp as wp

from surfaces.explicit.accelerators.protocol import Config
from ._warp_layout import _pack_ray_nodes, _validate_tree
from .sah_jax import _build_payload


STACK_CAPACITY = 64


@wp.struct
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


def build_tree(primitive, data, config=None):
    """Return tree and leaf-ordered geometry; preserve original primitive IDs."""
    config = config or Config()
    if primitive.dimension not in (2, 3):
        raise ValueError("BVH bounds support 2D or 3D primitives")
    if not 1 <= config.max_stack_depth <= STACK_CAPACITY or config.leaf_size_threshold < 1:
        raise ValueError("invalid leaf size or stack depth")
    raw = primitive.bounds(data)
    if any(np.iscomplexobj(v) for v in raw):
        raise ValueError("primitive bounds must be real")
    lo, hi = (np.asarray(v, np.float64) for v in raw)
    if lo.ndim != 2 or lo.shape[1] != primitive.dimension or hi.shape != lo.shape:
        raise ValueError("primitive bounds must have matching shape (N, dimension)")
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo > hi):
        raise ValueError("primitive bounds must be finite and ordered")
    with np.errstate(over="ignore"):
        low32, high32 = lo.astype(np.float32), hi.astype(np.float32)
    low32 = np.where(low32 > lo, np.nextafter(low32, np.float32(-np.inf)), low32)
    high32 = np.where(high32 < hi, np.nextafter(high32, np.float32(np.inf)), high32)
    lo, hi = low32, high32
    if not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError("primitive bounds exceed finite float32 range")
    if not len(lo):
        empty = jnp.empty((0,), jnp.int32)
        bounds = jnp.empty((0, 3), jnp.float32)
        packed = jnp.empty((0, 4), jnp.float32)
        return TreeData(bounds, bounds, empty, empty, empty, empty, empty, packed, packed), primitive.pack(data, empty)
    if len(lo) >= 2**31:
        raise ValueError("primitive IDs must fit int32")
    if primitive.dimension == 2:
        lo, hi = (np.pad(v, ((0, 0), (0, 1))) for v in (lo, hi))
    built = _build_payload(lo, hi, int(config.leaf_size_threshold))
    _, depths, _ = _validate_tree(built, len(lo))
    if depths.max() > config.max_stack_depth:
        raise ValueError("tree exceeds configured stack depth")
    packed = _pack_ray_nodes(np.asarray(built.node_lower), np.asarray(built.node_upper),
                             np.asarray(built.node_left), np.asarray(built.node_right),
                             np.asarray(built.node_leaf_offset), np.asarray(built.node_leaf_count))
    tree = TreeData(*(getattr(built, name) for name in TreeData._fields[:7]),
                    *(jnp.asarray(a) for a in packed))
    return tree, primitive.pack(data, tree.prim_ids)
