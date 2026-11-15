"""Static crossing and classification operations."""
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

from ._warp_types import WarpFunction, WarpType, warp_func, warp_struct

import warp as wp


@warp_struct
class Ray:
    origin: wp.vec3
    direction: wp.vec3
    inverse: wp.vec3
    fast: bool
    axes: wp.vec3i
    frame_origin: wp.vec3
    frame_direction: wp.vec3
    shear: wp.vec2


@warp_func
def prepare_ray(origin: wp.vec3, direction: wp.vec3, dimension: int) -> Ray:
    ray = Ray()
    ray.origin = origin
    ray.direction = direction
    components = direction
    if dimension == 2:
        components[2] = 1.0
    ray.fast = wp.min(wp.abs(components)) >= 1.e-20
    if ray.fast:
        ray.inverse = wp.cw_div(wp.vec3(1.0), components)
    kz = int(0)
    if wp.abs(direction[1]) > wp.abs(direction[kz]):
        kz = 1
    if wp.abs(direction[2]) > wp.abs(direction[kz]):
        kz = 2
    kx, ky = (kz+1)%3, (kz+2)%3
    ray.axes = wp.vec3i(kx, ky, kz)
    ray.frame_origin = wp.vec3(origin[kx], origin[ky], origin[kz])
    ray.frame_direction = wp.vec3(direction[kx], direction[ky], direction[kz])
    ray.shear = wp.vec2(direction[kx]/direction[kz], direction[ky]/direction[kz])
    return ray


DeviceDataT = TypeVar("DeviceDataT")
StateT = TypeVar("StateT")
PointT = TypeVar("PointT")
MetaT = TypeVar("MetaT")


@dataclass(frozen=True)
class RayOps(Generic[DeviceDataT, StateT]):
    """Single hit or next_hit -> (state, t, flag): 1 hit, 0 done, -1 failure."""
    intersect: WarpFunction[[Ray, float, float, DeviceDataT, int, bool, bool], wp.vec2] | None = None
    state_type: WarpType[StateT] = cast(WarpType[StateT], wp.int32)
    next_hit: WarpFunction[[Ray, float, float, DeviceDataT, int, StateT, bool, bool], tuple[StateT, float, int]] | None = None


@dataclass(frozen=True)
class ClassificationOps(Generic[PointT, DeviceDataT, MetaT]):
    """event(point, data, slot, id, meta): 0 miss, 1 parity, 2 contact, -1 failure."""
    metadata_type: WarpType[MetaT]
    valid: WarpFunction[[MetaT], bool]
    event: WarpFunction[[PointT, DeviceDataT, int, int, MetaT], int]


def builtin_classification(primitive, metadata_type):
    Vec, Geometry, Meta = primitive.point_type, primitive.data_type, metadata_type

    @wp.func
    def valid(meta: Meta):
        if meta.closed.shape[0] != 1:
            return False
        return meta.closed[0] != 0

    @wp.func_native("return bvh::classification_event(point, meta, slot);")
    def event(point: Vec, geometry: Geometry, slot: int, original: int, meta: Meta) -> int: ...

    return ClassificationOps(Meta, valid, event)
