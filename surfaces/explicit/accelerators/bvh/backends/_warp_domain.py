"""Static crossing and classification operations."""
from dataclasses import dataclass
from typing import Any

import warp as wp


@wp.struct
class Ray:
    origin: wp.vec3
    direction: wp.vec3
    inverse: wp.vec3
    fast: bool
    axes: wp.vec3i
    frame_origin: wp.vec3
    frame_direction: wp.vec3
    shear: wp.vec2


@wp.func
def prepare_ray(origin: wp.vec3, direction: wp.vec3):
    ray = Ray()
    ray.origin = origin
    ray.direction = direction
    ray.fast = wp.min(wp.abs(direction)) >= 1.e-20
    if ray.fast:
        ray.inverse = wp.cw_div(wp.vec3(1.0), direction)
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


@dataclass(frozen=True)
class RayOps:
    """Single hit or next_hit -> (state, t, flag): 1 hit, 0 done, -1 failure."""
    intersect: Any = None
    state_type: Any = wp.int32
    next_hit: Any = None


@dataclass(frozen=True)
class ClassificationOps:
    """event(point, data, slot, id, meta): 0 miss, 1 parity, 2 contact, -1 failure."""
    metadata_type: Any
    valid: Any
    event: Any


def builtin_classification(primitive, metadata_type):
    from ._warp_geometry import _NATIVE_EDGE_PREDICATES
    from ._warp_inside import _triangle_event_source

    Vec, Geometry, Meta = primitive.point_type, primitive.data_type, metadata_type

    @wp.func
    def valid(meta: Meta):
        if meta.closed.shape[0] != 1:
            return False
        return meta.closed[0] != 0

    if primitive.dimension == 2:
        source = _NATIVE_EDGE_PREDICATES + r"""
            const wp::vec3 a = meta.vertices.data[meta.indices.data[2*slot]];
            const wp::vec3 b = meta.vertices.data[meta.indices.data[2*slot+1]];
            const double ax=double(a[0])-double(point[0]), ay=double(a[1])-double(point[1]);
            const double bx=double(b[0])-double(point[0]), by=double(b[1])-double(point[1]);
            const double cross=cross2(ax,ay,bx,by);
            if (cross == 0.0 && wp::min(ax,bx) <= 0 && wp::max(ax,bx) >= 0
                && wp::min(ay,by) <= 0 && wp::max(ay,by) >= 0) return 2;
            if ((ay <= 0 && by > 0) || (by <= 0 && ay > 0)) return cross/(by-ay) > 0 ? 1 : 0;
            return 0;
        """
    else:
        source = _NATIVE_EDGE_PREDICATES + r"""
            const auto p = point;
            const wp::vec3i ids(meta.indices.data[3*slot],
                                meta.indices.data[3*slot+1], meta.indices.data[3*slot+2]);
        """ + _triangle_event_source("meta.vertices.data[ids[$]]")

    @wp.func_native(source)
    def event(point: Vec, geometry: Geometry, slot: int, original: int, meta: Meta) -> int: ...

    return ClassificationOps(Meta, valid, event)
