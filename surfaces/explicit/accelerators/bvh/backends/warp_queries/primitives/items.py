"""Concrete builtin item annotations shared by host typing and Warp codegen."""
import warp as wp

from ..contracts import warp_struct


@warp_struct
class PointItem2:
    a: wp.vec2


@warp_struct
class PointItem3:
    a: wp.vec3


@warp_struct
class SegmentItem2:
    a: wp.vec2
    b: wp.vec2


@warp_struct
class SegmentItem3:
    a: wp.vec3
    b: wp.vec3


@warp_struct
class TriangleItem3:
    a: wp.vec3
    e1: wp.vec3
    e2: wp.vec3


@warp_struct
class SilhouetteItem3:
    a: wp.vec3
    b: wp.vec3
    n0: wp.vec3
    n1: wp.vec3
