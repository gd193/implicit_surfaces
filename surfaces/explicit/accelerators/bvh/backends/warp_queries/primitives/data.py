"""Concrete built-in array schemas shared by Warp and Python type checking."""
from typing import TYPE_CHECKING, NamedTuple

import jax
import warp as wp

from ..contracts import DeviceArray, warp_struct

if TYPE_CHECKING:
    _Vec2Array = DeviceArray[wp.vec2]
    _Vec3Array = DeviceArray[wp.vec3]
    _Vec2iArray = DeviceArray[wp.vec2i]
    _Vec3iArray = DeviceArray[wp.vec3i]
else:
    _Vec2Array = wp.array(dtype=wp.vec2)
    _Vec3Array = wp.array(dtype=wp.vec3)
    _Vec2iArray = wp.array(dtype=wp.vec2i)
    _Vec3iArray = wp.array(dtype=wp.vec3i)


@warp_struct
class PointsData2:
    a: _Vec2Array


@warp_struct
class PointsData3:
    a: _Vec3Array


@warp_struct
class SegmentsData2:
    a: _Vec2Array
    b: _Vec2Array
    vertices: _Vec2Array
    indices: _Vec2iArray


@warp_struct
class SegmentsData3:
    a: _Vec3Array
    b: _Vec3Array
    vertices: _Vec3Array
    indices: _Vec2iArray


@warp_struct
class TrianglesData3:
    a: _Vec3Array
    e1: _Vec3Array
    e2: _Vec3Array
    vertices: _Vec3Array
    indices: _Vec3iArray


@warp_struct
class SilhouetteEdgesData3:
    a: _Vec3Array
    b: _Vec3Array
    n0: _Vec3Array
    n1: _Vec3Array


class PointsArrays2(NamedTuple):
    a: jax.Array


class PointsArrays3(NamedTuple):
    a: jax.Array


class SegmentsArrays2(NamedTuple):
    a: jax.Array
    b: jax.Array
    vertices: jax.Array
    indices: jax.Array


class SegmentsArrays3(NamedTuple):
    a: jax.Array
    b: jax.Array
    vertices: jax.Array
    indices: jax.Array


class TrianglesArrays3(NamedTuple):
    a: jax.Array
    e1: jax.Array
    e2: jax.Array
    vertices: jax.Array
    indices: jax.Array


class SilhouetteEdgesArrays3(NamedTuple):
    a: jax.Array
    b: jax.Array
    n0: jax.Array
    n1: jax.Array
