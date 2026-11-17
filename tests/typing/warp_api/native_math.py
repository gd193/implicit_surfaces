"""Narrow native operation boundary for incomplete upstream vector stubs."""
import warp as wp


@wp.func_native("return wp::length_sq(a-b);")
def squared_distance(a: wp.vec3, b: wp.vec3) -> float: ...


@wp.func_native("return a[0];")
def vector_x(a: wp.vec3) -> float: ...


@wp.func_native("return lower[axis] <= point[axis] && point[axis] <= upper[axis];")
def within_axis(point: wp.vec3, lower: wp.vec3, upper: wp.vec3, axis: int) -> bool: ...
