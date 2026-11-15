"""Narrow native operation boundary for incomplete upstream vector stubs."""
import warp as wp


@wp.func_native("return wp::length_sq(a-b);")
def squared_distance(a: wp.vec3, b: wp.vec3) -> float: ...
