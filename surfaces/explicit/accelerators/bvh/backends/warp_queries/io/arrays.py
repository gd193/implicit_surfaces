"""Native array access preserves the existing load/store addressing ABI."""
from typing import Any

import warp as wp


@wp.func_native("return bvh::load_array(values, index);")
def load_array(values: Any, index: int):
    return values[index]


@wp.func_native("bvh::store_array(values, index, value);")
def store_array(values: Any, index: Any, value: Any): ...


@wp.func_native("return bvh::load_node(values, index);")
def load_node(values: wp.array(dtype=wp.vec4), index: int) -> wp.vec4: ...
