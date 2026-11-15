"""Array schemas used by query callbacks; construction algorithms are unchanged."""
from typing import NamedTuple, TypeVar

import jax
import warp as wp

from surfaces.explicit.accelerators.protocol import Config
from ._warp_types import DeviceArray, PrimitiveSpec

H = TypeVar("H")
D = TypeVar("D")
I = TypeVar("I")
P = TypeVar("P")

STACK_CAPACITY: int


class DeviceTree:
    node_lower: DeviceArray[wp.vec3]
    node_upper: DeviceArray[wp.vec3]
    node_left: DeviceArray[int]
    node_right: DeviceArray[int]
    node_leaf_offset: DeviceArray[int]
    node_leaf_count: DeviceArray[int]
    prim_ids: DeviceArray[int]
    node_ray_lower: DeviceArray[wp.vec4]
    node_ray_upper: DeviceArray[wp.vec4]


class TreeData(NamedTuple):
    node_lower: jax.Array
    node_upper: jax.Array
    node_left: jax.Array
    node_right: jax.Array
    node_leaf_offset: jax.Array
    node_leaf_count: jax.Array
    prim_ids: jax.Array
    node_ray_lower: jax.Array
    node_ray_upper: jax.Array

    @classmethod
    def from_payload(cls, payload: object) -> TreeData: ...


def build_tree(primitive: PrimitiveSpec[H, D, I, P], data: H,
               config: Config | None = ...) -> tuple[TreeData, H]: ...
