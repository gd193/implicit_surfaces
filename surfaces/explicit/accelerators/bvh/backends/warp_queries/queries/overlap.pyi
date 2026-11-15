"""Primitive-AABB overlap signatures; box shapes are checked at runtime."""
from typing import Generic, TypeVar

import jax
from jax.typing import ArrayLike

from ..launch import BoundQuery
from ..contracts import PrimitiveSpec, WarpFunction

HostDataT = TypeVar("HostDataT")
DeviceDataT = TypeVar("DeviceDataT")
ItemT = TypeVar("ItemT")
PointT = TypeVar("PointT")


class OverlapQuery(BoundQuery, Generic[HostDataT]):
    device: WarpFunction[..., tuple[int, int]]
    def __call__(self, tree: object, geometry: HostDataT, lower: ArrayLike,
                 upper: ArrayLike) -> tuple[jax.Array, jax.Array]: ...


def make_overlap(primitive: PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT], *,
                 stack_capacity: int = ..., block_dim: int = ...
                 ) -> OverlapQuery[HostDataT]: ...
