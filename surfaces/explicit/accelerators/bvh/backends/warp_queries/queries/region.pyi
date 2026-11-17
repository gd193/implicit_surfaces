"""Typed exact closed-box query signatures."""
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar, overload

import jax
import warp as wp
from jax.typing import ArrayLike

from ..contracts import PrimitiveSpec, WarpFunction
from ..launch import BoundQuery
from ..domain import ClassificationOps
from ..._warp_tree import DeviceTree
from .classification import WithCore
from ..primitives.data import PointsData2, PointsData3, SegmentsData2, SegmentsData3, TrianglesData3

HostDataT = TypeVar("HostDataT")
DeviceDataT = TypeVar("DeviceDataT")
ItemT = TypeVar("ItemT")
PointT = TypeVar("PointT")
MetaT = TypeVar("MetaT")
HostMetaT = TypeVar("HostMetaT")

@dataclass(frozen=True)
class RegionIntersectionPolicy(Generic[DeviceDataT]):
    intersects: WarpFunction[[DeviceDataT, int, wp.vec3, wp.vec3], bool]

@overload
def point_region_policy(dimension: Literal[2]) -> RegionIntersectionPolicy[PointsData2]: ...
@overload
def point_region_policy(dimension: Literal[3]) -> RegionIntersectionPolicy[PointsData3]: ...
@overload
def point_region_policy(dimension: int) -> RegionIntersectionPolicy[Any]: ...
@overload
def segment_region_policy(dimension: Literal[2]) -> RegionIntersectionPolicy[SegmentsData2]: ...
@overload
def segment_region_policy(dimension: Literal[3]) -> RegionIntersectionPolicy[SegmentsData3]: ...
@overload
def segment_region_policy(dimension: int) -> RegionIntersectionPolicy[Any]: ...
def triangle_region_policy() -> RegionIntersectionPolicy[TrianglesData3]: ...

@dataclass(frozen=True)
class RegionQuery(BoundQuery, Generic[HostDataT, DeviceDataT, PointT]):
    mode: str
    device: WarpFunction[[PointT, PointT, DeviceTree, DeviceDataT], tuple[int, int]]
    def __call__(self, tree: object, geometry: HostDataT, lower: ArrayLike,
                 upper: ArrayLike) -> tuple[jax.Array, jax.Array]: ...

@dataclass(frozen=True)
class RegionClassifyQuery(BoundQuery, Generic[HostDataT, DeviceDataT, PointT, HostMetaT, MetaT]):
    device: WarpFunction[[PointT, PointT, DeviceTree, DeviceDataT, MetaT], tuple[int, int]]
    def __call__(self, tree: object, geometry: HostDataT, boundary: HostMetaT | WithCore[HostMetaT],
                 lower: ArrayLike, upper: ArrayLike) -> tuple[jax.Array, jax.Array]: ...

def make_region(primitive: PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT],
                policy: RegionIntersectionPolicy[DeviceDataT], *, mode: Literal["any", "count"] = ...,
                count_limit: int | None = ..., stack_capacity: int = ...,
                block_dim: int = ...) -> RegionQuery[HostDataT, DeviceDataT, PointT]: ...

@overload
def make_region_classify(primitive: PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT],
                         region_policy: RegionIntersectionPolicy[DeviceDataT],
                         classify_policy: ClassificationOps[PointT, DeviceDataT, MetaT], *,
                         metadata_host_type: type[HostMetaT],
                         stack_capacity: int = ..., block_dim: int = ...
                         ) -> RegionClassifyQuery[HostDataT, DeviceDataT, PointT, HostMetaT, MetaT]: ...
@overload
def make_region_classify(primitive: PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT],
                         region_policy: RegionIntersectionPolicy[DeviceDataT],
                         classify_policy: ClassificationOps[PointT, DeviceDataT, MetaT], *,
                         metadata_host_type: None = ...,
                         stack_capacity: int = ..., block_dim: int = ...
                         ) -> RegionClassifyQuery[HostDataT, DeviceDataT, PointT, MetaT, MetaT]: ...
