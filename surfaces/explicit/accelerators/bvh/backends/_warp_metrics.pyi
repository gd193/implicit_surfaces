"""Host signatures for runtime-specialized device operations.

Warp's dynamically selected vector/matrix annotations are compiled by Warp,
not interpreted as Python generics. This stub retains the host connections.
"""
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar
import warp as wp
from ._warp_types import PrimitiveSpec, WarpFunction, WarpType

HostDataT = TypeVar("HostDataT")
DeviceDataT = TypeVar("DeviceDataT")
ItemT = TypeVar("ItemT")
PointT = TypeVar("PointT")
QueryT = TypeVar("QueryT")

class AngularQuery(Protocol):
    """Angular matrix value; dimensions remain a runtime Warp constraint."""
    def __getitem__(self, index: tuple[int, int]) -> float: ...

@dataclass(frozen=True)
class NearestOps(Generic[QueryT, ItemT, PointT]):
    query_type: WarpType[QueryT]
    query_shape: tuple[int, ...]
    valid: WarpFunction[[QueryT], bool]
    node_bound: WarpFunction[[QueryT, wp.vec3, wp.vec3], float]
    closest: WarpFunction[[QueryT, ItemT], tuple[PointT, float]]
    to_score: WarpFunction[[float], float]
    to_distance: WarpFunction[[float], float]
    cull_limit: WarpFunction[[QueryT, float], float]
    # Optional traversal hooks have runtime-specific tree/geometry schemas.
    node_bound_with_data: WarpFunction[..., float] | None = ...
    viewpoint: WarpFunction[[QueryT], wp.vec3] | None = ...
    cull_items: bool = ...
    node_filter: WarpFunction[..., bool] | None = ...
    group_leaves: bool = ...

class NearestPolicy(Protocol[DeviceDataT, ItemT, PointT, QueryT]):
    def bind(self, primitive: PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT]) -> NearestOps[QueryT, ItemT, PointT]: ...

MetricSpec = NearestPolicy

class NearestPredicate(Protocol[DeviceDataT, ItemT, PointT, QueryT]):
    def bind(self, primitive: PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT],
             ops: NearestOps[QueryT, ItemT, PointT]) -> WarpFunction[[QueryT, ItemT], bool]: ...

@dataclass(frozen=True)
class NearestCallbacks(Generic[ItemT, PointT]):
    evaluate: WarpFunction[[PointT, ItemT], tuple[PointT, float]]
    lower_bound: WarpFunction[[PointT, wp.vec3, wp.vec3], float]
    def bind(self, primitive: PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT]) -> NearestOps[PointT, ItemT, PointT]: ...

@dataclass(frozen=True)
class Euclidean:
    def bind(self, primitive: PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT]) -> NearestOps[PointT, ItemT, PointT]: ...

@dataclass(frozen=True)
class Angular:
    def bind(self, primitive: PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT]) -> NearestOps[AngularQuery, ItemT, PointT]: ...

@dataclass(frozen=True)
class Silhouette:
    def bind(self, primitive: PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT],
             ops: NearestOps[QueryT, ItemT, PointT]) -> WarpFunction[[QueryT, ItemT], bool]: ...
