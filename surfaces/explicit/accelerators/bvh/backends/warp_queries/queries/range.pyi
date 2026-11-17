from dataclasses import dataclass
from typing import Generic, Literal, TypeVar, overload
import jax
from jax.typing import ArrayLike
import warp as wp
from ..._warp_tree import DeviceTree
from ..contracts import PrimitiveSpec, WarpFunction, WarpRef, WarpType
from ..launch import BoundQuery
from ..metrics import NearestOps, NearestPolicy, NearestPredicate

HostDataT = TypeVar("HostDataT")
DeviceDataT = TypeVar("DeviceDataT")
ItemT = TypeVar("ItemT")
PointT = TypeVar("PointT")
QueryT = TypeVar("QueryT")
StateT = TypeVar("StateT")
ResultT = TypeVar("ResultT")

@dataclass(frozen=True)
class RadiusAccumulator(Generic[QueryT, ItemT, PointT, StateT, ResultT]):
    state_type: WarpType[StateT]
    initialize: WarpFunction[[QueryT, float], StateT]
    update: WarpFunction[[QueryT, ItemT, PointT, float, int, WarpRef[StateT]], None]
    finish: WarpFunction[[StateT], ResultT]

class RadiusQuery(BoundQuery, Generic[HostDataT, DeviceDataT, ItemT, PointT, QueryT]):
    device: WarpFunction[[QueryT, float, DeviceTree, DeviceDataT], tuple[int, int]]
    ops: NearestOps[QueryT, ItemT, PointT]
    mode: str
    def __call__(self, tree: object, geometry: HostDataT, queries: ArrayLike,
                 radius: ArrayLike) -> tuple[jax.Array, jax.Array]: ...

class RadiusReduction(BoundQuery, Generic[HostDataT, DeviceDataT, ItemT, PointT, QueryT, ResultT]):
    device: WarpFunction[[QueryT, float, DeviceTree, DeviceDataT], tuple[ResultT, int]]
    ops: NearestOps[QueryT, ItemT, PointT]
    mode: str

@overload
def make_radius(primitive: PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT],
                metric: NearestPolicy[DeviceDataT, ItemT, PointT, QueryT], *,
                predicate: WarpFunction[[QueryT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, QueryT] | None = ...,
                mode: Literal["any", "count"] = ..., accumulator: None = ...,
                stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool = ...,
                packed_nodes: bool = ..., group_leaves: bool | None = ...
                ) -> RadiusQuery[HostDataT, DeviceDataT, ItemT, PointT, QueryT]: ...

@overload
def make_radius(primitive: PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT],
                metric: NearestPolicy[DeviceDataT, ItemT, PointT, QueryT], *,
                predicate: WarpFunction[[QueryT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, QueryT] | None = ...,
                mode: Literal["accumulate"],
                accumulator: RadiusAccumulator[QueryT, ItemT, PointT, StateT, ResultT],
                stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool = ...,
                packed_nodes: bool = ..., group_leaves: bool | None = ...
                ) -> RadiusReduction[HostDataT, DeviceDataT, ItemT, PointT, QueryT, ResultT]: ...

@overload
def make_radius(primitive: PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT],
                metric: None = ..., *,
                evaluate: WarpFunction[[PointT, ItemT], tuple[PointT, float]],
                lower_bound: WarpFunction[[PointT, wp.vec3, wp.vec3], float],
                predicate: WarpFunction[[PointT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, PointT] | None = ...,
                mode: Literal["any", "count"] = ..., accumulator: None = ...,
                stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool = ...,
                packed_nodes: bool = ..., group_leaves: bool | None = ...
                ) -> RadiusQuery[HostDataT, DeviceDataT, ItemT, PointT, PointT]: ...

@overload
def make_radius(primitive: PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT],
                metric: None = ..., *,
                evaluate: WarpFunction[[PointT, ItemT], tuple[PointT, float]],
                lower_bound: WarpFunction[[PointT, wp.vec3, wp.vec3], float],
                predicate: WarpFunction[[PointT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, PointT] | None = ...,
                mode: Literal["accumulate"],
                accumulator: RadiusAccumulator[PointT, ItemT, PointT, StateT, ResultT],
                stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool = ...,
                packed_nodes: bool = ..., group_leaves: bool | None = ...
                ) -> RadiusReduction[HostDataT, DeviceDataT, ItemT, PointT, PointT, ResultT]: ...
