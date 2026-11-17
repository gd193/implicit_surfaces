"""Host specialization signatures; JAX arrays do not encode vector dimensions."""
from ..._warp_tree import DeviceTree
from typing import Generic, TypeVar, overload
import jax
from jax.typing import ArrayLike
import warp as wp
from ..launch import BoundQuery
from ..metrics import NearestOps, NearestPolicy, NearestPredicate
from ..contracts import PrimitiveSpec, WarpFunction
from .results import ResultPolicy, IdPoint, IdPointResult

HostDataT = TypeVar("HostDataT")
DeviceDataT = TypeVar("DeviceDataT")
ItemT = TypeVar("ItemT")
PointT = TypeVar("PointT")
QueryT = TypeVar("QueryT")
ResultT = TypeVar("ResultT")

class NearestQuery(BoundQuery, Generic[HostDataT, DeviceDataT, ItemT, PointT, QueryT]):
    device: WarpFunction[[QueryT, float, DeviceTree, DeviceDataT], tuple[PointT, float, int, int]]
    ops: NearestOps[QueryT, ItemT, PointT]
    def __call__(self, tree: object, geometry: HostDataT, queries: ArrayLike,
                 max_distance: ArrayLike = ...) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]: ...

class SelectedNearestQuery(BoundQuery,
                           Generic[HostDataT, DeviceDataT, ItemT, PointT, QueryT, ResultT]):
    device: WarpFunction[[QueryT, float, DeviceTree, DeviceDataT], tuple[ResultT, int]]
    result_policy: ResultPolicy[QueryT, ItemT, PointT, ResultT]
    ops: NearestOps[QueryT, ItemT, PointT]

@overload
def make_nearest(primitive: PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT],
                 metric: NearestPolicy[DeviceDataT, ItemT, PointT, QueryT], *,
                 result_policy: ResultPolicy[QueryT, ItemT, PointT, ResultT],
                 predicate: WarpFunction[[QueryT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, QueryT] | None = ...,
                 stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool = ...,
                 packed_nodes: bool = ..., group_leaves: bool | None = ...
                 ) -> SelectedNearestQuery[HostDataT, DeviceDataT, ItemT, PointT, QueryT, ResultT]: ...
@overload
def make_nearest(primitive: PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT],
                 metric: None = ..., *,
                 evaluate: WarpFunction[[PointT, ItemT], tuple[PointT, float]],
                 lower_bound: WarpFunction[[PointT, wp.vec3, wp.vec3], float],
                 result_policy: ResultPolicy[PointT, ItemT, PointT, ResultT],
                 predicate: WarpFunction[[PointT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, PointT] | None = ...,
                 stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool = ...,
                 packed_nodes: bool = ..., group_leaves: bool | None = ...
                 ) -> SelectedNearestQuery[HostDataT, DeviceDataT, ItemT, PointT, PointT, ResultT]: ...

@overload
def make_nearest(primitive: PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT],
                 metric: NearestPolicy[DeviceDataT, ItemT, PointT, QueryT], *,
                 result_policy: IdPoint,
                 predicate: WarpFunction[[QueryT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, QueryT] | None = ...,
                 stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool = ...,
                 packed_nodes: bool = ..., group_leaves: bool | None = ...
                 ) -> SelectedNearestQuery[HostDataT, DeviceDataT, ItemT, PointT, QueryT, IdPointResult[PointT]]: ...
@overload
def make_nearest(primitive: PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT],
                 metric: None = ..., *,
                 evaluate: WarpFunction[[PointT, ItemT], tuple[PointT, float]],
                 lower_bound: WarpFunction[[PointT, wp.vec3, wp.vec3], float],
                 result_policy: IdPoint,
                 predicate: WarpFunction[[PointT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, PointT] | None = ...,
                 stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool = ...,
                 packed_nodes: bool = ..., group_leaves: bool | None = ...
                 ) -> SelectedNearestQuery[HostDataT, DeviceDataT, ItemT, PointT, PointT, IdPointResult[PointT]]: ...

@overload
def make_nearest(primitive: PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT],
                 metric: NearestPolicy[DeviceDataT, ItemT, PointT, QueryT], *,
                 predicate: WarpFunction[[QueryT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, QueryT] | None = ...,
                 result_policy: None = ...,
                 stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool = ...,
                 packed_nodes: bool = ..., group_leaves: bool | None = ...
                 ) -> NearestQuery[HostDataT, DeviceDataT, ItemT, PointT, QueryT]: ...
@overload
def make_nearest(primitive: PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT],
                 metric: None = ..., *,
                 evaluate: WarpFunction[[PointT, ItemT], tuple[PointT, float]],
                 lower_bound: WarpFunction[[PointT, wp.vec3, wp.vec3], float],
                 predicate: WarpFunction[[PointT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, PointT] | None = ...,
                 result_policy: None = ...,
                 stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool = ...,
                 packed_nodes: bool = ..., group_leaves: bool | None = ...
                 ) -> NearestQuery[HostDataT, DeviceDataT, ItemT, PointT, PointT]: ...

def prepare_nearest(ops: NearestOps[QueryT, ItemT, PointT], queries: ArrayLike,
                    max_distance: ArrayLike) -> tuple[jax.Array, jax.Array, tuple[int, ...]]: ...


def _bind_nearest(primitive: PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT],
                  metric: NearestPolicy[DeviceDataT, ItemT, PointT, QueryT]
                  ) -> NearestOps[QueryT, ItemT, PointT]: ...
