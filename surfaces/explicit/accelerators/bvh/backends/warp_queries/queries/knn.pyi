"""Host specialization signatures; JAX arrays do not encode vector dimensions."""
from ..._warp_tree import DeviceTree
from typing import Generic, Literal, TypeVar, overload
import jax
from jax.typing import ArrayLike
import warp as wp
from ..launch import BoundQuery
from ..metrics import NearestOps, NearestPolicy, NearestPredicate
from ..contracts import PrimitiveSpec, WarpFunction, DeviceArray2D
from .results import ResultPolicy, IdPoint, IdPointResult

HostDataT = TypeVar("HostDataT")
DeviceDataT = TypeVar("DeviceDataT")
ItemT = TypeVar("ItemT")
PointT = TypeVar("PointT")
QueryT = TypeVar("QueryT")
ResultT = TypeVar("ResultT")

class KnnQuery(BoundQuery, Generic[HostDataT, DeviceDataT, ItemT, PointT, QueryT]):
    device: WarpFunction[[QueryT, float, DeviceTree, DeviceDataT, DeviceArray2D[float], DeviceArray2D[int] | DeviceArray2D[PointT], int], tuple[int, int]]
    k: int
    return_points: bool
    ops: NearestOps[QueryT, ItemT, PointT]
    def __call__(self, tree: object, geometry: HostDataT, queries: ArrayLike,
                 max_distance: ArrayLike = ...) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]: ...

class SelectedKnnQuery(BoundQuery,
                       Generic[HostDataT, DeviceDataT, ItemT, PointT, QueryT, ResultT]):
    device: WarpFunction[[QueryT, float, DeviceTree, DeviceDataT, DeviceArray2D[float], DeviceArray2D[ResultT], int], tuple[int, int]]
    result_policy: ResultPolicy[QueryT, ItemT, PointT, ResultT]
    ops: NearestOps[QueryT, ItemT, PointT]
    k: int

@overload
def make_knn(primitive: PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT],
             k: int, metric: NearestPolicy[DeviceDataT, ItemT, PointT, QueryT], *,
             result_policy: ResultPolicy[QueryT, ItemT, PointT, ResultT],
             predicate: WarpFunction[[QueryT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, QueryT] | None = ...,
             stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool = ...,
             packed_nodes: bool = ..., group_leaves: bool | None = ..., return_points: Literal[False] = ...
             ) -> SelectedKnnQuery[HostDataT, DeviceDataT, ItemT, PointT, QueryT, ResultT]: ...
@overload
def make_knn(primitive: PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT],
             k: int, metric: None = ..., *,
             evaluate: WarpFunction[[PointT, ItemT], tuple[PointT, float]],
             lower_bound: WarpFunction[[PointT, wp.vec3, wp.vec3], float],
             result_policy: ResultPolicy[PointT, ItemT, PointT, ResultT],
             predicate: WarpFunction[[PointT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, PointT] | None = ...,
             stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool = ...,
             packed_nodes: bool = ..., group_leaves: bool | None = ..., return_points: Literal[False] = ...
             ) -> SelectedKnnQuery[HostDataT, DeviceDataT, ItemT, PointT, PointT, ResultT]: ...

@overload
def make_knn(primitive: PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT],
             k: int, metric: NearestPolicy[DeviceDataT, ItemT, PointT, QueryT], *,
             result_policy: IdPoint,
             predicate: WarpFunction[[QueryT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, QueryT] | None = ...,
             stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool = ...,
             packed_nodes: bool = ..., group_leaves: bool | None = ..., return_points: Literal[False] = ...
             ) -> SelectedKnnQuery[HostDataT, DeviceDataT, ItemT, PointT, QueryT, IdPointResult[PointT]]: ...
@overload
def make_knn(primitive: PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT],
             k: int, metric: None = ..., *,
             evaluate: WarpFunction[[PointT, ItemT], tuple[PointT, float]],
             lower_bound: WarpFunction[[PointT, wp.vec3, wp.vec3], float],
             result_policy: IdPoint,
             predicate: WarpFunction[[PointT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, PointT] | None = ...,
             stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool = ...,
             packed_nodes: bool = ..., group_leaves: bool | None = ..., return_points: Literal[False] = ...
             ) -> SelectedKnnQuery[HostDataT, DeviceDataT, ItemT, PointT, PointT, IdPointResult[PointT]]: ...

@overload
def make_knn(primitive: PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT],
                 k: int, metric: NearestPolicy[DeviceDataT, ItemT, PointT, QueryT], *,
                 predicate: WarpFunction[[QueryT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, QueryT] | None = ...,
                 result_policy: None = ...,
                 stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool = ...,
                 packed_nodes: bool = ..., group_leaves: bool | None = ..., return_points: bool = ...
                 ) -> KnnQuery[HostDataT, DeviceDataT, ItemT, PointT, QueryT]: ...
@overload
def make_knn(primitive: PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT],
                 k: int, metric: None = ..., *,
                 evaluate: WarpFunction[[PointT, ItemT], tuple[PointT, float]],
                 lower_bound: WarpFunction[[PointT, wp.vec3, wp.vec3], float],
                 predicate: WarpFunction[[PointT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, PointT] | None = ...,
                 result_policy: None = ...,
                 stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool = ...,
                 packed_nodes: bool = ..., group_leaves: bool | None = ..., return_points: bool = ...
                 ) -> KnnQuery[HostDataT, DeviceDataT, ItemT, PointT, PointT]: ...
