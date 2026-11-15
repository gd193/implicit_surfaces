"""Host specialization signatures; JAX arrays do not encode vector dimensions."""
from typing import Generic, TypeVar, overload
import jax
from jax.typing import ArrayLike
import warp as wp
from ._warp_factory import BoundQuery
from ._warp_metrics import NearestOps, NearestPolicy, NearestPredicate
from ._warp_types import PrimitiveSpec, WarpFunction

HostDataT = TypeVar("HostDataT")
DeviceDataT = TypeVar("DeviceDataT")
ItemT = TypeVar("ItemT")
PointT = TypeVar("PointT")
QueryT = TypeVar("QueryT")

class KnnQuery(BoundQuery, Generic[HostDataT, DeviceDataT, ItemT, PointT, QueryT]):
    k: int
    ops: NearestOps[QueryT, ItemT, PointT]
    def __call__(self, tree: object, geometry: HostDataT, queries: ArrayLike,
                 max_distance: ArrayLike = ...) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]: ...

@overload
def make_knn(primitive: PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT],
                 k: int, metric: NearestPolicy[DeviceDataT, ItemT, PointT, QueryT], *,
                 predicate: WarpFunction[[QueryT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, QueryT] | None = ...,
                 stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool = ...,
                 packed_nodes: bool = ..., group_leaves: bool | None = ...
                 ) -> KnnQuery[HostDataT, DeviceDataT, ItemT, PointT, QueryT]: ...
@overload
def make_knn(primitive: PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT],
                 k: int, metric: None = ..., *,
                 evaluate: WarpFunction[[PointT, ItemT], tuple[PointT, float]],
                 lower_bound: WarpFunction[[PointT, wp.vec3, wp.vec3], float],
                 predicate: WarpFunction[[PointT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, PointT] | None = ...,
                 stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool = ...,
                 packed_nodes: bool = ..., group_leaves: bool | None = ...
                 ) -> KnnQuery[HostDataT, DeviceDataT, ItemT, PointT, PointT]: ...
