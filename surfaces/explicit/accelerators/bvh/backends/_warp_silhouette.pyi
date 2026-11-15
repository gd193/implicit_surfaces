from dataclasses import dataclass
from typing import Generic, TypeVar, overload
import jax
from jax.typing import ArrayLike
from ._warp_types import PrimitiveSpec
from ._warp_boundary import BoundaryAdapter, BoundaryData, BoundaryPrimitive
from ._warp_metrics import NearestPolicy
from ._warp_nearest import NearestQuery
H = TypeVar("H")
D = TypeVar("D")
ItemT = TypeVar("ItemT")
P = TypeVar("P")
Q = TypeVar("Q")
CH = TypeVar("CH")
CD = TypeVar("CD")
FH = TypeVar("FH")
FD = TypeVar("FD")
FI = TypeVar("FI")

@dataclass(frozen=True)
class SilhouetteQuery(Generic[H, D, ItemT, P, Q, CH, CD, FH, FD, FI]):
    query: NearestQuery[FH, FD, FI, P, Q]
    @property
    def kernel(self) -> object: ...
    def __call__(self, tree: object, geometry: H, boundary: BoundaryData[CH, FH] | FH,
                 queries: ArrayLike, max_distance: ArrayLike = ...
                 ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]: ...
@dataclass(frozen=True)
class StarRadiusQuery(Generic[H, D, ItemT, P, CH, CD, FH, FD, FI]):
    silhouette: SilhouetteQuery[H, D, ItemT, P, P, CH, CD, FH, FD, FI]
    def __call__(self, tree: object, geometry: H, boundary: BoundaryData[CH, FH] | FH,
                 points: ArrayLike, cap: ArrayLike) -> tuple[jax.Array, jax.Array]: ...

@overload
def make_silhouette(primitive: PrimitiveSpec[H, D, ItemT, P], metric: None = ..., *,
    adapter: BoundaryAdapter[H, D, ItemT, P, CH, CD, FH, FD, FI],
    stack_capacity: int = ..., block_dim: int = ...
    ) -> SilhouetteQuery[H, D, ItemT, P, P, CH, CD, FH, FD, FI]: ...
@overload
def make_silhouette(primitive: PrimitiveSpec[H, D, ItemT, P], metric: NearestPolicy[FD, FI, P, Q], *,
    adapter: BoundaryAdapter[H, D, ItemT, P, CH, CD, FH, FD, FI],
    stack_capacity: int = ..., block_dim: int = ...
    ) -> SilhouetteQuery[H, D, ItemT, P, Q, CH, CD, FH, FD, FI]: ...
@overload
def make_silhouette(primitive: BoundaryPrimitive[H, D, ItemT, P, CH, CD, FH, FD, FI], metric: None = ..., *,
    adapter: None = ..., stack_capacity: int = ..., block_dim: int = ...
    ) -> SilhouetteQuery[H, D, ItemT, P, P, CH, CD, FH, FD, FI]: ...
@overload
def make_silhouette(primitive: BoundaryPrimitive[H, D, ItemT, P, CH, CD, FH, FD, FI], metric: NearestPolicy[FD, FI, P, Q], *,
    adapter: None = ..., stack_capacity: int = ..., block_dim: int = ...
    ) -> SilhouetteQuery[H, D, ItemT, P, Q, CH, CD, FH, FD, FI]: ...

@overload
def make_star_radius(primitive: PrimitiveSpec[H, D, ItemT, P], *,
    adapter: BoundaryAdapter[H, D, ItemT, P, CH, CD, FH, FD, FI],
    stack_capacity: int = ..., block_dim: int = ...
    ) -> StarRadiusQuery[H, D, ItemT, P, CH, CD, FH, FD, FI]: ...
@overload
def make_star_radius(primitive: BoundaryPrimitive[H, D, ItemT, P, CH, CD, FH, FD, FI], *,
    adapter: None = ..., stack_capacity: int = ..., block_dim: int = ...
    ) -> StarRadiusQuery[H, D, ItemT, P, CH, CD, FH, FD, FI]: ...
