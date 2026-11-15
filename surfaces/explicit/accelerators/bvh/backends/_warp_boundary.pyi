from dataclasses import dataclass
from typing import Callable, Generic, NamedTuple, Protocol, TypeVar, overload
import warp as wp
from jax.typing import ArrayLike
from .warp_queries.contracts import DeviceArray, PrimitiveSpec, WarpFunction
from ._warp_tree import DeviceTree, TreeData
from .warp_queries.queries.boundary import MeasureOps, RobinOps
from .warp_queries.metrics import NearestPredicate
from .warp_queries.domain import ClassificationOps
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
class _BoundaryIndices(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...
    def __getitem__(self, index: tuple[int, int]) -> int: ...

class BoundaryCore:
    vertices: DeviceArray[wp.vec3]
    indices: _BoundaryIndices
    closed: DeviceArray[int]
    primitive_normal: DeviceArray[wp.vec3]
    primitive_measure: DeviceArray[float]
class SilhouetteMeta:
    feature_a: DeviceArray[wp.vec3]
    feature_b: DeviceArray[wp.vec3]
    feature_n0: DeviceArray[wp.vec3]
    feature_n1: DeviceArray[wp.vec3]
    feature_refs: DeviceArray[int]
    node_feature_offset: DeviceArray[int]
    node_feature_count: DeviceArray[int]
    node_cone_axis: DeviceArray[wp.vec3]
    node_cone_sin_cos: DeviceArray[wp.vec2]
    node_cone_radius: DeviceArray[float]
class BoundaryCoreData(NamedTuple):
    vertices: ArrayLike
    indices: ArrayLike
    closed: ArrayLike
    primitive_normal: ArrayLike
    primitive_measure: ArrayLike
class SilhouetteData(NamedTuple):
    feature_a: ArrayLike
    feature_b: ArrayLike
    feature_n0: ArrayLike
    feature_n1: ArrayLike
    feature_refs: ArrayLike
    node_feature_offset: ArrayLike
    node_feature_count: ArrayLike
    node_cone_axis: ArrayLike
    node_cone_sin_cos: ArrayLike
    node_cone_radius: ArrayLike

class BoundaryData(NamedTuple, Generic[CH, FH]):
    core: CH
    silhouette: FH | None = ...

@dataclass(frozen=True)
class BoundaryAdapter(Generic[H, D, ItemT, P, CH, CD, FH, FD, FI]):
    prepare: Callable[[PrimitiveSpec[H, D, ItemT, P], H], CH]
    pack_core: Callable[[CH, H, TreeData, H], CH]
    feature_primitive: PrimitiveSpec[FH, FD, FI, P] | None = ...
    build_silhouette: Callable[[CH, TreeData, H], FH] | None = ...
    visibility: WarpFunction[[wp.vec3, DeviceTree, FD, int], bool] | None = ...
    classification: Callable[[PrimitiveSpec[H, D, ItemT, P]], ClassificationOps[P, D, CD]] | None = ...
    predicate: NearestPredicate[FD, FI, P, P] | WarpFunction[[P, FI], bool] | None = ...
    measure: Callable[[PrimitiveSpec[H, D, ItemT, P]], MeasureOps[ItemT, P, CH]] | None = ...
    robin: Callable[[PrimitiveSpec[H, D, ItemT, P]], RobinOps[P, D, ItemT, CD, FD]] | None = ...

cone_visibility: WarpFunction[[wp.vec3, DeviceTree, SilhouetteMeta, int], bool]

class BoundaryPrimitive(PrimitiveSpec[H, D, ItemT, P], Protocol[H, D, ItemT, P, CH, CD, FH, FD, FI]):
    @property
    def boundary_adapter(self) -> BoundaryAdapter[H, D, ItemT, P, CH, CD, FH, FD, FI] | None: ...

@overload
def resolve_boundary_adapter(primitive: PrimitiveSpec[H, D, ItemT, P],
    adapter: BoundaryAdapter[H, D, ItemT, P, CH, CD, FH, FD, FI]
    ) -> BoundaryAdapter[H, D, ItemT, P, CH, CD, FH, FD, FI]: ...
@overload
def resolve_boundary_adapter(primitive: BoundaryPrimitive[H, D, ItemT, P, CH, CD, FH, FD, FI],
    adapter: None = ...) -> BoundaryAdapter[H, D, ItemT, P, CH, CD, FH, FD, FI]: ...

@overload
def build_boundary(primitive: PrimitiveSpec[H, D, ItemT, P], data: H, config: object = ..., *,
    adapter: BoundaryAdapter[H, D, ItemT, P, CH, CD, FH, FD, FI],
    silhouettes: bool = ...) -> tuple[TreeData, H, BoundaryData[CH, FH]]: ...
@overload
def build_boundary(primitive: BoundaryPrimitive[H, D, ItemT, P, CH, CD, FH, FD, FI], data: H, config: object = ..., *,
    adapter: None = ..., silhouettes: bool = ...) -> tuple[TreeData, H, BoundaryData[CH, FH]]: ...
