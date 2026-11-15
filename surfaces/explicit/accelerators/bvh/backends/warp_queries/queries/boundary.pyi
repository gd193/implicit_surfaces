from dataclasses import dataclass
from typing import Callable, Generic, NamedTuple, Protocol, TypeVar, overload
import jax
import warp as wp
from jax.typing import ArrayLike
from ..contracts import DeviceArray, PrimitiveSpec, WarpFunction, WarpType
from ..._warp_tree import DeviceTree
from ..launch import BoundQuery
from ..._warp_boundary import BoundaryData, BoundaryCoreData, SilhouetteData, BoundaryCore, SilhouetteMeta
from ..metrics import Euclidean, NearestPolicy
H = TypeVar("H")
D = TypeVar("D")
ItemT = TypeVar("ItemT")
P = TypeVar("P")
PointContra = TypeVar("PointContra", contravariant=True)
Q = TypeVar("Q")
CH = TypeVar("CH")
CD = TypeVar("CD")
FH = TypeVar("FH")
FD = TypeVar("FD")
FI = TypeVar("FI")

class SamplingMeta:
    primitive_weight: DeviceArray[float]
    node_weight: DeviceArray[float]
class CoefficientMeta:
    primitive_bounds: DeviceArray[wp.vec2]
    node_bounds: DeviceArray[wp.vec2]
class SamplingData(NamedTuple):
    primitive_weight: ArrayLike
    node_weight: ArrayLike
class CoefficientData(NamedTuple):
    primitive_bounds: ArrayLike
    node_bounds: ArrayLike
class RobinData(NamedTuple, Generic[CH, FH]):
    core: CH
    silhouette: FH
    coefficients: CoefficientData

@dataclass(frozen=True)
class MeasureOps(Generic[ItemT, P, CH]):
    sample: WarpFunction[[ItemT, wp.vec2], tuple[P, P, float]]
    weights: Callable[[CH], ArrayLike] | None = ...
@dataclass(frozen=True)
class RegionOps(Generic[Q, P]):
    query_type: WarpType[Q]
    query_shape: tuple[int, ...]
    valid: WarpFunction[[Q, float], bool]
    overlaps: WarpFunction[[Q, float, wp.vec3, wp.vec3], bool]
    contains: WarpFunction[[Q, float, P], bool]
@dataclass(frozen=True)
class RobinOps(Generic[Q, D, ItemT, CD, FD]):
    query_type: WarpType[Q]
    query_shape: tuple[int, ...]
    valid: WarpFunction[[Q], bool]
    node_bound: WarpFunction[[Q, wp.vec3, wp.vec3, wp.vec2], float] | None
    primitive_bound: WarpFunction[[Q, ItemT, float], float] | None
    feature_bound: WarpFunction[[Q, wp.vec3, wp.vec3, wp.vec3, wp.vec3], float] | None
    node_bound_with_data: WarpFunction[[Q, DeviceTree, D, CD, FD, CoefficientMeta, int], float] | None = ...
    feature_bound_with_data: WarpFunction[[Q, DeviceTree, D, CD, FD, CoefficientMeta, int, int], float] | None = ...
    core_type: WarpType[CD] | None = ...
    feature_type: WarpType[FD] | None = ...
    to_score: WarpFunction[[float], float] | None = ...
    to_distance: WarpFunction[[float], float] | None = ...
    leaf_bound_with_data: WarpFunction[[Q, D, CD, CoefficientMeta, int, int, float], float] | None = ...
    node_visit: WarpFunction[[Q, DeviceTree, D, CD, FD, CoefficientMeta, int, wp.vec3, wp.vec3, float], tuple[float, float, bool, bool]] | None = ...

class RobinPolicy(Protocol[D, ItemT, P, Q, CD, FD]):
    def bind(self, primitive: PrimitiveSpec[H, D, ItemT, P], metric: NearestPolicy[D, ItemT, P, Q]) -> RobinOps[Q, D, ItemT, CD, FD]: ...

class RobinMetric(Protocol[D, ItemT, PointContra, Q, CD, FD]):
    def bind_robin(self, primitive: PrimitiveSpec[H, D, ItemT, PointContra]) -> RobinOps[Q, D, ItemT, CD, FD]: ...

class BoundarySampleQuery(BoundQuery, Generic[H, D, ItemT, P, Q]):
    device: WarpFunction[[Q, float, wp.vec3, DeviceTree, D, SamplingMeta], tuple[P, P, float, int, int]]
    region: RegionOps[Q, P]
    def __call__(self, tree: object, geometry: H, metadata: SamplingData,
                 queries: ArrayLike, radii: ArrayLike, randoms: ArrayLike
                 ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]: ...
class RobinQuery(BoundQuery, Generic[H, D, ItemT, Q, CH, CD, FH, FD]):
    device: WarpFunction[[Q, float, DeviceTree, D, CD, FD, CoefficientMeta], tuple[float, int]]
    ops: RobinOps[Q, D, ItemT, CD, FD]
    def __call__(self, tree: object, geometry: H, metadata: RobinData[CH, FH],
                 queries: ArrayLike, cap: ArrayLike = ...
                 ) -> tuple[jax.Array, jax.Array]: ...

def prepare_boundary_sample(tree: object, boundary: BoundaryData[CH, FH] | CH,
    measure: MeasureOps[ItemT, P, CH] | None = ...) -> SamplingData: ...
def prepare_robin_coefficients(tree: object, minimum: ArrayLike, maximum: ArrayLike | None = ...) -> CoefficientData: ...
def prepare_robin(tree: object, boundary: BoundaryData[CH, FH], minimum: ArrayLike,
                  maximum: ArrayLike | None = ...) -> RobinData[CH, FH]: ...

@overload
def make_boundary_sample(primitive: PrimitiveSpec[H, D, ItemT, P], *, region: RegionOps[Q, P],
    block_dim: int = ...) -> BoundarySampleQuery[H, D, ItemT, P, Q]: ...
@overload
def make_boundary_sample(primitive: PrimitiveSpec[H, D, ItemT, P],
    measure: MeasureOps[ItemT, P, CH] | None = ..., region: None = ...,
    *, block_dim: int = ...) -> BoundarySampleQuery[H, D, ItemT, P, P]: ...
@overload
def make_boundary_sample(primitive: PrimitiveSpec[H, D, ItemT, P],
    measure: MeasureOps[ItemT, P, CH] | None, region: RegionOps[Q, P],
    *, block_dim: int = ...) -> BoundarySampleQuery[H, D, ItemT, P, Q]: ...

@overload
def make_robin_radius(primitive: PrimitiveSpec[H, D, ItemT, P],
    policy: None = ..., *, metric: RobinMetric[D, ItemT, P, Q, CD, FD],
    core_host_type: type[CH], feature_host_type: type[FH],
    stack_capacity: int = ..., block_dim: int = ...
    ) -> RobinQuery[H, D, ItemT, Q, CH, CD, FH, FD]: ...
@overload
def make_robin_radius(primitive: PrimitiveSpec[H, D, ItemT, P],
    policy: None = ..., *, metric: RobinMetric[D, ItemT, P, Q, CD, FD],
    core_host_type: type[BoundaryCoreData] = ..., feature_host_type: type[SilhouetteData] = ...,
    stack_capacity: int = ..., block_dim: int = ...
    ) -> RobinQuery[H, D, ItemT, Q, BoundaryCoreData, CD, SilhouetteData, FD]: ...
@overload
def make_robin_radius(primitive: PrimitiveSpec[H, D, ItemT, P],
    policy: None, metric: RobinMetric[D, ItemT, P, Q, CD, FD], *,
    core_host_type: type[CH], feature_host_type: type[FH],
    stack_capacity: int = ..., block_dim: int = ...
    ) -> RobinQuery[H, D, ItemT, Q, CH, CD, FH, FD]: ...
@overload
def make_robin_radius(primitive: PrimitiveSpec[H, D, ItemT, P],
    policy: None, metric: RobinMetric[D, ItemT, P, Q, CD, FD], *,
    core_host_type: type[BoundaryCoreData] = ..., feature_host_type: type[SilhouetteData] = ...,
    stack_capacity: int = ..., block_dim: int = ...
    ) -> RobinQuery[H, D, ItemT, Q, BoundaryCoreData, CD, SilhouetteData, FD]: ...
@overload
def make_robin_radius(primitive: PrimitiveSpec[H, D, ItemT, P],
    policy: RobinOps[Q, D, ItemT, CD, FD] | RobinPolicy[D, ItemT, P, Q, CD, FD],
    metric: NearestPolicy[D, ItemT, P, Q] | None = ..., *, core_host_type: type[CH],
    feature_host_type: type[FH], stack_capacity: int = ...,
    block_dim: int = ...) -> RobinQuery[H, D, ItemT, Q, CH, CD, FH, FD]: ...
@overload
def make_robin_radius(primitive: PrimitiveSpec[H, D, ItemT, P],
    policy: RobinOps[Q, D, ItemT, CD, FD] | RobinPolicy[D, ItemT, P, Q, CD, FD],
    metric: NearestPolicy[D, ItemT, P, Q] | None = ..., *,
    core_host_type: type[BoundaryCoreData] = ..., feature_host_type: type[SilhouetteData] = ...,
    stack_capacity: int = ..., block_dim: int = ...
    ) -> RobinQuery[H, D, ItemT, Q, BoundaryCoreData, CD, SilhouetteData, FD]: ...
@overload
def make_robin_radius(primitive: PrimitiveSpec[H, D, ItemT, P],
    policy: None = ..., metric: Euclidean | None = ..., *,
    core_host_type: type[BoundaryCoreData] = ..., feature_host_type: type[SilhouetteData] = ...,
    stack_capacity: int = ..., block_dim: int = ...
    ) -> RobinQuery[H, D, ItemT, P, BoundaryCoreData, BoundaryCore, SilhouetteData, SilhouetteMeta]: ...
