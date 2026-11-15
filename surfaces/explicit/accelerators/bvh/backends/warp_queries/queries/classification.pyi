"""Static classification callback and host/device metadata connections."""
from ..._warp_tree import DeviceTree
from typing import Any, Generic, Protocol, TypeVar, overload
import jax
from jax.typing import ArrayLike
from ..._warp_boundary import BoundaryAdapter, BoundaryCore, BoundaryCoreData
from ..domain import ClassificationOps
from ..launch import BoundQuery
from ..contracts import PrimitiveSpec, WarpFunction

H = TypeVar("H")
D = TypeVar("D")
I = TypeVar("I")
P = TypeVar("P")
M = TypeVar("M")
HM = TypeVar("HM")
M_co = TypeVar("M_co", covariant=True)

class WithCore(Protocol[M_co]):
    @property
    def core(self) -> M_co: ...

class ClassifyQuery(BoundQuery, Generic[H, D, I, P, HM, M]):
    device: WarpFunction[[P, DeviceTree, D, M], tuple[float, int]]
    policy: ClassificationOps[P, D, M]
    def __call__(self, tree: object, geometry: H, boundary: HM | WithCore[HM], points: ArrayLike) -> tuple[jax.Array, jax.Array]: ...

@overload
def make_classify(primitive: PrimitiveSpec[H, D, I, P], policy: ClassificationOps[P, D, M], *, metadata_host_type: type[HM], adapter: object = ..., stack_capacity: int = ..., block_dim: int = ...) -> ClassifyQuery[H, D, I, P, HM, M]: ...
@overload
def make_classify(primitive: PrimitiveSpec[H, D, I, P], policy: ClassificationOps[P, D, M], *, metadata_host_type: None = ..., adapter: object = ..., stack_capacity: int = ..., block_dim: int = ...) -> ClassifyQuery[H, D, I, P, M, M]: ...
@overload
def make_classify(primitive: PrimitiveSpec[H, D, I, P], policy: None = ..., *, adapter: BoundaryAdapter[H, D, I, P, HM, M, Any, Any, Any], metadata_host_type: None = ..., stack_capacity: int = ..., block_dim: int = ...) -> ClassifyQuery[H, D, I, P, HM, M]: ...
@overload
def make_classify(primitive: PrimitiveSpec[H, D, I, P], policy: None = ..., *, adapter: None = ..., metadata_host_type: None = ..., stack_capacity: int = ..., block_dim: int = ...) -> ClassifyQuery[H, D, I, P, BoundaryCoreData, BoundaryCore]: ...
