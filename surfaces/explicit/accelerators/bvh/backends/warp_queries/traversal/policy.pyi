from dataclasses import dataclass
from typing import Any, Generic, TypeVar
import warp as wp
from ..._warp_tree import DeviceTree
from ..contracts import WarpFunction, WarpRef, WarpType

ContextT = TypeVar("ContextT")
StateT = TypeVar("StateT")

@dataclass(frozen=True)
class TraversalPolicy(Generic[ContextT, StateT]):
    device: WarpFunction[[DeviceTree, ContextT, WarpRef[StateT]], None]
    state_type: WarpType[StateT]
    _device_kernel: wp.Kernel

def make_traversal_state(name: str, extra_fields: dict[str, Any] | None = ...) -> WarpType[Any]: ...

def make_traversal(Context: WarpType[ContextT], State: WarpType[StateT],
                   bounds: WarpFunction[[ContextT, DeviceTree, int, wp.vec3, wp.vec3, WarpRef[StateT]], tuple[float, bool, bool]],
                   visit: WarpFunction[[ContextT, DeviceTree, int, int, int, bool, WarpRef[StateT]], None], *,
                   stack_capacity: int = ..., near_first: bool = ..., packed_nodes: bool = ...,
                   cache_bounds: bool = ..., group_leaves: bool = ..., test_root: bool = ...,
                   resume: WarpFunction[[ContextT, DeviceTree, int, wp.vec3, wp.vec3], float] | None = ...
                   ) -> TraversalPolicy[ContextT, StateT]: ...
