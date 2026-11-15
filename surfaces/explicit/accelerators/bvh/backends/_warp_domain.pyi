"""Typed contracts for native ray and classification operations."""
from dataclasses import dataclass
from typing import Generic, TypeVar, overload
import warp as wp
from ._warp_types import PrimitiveSpec, WarpFunction, WarpType

H = TypeVar("H")
D = TypeVar("D")
I = TypeVar("I")
P = TypeVar("P")
S = TypeVar("S")
M = TypeVar("M")

class Ray:
    origin: wp.vec3
    direction: wp.vec3
    inverse: wp.vec3
    fast: bool
    axes: wp.vec3i
    frame_origin: wp.vec3
    frame_direction: wp.vec3
    shear: wp.vec2

prepare_ray: WarpFunction[[wp.vec3, wp.vec3, int], Ray]

@dataclass(frozen=True)
class RayOps(Generic[D, S]):
    intersect: WarpFunction[[Ray, float, float, D, int, bool, bool], wp.vec2] | None
    state_type: WarpType[S]
    next_hit: WarpFunction[[Ray, float, float, D, int, S, bool, bool], tuple[S, float, int]] | None
    # An explicit RayOps[D, S] can supply D when no callback is supplied.
    @overload
    def __init__(self, intersect: WarpFunction[[Ray, float, float, D, int, bool, bool], wp.vec2] | None = ..., *, state_type: WarpType[S], next_hit: WarpFunction[[Ray, float, float, D, int, S, bool, bool], tuple[S, float, int]] | None = ...) -> None: ...  # pyright: ignore[reportInvalidTypeVarUse]
    @overload
    def __init__(self, intersect: WarpFunction[[Ray, float, float, D, int, bool, bool], wp.vec2] | None, state_type: WarpType[S], next_hit: WarpFunction[[Ray, float, float, D, int, S, bool, bool], tuple[S, float, int]] | None = ...) -> None: ...
    @overload
    def __init__(self, intersect: WarpFunction[[Ray, float, float, D, int, bool, bool], wp.vec2] | None = ..., *, next_hit: None = ...) -> None: ...  # pyright: ignore[reportInvalidTypeVarUse]

@dataclass(frozen=True)
class ClassificationOps(Generic[P, D, M]):
    metadata_type: WarpType[M]
    valid: WarpFunction[[M], bool]
    event: WarpFunction[[P, D, int, int, M], int]

def builtin_classification(primitive: PrimitiveSpec[H, D, I, P], metadata_type: WarpType[M]) -> ClassificationOps[P, D, M]: ...
