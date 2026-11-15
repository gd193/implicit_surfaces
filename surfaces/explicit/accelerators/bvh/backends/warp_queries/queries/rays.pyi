"""Host contracts for the four statically specialized ray modes."""
from typing import Generic, Literal, TypeVar, overload
import jax
from jax.typing import ArrayLike
import warp as wp
from ..domain import Ray, RayOps
from ..launch import BoundQuery
from ..contracts import PrimitiveSpec, WarpFunction

H = TypeVar("H")
D = TypeVar("D")
I = TypeVar("I")
P = TypeVar("P")
S = TypeVar("S")
Mode = TypeVar("Mode", Literal["first"], Literal["any"], Literal["count"], Literal["all"])
RayMode = Literal["first", "any", "count", "all"]

class RayQuery(BoundQuery, Generic[H, D, I, P, Mode]):
    mode: Mode
    exclusion: bool
    @overload
    def __call__(self: RayQuery[H, D, I, P, Literal["first"]], tree: object, geometry: H, origin: ArrayLike, direction: ArrayLike, max_t: ArrayLike = ..., *, min_t: ArrayLike = ..., exclude_face: ArrayLike | None = ..., max_hits: None = ...) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]: ...
    @overload
    def __call__(self: RayQuery[H, D, I, P, Literal["any"]], tree: object, geometry: H, origin: ArrayLike, direction: ArrayLike, max_t: ArrayLike = ..., *, min_t: ArrayLike = ..., exclude_face: ArrayLike | None = ..., max_hits: None = ...) -> tuple[jax.Array, jax.Array]: ...
    @overload
    def __call__(self: RayQuery[H, D, I, P, Literal["count"]], tree: object, geometry: H, origin: ArrayLike, direction: ArrayLike, max_t: ArrayLike = ..., *, min_t: ArrayLike = ..., exclude_face: ArrayLike | None = ..., max_hits: None = ...) -> tuple[jax.Array, jax.Array]: ...
    @overload
    def __call__(self: RayQuery[H, D, I, P, Literal["all"]], tree: object, geometry: H, origin: ArrayLike, direction: ArrayLike, max_t: ArrayLike = ..., *, min_t: ArrayLike = ..., exclude_face: ArrayLike | None = ..., max_hits: int) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]: ...

@overload
def make_ray_query(primitive: PrimitiveSpec[H, D, I, P], mode: Mode = "first", *, ops: RayOps[D, S], exclude: bool = ..., stack_capacity: int = ..., block_dim: int = ...) -> RayQuery[H, D, I, P, Mode]: ...
@overload
def make_ray_query(primitive: PrimitiveSpec[H, D, I, P], mode: Mode = "first", *, ops: None = ..., exclude: bool = ..., stack_capacity: int = ..., block_dim: int = ...) -> RayQuery[H, D, I, P, Mode]: ...

ray_box: WarpFunction[[Ray, wp.vec3, wp.vec3, float, float, int], wp.vec2]
# Native vector overloads accept both dimensions.
valid_ray: WarpFunction[[wp.vec2 | wp.vec3, wp.vec2 | wp.vec3, float, float, int, int, bool, bool], bool]
