"""Static result builders for selected nearest and kNN neighbors."""
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Generic, Protocol, TypeVar

import warp as wp

from ..contracts import make_struct


QueryT = TypeVar("QueryT")
ItemT = TypeVar("ItemT")
PointT = TypeVar("PointT")
ResultT = TypeVar("ResultT")
PointCo = TypeVar("PointCo", covariant=True)


class IdPointResult(Protocol[PointCo]):
    @property
    def id(self) -> int: ...
    @property
    def point(self) -> PointCo: ...


@dataclass(frozen=True)
class ResultPolicy(Generic[QueryT, ItemT, PointT, ResultT]):
    """Build a Warp struct from (query, item, original_id, point, reported_value).

    ``build`` must be a decorated Warp function returning ``result_type``.
    It runs once per selected neighbor, after traversal and sorting.
    ``reported_value`` is the converted distance, not the internal score.
    """

    result_type: object
    build: object

    def __post_init__(self):
        if not isinstance(self.build, wp.Function):
            raise TypeError("result build must be a decorated Warp function")
        if not hasattr(self.result_type, "vars") or not hasattr(self.result_type, "ctype"):
            raise TypeError("result_type must be a decorated Warp struct")


@dataclass(frozen=True)
class IdPoint:
    """Built-in selected result with original ID and selected point."""


@lru_cache(maxsize=None)
def resolve_result(policy, primitive, query_type):
    if not isinstance(policy, IdPoint):
        if not isinstance(policy, ResultPolicy):
            raise TypeError("result_policy must be ResultPolicy or IdPoint")
        return policy
    point_type = primitive.point_type
    Result = make_struct("IdPointResult", {"id": wp.int32, "point": point_type})

    @wp.func
    def build(query: Any, item: Any, original_id: int,
              selected_point: point_type, reported_value: float) -> Result:
        result = Result()
        result.id = original_id
        result.point = selected_point
        return result

    return ResultPolicy(Result, build)
