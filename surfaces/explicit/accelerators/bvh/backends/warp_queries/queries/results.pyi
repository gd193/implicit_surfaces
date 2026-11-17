"""Typed selected-result policy contract."""
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar
from ..contracts import WarpFunction, WarpType

QueryT = TypeVar("QueryT")
ItemT = TypeVar("ItemT")
PointT = TypeVar("PointT")
ResultT = TypeVar("ResultT")

@dataclass(frozen=True)
class ResultPolicy(Generic[QueryT, ItemT, PointT, ResultT]):
    result_type: WarpType[ResultT]
    build: WarpFunction[[QueryT, ItemT, int, PointT, float], ResultT]

@dataclass(frozen=True)
class IdPoint:
    pass

PointCo = TypeVar("PointCo", covariant=True)
class IdPointResult(Protocol[PointCo]):
    @property
    def id(self) -> int: ...
    @property
    def point(self) -> PointCo: ...
