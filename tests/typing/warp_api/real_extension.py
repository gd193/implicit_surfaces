"""Real third-party Warp extension, shared by strict fixtures and CPU tests."""
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Protocol
import jax
import numpy as np
import warp as wp
from surfaces.explicit.accelerators.bvh.warp import PointItem3, WarpFunction, WarpType, warp_func, warp_struct
from tests.typing.warp_api.native_math import squared_distance

class VectorArray(Protocol):
    def __getitem__(self, index: int) -> wp.vec3: ...


@warp_struct
class ExternalData:
    if TYPE_CHECKING:
        positions: VectorArray
    else:
        positions: wp.array(dtype=wp.vec3)


@warp_struct
class ExternalItem:
    position: wp.vec3


@warp_func
def external_load(data: ExternalData, index: int) -> ExternalItem:
    item = ExternalItem()
    item.position = data.positions[index]
    return item


@warp_func
def external_bounds(item: ExternalItem) -> tuple[wp.vec3, wp.vec3]:
    return item.position, item.position


@warp_func
def external_evaluate(query: wp.vec3, item: ExternalItem) -> tuple[wp.vec3, float]:
    return item.position, squared_distance(query, item.position)


@warp_func
def external_bound(query: wp.vec3, lo: wp.vec3, hi: wp.vec3) -> float:
    # Conservative zero bound keeps the extension independent of upstream
    # Warp's incomplete vector-operator stubs; other tests exercise pruning.
    return 0.0



class ExternalArrays(NamedTuple):
    positions: jax.Array


@dataclass(frozen=True)
class ExternalPoints:
    """No inheritance, built-in item layout, or private imports required."""
    dimension: int = 3
    point_type: WarpType[wp.vec3] = wp.vec3
    data_type: WarpType[ExternalData] = ExternalData
    load: WarpFunction[[ExternalData, int], ExternalItem] = external_load
    bounds_of_item: WarpFunction[[ExternalItem], tuple[wp.vec3, wp.vec3]] = external_bounds

    def bounds(self, data: ExternalArrays) -> tuple[np.ndarray, np.ndarray]:
        positions = np.asarray(data.positions)
        return positions, positions

    def pack(self, data: ExternalArrays, ids: np.ndarray | jax.Array) -> ExternalArrays:
        return ExternalArrays(data.positions[ids])




@warp_func
def builtin_evaluate(query: wp.vec3, item: PointItem3) -> tuple[wp.vec3, float]:
    return item.a, squared_distance(query, item.a)
