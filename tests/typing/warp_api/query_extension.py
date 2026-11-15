"""Executable third-party policies shared by typing checks and CPU tests.

The tiny point policy deliberately has two events per primitive, making it
possible to test stateful iteration independently of built-in intersections.
"""
from surfaces.explicit.accelerators.bvh.warp import Ray, warp_func, warp_struct
from typing import TYPE_CHECKING, NamedTuple, Protocol, TypeVar
import jax
import warp as wp
from tests.typing.warp_api.real_extension import ExternalData, ExternalItem
from tests.typing.warp_api.native_math import squared_distance, vector_x
from dataclasses import dataclass
import numpy as np
from surfaces.explicit.accelerators.bvh.warp import (
    BoundaryAdapter, BoundaryCore, BoundaryCoreData, PrimitiveSpec, RobinOps, SilhouetteMeta,
    TreeData, WarpFunction, WarpType,
)
from tests.typing.warp_api.real_extension import (
    ExternalArrays, VectorArray, external_bounds,
)
from surfaces.explicit.accelerators.bvh.warp import PointsData3

@warp_struct
class Cursor:
    index: int


class IntArray(Protocol):
    def __getitem__(self, index: int) -> int: ...


@warp_struct
class ExternalMetadata:
    if TYPE_CHECKING:
        enabled: IntArray
    else:
        enabled: wp.array(dtype=wp.int32)


class MetadataArrays(NamedTuple):
    enabled: jax.Array


@warp_func
def intersect(ray: Ray, minimum: float, maximum: float, data: ExternalData,
              slot: int, half_open: bool, store_times: bool) -> wp.vec2:
    t = float(slot + 1)
    if minimum <= t and t <= maximum:
        return wp.vec2(t, 1.0)
    return wp.vec2(0.0)


@warp_func
def next_hit(ray: Ray, minimum: float, maximum: float, data: ExternalData,
             slot: int, state: Cursor, half_open: bool,
             store_times: bool) -> tuple[Cursor, float, int]:
    result = Cursor()
    result.index = state.index + 1
    t = float(state.index + 1)
    if state.index < 2 and minimum <= t and t <= maximum:
        return result, t, 1
    return result, 0.0, 0


@warp_func
def metadata_valid(meta: ExternalMetadata) -> bool:
    return meta.enabled[0] != 0


@warp_func
def classification_event(point: wp.vec3, data: ExternalData, slot: int,
                         original: int, meta: ExternalMetadata) -> int:
    return 1


@warp_func
def sample(item: ExternalItem, uniforms: wp.vec2) -> tuple[wp.vec3, wp.vec3, float]:
    return item.position, wp.vec3(0.0, 1.0, 0.0), 1.0


@warp_func
def region_valid(query: float, cap: float) -> bool:
    return query >= 0.0 and cap >= 0.0


@warp_func
def region_overlaps(query: float, cap: float, lower: wp.vec3, upper: wp.vec3) -> bool:
    return True


@warp_func
def region_contains(query: float, cap: float, point: wp.vec3) -> bool:
    return True


@warp_func
def robin_valid(query: wp.vec3) -> bool:
    return True


@warp_func
def robin_node(query: wp.vec3, lower: wp.vec3, upper: wp.vec3,
               coefficients: wp.vec2) -> float:
    return 0.0


@warp_func
def robin_primitive(query: wp.vec3, item: ExternalItem, coefficient: float) -> float:
    # A point boundary uses its Euclidean distance as a conservative radius.
    return squared_distance(query, item.position)


@warp_func
def robin_feature(query: wp.vec3, a: wp.vec3, b: wp.vec3,
                  n0: wp.vec3, n1: wp.vec3) -> float:
    return 1.e30


@warp_func
def score_to_radius(score: float) -> float:
    return wp.sqrt(score)


@warp_func
def radius_to_score(radius: float) -> float:
    return radius * radius




@warp_struct
class FeatureData:
    if TYPE_CHECKING:
        positions: VectorArray
        feature_refs: IntArray
        node_feature_offset: IntArray
        node_feature_count: IntArray
    else:
        positions: wp.array(dtype=wp.vec3)
        feature_refs: wp.array(dtype=wp.int32)
        node_feature_offset: wp.array(dtype=wp.int32)
        node_feature_count: wp.array(dtype=wp.int32)


class FeatureArrays(NamedTuple):
    positions: jax.Array
    feature_refs: np.ndarray | jax.Array
    node_feature_offset: np.ndarray | jax.Array
    node_feature_count: np.ndarray | jax.Array


@warp_func
def feature_load(data: FeatureData, index: int) -> ExternalItem:
    item = ExternalItem()
    item.position = data.positions[data.feature_refs[index]]
    return item


@warp_func
def feature_closest(query: wp.vec3, item: ExternalItem) -> wp.vec3:
    return item.position


@dataclass(frozen=True)
class ExternalFeatures:
    dimension: int = 3
    point_type: WarpType[wp.vec3] = wp.vec3
    data_type: WarpType[FeatureData] = FeatureData
    load: WarpFunction[[FeatureData, int], ExternalItem] = feature_load
    bounds_of_item: WarpFunction[[ExternalItem], tuple[wp.vec3, wp.vec3]] = external_bounds
    closest_euclidean: WarpFunction[[wp.vec3, ExternalItem], wp.vec3] = feature_closest

    def bounds(self, data: FeatureArrays) -> tuple[np.ndarray, np.ndarray]:
        positions = np.asarray(data.positions)
        return positions, positions

    def pack(self, data: FeatureArrays, ids: np.ndarray | jax.Array) -> FeatureArrays:
        return FeatureArrays(data.positions[ids], data.feature_refs, data.node_feature_offset,
                             data.node_feature_count)


def prepare_core(primitive: PrimitiveSpec[ExternalArrays, ExternalData, ExternalItem, wp.vec3],
                 data: ExternalArrays) -> BoundaryCoreData:
    count = data.positions.shape[0]
    return BoundaryCoreData(data.positions, np.zeros((count, 0), dtype=np.int32),
        np.ones((1,), dtype=np.int32), np.zeros(data.positions.shape, dtype=np.float32), np.ones((count,)))


def pack_core(core: BoundaryCoreData, data: ExternalArrays, tree: TreeData,
              geometry: ExternalArrays) -> BoundaryCoreData:
    ids = tree.prim_ids
    return BoundaryCoreData(np.asarray(core.vertices), np.asarray(core.indices)[ids],
        np.asarray(core.closed), np.asarray(core.primitive_normal)[ids],
        np.asarray(core.primitive_measure)[ids])


# This small adapter describes a single-leaf point boundary. Production adapters
# would construct feature references for each tree node, as the builtins do.
def build_single_leaf_features(core: BoundaryCoreData, tree: object,
                               data: ExternalArrays) -> FeatureArrays:
    return FeatureArrays(data.positions, np.arange(len(data.positions), dtype=np.int32),
        np.zeros((1,), dtype=np.int32), np.array([len(data.positions)], dtype=np.int32))


external_adapter: BoundaryAdapter[
    ExternalArrays, ExternalData, ExternalItem, wp.vec3,
    BoundaryCoreData, BoundaryCore, FeatureArrays, FeatureData, ExternalItem,
] = BoundaryAdapter(prepare_core, pack_core, ExternalFeatures(), build_single_leaf_features)




@warp_func
def builtin_intersect(ray: Ray, minimum: float, maximum: float, data: PointsData3,
                      slot: int, half_open: bool, store_times: bool) -> wp.vec2:
    # A point on the +X unit ray intersects at its X coordinate.
    t = vector_x(data.a[slot])
    if minimum <= t and t <= maximum:
        return wp.vec2(t, 1.0)
    return wp.vec2(0.0)


HostT = TypeVar("HostT")


@dataclass(frozen=True)
class ExternalRobinMetric:
    """An external metric opts into Robin queries through the same capability."""

    def bind_robin(self, primitive: PrimitiveSpec[HostT, ExternalData, ExternalItem, wp.vec3]
                   ) -> RobinOps[wp.vec3, ExternalData, ExternalItem, BoundaryCore, SilhouetteMeta]:
        return RobinOps(wp.vec3, (3,), robin_valid, robin_node, robin_primitive, robin_feature,
                        core_type=BoundaryCore, feature_type=SilhouetteMeta,
                        to_score=radius_to_score, to_distance=score_to_radius)
