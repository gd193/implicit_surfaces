"""Exact diagnostics prove query callback and payload links are enforced."""
import jax
import warp as wp
from sengi.bvh.warp import (
    ClassificationOps, MeasureOps, Ray, RayOps, RegionOps, warp_func,
    make_overlap, make_ray_query,
)
from tests.typing.warp_api.real_extension import ExternalData, ExternalItem, ExternalPoints
from tests.typing.warp_api.query_extension import (
    Cursor, ExternalMetadata, classification_event, intersect, metadata_valid,
    region_overlaps, region_valid,
)
from sengi.bvh.warp import (
    BoundaryCore, SilhouetteMeta, RobinOps, make_boundary_sample, make_classify,
    make_robin_radius,
)
from tests.typing.warp_api.query_extension import robin_valid, robin_node, robin_feature
from sengi.bvh.warp import Points, PointsData3
from tests.typing.warp_api.query_extension import builtin_intersect
from sengi.bvh.warp import Primitive, DeviceTree, CoefficientMeta
from tests.typing.warp_api.real_extension import ExternalArrays, external_load, external_bounds

class OtherData: ...
class OtherItem: ...
class OtherMetadata: ...

@warp_func
def wrong_intersect(ray: Ray, minimum: float, maximum: float, data: OtherData,
                    slot: int, half_open: bool, store_times: bool) -> wp.vec2:
    return wp.vec2(0.0)

@warp_func
def wrong_state(ray: Ray, minimum: float, maximum: float, data: ExternalData,
                slot: int, state: int, half_open: bool,
                store_times: bool) -> tuple[int, float, int]:
    return state, 0.0, 0

@warp_func
def wrong_event(point: wp.vec3, data: OtherData, slot: int, original: int,
                meta: ExternalMetadata) -> int:
    return 0

@warp_func
def wrong_valid(meta: OtherMetadata) -> bool:
    return True

@warp_func
def wrong_sample(item: OtherItem, uniforms: wp.vec2) -> tuple[wp.vec3, wp.vec3, float]:
    return wp.vec3(0.0), wp.vec3(0.0), 1.0

@warp_func
def wrong_contains(query: float, cap: float, point: wp.vec2) -> bool:
    return True

bad_geometry: RayOps[ExternalData, int] = RayOps[ExternalData, int](intersect=wrong_intersect)  # expect: reportArgumentType
bad_state: RayOps[ExternalData, Cursor] = RayOps[ExternalData, Cursor](state_type=Cursor, next_hit=wrong_state)  # expect: reportArgumentType
bad_event: ClassificationOps[wp.vec3, ExternalData, ExternalMetadata] = ClassificationOps[wp.vec3, ExternalData, ExternalMetadata](ExternalMetadata, metadata_valid, wrong_event)  # expect: reportArgumentType
bad_metadata: ClassificationOps[wp.vec3, ExternalData, ExternalMetadata] = ClassificationOps[wp.vec3, ExternalData, ExternalMetadata](ExternalMetadata, wrong_valid, classification_event)  # expect: reportArgumentType
bad_item: MeasureOps[ExternalItem, wp.vec3, object] = MeasureOps[ExternalItem, wp.vec3, object](wrong_sample)  # expect: reportArgumentType
bad_point: RegionOps[float, wp.vec3] = RegionOps[float, wp.vec3](float, (), region_valid, region_overlaps, wrong_contains)  # expect: reportArgumentType
make_ray_query(ExternalPoints(), ops=RayOps[OtherData, int](intersect=wrong_intersect))  # expect: reportArgumentType # expect: reportCallIssue


def payloads(tree: object, array: jax.Array, wrong: OtherData) -> None:
    ray = make_ray_query(ExternalPoints(), ops=RayOps[ExternalData, int](intersect=intersect))
    ray(tree, wrong, array, array)  # expect: reportArgumentType
    overlap = make_overlap(ExternalPoints())
    overlap(tree, wrong, array, array)  # expect: reportArgumentType


@warp_func
def wrong_return(ray: Ray, minimum: float, maximum: float, data: ExternalData,
                 slot: int, half_open: bool, store_times: bool) -> float:
    return 0.0

@warp_func
def wrong_cursor_return(ray: Ray, minimum: float, maximum: float, data: ExternalData,
                        slot: int, state: Cursor, half_open: bool,
                        store_times: bool) -> tuple[int, float, int]:
    return 0, 0.0, 0

@warp_func
def wrong_sample_return(item: ExternalItem, uniforms: wp.vec2) -> tuple[wp.vec3, wp.vec2, float]:
    return wp.vec3(0.0), wp.vec2(0.0), 1.0

@warp_func
def wrong_robin(query: wp.vec3, item: OtherItem, coefficient: float) -> float:
    return 0.0

bad_ray_return: RayOps[ExternalData, int] = RayOps[ExternalData, int](intersect=wrong_return)  # expect: reportArgumentType
bad_cursor_return: RayOps[ExternalData, Cursor] = RayOps[ExternalData, Cursor](state_type=Cursor, next_hit=wrong_cursor_return)  # expect: reportArgumentType
bad_sample_return: MeasureOps[ExternalItem, wp.vec3, object] = MeasureOps[ExternalItem, wp.vec3, object](wrong_sample_return)  # expect: reportArgumentType
bad_robin_item: RobinOps[wp.vec3, ExternalData, ExternalItem, BoundaryCore, SilhouetteMeta] = RobinOps[wp.vec3, ExternalData, ExternalItem, BoundaryCore, SilhouetteMeta](wp.vec3, (3,), robin_valid, robin_node, wrong_robin, robin_feature)  # expect: reportArgumentType


def more_payloads(tree: object, array: jax.Array, wrong: OtherData) -> None:
    from tests.typing.warp_api.query_valid import classification, measure, region, robin
    sample_query = make_boundary_sample(ExternalPoints(), measure, region)
    classify_query = make_classify(ExternalPoints(), classification)
    robin_query = make_robin_radius(ExternalPoints(), robin)
    from sengi.bvh.warp import SamplingData, RobinData, CoefficientData, BoundaryCoreData, SilhouetteData
    sampling = SamplingData(array, array)
    sample_query(tree, wrong, sampling, array, 1.0, array)  # expect: reportArgumentType
    classify_query(tree, wrong, ExternalMetadata(), array)  # expect: reportArgumentType
    metadata = RobinData(BoundaryCoreData(array, array, array, array, array),
        SilhouetteData(array, array, array, array, array, array, array, array, array, array),
        CoefficientData(array, array))
    robin_query(tree, wrong, metadata, array)  # expect: reportArgumentType

make_ray_query(Points(3), ops=RayOps[ExternalData, int](intersect=intersect))  # expect: reportArgumentType # expect: reportCallIssue
make_ray_query(ExternalPoints(), ops=RayOps[PointsData3, int](intersect=builtin_intersect))  # expect: reportArgumentType # expect: reportCallIssue


@warp_func
def wrong_closest(query: wp.vec3, item: OtherItem) -> wp.vec3:
    return query

@warp_func
def wrong_robin_metadata(query: wp.vec3, tree: DeviceTree, data: ExternalData,
                         core: OtherMetadata, feature: SilhouetteMeta,
                         coefficients: CoefficientMeta, node: int) -> float:
    return 0.0

bad_robin_metadata: RobinOps[wp.vec3, ExternalData, ExternalItem, BoundaryCore, SilhouetteMeta] = RobinOps[wp.vec3, ExternalData, ExternalItem, BoundaryCore, SilhouetteMeta](wp.vec3, (3,), robin_valid, robin_node, None, robin_feature, node_bound_with_data=wrong_robin_metadata)  # expect: reportArgumentType


def descriptor_capabilities() -> None:
    host = ExternalPoints()
    Primitive[ExternalArrays, ExternalData, ExternalItem, wp.vec3](dimension=3,
        point_type=wp.vec3, data_type=ExternalData, load=external_load,
        bounds_of_item=external_bounds, host_bounds=host.bounds, host_pack=host.pack,
        closest_euclidean=wrong_closest)  # expect: reportArgumentType
    Primitive[ExternalArrays, ExternalData, ExternalItem, wp.vec3](dimension=3,
        point_type=wp.vec3, data_type=ExternalData, load=external_load,
        bounds_of_item=external_bounds, host_bounds=host.bounds, host_pack=host.pack,
        intersect=wrong_intersect)  # expect: reportArgumentType


def feature_payloads(tree: object, array: jax.Array, wrong: OtherData,
                     geometry: ExternalArrays) -> None:
    from tests.typing.warp_api.query_valid import silhouette_query, star_query, classify
    silhouette_query(tree, wrong, geometry, array)  # expect: reportArgumentType # expect: reportArgumentType
    star_query(tree, wrong, geometry, array, 1.0)  # expect: reportArgumentType # expect: reportArgumentType
    classify(tree, geometry, OtherMetadata(), array)  # expect: reportArgumentType


def wrong_robin_host_metadata(tree: object, geometry: ExternalArrays, array: jax.Array) -> None:
    from tests.typing.warp_api.query_valid import robin_query
    from sengi.bvh.warp import RobinData, CoefficientData
    robin_query(tree, geometry, RobinData(OtherMetadata(), OtherMetadata(), CoefficientData(array, array)), array)  # expect: reportArgumentType
