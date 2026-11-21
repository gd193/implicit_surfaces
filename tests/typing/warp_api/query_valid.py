"""All query families retain concrete third-party callback and payload types."""
import warp as wp
from sengi.bvh.warp import (
    ClassificationOps, MeasureOps, RayOps, RegionOps,
    make_boundary_sample, make_classify, make_overlap, make_ray_query,
)
from tests.typing.warp_api.real_extension import ExternalArrays, ExternalData, ExternalItem, ExternalPoints
from tests.typing.warp_api.query_extension import (
    Cursor, ExternalMetadata, ExternalRobinMetric, MetadataArrays, classification_event, intersect, metadata_valid,
    next_hit, region_contains, region_overlaps, region_valid, sample,
)
from sengi.bvh.warp import (
    BoundaryCore, SilhouetteMeta, RobinOps, make_robin_radius,
)
from tests.typing.warp_api.query_extension import (
    radius_to_score, robin_feature, robin_node, robin_primitive,
    robin_valid, score_to_radius,
)
import jax
from sengi.bvh.warp import SamplingData
from sengi.bvh.warp import make_silhouette, make_star_radius, NearestCallbacks
from tests.typing.warp_api.query_extension import external_adapter, FeatureArrays
from tests.typing.warp_api.real_extension import external_bound, external_evaluate
from sengi.bvh.warp import Points, PointsData3
from tests.typing.warp_api.query_extension import builtin_intersect
from sengi.bvh.warp import BoundaryCoreData, SilhouetteData, RobinData, build_boundary

single: RayOps[ExternalData, int] = RayOps(intersect=intersect)
multiple: RayOps[ExternalData, Cursor] = RayOps(state_type=Cursor, next_hit=next_hit)
classification: ClassificationOps[wp.vec3, ExternalData, ExternalMetadata] = ClassificationOps(
    ExternalMetadata, metadata_valid, classification_event)
measure: MeasureOps[ExternalItem, wp.vec3, ExternalArrays] = MeasureOps(sample)
region: RegionOps[float, wp.vec3] = RegionOps(float, (), region_valid, region_overlaps, region_contains)
primitive = ExternalPoints()
ray = make_ray_query(primitive, ops=single)
stateful_ray = make_ray_query(primitive, "all", ops=multiple)
classify = make_classify(primitive, classification, metadata_host_type=MetadataArrays)
sampler = make_boundary_sample(primitive, measure, region)
overlap = make_overlap(primitive)

robin: RobinOps[wp.vec3, ExternalData, ExternalItem, BoundaryCore, SilhouetteMeta] = RobinOps(
    wp.vec3, (3,), robin_valid, robin_node, robin_primitive, robin_feature,
    core_type=BoundaryCore, feature_type=SilhouetteMeta,
    to_score=radius_to_score, to_distance=score_to_radius)
robin_query = make_robin_radius(primitive, robin)
robin_from_metric = make_robin_radius(primitive, metric=ExternalRobinMetric())


def invoke(tree: object, geometry: ExternalArrays, arrays: jax.Array,
           sampling: SamplingData, metadata: MetadataArrays) -> None:
    classify(tree, geometry, metadata, arrays)
    ray(tree, geometry, arrays, arrays)
    stateful_ray(tree, geometry, arrays, arrays, max_hits=4)
    sampler(tree, geometry, sampling, arrays, 1.0, arrays)
    overlap(tree, geometry, arrays, arrays)

silhouette_query = make_silhouette(primitive, NearestCallbacks(external_evaluate, external_bound), adapter=external_adapter)
star_query = make_star_radius(primitive, adapter=external_adapter)

def invoke_features(tree: object, geometry: ExternalArrays, features: FeatureArrays, arrays: jax.Array) -> None:
    silhouette_query(tree, geometry, features, arrays)
    star_query(tree, geometry, features, arrays, 10.0)

builtin_ray_ops: RayOps[PointsData3, int] = RayOps(intersect=builtin_intersect)
builtin_ray = make_ray_query(Points(3), ops=builtin_ray_ops)


def invoke_robin(tree: object, geometry: ExternalArrays, metadata: RobinData[BoundaryCoreData, SilhouetteData],
                 arrays: jax.Array) -> None:
    robin_query(tree, geometry, metadata, arrays)
    robin_from_metric(tree, geometry, metadata, arrays)


def construct_boundary(geometry: ExternalArrays) -> tuple[BoundaryCoreData, FeatureArrays | None]:
    _, _, boundary = build_boundary(primitive, geometry, adapter=external_adapter)
    return boundary.core, boundary.silhouette
