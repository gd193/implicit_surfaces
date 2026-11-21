"""Caller-specialized Warp BVH queries; primitive and metric arguments are static."""
from .backends.warp_queries.device import configure_query_kernel
from .backends.warp_queries.metrics import (
    Angular, AngularQuery, Euclidean, MetricSpec, NearestCallbacks, NearestOps, NearestPolicy,
    NearestPredicate, Silhouette,
)
from .backends.warp_queries.queries.nearest import NearestQuery, SelectedNearestQuery, make_nearest
from .backends.warp_queries.queries.knn import KnnQuery, SelectedKnnQuery, make_knn
from .backends.warp_queries.queries.results import ResultPolicy, IdPoint, IdPointResult
from .backends.warp_queries.traversal.policy import (
    TraversalPolicy, make_traversal, make_traversal_state,
)
from .backends.warp_queries.queries.range import (
    RadiusAccumulator, RadiusQuery, RadiusReduction, make_radius,
)
from .backends.warp_queries.contracts import (
    DeviceArray, DeviceArray2D, FusedNearestOps, PrimitiveSpec, WarpFunction, WarpRef,
    WarpType, as_warp_function, warp_func, warp_struct,
)
from .backends.warp_queries.primitives.builtin import (
    Points, Primitive, Segments, SilhouetteEdges, Triangles,
)
from .backends.warp_queries.primitives.items import (
    PointItem2, PointItem3, SegmentItem2, SegmentItem3, SilhouetteItem3, TriangleItem3,
)
from .backends.warp_queries.primitives.data import (
    PointsArrays2, PointsArrays3, PointsData2, PointsData3, SegmentsArrays2, SegmentsArrays3,
    SegmentsData2, SegmentsData3, TrianglesArrays3, TrianglesData3, SilhouetteEdgesArrays3,
    SilhouetteEdgesData3,
)
from .backends.warp_queries.queries.rays import RayQuery, make_ray_query
from .backends._warp_tree import DeviceTree, TreeData, build_tree
from .backends._warp_boundary import (
    BoundaryAdapter, BoundaryCore, BoundaryCoreData, BoundaryData, BoundaryPrimitive, SilhouetteData,
    SilhouetteMeta, build_boundary,
)
from .backends.warp_queries.queries.boundary import (
    BoundarySampleQuery, CoefficientData, CoefficientMeta, MeasureOps, RegionOps, RobinData,
    RobinMetric, RobinOps, RobinPolicy, RobinQuery, SamplingData, SamplingMeta,
    make_boundary_sample, make_robin_radius, prepare_boundary_sample, prepare_robin,
)
from .backends.warp_queries.domain import ClassificationOps, Ray, RayOps
from .backends.warp_queries.queries.classification import ClassifyQuery, make_classify
from .backends.warp_queries.queries.overlap import OverlapQuery, make_overlap
from .backends.warp_queries.queries.region import (
    RegionIntersectionPolicy, RegionQuery, RegionClassifyQuery, make_region, make_region_classify,
    point_region_policy, segment_region_policy, triangle_region_policy,
)
from .backends.warp_queries.queries.silhouette import (
    SilhouetteQuery, StarRadiusQuery, make_silhouette, make_star_radius,
)

__all__ = [
    "ResultPolicy", "IdPoint", "IdPointResult", "SelectedNearestQuery", "SelectedKnnQuery",
    "RegionIntersectionPolicy", "RegionQuery", "RegionClassifyQuery", "make_region", "make_region_classify",
    "point_region_policy", "segment_region_policy", "triangle_region_policy",
    "TraversalPolicy", "make_traversal", "make_traversal_state",
    "RadiusAccumulator", "RadiusQuery", "RadiusReduction", "make_radius",
    "Angular", "AngularQuery", "BoundaryAdapter", "BoundaryCore", "BoundaryCoreData",
    "BoundaryData", "BoundaryPrimitive", "BoundarySampleQuery", "ClassificationOps", "ClassifyQuery",
    "CoefficientData", "CoefficientMeta", "DeviceArray", "DeviceArray2D", "DeviceTree", "Euclidean",
    "FusedNearestOps", "MeasureOps",
    "KnnQuery", "MetricSpec", "NearestCallbacks", "NearestOps", "NearestPolicy", "NearestPredicate",
    "NearestQuery", "OverlapQuery", "PointItem2", "PointItem3", "Points", "PointsArrays2",
    "PointsArrays3", "PointsData2", "PointsData3", "Primitive", "PrimitiveSpec", "Ray", "RayOps",
    "RayQuery", "RegionOps", "RobinData", "RobinMetric", "RobinOps", "RobinPolicy", "RobinQuery", "SamplingData",
    "SamplingMeta", "SegmentItem2", "SegmentItem3", "Segments", "SegmentsArrays2",
    "SegmentsArrays3", "SegmentsData2", "SegmentsData3", "Silhouette", "SilhouetteData",
    "SilhouetteEdges", "SilhouetteEdgesArrays3", "SilhouetteEdgesData3", "SilhouetteItem3",
    "SilhouetteMeta", "SilhouetteQuery", "StarRadiusQuery", "TreeData", "TriangleItem3",
    "Triangles", "TrianglesArrays3", "TrianglesData3", "WarpFunction", "WarpRef", "WarpType",
    "as_warp_function", "build_boundary", "build_tree", "configure_query_kernel", "make_boundary_sample", "make_classify",
    "make_knn", "make_nearest", "make_overlap", "make_ray_query", "make_robin_radius", "make_silhouette",
    "make_star_radius", "prepare_boundary_sample", "prepare_robin", "warp_func", "warp_struct",
]
