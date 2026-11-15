"""Caller-specialized Warp BVH queries; primitive and metric arguments are static."""
from .backends._warp_metrics import (
    Angular, AngularQuery, Euclidean, MetricSpec, NearestCallbacks, NearestOps,
    NearestPolicy, NearestPredicate, Silhouette,
)
from .backends._warp_nearest import NearestQuery, make_nearest
from .backends._warp_types import DeviceArray, PrimitiveSpec, WarpFunction, WarpType, as_warp_function, warp_func, warp_struct
from .backends._warp_primitives import (
    Points, Primitive, Segments, SilhouetteEdges, Triangles,
)
from .backends._warp_builtin_items import (
    PointItem2, PointItem3, SegmentItem2, SegmentItem3, SilhouetteItem3, TriangleItem3,
)
from .backends._warp_builtin_data import (
    PointsArrays2, PointsArrays3, PointsData2, PointsData3,
    SegmentsArrays2, SegmentsArrays3, SegmentsData2, SegmentsData3,
    TrianglesArrays3, TrianglesData3, SilhouetteEdgesArrays3, SilhouetteEdgesData3,
)
from .backends._warp_rays import RayQuery, make_ray_query
from .backends._warp_tree import DeviceTree, TreeData, build_tree
from .backends._warp_boundary import (
    BoundaryAdapter, BoundaryCore, BoundaryCoreData, BoundaryData, BoundaryPrimitive, SilhouetteData,
    SilhouetteMeta, build_boundary,
)
from .backends._warp_boundary_queries import (
    BoundarySampleQuery, CoefficientData, CoefficientMeta, MeasureOps, RegionOps,
    RobinData, RobinMetric, RobinOps, RobinPolicy, RobinQuery, SamplingData, SamplingMeta,
    make_boundary_sample, make_robin_radius,
    prepare_boundary_sample, prepare_robin,
)
from .backends._warp_domain import ClassificationOps, Ray, RayOps
from .backends._warp_inside import ClassifyQuery, make_classify
from .backends._warp_overlap import OverlapQuery, make_overlap
from .backends._warp_silhouette import SilhouetteQuery, StarRadiusQuery, make_silhouette, make_star_radius

__all__ = [
    "Angular", "AngularQuery", "BoundaryAdapter", "BoundaryCore", "BoundaryCoreData",
    "BoundaryData", "BoundaryPrimitive", "BoundarySampleQuery", "ClassificationOps", "ClassifyQuery",
    "CoefficientData", "CoefficientMeta", "DeviceArray", "DeviceTree", "Euclidean", "MeasureOps",
    "MetricSpec", "NearestCallbacks", "NearestOps", "NearestPolicy", "NearestPredicate",
    "NearestQuery", "OverlapQuery", "PointItem2", "PointItem3", "Points", "PointsArrays2",
    "PointsArrays3", "PointsData2", "PointsData3", "Primitive", "PrimitiveSpec", "Ray", "RayOps",
    "RayQuery", "RegionOps", "RobinData", "RobinMetric", "RobinOps", "RobinPolicy", "RobinQuery", "SamplingData",
    "SamplingMeta", "SegmentItem2", "SegmentItem3", "Segments", "SegmentsArrays2",
    "SegmentsArrays3", "SegmentsData2", "SegmentsData3", "Silhouette", "SilhouetteData",
    "SilhouetteEdges", "SilhouetteEdgesArrays3", "SilhouetteEdgesData3", "SilhouetteItem3",
    "SilhouetteMeta", "SilhouetteQuery", "StarRadiusQuery", "TreeData", "TriangleItem3",
    "Triangles", "TrianglesArrays3", "TrianglesData3", "WarpFunction", "WarpType",
    "as_warp_function", "build_boundary", "build_tree", "make_boundary_sample", "make_classify",
    "make_nearest", "make_overlap", "make_ray_query", "make_robin_radius", "make_silhouette",
    "make_star_radius", "prepare_boundary_sample", "prepare_robin", "warp_func", "warp_struct",
]
