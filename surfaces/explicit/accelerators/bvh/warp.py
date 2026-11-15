"""Caller-specialized Warp BVH queries; primitive and metric arguments are static."""
from .backends._warp_metrics import (
    Angular, AngularQuery, Euclidean, MetricSpec, NearestCallbacks, NearestOps,
    NearestPolicy, NearestPredicate, Silhouette,
)
from .backends._warp_nearest import NearestQuery, make_nearest
from .backends._warp_types import PrimitiveSpec, WarpFunction, WarpType, as_warp_function, warp_func, warp_struct
from .backends._warp_primitives import (
    Points, Primitive, Segments, SilhouetteEdges, Triangles,
)
from .backends._warp_builtin_items import (
    PointItem2, PointItem3, SegmentItem2, SegmentItem3, SilhouetteItem3, TriangleItem3,
)
from .backends._warp_rays import make_ray_query
from .backends._warp_tree import TreeData, build_tree
from .backends._warp_boundary import (
    BoundaryAdapter, BoundaryCore, BoundaryCoreData, BoundaryData, SilhouetteData,
    SilhouetteMeta, build_boundary,
)
from .backends._warp_boundary_queries import (
    MeasureOps, RegionOps, RobinOps, make_boundary_sample, make_robin_radius,
    prepare_boundary_sample, prepare_robin,
)
from .backends._warp_domain import ClassificationOps, Ray, RayOps
from .backends._warp_inside import make_classify
from .backends._warp_overlap import make_overlap
from .backends._warp_silhouette import make_silhouette, make_star_radius

__all__ = ["Angular", "AngularQuery", "BoundaryAdapter", "BoundaryCore", "BoundaryCoreData", "BoundaryData",
           "ClassificationOps", "Euclidean", "MeasureOps", "MetricSpec", "NearestCallbacks", "NearestOps", "NearestPolicy", "NearestPredicate", "NearestQuery", "PointItem2", "PointItem3", "Points",
           "WarpFunction", "WarpType", "as_warp_function", "warp_func", "warp_struct",
           "Primitive", "PrimitiveSpec", "Ray", "RayOps", "RegionOps", "RobinOps", "SegmentItem2", "SegmentItem3", "Segments", "Silhouette",
           "SilhouetteData", "SilhouetteEdges", "SilhouetteItem3", "SilhouetteMeta", "TreeData", "TriangleItem3", "Triangles",
           "build_boundary", "build_tree", "make_boundary_sample", "make_classify", "make_nearest",
           "make_overlap", "make_ray_query", "make_robin_radius", "make_silhouette", "make_star_radius",
           "prepare_boundary_sample", "prepare_robin"]
