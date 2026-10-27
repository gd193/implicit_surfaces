"""Caller-specialized Warp BVH queries; primitive and metric arguments are static."""
from .backends._warp_metrics import Angular, Euclidean, MetricSpec, NearestOps, Silhouette
from .backends._warp_nearest import make_nearest
from .backends._warp_primitives import Points, PrimitiveSpec, Segments, SilhouetteEdges, Triangles
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

__all__ = ["Angular", "BoundaryAdapter", "BoundaryCore", "BoundaryCoreData", "BoundaryData",
           "ClassificationOps", "Euclidean", "MeasureOps", "MetricSpec", "NearestOps", "Points",
           "PrimitiveSpec", "Ray", "RayOps", "RegionOps", "RobinOps", "Segments", "Silhouette",
           "SilhouetteData", "SilhouetteEdges", "SilhouetteMeta", "TreeData", "Triangles",
           "build_boundary", "build_tree", "make_boundary_sample", "make_classify", "make_nearest",
           "make_overlap", "make_ray_query", "make_robin_radius", "make_silhouette", "make_star_radius",
           "prepare_boundary_sample", "prepare_robin"]
