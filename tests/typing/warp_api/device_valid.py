"""Device calls keep query, geometry, metadata and buffer types."""
from typing import assert_type
import warp as wp
from sengi.bvh.warp import (
    DeviceArray, DeviceArray2D, DeviceTree, Euclidean, Points, PointsData3,
    Triangles, TrianglesData3, BoundaryCore, SilhouetteMeta, SamplingMeta, CoefficientMeta,
    make_nearest, make_knn, make_ray_query, make_classify, make_boundary_sample,
    make_silhouette, make_star_radius, make_robin_radius,
)


def point_queries(tree: DeviceTree, geometry: PointsData3, point: wp.vec3,
                  values: DeviceArray2D[float], ids: DeviceArray2D[int],
                  points: DeviceArray2D[wp.vec3]) -> None:
    nearest = make_nearest(Points(), Euclidean())
    assert_type(nearest.device(point, 10., tree, geometry), tuple[wp.vec3, float, int, int])
    knn = make_knn(Points(), 4, Euclidean())
    assert_type(knn.device(point, 10., tree, geometry, values, ids, 0), tuple[int, int])
    knn_points = make_knn(Points(), 4, Euclidean(), return_points=True)
    assert_type(knn_points.device(point, 10., tree, geometry, values, points, 0), tuple[int, int])


def boundaries(tree: DeviceTree, geometry: TrianglesData3, point: wp.vec3,
               core: BoundaryCore, feature: SilhouetteMeta, sampling: SamplingMeta, coefficients: CoefficientMeta,
               times: DeviceArray[float], ids: DeviceArray[int], counts: DeviceArray[int], statuses: DeviceArray[int],
               all_times: DeviceArray2D[float], all_ids: DeviceArray2D[int]) -> None:
    first = make_ray_query(Triangles(), 'first')
    first.device(point, point, 0., 10., -1, tree, geometry, times, ids, counts, statuses, 0)
    make_ray_query(Triangles(), 'any').device(point, point, 0., 10., -1, tree, geometry, counts, statuses, 0)
    make_ray_query(Triangles(), 'count').device(point, point, 0., 10., -1, tree, geometry, counts, statuses, 0)
    make_ray_query(Triangles(), 'all').device(point, point, 0., 10., -1, tree, geometry, all_times, all_ids, counts, statuses, 0)
    assert_type(make_classify(Triangles()).device(point, tree, geometry, core), tuple[float, int])
    assert_type(make_silhouette(Triangles()).device(point, 10., tree, feature), tuple[wp.vec3, float, int, int])
    assert_type(make_star_radius(Triangles()).device(point, 10., tree, feature), tuple[float, int])
    assert_type(make_robin_radius(Triangles()).device(point, 10., tree, geometry, core, feature, coefficients), tuple[float, int])
    assert_type(make_boundary_sample(Triangles()).device(point, 10., point, tree, geometry, sampling), tuple[wp.vec3, wp.vec3, float, int, int])
