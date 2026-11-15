"""Reject host payloads, incorrect geometry and wrong device result buffers."""
import warp as wp
from surfaces.explicit.accelerators.bvh.warp import (
    DeviceArray, DeviceArray2D, DeviceTree, Euclidean, Points, PointsData3,
    Triangles, TrianglesData3, PointsArrays3, BoundaryCore, SilhouetteMeta,
    make_nearest, make_knn, make_ray_query, make_classify, make_star_radius, StarRadiusQuery, TrianglesArrays3, TriangleItem3,
    BoundaryCoreData, SilhouetteData, SilhouetteItem3,
)


def invalid(tree: DeviceTree, geometry: PointsData3, triangles: TrianglesData3,
            host: PointsArrays3, point: wp.vec3, core: BoundaryCore, feature: SilhouetteMeta,
            values: DeviceArray2D[float], ids: DeviceArray2D[int], scalar: DeviceArray[int]) -> None:
    make_nearest(Points(), Euclidean()).device(point, 10., tree, host)  # expect: reportArgumentType
    make_nearest(Points(), Euclidean()).device(point, 10., tree, triangles)  # expect: reportArgumentType
    make_knn(Points(), 4, Euclidean()).device(point, 10., tree, triangles, values, ids, 0)  # expect: reportArgumentType
    make_knn(Points(), 4, Euclidean()).device(point, 10., tree, geometry, values, scalar, 0)  # expect: reportArgumentType
    make_ray_query(Triangles(), 'all').device(point, point, 0., 10., -1, tree, triangles, values, scalar, scalar, scalar, 0)  # expect: reportArgumentType
    make_ray_query(Triangles(), 'any').device(point, point, 0., 10., -1, tree, geometry, scalar, scalar, 0)  # expect: reportArgumentType
    make_classify(Triangles()).device(point, tree, triangles, feature)  # expect: reportArgumentType
    star: StarRadiusQuery[TrianglesArrays3, TrianglesData3, TriangleItem3, wp.vec3, BoundaryCoreData, BoundaryCore, SilhouetteData, SilhouetteMeta, SilhouetteItem3] = make_star_radius(Triangles())
    star.device(point, 10., tree, core)  # expect: reportArgumentType
