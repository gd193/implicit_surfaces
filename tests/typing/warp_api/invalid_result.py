"""Custom result queries are device-only and have typed output buffers."""
import warp as wp
from sengi.bvh.warp import (
    DeviceArray2D, DeviceTree, Euclidean, Points, make_nearest,
)
from tests.typing.warp_api.real_extension import ExternalData
from tests.typing.warp_api.result_valid import nearest, knn, policy


def invalid(tree: DeviceTree, data: ExternalData, point: wp.vec3,
            scores: DeviceArray2D[float], ids: DeviceArray2D[int]) -> None:
    nearest(tree, data, point)  # expect: reportCallIssue
    knn(tree, data, point)  # expect: reportCallIssue
    knn.device(point, 1.0, tree, data, scores, ids, 0)  # expect: reportArgumentType
    nearest.device(wp.vec2(), 1.0, tree, data)  # expect: reportArgumentType
    make_nearest(Points(), Euclidean(), result_policy=policy)  # expect: reportCallIssue  # expect: reportArgumentType
