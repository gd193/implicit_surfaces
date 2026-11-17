"""Custom selected structs preserve callback, primitive and output types."""
from typing import assert_type
import warp as wp
from surfaces.explicit.accelerators.bvh.warp import (
    DeviceArray2D, DeviceTree, Euclidean, IdPoint, IdPointResult, Points,
    PointsData3, ResultPolicy, make_knn, make_nearest, warp_func, warp_struct,
)
from tests.typing.warp_api.real_extension import (
    ExternalPoints, ExternalItem, ExternalData, external_evaluate, external_bound,
)


@warp_struct
class Selected:
    original_id: int
    point: wp.vec3
    score: float


@warp_func
def build(query: wp.vec3, item: ExternalItem, index: int, point: wp.vec3, value: float) -> Selected:
    selected = Selected()
    selected.original_id = index
    selected.point = point
    selected.score = value
    return selected


policy = ResultPolicy(Selected, build)
nearest = make_nearest(ExternalPoints(), evaluate=external_evaluate,
                       lower_bound=external_bound, result_policy=policy)
knn = make_knn(ExternalPoints(), 4, evaluate=external_evaluate,
              lower_bound=external_bound, result_policy=policy)


def signatures(tree: DeviceTree, data: ExternalData, points: PointsData3, query: wp.vec3,
               scores: DeviceArray2D[float], selected: DeviceArray2D[Selected],
               id_points: DeviceArray2D[IdPointResult[wp.vec3]]) -> None:
    assert_type(nearest.device(query, 2.0, tree, data), tuple[Selected, int])
    assert_type(knn.device(query, 2.0, tree, data, scores, selected, 0), tuple[int, int])
    plain = make_nearest(Points(), Euclidean(), result_policy=IdPoint())
    assert_type(plain.device(query, 2.0, tree, points), tuple[IdPointResult[wp.vec3], int])
    both = make_knn(Points(), 4, Euclidean(), result_policy=IdPoint())
    assert_type(both.device(query, 2.0, tree, points, scores, id_points, 0), tuple[int, int])
