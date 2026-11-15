"""Concrete kNN mismatches must be rejected by the public signatures."""
import jax
import warp as wp
from surfaces.explicit.accelerators.bvh.warp import (
    Euclidean, NearestCallbacks, NearestPolicy, Points, PrimitiveSpec, make_knn, make_nearest, warp_func,
)
from tests.typing.warp_api.valid import Device, Host, Item, OtherItem, evaluate, lower_bound
from tests.typing.warp_api.real_extension import external_evaluate, external_bound

@warp_func
def wrong_item(query: wp.vec3, item: OtherItem) -> tuple[wp.vec3, float]:
    return query, 0.0

@warp_func
def wrong_predicate(query: wp.vec3, item: OtherItem) -> bool:
    return True

@warp_func
def wrong_state(query: float, item: Item) -> bool:
    return True


def mismatches(primitive: PrimitiveSpec[Host, Device, Item, wp.vec3],
               policy: NearestPolicy[Device, OtherItem, wp.vec3, wp.vec3],
               tree: object, queries: jax.Array) -> None:
    make_knn(primitive, 3, policy)  # expect: reportArgumentType
    make_knn(primitive, 3, evaluate=wrong_item, lower_bound=lower_bound)  # expect: reportArgumentType
    make_knn(primitive, 3, NearestCallbacks(evaluate, lower_bound), predicate=wrong_predicate)  # expect: reportArgumentType
    make_knn(primitive, 3, NearestCallbacks(evaluate, lower_bound), predicate=wrong_state)  # expect: reportArgumentType
    query = make_knn(primitive, 3, evaluate=evaluate, lower_bound=lower_bound)
    query(tree, object(), queries)  # expect: reportArgumentType

make_knn(Points(3), 3, evaluate=external_evaluate, lower_bound=external_bound)  # expect: reportArgumentType
make_knn(Points(3), 3, NearestCallbacks(external_evaluate, external_bound))  # expect: reportArgumentType
make_knn(Points(3), 3, Euclidean(), packed_nodes="packed")  # expect: reportArgumentType
make_nearest(Points(3), Euclidean(), group_leaves="flat")  # expect: reportArgumentType
