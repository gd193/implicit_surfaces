"""kNN preserves primitive, callback, policy and host payload types."""
from typing import assert_type
import jax
import warp as wp
from surfaces.explicit.accelerators.bvh.warp import (
    Euclidean, NearestCallbacks, NearestPolicy, Points, PrimitiveSpec,
    make_knn, warp_func,
)
from tests.typing.warp_api.real_extension import (
    ExternalArrays, ExternalData, ExternalItem, ExternalPoints,
    external_bound, external_evaluate,
)

@warp_func
def accepts(query: wp.vec3, item: ExternalItem) -> bool:
    return True


def extension(primitive: PrimitiveSpec[ExternalArrays, ExternalData, ExternalItem, wp.vec3],
              tree: object, data: ExternalArrays, queries: jax.Array) -> None:
    callbacks: NearestCallbacks[ExternalItem, wp.vec3] = NearestCallbacks(external_evaluate, external_bound)
    policy: NearestPolicy[ExternalData, ExternalItem, wp.vec3, wp.vec3] = callbacks
    query = make_knn(primitive, 3, policy, predicate=accepts)
    assert_type(query(tree, data, queries), tuple[jax.Array, jax.Array, jax.Array, jax.Array])
    make_knn(primitive, 3, evaluate=external_evaluate, lower_bound=external_bound, predicate=accepts)

make_knn(ExternalPoints(), 3, NearestCallbacks(external_evaluate, external_bound))
make_knn(Points(3), 3, Euclidean())
