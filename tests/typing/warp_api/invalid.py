"""Each marked line must produce a real diagnostic; no ignore comments."""
import warp as wp
from sengi.bvh.warp import (
    NearestCallbacks, NearestPolicy, Points, PrimitiveSpec, make_nearest, warp_func,
)
from tests.typing.warp_api.valid import Device, Host, Item, OtherItem, evaluate, lower_bound
from tests.typing.warp_api.real_extension import external_evaluate, external_bound

@warp_func
def wrong_item(query: wp.vec3, item: OtherItem) -> tuple[wp.vec3, float]:
    return query, 0.0

@warp_func
def wrong_query(query: float, item: Item) -> tuple[wp.vec3, float]:
    return wp.vec3(query), 0.0

@warp_func
def wrong_return(query: wp.vec3, item: Item) -> tuple[wp.vec3, str]:
    return query, "not a score"

@warp_func
def wrong_bound(query: wp.vec3, lower: wp.vec3, upper: wp.vec3) -> str:
    return "not a bound"

bad_item: NearestCallbacks[Item, wp.vec3] = NearestCallbacks[Item, wp.vec3](wrong_item, lower_bound)  # expect: reportArgumentType
bad_query: NearestCallbacks[Item, wp.vec3] = NearestCallbacks[Item, wp.vec3](wrong_query, lower_bound)  # expect: reportArgumentType
bad_return: NearestCallbacks[Item, wp.vec3] = NearestCallbacks[Item, wp.vec3](wrong_return, lower_bound)  # expect: reportArgumentType
bad_bound: NearestCallbacks[Item, wp.vec3] = NearestCallbacks[Item, wp.vec3](evaluate, wrong_bound)  # expect: reportArgumentType

def mismatches(primitive: PrimitiveSpec[Host, Device, Item, wp.vec3],
               other: PrimitiveSpec[Host, Device, OtherItem, wp.vec3],
               policy: NearestPolicy[Device, OtherItem, wp.vec3, wp.vec3]) -> None:
    callbacks: NearestCallbacks[Item, wp.vec3] = NearestCallbacks(evaluate, lower_bound)
    callbacks.bind(other)  # expect: reportArgumentType
    make_nearest(primitive, policy)  # expect: reportArgumentType
    make_nearest(primitive, evaluate=wrong_item, lower_bound=lower_bound)  # expect: reportArgumentType
    make_nearest(primitive, evaluate=wrong_query, lower_bound=lower_bound)  # expect: reportArgumentType
    make_nearest(primitive, evaluate=wrong_return, lower_bound=lower_bound)  # expect: reportArgumentType

make_nearest(Points(3), evaluate=external_evaluate, lower_bound=external_bound)  # expect: reportArgumentType
make_nearest(Points(3), NearestCallbacks(external_evaluate, external_bound))  # expect: reportArgumentType
