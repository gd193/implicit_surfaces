"""Concrete structural and descriptor extensions agree on the four type axes."""
from typing import assert_type
import warp as wp
from sengi.bvh.warp import (
    Euclidean, NearestOps, Primitive, PrimitiveSpec, make_nearest, warp_func,
)
from tests.typing.warp_api.valid import Device, Host, Item, callbacks

from tests.typing.warp_api.real_extension import (
    ExternalPoints, external_load as load, external_bounds as bounds_of_item,
)

@warp_func
def closest(query: wp.vec3, item: Item) -> wp.vec3:
    return item.position

external = ExternalPoints()
structural: PrimitiveSpec[Host, Device, Item, wp.vec3] = external
bounds = external.bounds
pack = external.pack
descriptor: Primitive[Host, Device, Item, wp.vec3] = Primitive(
    dimension=3, point_type=wp.vec3, data_type=Device, load=load,
    bounds_of_item=bounds_of_item, host_bounds=bounds, host_pack=pack,
    closest_euclidean=closest,
)
assert_type(callbacks.bind(structural), NearestOps[wp.vec3, Item, wp.vec3])
assert_type(Euclidean().bind(descriptor), NearestOps[wp.vec3, Item, wp.vec3])
make_nearest(structural, callbacks)
make_nearest(descriptor, Euclidean())
