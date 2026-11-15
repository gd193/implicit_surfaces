"""Resolved query defaults share device specializations without merging callbacks."""
from dataclasses import replace
from typing import Any

import pytest

wp = pytest.importorskip("warp")

from surfaces.explicit.accelerators.bvh.backends._warp_boundary import resolve_boundary_adapter  # noqa: E402
from surfaces.explicit.accelerators.bvh.backends._warp_boundary_queries import (  # noqa: E402
    _ball_region, make_boundary_sample, make_robin_radius,
)
from surfaces.explicit.accelerators.bvh.warp import (  # noqa: E402
    Euclidean, RayOps, Segments, make_ray_query, make_silhouette, make_star_radius,
)
from surfaces.explicit.accelerators.bvh.backends._warp_domain import Ray  # noqa: E402


@pytest.mark.parametrize("mode", ["first", "any", "count", "all"])
def test_ray_resolved_defaults_and_options(mode):
    primitive = Segments(2)
    query = make_ray_query(primitive, mode)
    assert query is make_ray_query(primitive=primitive, mode=mode, ops=RayOps(intersect=primitive.intersect),
                                   exclude=False, stack_capacity=64, block_dim=128)
    assert query is not make_ray_query(primitive, mode, block_dim=64)
    assert make_ray_query(primitive, mode, block_dim=64).block_dim == 64
    assert query is not make_ray_query(primitive, mode, exclude=True)


def test_ray_callback_identity():
    primitive = Segments(2)

    def callback(value):
        @wp.func(module="unique")
        def intersect(ray: Ray, minimum: float, maximum: float, data: Any,
                      slot: int, half_open: bool, store_times: bool):
            return wp.vec2(wp.static(value), 1.0)
        return intersect

    first, second = callback(1.0), callback(2.0)
    query = make_ray_query(primitive, ops=RayOps(intersect=first))
    assert query is make_ray_query(primitive, ops=RayOps(intersect=first))
    assert query is not make_ray_query(primitive, ops=RayOps(intersect=callback(1.0)))
    assert query.kernel is not make_ray_query(primitive, ops=RayOps(intersect=second)).kernel


def test_boundary_resolved_defaults_and_host_markers():
    primitive = Segments(2)
    adapter = resolve_boundary_adapter(primitive)
    metric = Euclidean()
    robin = make_robin_radius(primitive)
    assert robin is make_robin_radius(primitive, metric=metric)
    assert robin is make_robin_radius(primitive, adapter.robin(primitive), metric,
                                      core_host_type=object, feature_host_type=tuple)
    assert robin is not make_robin_radius(primitive, block_dim=128)
    class BoundPolicy:
        def bind(self, primitive, metric):
            return robin.ops

    assert robin is make_robin_radius(primitive, BoundPolicy())
    assert robin.kernel is not make_robin_radius(primitive, replace(robin.ops, to_score=_score,
                                                                to_distance=_score)).kernel
    sample = make_boundary_sample(primitive)
    assert sample is make_boundary_sample(primitive, adapter.measure(primitive), _ball_region(primitive), block_dim=256)


@wp.func
def _score(value: float):
    return value


def test_silhouette_resolved_defaults():
    primitive = Segments(2)
    adapter = resolve_boundary_adapter(primitive)
    query = make_silhouette(primitive)
    assert query is make_silhouette(primitive, Euclidean(), adapter=adapter,
                                       stack_capacity=64, block_dim=64)
    assert query is not make_silhouette(primitive, block_dim=128)
    assert make_star_radius(primitive) is make_star_radius(primitive, adapter=adapter)
