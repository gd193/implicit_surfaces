"""Static nearest-query policies; bounds and minimizers share score units."""
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol, TypeVar

import warp as wp


@dataclass(frozen=True)
class NearestOps:
    query_type: Any
    query_shape: tuple[int, ...]
    valid: Any
    node_bound: Any
    closest: Any
    to_score: Any
    to_distance: Any
    cull_limit: Any
    node_bound_with_data: Any = None
    viewpoint: Any = None
    cull_items: bool = True
    node_filter: Any = None
    group_leaves: bool = False


PrimitiveT = TypeVar("PrimitiveT", contravariant=True)


class MetricSpec(Protocol[PrimitiveT]):
    def bind(self, primitive: PrimitiveT) -> NearestOps: ...


@dataclass(frozen=True)
class Euclidean:
    @lru_cache(maxsize=None)
    def bind(self, primitive):
        closest_primitive = getattr(primitive, "closest_euclidean", None)
        if closest_primitive is None:
            raise TypeError("primitive has no Euclidean minimizer")
        Vec, dimension = primitive.point_type, primitive.dimension

        @wp.func
        def valid(query: Vec):
            return wp.isfinite(query)

        @wp.func
        def bound(query: Vec, lower: wp.vec3, upper: wp.vec3):
            x = wp.min(wp.max(query[0], lower[0]), upper[0])-query[0]
            y = wp.min(wp.max(query[1], lower[1]), upper[1])-query[1]
            score = x*x+y*y
            if wp.static(dimension == 3):
                z = wp.min(wp.max(query[2], lower[2]), upper[2])-query[2]
                score = score+z*z
            return score

        @wp.func
        def closest(query: Vec, item: Any):
            point = wp.static(closest_primitive)(query, item)
            delta = query-point
            score = wp.dot(delta, delta)
            return point, wp.where(score < wp.inf, score, wp.nan)

        @wp.func
        def to_score(distance: float):
            return distance*distance

        @wp.func
        def to_distance(score: float):
            return wp.sqrt(score)

        @wp.func
        def cull_limit(query: Vec, score: float):
            return score*1.000002 + 1.e-12*wp.dot(query, query)

        @wp.func
        def viewpoint(query: Vec):
            if wp.static(dimension == 2):
                return wp.vec3(query[0], query[1], 0.0)
            else:
                return query

        return NearestOps(Vec, (dimension,), valid, bound, closest, to_score, to_distance,
                          cull_limit, viewpoint=viewpoint, group_leaves=getattr(primitive, "point_bounds", False))


@dataclass(frozen=True)
class Angular:
    @lru_cache(maxsize=None)
    def bind(self, primitive):
        endpoints = getattr(primitive, "segment_endpoints", None)
        if endpoints is None:
            raise TypeError("Angular requires a point or segment endpoint operation")
        dimension, Vec = primitive.dimension, primitive.point_type
        Query = wp.types.matrix((2, dimension), wp.float32)
        @wp.func
        def valid(query: Query):
            return wp.isfinite(query) and wp.max(wp.abs(query[1])) > 0.0

        @wp.func_native("return bvh::angular_bound(query, lower, upper);")
        def bound(query: Query, lower: wp.vec3, upper: wp.vec3) -> float: ...

        @wp.func_native("return bvh::angular_segment(query, a, b);")
        def segment(query: Query, a: Vec, b: Vec) -> wp.vec4: ...

        @wp.func
        def closest(query: Query, item: Any):
            a, b = wp.static(endpoints)(item)
            result = segment(query, a, b)
            if wp.static(dimension == 2):
                return wp.vec2(result[0], result[1]), result[3]
            else:
                return wp.vec3(result[0], result[1], result[2]), result[3]

        @wp.func
        def identity(value: float):
            return value

        @wp.func
        def cull_limit(query: Query, score: float):
            return score + 4.e-7

        @wp.func
        def viewpoint(query: Query):
            if wp.static(dimension == 2):
                return wp.vec3(query[0, 0], query[0, 1], 0.0)
            else:
                return wp.vec3(query[0, 0], query[0, 1], query[0, 2])

        return NearestOps(Query, (2, dimension), valid, bound, closest,
                          identity, identity, cull_limit, viewpoint=viewpoint)


@dataclass(frozen=True)
class Silhouette:
    def bind(self, primitive, ops):
        normal_sides = getattr(primitive, "normal_sides", None)
        if normal_sides is None or ops.viewpoint is None:
            raise TypeError("Silhouette requires adjacent normals and a metric viewpoint")
        Query, viewpoint = ops.query_type, ops.viewpoint

        @wp.func
        def accepts(query: Query, item: Any):
            anchor, n0, n1 = wp.static(normal_sides)(item)
            view = wp.static(viewpoint)(query)-anchor
            a, b = wp.dot(view, n0), wp.dot(view, n1)
            return (a <= 0.0 and b >= 0.0) or (a >= 0.0 and b <= 0.0)

        return accepts
