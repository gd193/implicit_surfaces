"""Static nearest-query policies; bounds and minimizers share score units."""
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol, TypeVar

import warp as wp

from ._warp_geometry import _NATIVE_POINT_GEOMETRY


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

        if dimension == 3:
            @wp.func_native(_NATIVE_POINT_GEOMETRY + "return box_distance_sq(lower, upper);")
            def bound(query: Vec, lower: wp.vec3, upper: wp.vec3) -> float: ...
        else:
            @wp.func
            def bound(query: Vec, lower: wp.vec3, upper: wp.vec3):
                x = wp.max(wp.max(lower[0]-query[0], 0.0), query[0]-upper[0])
                y = wp.max(wp.max(lower[1]-query[1], 0.0), query[1]-upper[1])
                return x*x+y*y

        @wp.func
        def closest(query: Vec, item: Any):
            point = wp.static(closest_primitive)(query, item)
            delta = query-point
            score = wp.dot(delta, delta)
            return point, wp.where(wp.isfinite(score), score, wp.nan)

        @wp.func
        def to_score(distance: float):
            return distance*distance

        @wp.func
        def to_distance(score: float):
            return wp.sqrt(score)

        @wp.func
        def cull_limit(query: Vec, score: float):
            return score*1.000002 + 1.e-12*wp.dot(query, query)

        return NearestOps(Vec, (dimension,), valid, bound, closest, to_score, to_distance, cull_limit)


AngularQuery = wp.types.matrix((2, 3), wp.float32)

_ANGLE = r"""
    using V = wp::vec_t<3,double>;
    const V origin(query.data[0][0],query.data[0][1],query.data[0][2]);
    const V direction(query.data[1][0],query.data[1][1],query.data[1][2]);
    const auto angle = [&](const V& view) -> double {
        return wp::dot(view,view) > 0
            ? atan2(wp::length(wp::cross(view,direction)),wp::dot(view,direction)) : INFINITY;
    };
"""


@dataclass(frozen=True)
class Angular:
    @lru_cache(maxsize=None)
    def bind(self, primitive):
        if getattr(primitive, "name", None) not in ("segments", "silhouette_edges") or primitive.dimension != 3:
            raise TypeError("Angular supports 3D segments")
        @wp.func
        def valid(query: AngularQuery):
            return wp.isfinite(query) and wp.max(wp.abs(query[1])) > 0.0

        @wp.func_native(_ANGLE + r"""
            const V view=.5*(V(lower)+V(upper))-origin, extent=.5*(V(upper)-V(lower));
            const double length2=wp::dot(view,view), radius2=wp::dot(extent,extent);
            if (length2 <= radius2) return 0.f;
            const double lower_angle=angle(view)-asin(sqrt(radius2/length2));
            return static_cast<float>(wp::max(0.0,lower_angle-4.e-7));
        """)
        def bound(query: AngularQuery, lower: wp.vec3, upper: wp.vec3) -> float: ...

        @wp.func_native(_ANGLE + r"""
            wp::vec3 best = a;
            float score = static_cast<float>(angle(V(a)-origin));
            float other = static_cast<float>(angle(V(b)-origin));
            if (other < score) { best=b; score=other; }
            const V view=V(a)-origin, edge=V(b)-V(a);
            const double alpha=wp::dot(direction,view), beta=wp::dot(direction,edge);
            const double c=wp::dot(view,view), d=wp::dot(view,edge), f=wp::dot(edge,edge);
            const double denominator=beta*d-alpha*f;
            if (denominator != 0) {
                const double t=(alpha*d-beta*c)/denominator;
                if (t > 0 && t < 1) {
                    const wp::vec3 point(V(a)+t*edge);
                    other=static_cast<float>(angle(V(point)-origin));
                    if (other < score) { best=point; score=other; }
                }
            }
            return wp::vec4(best[0],best[1],best[2],score);
        """)
        def segment(query: AngularQuery, a: wp.vec3, b: wp.vec3) -> wp.vec4: ...

        @wp.func
        def closest(query: AngularQuery, item: Any):
            result = segment(query, item.a, item.b)
            return wp.vec3(result[0], result[1], result[2]), result[3]

        @wp.func
        def identity(value: float):
            return value

        @wp.func
        def cull_limit(query: AngularQuery, score: float):
            return score + 4.e-7

        return NearestOps(AngularQuery, (2, 3), valid, bound, closest,
                          identity, identity, cull_limit)


@dataclass(frozen=True)
class Silhouette:
    def bind(self, primitive, ops):
        if getattr(primitive, "name", None) != "silhouette_edges":
            raise TypeError("Silhouette requires edge endpoints and adjacent normals")
        Query = ops.query_type
        angular = ops.query_shape == (2, 3)

        @wp.func
        def accepts(query: Query, item: Any):
            if wp.static(angular):
                origin = wp.vec3(query[0, 0], query[0, 1], query[0, 2])
            else:
                origin = query
            view = origin-item.a
            a, b = wp.dot(view, item.n0), wp.dot(view, item.n1)
            return (a <= 0.0 and b >= 0.0) or (a >= 0.0 and b <= 0.0)

        return accepts
