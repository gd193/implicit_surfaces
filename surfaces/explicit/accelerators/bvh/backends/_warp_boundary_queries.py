"""Static boundary sampling and Robin radius policies."""
from collections import namedtuple
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import jax.numpy as jnp
import numpy as np
import warp as wp

from ._warp_factory import BoundQuery, bind_query
from ._warp_metrics import Euclidean
from ._warp_nearest import prepare_nearest
from ._warp_layout import _validate_tree
from ._warp_traversal import load_array, make_walk
from ._warp_primitives import _struct
from ._warp_tree import DeviceTree, STACK_CAPACITY


@wp.struct
class SamplingMeta:
    primitive_weight: wp.array(dtype=wp.float32)
    node_weight: wp.array(dtype=wp.float32)


@wp.struct
class CoefficientMeta:
    primitive_bounds: wp.array(dtype=wp.vec2)
    node_bounds: wp.array(dtype=wp.vec2)


SamplingData = namedtuple("SamplingData", SamplingMeta.vars)
CoefficientData = namedtuple("CoefficientData", CoefficientMeta.vars)
RobinData = namedtuple("RobinData", "core silhouette coefficients")


@dataclass(frozen=True)
class MeasureOps:
    """sample(item, uniforms2) returns point, normal and conditional density."""
    sample: Any
    weights: Any = None


@dataclass(frozen=True)
class RegionOps:
    """overlaps is conservative; contains tests the sampled point itself."""
    query_type: Any
    query_shape: tuple
    valid: Any
    overlaps: Any
    contains: Any


@dataclass(frozen=True)
class RobinOps:
    """Node bounds include features; node_visit returns (score, limit, skip_features, accepted). Scores default to radii."""
    query_type: Any
    query_shape: tuple
    valid: Any
    node_bound: Any
    primitive_bound: Any
    feature_bound: Any
    node_bound_with_data: Any = None
    feature_bound_with_data: Any = None
    core_type: Any = None
    feature_type: Any = None
    to_score: Any = None
    to_distance: Any = None
    leaf_bound_with_data: Any = None
    node_visit: Any = None


def _reduce_nodes(tree, primitive, combine):
    result = np.empty((tree.node_left.shape[0],)+primitive.shape[1:], np.float32)
    if not len(result):
        return result
    arrays, _, order = _validate_tree(tree, len(primitive))
    left, right, offset, count = (arrays[name] for name in
        ("node_left", "node_right", "node_leaf_offset", "node_leaf_count"))
    for node in order:
        if count[node]:
            result[node] = combine(primitive[offset[node]:offset[node]+count[node]])
        else:
            result[node] = combine(result[[left[node], right[node]]])
    return result


def prepare_boundary_sample(tree, boundary, measure=None):
    """Use the same static measure policy when preparing and querying."""
    core = boundary.core if hasattr(boundary, "core") else boundary
    primitive_weight = core.primitive_measure
    if measure is not None and measure.weights is not None:
        raw = measure.weights(core)
        if np.iscomplexobj(raw):
            raise ValueError("sampling weights must be real")
        raw = np.asarray(raw, np.float64)
        if (raw.shape != tree.prim_ids.shape or not np.isfinite(raw).all() or np.any(raw < 0)
                or np.any(raw > np.finfo(np.float32).max)):
            raise ValueError("sampling weights must have one finite float32 value per leaf primitive")
        primitive_weight = jnp.asarray(raw, jnp.float32, device=tree.node_lower.sharding)
    values = np.asarray(primitive_weight)
    if values.shape != tree.prim_ids.shape or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("primitive measure must be finite, nonnegative and match tree slots")
    nodes = _reduce_nodes(tree, values, lambda a: a.sum(axis=0, dtype=np.float32))
    if not np.isfinite(nodes).all():
        raise ValueError("subtree sampling weight exceeds float32 range")
    return SamplingData(primitive_weight, jnp.asarray(nodes, device=tree.node_lower.sharding))


def prepare_robin_coefficients(tree, minimum, maximum=None):
    values = []
    for raw in (minimum, minimum if maximum is None else maximum):
        if np.iscomplexobj(raw):
            raise ValueError("Robin coefficients must be real")
        raw = np.asarray(raw, np.float64)
        if np.isnan(raw).any() or np.any(raw < 0) or np.any(np.isfinite(raw) & (raw > np.finfo(np.float32).max)):
            raise ValueError("Robin coefficients must be nonnegative float32 or infinity")
        values.append(np.broadcast_to(raw, tree.prim_ids.shape))
    lower, upper = values
    if np.any(lower > upper):
        raise ValueError("Robin coefficient minimum exceeds maximum")
    lo, hi = lower.astype(np.float32), upper.astype(np.float32)
    lo = np.where(lo.astype(np.float64) > lower, np.nextafter(lo, np.float32(-np.inf)), lo)
    hi = np.where(hi.astype(np.float64) < upper, np.nextafter(hi, np.float32(np.inf)), hi)
    primitive = np.stack((lo, hi), axis=-1)[np.asarray(tree.prim_ids)]
    nodes = _reduce_nodes(tree, primitive, lambda a: np.array([a[:, 0].min(), a[:, 1].max()], np.float32))
    return CoefficientData(*(jnp.asarray(a, device=tree.node_lower.sharding) for a in (primitive, nodes)))


def prepare_robin(tree, boundary, minimum, maximum=None):
    if boundary.silhouette is None:
        raise ValueError("Robin radius requires silhouette metadata")
    return RobinData(boundary.core, boundary.silhouette, prepare_robin_coefficients(tree, minimum, maximum))


@lru_cache(maxsize=None)
def _ball_region(primitive):
    Vec, dimension = primitive.point_type, primitive.dimension

    @wp.func
    def valid(query: Vec, radius: float):
        return wp.isfinite(query) and not (radius != radius) and radius >= 0.0

    @wp.func
    def overlaps(query: Vec, radius: float, lower: wp.vec3, upper: wp.vec3):
        distance2 = float(0.0)
        for j in range(wp.static(dimension)):
            delta = wp.max(wp.max(lower[j]-query[j], 0.0), query[j]-upper[j])
            distance2 += delta*delta
        return distance2 <= radius*radius

    @wp.func
    def contains(query: Vec, radius: float, point: Vec):
        delta = point-query
        return wp.dot(delta, delta) <= radius*radius

    return RegionOps(Vec, (dimension,), valid, overlaps, contains)


def _boundary_dimension(primitive):
    from ._warp_primitives import Primitive, Segments, Triangles

    if type(primitive) is Primitive:
        if primitive == Segments(2):
            return 2
        if primitive == Triangles():
            return 3
    raise TypeError("this primitive requires explicit boundary query policies")


@lru_cache(maxsize=None)
def _uniform_measure(primitive):
    Vec, Item = primitive.point_type, Any
    dimension = _boundary_dimension(primitive)
    if dimension == 2:
        @wp.func
        def sample(item: Item, uniforms: wp.vec2):
            edge = item.b-item.a
            length = wp.length(edge)
            normal = Vec(edge[1], -edge[0])/length
            return item.a+uniforms[0]*edge, normal, 1.0/length
    else:
        @wp.func
        def sample(item: Item, uniforms: wp.vec2):
            normal = wp.cross(item.e1, item.e2)
            twice_area = wp.length(normal)
            root = wp.sqrt(uniforms[0])
            point = item.a+root*(1.0-uniforms[1])*item.e1+root*uniforms[1]*item.e2
            return point, normal/twice_area, 2.0/twice_area
    return MeasureOps(sample)


@dataclass(frozen=True)
class BoundarySampleQuery(BoundQuery):
    region: object

    def __call__(self, tree, geometry, metadata, queries, radii, randoms):
        queries, caps, shape = prepare_nearest(self.region, queries, radii)
        randoms = jnp.asarray(randoms, jnp.float32)
        if randoms.shape[-1:] != (3,):
            raise ValueError("randoms must have trailing shape (3,)")
        randoms = jnp.broadcast_to(randoms, shape+(3,)).reshape(-1, 3)
        return self.launch(tree, geometry, shape, queries, caps, randoms, metadata=(metadata,))


@lru_cache(maxsize=None)
def make_boundary_sample(primitive, measure=None, region=None, *, block_dim=256):
    if not 1 <= block_dim <= 1024:
        raise ValueError("invalid block dimension")
    measure = _uniform_measure(primitive) if measure is None else measure
    region = _ball_region(primitive) if region is None else region
    Vec, Query, Geometry = primitive.point_type, region.query_type, primitive.data_type

    @wp.func
    def sample(query: Query, radius: float, randoms: wp.vec3, tree: DeviceTree,
               geometry: Geometry, meta: SamplingMeta):
        point, normal, density, index, status = Vec(0.0), Vec(0.0), float(0.0), int(-1), int(0)
        if (not wp.static(region.valid)(query, radius) or not wp.isfinite(randoms)
                or wp.min(randoms) < 0.0 or wp.max(randoms) >= 1.0):
            return point, normal, density, index, -1
        if tree.node_left.shape[0] == 0 or meta.node_weight[0] <= 0.0:
            return point, normal, density, index, status
        if not wp.static(region.overlaps)(query, radius, tree.node_lower[0], tree.node_upper[0]):
            return point, normal, density, index, status
        node, u, probability = int(0), float(randoms[0]), float(1.0)
        while tree.node_leaf_count[node] == 0:
            left, right = tree.node_left[node]+0, tree.node_right[node]+0
            a, b = float(0.0), float(0.0)
            if wp.static(region.overlaps)(query, radius, tree.node_lower[left], tree.node_upper[left]):
                a = meta.node_weight[left]
            if wp.static(region.overlaps)(query, radius, tree.node_lower[right], tree.node_upper[right]):
                b = meta.node_weight[right]
            if a+b <= 0.0:
                return point, normal, density, index, status
            threshold = a/(a+b)
            if b <= 0.0 or (a > 0.0 and u < threshold):
                probability *= threshold
                u /= threshold
                node = left
            else:
                probability *= 1.0-threshold
                u = (u-threshold)/(1.0-threshold)
                node = right
            u = wp.clamp(u, 0.0, 0.99999994)
        offset, count = tree.node_leaf_offset[node], tree.node_leaf_count[node]
        total, selected = float(0.0), int(-1)
        for j in range(count):
            slot = offset+j
            item = wp.static(primitive.load)(geometry, slot)
            lower, upper = wp.static(primitive.bounds_of_item)(item)
            if meta.primitive_weight[slot] > 0.0 and wp.static(region.overlaps)(query, radius, lower, upper):
                total += meta.primitive_weight[slot]
                selected = slot
        if total <= 0.0:
            return point, normal, density, index, status
        threshold, accumulated = u*total, float(0.0)
        for j in range(count):
            slot = offset+j
            item = wp.static(primitive.load)(geometry, slot)
            lower, upper = wp.static(primitive.bounds_of_item)(item)
            if meta.primitive_weight[slot] > 0.0 and wp.static(region.overlaps)(query, radius, lower, upper):
                accumulated += meta.primitive_weight[slot]
                if threshold < accumulated:
                    selected = slot
                    break
        candidate, candidate_normal, local_pdf = wp.static(measure.sample)(
            wp.static(primitive.load)(geometry, selected), wp.vec2(randoms[1], randoms[2]))
        if not wp.isfinite(candidate) or not wp.isfinite(candidate_normal) or not wp.isfinite(local_pdf) or local_pdf < 0.0:
            return point, normal, density, index, -1
        if local_pdf == 0.0:
            return point, normal, density, index, status
        if not wp.static(region.contains)(query, radius, candidate):
            return point, normal, density, index, status
        density = probability*(meta.primitive_weight[selected]/total)*local_pdf
        if density <= 0.0:
            return point, normal, float(0.0), index, -1
        return candidate, candidate_normal, density, tree.prim_ids[selected], 1

    inputs = [("points", wp.array(dtype=Query)), ("radii", wp.array(dtype=wp.float32)),
              ("randoms", wp.array(dtype=wp.vec3))]
    outputs = [("out_point", wp.array(dtype=Vec)), ("out_normal", wp.array(dtype=Vec)),
               ("out_pdf", wp.array(dtype=wp.float32)), ("out_id", wp.array(dtype=wp.int32)),
               ("out_status", wp.array(dtype=wp.int32))]
    body = ["p, n, pdf, index, status = wp.static(sample)(load_array(points, i), load_array(radii, i), load_array(randoms, i), tree, geometry, meta)",
            "store_array(out_point, i, p)", "store_array(out_normal, i, n)", "store_array(out_pdf, i, pdf)", "store_array(out_id, i, index)", "store_array(out_status, i, status)"]
    return bind_query(BoundarySampleQuery, "boundary_sample_kernel", primitive, inputs, outputs, body,
        {"sample": sample}, block_dim, groups=(("meta", SamplingMeta),), region=region)


@lru_cache(maxsize=None)
def _euclidean_robin(primitive):
    from ._warp_boundary import BoundaryCore, SilhouetteMeta, cone_visibility

    Vec, dimension = primitive.point_type, _boundary_dimension(primitive)
    Geometry = primitive.data_type
    nearest = Euclidean().bind(primitive)
    leaf_with_data = visit = bound_with_data = primitive_bound = None

    @wp.func
    def valid(query: Vec):
        return wp.isfinite(query)

    @wp.func
    def node_bound(query: Vec, lower: wp.vec3, upper: wp.vec3, coefficients: wp.vec2):
        return wp.static(nearest.node_bound)(query, lower, upper)

    if dimension == 3:
        @wp.func_native("return bvh::robin_leaf(query, geometry, core, coefficients, offset, count, best);")
        def leaf_with_data(query: Vec, geometry: Geometry, core: BoundaryCore,
                           coefficients: CoefficientMeta, offset: int, count: int, best: float) -> float: ...

        @wp.func_native("bvh::robin_node(query, feature, meta, node, lower, upper, best, score, bound, skip, accepted);")
        def visit_values(query: Vec, tree: DeviceTree, geometry: Geometry, core: BoundaryCore,
                         feature: SilhouetteMeta, meta: CoefficientMeta, node: int,
                         lower: wp.vec3, upper: wp.vec3, best: float, score: wp.ref[float], bound: wp.ref[float],
                         skip: wp.ref[bool], accepted: wp.ref[bool]): ...

        @wp.func
        def visit(query: Vec, tree: DeviceTree, geometry: Geometry, core: BoundaryCore,
                  feature: SilhouetteMeta, meta: CoefficientMeta, node: int,
                  lower: wp.vec3, upper: wp.vec3, best: float):
            score, bound, skip, accepted = float(0.0), float(best), bool(False), bool(False)
            visit_values(query, tree, geometry, core, feature, meta, node, lower, upper, best, score, bound, skip, accepted)
            return score, bound, skip, accepted
    else:
        @wp.func
        def bound_with_data(query: Vec, tree: DeviceTree, geometry: Geometry, core: BoundaryCore,
                            feature: SilhouetteMeta, meta: CoefficientMeta, node: int):
            lower, upper = tree.node_lower[node], tree.node_upper[node]
            box2 = node_bound(query, lower, upper, meta.node_bounds[node])
            point = wp.vec3(query[0], query[1], 0.0)
            coefficient = meta.node_bounds[node][1]
            if coefficient == 0.0:
                return wp.where(cone_visibility(point, tree, feature, node), box2, wp.inf)
            angles = feature.node_cone_sin_cos[node]
            if angles[0] < 0.0 or angles[1] <= 0.0 or wp.isinf(coefficient):
                return box2
            view = .5*(lower+upper)-point
            axis = feature.node_cone_axis[node]
            projection = wp.dot(axis, view)
            extent = .5*(upper-lower)
            gap = wp.abs(projection)-wp.dot(wp.abs(axis), extent)
            if gap <= 0.0:
                return box2
            radius = feature.node_cone_radius[node]
            h = gap*angles[1]-angles[0]*(wp.length(view-projection*axis)+radius)-4e-6*(wp.length(view)+radius)
            if not (h > 0.0 and wp.isfinite(h)):
                return box2
            far = wp.max(wp.abs(lower-point), wp.abs(upper-point))
            distance2 = wp.clamp(2.0*(h/coefficient), box2, wp.dot(far, far))
            bound = distance2*wp.exp(2.0*h/(coefficient*distance2))*0.999996
            return wp.max(box2, bound)

        @wp.func
        def primitive_bound(query: Vec, item: Any, coefficient: float):
            if coefficient == 0.0:
                return wp.inf
            closest, near2 = wp.static(nearest.closest)(query, item)
            if not wp.isfinite(near2):
                return wp.nan
            if wp.isinf(coefficient) or near2 == 0.0:
                return near2
            edge = item.b-item.a
            normal = Vec(edge[1], -edge[0])/wp.length(edge)
            height = wp.abs(wp.dot(query-item.a, normal))
            if height == 0.0:
                return near2
            a, b = query-item.a, query-item.b
            far2 = wp.max(wp.dot(a, a), wp.dot(b, b))
            distance2 = wp.clamp(2.0*(height/coefficient), near2, far2)
            bound = distance2*wp.exp(2.0*height/(coefficient*distance2))*0.999996
            return wp.max(near2, bound)

    @wp.func
    def feature_bound(query: Vec, a: wp.vec3, b: wp.vec3, n0: wp.vec3, n1: wp.vec3):
        point = wp.vec3(0.0)
        for j in range(wp.static(dimension)):
            point[j] = query[j]
        view, edge = point-a, b-a
        side0, side1 = wp.dot(view, n0), wp.dot(view, n1)
        if (side0 < 0.0 and side1 < 0.0) or (side0 > 0.0 and side1 > 0.0):
            return wp.inf
        t, denominator = float(0.0), wp.dot(edge, edge)
        if denominator > 0.0:
            t = wp.clamp(wp.dot(view, edge)/denominator, 0.0, 1.0)
        delta = point-a-t*edge
        return wp.dot(delta, delta)

    return RobinOps(Vec, (dimension,), valid, node_bound, primitive_bound, feature_bound, bound_with_data,
                    to_score=nearest.to_score, to_distance=nearest.to_distance,
                    leaf_bound_with_data=leaf_with_data, node_visit=visit)


@dataclass(frozen=True)
class RobinQuery(BoundQuery):
    ops: object

    def __call__(self, tree, geometry, metadata, queries, cap=jnp.inf):
        queries, caps, shape = prepare_nearest(self.ops, queries, cap)
        return self.launch(tree, geometry, shape, queries, caps,
                           metadata=(metadata.core, metadata.silhouette, metadata.coefficients))


@lru_cache(maxsize=None)
def make_robin_radius(primitive, policy=None, metric=None, *, stack_capacity=STACK_CAPACITY, block_dim=64):
    from ._warp_boundary import BoundaryCore, SilhouetteMeta

    if not 1 <= stack_capacity <= STACK_CAPACITY or not 1 <= block_dim <= 1024:
        raise ValueError("invalid stack capacity or block dimension")
    metric = Euclidean() if metric is None else metric
    if policy is None and type(metric) is not Euclidean:
        raise TypeError("this metric requires explicit conservative RobinOps")
    ops = _euclidean_robin(primitive) if policy is None else policy
    if hasattr(ops, "bind"):
        ops = ops.bind(primitive, metric)
    required = (ops.valid, ops.leaf_bound_with_data or ops.primitive_bound,
                ops.node_visit or ops.node_bound_with_data or ops.node_bound,
                ops.feature_bound_with_data or ops.feature_bound)
    if not all(callable(operation) for operation in required):
        raise TypeError("RobinOps requires validity, primitive, node and feature callbacks")
    if (ops.to_score is None) != (ops.to_distance is None):
        raise TypeError("RobinOps score conversions must be supplied together")
    Core, Feature = ops.core_type or BoundaryCore, ops.feature_type or SilhouetteMeta
    for name in ("feature_refs", "node_feature_offset", "node_feature_count"):
        field = Feature.vars.get(name)
        if field is None or not isinstance(field.type, wp.array) or field.type.dtype is not wp.int32 or field.type.ndim != 1:
            raise TypeError(f"Robin feature metadata requires int32 array {name}")
    if ops.feature_bound_with_data is None and any(name not in Feature.vars for name in
            ("feature_a", "feature_b", "feature_n0", "feature_n1")):
        raise TypeError("custom feature metadata requires feature_bound_with_data")
    Query, Geometry = ops.query_type, primitive.data_type
    Context = _struct("RobinContext", {"query": Query, "geometry": Geometry, "core": Core,
                                       "feature": Feature, "coefficients": CoefficientMeta})
    State = _struct("RobinState", {"limit": wp.float32, "failed": wp.bool, "stop": wp.bool})

    @wp.func
    def bounds(ctx: Context, tree: DeviceTree, node: int, lower: wp.vec3, upper: wp.vec3, state: wp.ref[State]):
        if wp.static(ops.node_visit is not None):
            score, bound, skip, eligible = wp.static(ops.node_visit)(ctx.query, tree, ctx.geometry,
                ctx.core, ctx.feature, ctx.coefficients, node, lower, upper, state.limit)
            state.failed = wp.where(bound >= 0.0, state.failed, True)
            state.limit = wp.min(state.limit, bound)
            return score, eligible, skip
        score = float(-wp.inf)
        if wp.static(ops.node_bound is not None):
            score = wp.static(ops.node_bound)(ctx.query, lower, upper, load_array(ctx.coefficients.node_bounds, node))
        if wp.static(ops.node_bound_with_data is not None):
            if score <= state.limit:
                score = wp.static(ops.node_bound_with_data)(ctx.query, tree, ctx.geometry,
                    ctx.core, ctx.feature, ctx.coefficients, node)
        return score, score < wp.inf, False

    @wp.func
    def visit(ctx: Context, tree: DeviceTree, node: int, begin: int, end: int, skip: bool, state: wp.ref[State]):
        if not skip:
            offset = load_array(ctx.feature.node_feature_offset, node)
            for j in range(load_array(ctx.feature.node_feature_count, node)):
                index = load_array(ctx.feature.feature_refs, offset+j)
                candidate = float(0.0)
                if wp.static(ops.feature_bound_with_data is not None):
                    candidate = wp.static(ops.feature_bound_with_data)(ctx.query, tree, ctx.geometry,
                        ctx.core, ctx.feature, ctx.coefficients, node, index)
                else:
                    candidate = wp.static(ops.feature_bound)(ctx.query,
                        load_array(ctx.feature.feature_a, index), load_array(ctx.feature.feature_b, index),
                        load_array(ctx.feature.feature_n0, index), load_array(ctx.feature.feature_n1, index))
                state.failed = wp.where(candidate >= 0.0, state.failed, True)
                state.limit = wp.min(state.limit, candidate)
        if wp.static(ops.leaf_bound_with_data is not None):
            candidate = wp.static(ops.leaf_bound_with_data)(ctx.query, ctx.geometry, ctx.core,
                                                          ctx.coefficients, begin, end-begin, state.limit)
            state.failed = wp.where(candidate >= 0.0, state.failed, True)
            state.limit = wp.min(state.limit, candidate)
        else:
            for slot in range(begin, end):
                candidate = wp.static(ops.primitive_bound)(ctx.query, wp.static(primitive.load)(ctx.geometry, slot),
                                                           load_array(ctx.coefficients.primitive_bounds, slot)[1])
                state.failed = wp.where(candidate >= 0.0, state.failed, True)
                state.limit = wp.min(state.limit, candidate)
        state.stop = state.limit == 0.0
        return

    @wp.func
    def resume(ctx: Context, tree: DeviceTree, node: int, lower: wp.vec3, upper: wp.vec3):
        if wp.static(ops.node_bound is not None):
            return wp.static(ops.node_bound)(ctx.query, lower, upper, load_array(ctx.coefficients.node_bounds, node))
        return -wp.inf

    walk = make_walk(Context, State, bounds, visit, stack_capacity=stack_capacity, resume=resume)

    @wp.func
    def radius(query: Query, cap: float, tree: DeviceTree, geometry: Geometry,
               core: Core, feature: Feature, meta: CoefficientMeta):
        if not wp.static(ops.valid)(query) or not (cap >= 0.0):
            return wp.nan, -1
        if cap == 0.0 or tree.node_left.shape[0] == 0:
            return cap, 1
        ctx = Context()
        ctx.query = query
        ctx.geometry = geometry
        ctx.core = core
        ctx.feature = feature
        ctx.coefficients = meta
        state = State()
        state.limit = cap
        if wp.static(ops.to_score is not None):
            state.limit = wp.static(ops.to_score)(cap)
        if state.limit != state.limit:
            return wp.nan, -1
        wp.static(walk)(tree, ctx, state)
        if state.failed:
            return wp.nan, -1
        value = state.limit
        if wp.static(ops.to_distance is not None):
            value = wp.static(ops.to_distance)(value)
        if not (value >= 0.0):
            return wp.nan, -1
        return wp.min(cap, value), 1

    inputs = [("queries", wp.array(dtype=Query)), ("caps", wp.array(dtype=wp.float32))]
    outputs = [("out_radius", wp.array(dtype=wp.float32)), ("out_status", wp.array(dtype=wp.int32))]
    body = ["value, status = wp.static(radius)(load_array(queries, i), load_array(caps, i), tree, geometry, core, feature, meta)",
            "store_array(out_radius, i, value)", "store_array(out_status, i, status)"]
    return bind_query(RobinQuery, "robin_radius_kernel", primitive, inputs, outputs, body, {"radius": radius},
        block_dim, ops=ops, groups=(("core", Core), ("feature", Feature), ("meta", CoefficientMeta)))
