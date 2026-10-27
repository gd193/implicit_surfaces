"""Even-odd classification with half-open +X ray intersections."""

from dataclasses import dataclass
from functools import lru_cache

import warp as wp

from ._warp_factory import BoundQuery, bind_query
from ._warp_jax import _flatten_vec_queries
from ._warp_primitives import _struct
from ._warp_traversal import load_array, make_walk
from ._warp_tree import DeviceTree, STACK_CAPACITY


_NATIVE_TRIANGLE_EVENT = r"""
    const wp::vec3 a = LOAD(0) - p;
    const wp::vec3 b = LOAD(1) - p;
    const wp::vec3 c = LOAD(2) - p;
    const wp::vec3 e_fast(b[1]*c[2]-b[2]*c[1], c[1]*a[2]-c[2]*a[1],
                          a[1]*b[2]-a[2]*b[1]);
    // Bounds include coordinate subtraction, products and summation.
    constexpr float rounding = 8.0f * 1.1920928955078125e-7f;
    const wp::vec3 error(
        wp::max(rounding*(wp::abs(b[1]*c[2])+wp::abs(b[2]*c[1])), 1.e-37f),
        wp::max(rounding*(wp::abs(c[1]*a[2])+wp::abs(c[2]*a[1])), 1.e-37f),
        wp::max(rounding*(wp::abs(a[1]*b[2])+wp::abs(a[2]*b[1])), 1.e-37f));
    bool positive = false, negative = false, uncertain = false;
    for (int i = 0; i < 3; ++i) {
        positive |= e_fast[i] > error[i];
        negative |= e_fast[i] < -error[i];
        uncertain |= !(wp::abs(e_fast[i]) > error[i]);
    }
    if (positive && negative) return 0;
    if (!uncertain) {
        const wp::vec3 terms(e_fast[0]*a[0], e_fast[1]*b[0], e_fast[2]*c[0]);
        const float t = terms[0] + terms[1] + terms[2];
        const float t_error = rounding * (wp::abs(terms[0]) + wp::abs(terms[1])
            + wp::abs(terms[2])) + error[0]*wp::abs(a[0])
            + error[1]*wp::abs(b[0]) + error[2]*wp::abs(c[0]);
        if (wp::abs(t) > t_error) return (t > 0.0f) == positive ? 1 : 0;
    }
    double v[3][3];
    for (int i = 0; i < 3; ++i) {
        const wp::vec3 vertex = LOAD(i);
        for (int j = 0; j < 3; ++j)
            v[i][j] = static_cast<double>(vertex[j]) - static_cast<double>(p[j]);
    }
    double e[3];
    for (int i = 0; i < 3; ++i)
        e[i] = cross2(v[(i+1)%3][1], v[(i+1)%3][2],
                      v[(i+2)%3][1], v[(i+2)%3][2]);
    double det = e[0] + e[1] + e[2];
    const bool parallel = det == 0.0;
    int axis = 0;
    if (parallel) {
        // A parallel face can contain the query without crossing the ray.
        double normal[3];
        for (int j = 0; j < 3; ++j) {
            const int u = (j+1)%3, w = (j+2)%3;
            normal[j] = cross2(v[1][u]-v[0][u], v[1][w]-v[0][w],
                               v[2][u]-v[0][u], v[2][w]-v[0][w]);
        }
        axis = wp::abs(normal[1]) > wp::abs(normal[2]) ? 1 : 2;
        const int u = (axis+1)%3, w = (axis+2)%3;
        for (int i = 0; i < 3; ++i)
            e[i] = cross2(v[(i+1)%3][u], v[(i+1)%3][w],
                          v[(i+2)%3][u], v[(i+2)%3][w]);
        det = e[0] + e[1] + e[2];
        if (det == 0.0) return 0;
    }
    const double orientation = det > 0.0 ? 1.0 : -1.0;
    for (int i = 0; i < 3; ++i) {
        e[i] *= orientation;
        if (e[i] < 0.0) return 0;
    }
    const double t = e[0]*v[0][axis] + e[1]*v[1][axis] + e[2]*v[2][axis];
    if (t == 0.0) return 2;
    if (parallel || t < 0.0) return 0;
    // Top-left ownership counts a shared edge or vertex exactly once.
    for (int i = 0; i < 3; ++i) {
        if (e[i] == 0.0) {
            const int a = (i+1)%3, b = (i+2)%3;
            if (!owns_edge(v[a][1], v[a][2], v[b][1], v[b][2], orientation)) return 0;
        }
    }
    return 1;
"""


def _triangle_event_source(load):
    source = _NATIVE_TRIANGLE_EVENT
    for index in ("0", "1", "2", "i"):
        source = source.replace(f"LOAD({index})", load.replace("$", index))
    return source


@dataclass(frozen=True)
class ClassifyQuery(BoundQuery):
    policy: object

    def __call__(self, tree, geometry, boundary, points):
        queries, shape = _flatten_vec_queries(points, self.primitive.dimension)
        return self.launch(tree, geometry, shape, queries, metadata=(getattr(boundary, "core", boundary),))


@lru_cache(maxsize=None)
def make_classify(primitive, policy=None, *, adapter=None, stack_capacity=STACK_CAPACITY, block_dim=128):
    if not 1 <= stack_capacity <= STACK_CAPACITY or not 1 <= block_dim <= 1024:
        raise ValueError("invalid stack capacity or block dimension")
    if policy is None:
        from ._warp_boundary import resolve_boundary_adapter
        adapter = resolve_boundary_adapter(primitive, adapter)
        if adapter.classification is None:
            raise TypeError("boundary adapter has no classification operation")
        policy = adapter.classification(primitive)
    Vec, Geometry, Meta = primitive.point_type, primitive.data_type, policy.metadata_type
    Context = _struct("ClassifyContext", {"point": Vec, "geometry": Geometry, "meta": Meta})
    State = _struct("ClassifyState", {"limit": float, "failed": bool, "stop": bool,
                                     "parity": int, "contact": bool})
    dimension = primitive.dimension

    @wp.func
    def bounds(ctx: Context, lo: wp.vec3, hi: wp.vec3, state: wp.ref[State]):
        eligible = ctx.point[0] <= hi[0]
        for axis in range(1, wp.static(dimension)):
            eligible = eligible and ctx.point[axis] >= lo[axis] and ctx.point[axis] <= hi[axis]
        return 0.0, eligible, False

    @wp.func
    def visit(ctx: Context, tree: DeviceTree, begin: int, end: int, state: wp.ref[State]):
        for slot in range(begin, end):
            event = wp.static(policy.event)(ctx.point, ctx.geometry, slot, load_array(tree.prim_ids, slot), ctx.meta)
            if event < 0 or event > 2:
                state.failed = True
                return
            if event == 2:
                state.contact = True
                state.stop = True
                return
            state.parity = state.parity ^ event
        return

    walk = make_walk(Context, State, bounds, visit, stack_capacity=stack_capacity, near_first=False, packed=True)

    @wp.func
    def classify(point: Vec, tree: DeviceTree, geometry: Geometry, meta: Meta):
        if (not wp.isfinite(point) or wp.max(wp.abs(point)) >= 3.4028234e38
                or not wp.static(policy.valid)(meta)):
            return wp.nan, -1
        if tree.node_left.shape[0] == 0:
            return 1.0, 0
        ctx = Context(point, geometry, meta)
        state = State()
        state.limit = 0.0
        wp.static(walk)(tree, ctx, state)
        if state.failed:
            return wp.nan, -1
        return wp.where(state.contact, 0.0, wp.where(state.parity != 0, -1.0, 1.0)), 1

    inputs = [("queries", wp.array(dtype=Vec))]
    outputs = [("out_sign", wp.array(dtype=wp.float32)), ("out_status", wp.array(dtype=wp.int32))]
    body = ["sign, status = wp.static(classify)(load_array(queries, i), tree, geometry, meta)",
            "store_array(out_sign, i, sign)", "store_array(out_status, i, status)"]
    return bind_query(ClassifyQuery, "classify_kernel", primitive, inputs, outputs, body,
                      {"classify": classify}, block_dim, groups=(("meta", Meta),), policy=policy)
