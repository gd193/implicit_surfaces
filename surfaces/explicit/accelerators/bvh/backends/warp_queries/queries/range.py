"""Exact score-threshold queries with static metric and result policies."""
from dataclasses import dataclass
from functools import lru_cache
from typing import Generic, TypeVar

import warp as wp

from ..._warp_tree import DeviceTree, STACK_CAPACITY
from ..io.arrays import load_array
from ..launch import BoundQuery, bind_query
from ..traversal.policy import make_traversal, make_traversal_state
from .nearest import prepare_nearest
from .nearest_ops import bind_nearest, nearest_functions, resolve_metric


HostDataT = TypeVar("HostDataT")
DeviceDataT = TypeVar("DeviceDataT")
ItemT = TypeVar("ItemT")
PointT = TypeVar("PointT")
QueryT = TypeVar("QueryT")
StateT = TypeVar("StateT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class RadiusAccumulator(Generic[QueryT, ItemT, PointT, StateT, ResultT]):
    """Warp callbacks for a custom fixed-size result.

    ``initialize(query, score_limit)`` returns a state with ``limit``,
    ``failed``, ``stop`` and ``matched`` fields. ``update`` receives each qualifying item,
    closest point, score and external primitive ID; it may mutate the state or
    set ``stop``. ``finish(state)`` returns a Warp scalar/vector/struct value.
    """
    state_type: object
    initialize: wp.Function
    update: wp.Function
    finish: wp.Function

    def __post_init__(self):
        for callback in (self.initialize, self.update, self.finish):
            if not isinstance(callback, wp.Function):
                raise TypeError("accumulator callbacks must be decorated Warp functions")


@dataclass(frozen=True)
class RadiusReduction(BoundQuery, Generic[HostDataT, DeviceDataT, ItemT, PointT, QueryT, ResultT]):
    """Device-only reduction with a caller-defined result type."""
    ops: object
    mode: str


@dataclass(frozen=True)
class RadiusQuery(BoundQuery, Generic[HostDataT, DeviceDataT, ItemT, PointT, QueryT]):
    ops: object
    mode: str

    def __call__(self, tree, geometry, queries, radius):
        queries, caps, shape = prepare_nearest(self.ops, queries, radius)
        value, status = self.launch(tree, geometry, shape, queries, caps)
        return value.astype(bool) if self.mode == "any" else value, status


def make_radius(primitive, metric=None, *, evaluate=None, lower_bound=None,
                predicate=None, mode="any", accumulator=None,
                stack_capacity=STACK_CAPACITY, block_dim=64,
                cache_bounds=False, packed_nodes=True, group_leaves=None):
    """Build an exact closed-radius any, count or custom accumulation query."""
    metric = resolve_metric(metric, evaluate, lower_bound, "make_radius")
    if mode not in ("any", "count", "accumulate"):
        raise ValueError("mode must be 'any', 'count' or 'accumulate'")
    if (mode == "accumulate") != (accumulator is not None):
        raise TypeError("accumulator is required only for mode='accumulate'")
    if group_leaves is None:
        group_leaves = bind_nearest(primitive, metric).group_leaves
    return _make_radius(primitive, metric, predicate, mode, accumulator,
                        stack_capacity, block_dim, cache_bounds, packed_nodes,
                        group_leaves)


@lru_cache(maxsize=None)
def _make_radius(primitive, metric, predicate, mode, accumulator, stack_capacity,
                 block_dim, cache_bounds, packed_nodes, group_leaves):
    if not 1 <= stack_capacity <= STACK_CAPACITY or not 1 <= block_dim <= 1024:
        raise ValueError("invalid stack capacity or block dimension")
    ops = bind_nearest(primitive, metric)
    Query, Geometry = ops.query_type, primitive.data_type
    from ..contracts import make_struct
    Context = make_struct("RadiusContext", {"query": Query, "geometry": Geometry,
                                           "initial": wp.float32})
    if mode == "accumulate":
        State = accumulator.state_type
        if ("matched" not in getattr(State, "vars", {})
                or State.vars["matched"].type != wp.bool):
            raise TypeError("accumulator state requires 'matched' boolean field")
    else:
        State = make_traversal_state("RadiusState", {"value": wp.int32})
    functions = nearest_functions(primitive, metric, predicate, Context, State)
    accepts, bounds, resume = functions.accepts, functions.bounds, functions.resume

    @wp.func
    def visit(ctx: Context, tree: DeviceTree, node: int, begin: int, end: int,
              tag: bool, state: wp.ref[State]):
        for slot in range(begin, end):
            item = wp.static(primitive.load)(ctx.geometry, slot)
            if wp.static(accepts is not None):
                if not wp.static(accepts)(ctx.query, item):
                    continue
            if wp.static(ops.cull_items):
                lo, hi = wp.static(primitive.bounds_of_item)(item)
                item_bound = wp.static(ops.node_bound)(ctx.query, lo, hi)
                if item_bound != item_bound:
                    state.failed = True
                    return
                if not (item_bound <= wp.static(ops.cull_limit)(ctx.query, state.limit)):
                    continue
            point, score = wp.static(ops.closest)(ctx.query, item)
            if score != score:
                state.failed = True
                return
            if score <= state.limit:
                if wp.static(mode == "any"):
                    state.value = 1
                    state.stop = True
                    return
                elif wp.static(mode == "count"):
                    if state.value == 2147483647:
                        state.failed = True
                        return
                    state.value += 1
                else:
                    state.matched = True
                    index = load_array(tree.prim_ids, slot)
                    wp.static(accumulator.update)(ctx.query, item, point, score,
                                                  index, state)
                    if state.stop or state.failed:
                        return

    walk = make_traversal(Context, State, bounds, visit,
                          stack_capacity=stack_capacity, packed_nodes=packed_nodes,
                          cache_bounds=cache_bounds,
                          resume=None if cache_bounds else resume,
                          group_leaves=group_leaves)

    if mode == "accumulate":
        @wp.func
        def radius(query: Query, cap: float, tree: DeviceTree, geometry: Geometry):
            state = State()
            if not wp.static(ops.valid)(query) or not (cap >= 0.0):
                return wp.static(accumulator.finish)(state), -1
            score_cap = wp.static(ops.to_score)(cap)
            if score_cap != score_cap:
                return wp.static(accumulator.finish)(state), -1
            state = wp.static(accumulator.initialize)(query, score_cap)
            state.limit = wp.min(score_cap, 3.4028234e38)
            state.matched = False
            ctx = Context(query, geometry, score_cap)
            wp.static(walk.device)(tree, ctx, state)
            if state.failed:
                return wp.static(accumulator.finish)(state), -1
            return wp.static(accumulator.finish)(state), wp.where(state.matched, 1, 0)
    else:
        @wp.func
        def radius(query: Query, cap: float, tree: DeviceTree, geometry: Geometry):
            if not wp.static(ops.valid)(query) or not (cap >= 0.0):
                return 0, -1
            score_cap = wp.static(ops.to_score)(cap)
            if score_cap != score_cap:
                return 0, -1
            state = State()
            state.limit = wp.min(score_cap, 3.4028234e38)
            ctx = Context(query, geometry, score_cap)
            wp.static(walk.device)(tree, ctx, state)
            if state.failed:
                return 0, -1
            return state.value, wp.where(state.value > 0, 1, 0)

    components = (("node_access", "packed" if packed_nodes else "indexed"),
                  ("scheduling", "grouped" if group_leaves else "flat"),
                  ("deferred", "cached_bound" if cache_bounds else "recomputed_bound"),
                  ("selection", mode))
    inputs = [("queries", wp.array(dtype=Query)), ("caps", wp.array(dtype=wp.float32))]
    if mode == "accumulate":
        # A carrier kernel registers the dependency for caller-owned fused
        # kernels; arbitrary Warp result structs do not have a JAX array ABI.
        return RadiusReduction(walk._device_kernel, primitive, 0, 0, block_dim,
                           device=radius, ops=ops, mode=mode,
                           components=components)
    outputs = [("out_value", wp.array(dtype=wp.int32)),
               ("out_status", wp.array(dtype=wp.int32))]
    body = ["value, status = wp.static(radius)(load_array(queries, i), load_array(caps, i), tree, geometry)",
            "store_array(out_value, i, value)", "store_array(out_status, i, status)"]
    return bind_query(RadiusQuery, "radius_kernel", primitive, inputs, outputs,
                      body, {"radius": radius}, block_dim, device=radius, ops=ops,
                      mode=mode, components=components)
