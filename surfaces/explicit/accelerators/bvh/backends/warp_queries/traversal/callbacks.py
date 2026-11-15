"""Bind the two query callback ABIs before device compilation."""
from dataclasses import dataclass
from functools import lru_cache

import warp as wp

from ..._warp_tree import DeviceTree


@dataclass(frozen=True)
class CallbackFunctions:
    test: wp.Function
    visit_leaf: wp.Function


@lru_cache(maxsize=None)
def callback_functions(Context, State, bounds, visit, packed: bool) -> CallbackFunctions:
    if packed:
        @wp.func
        def test(ctx: Context, tree: DeviceTree, node: int, lower: wp.vec3,
                 upper: wp.vec3, state: wp.ref[State]):
            return wp.static(bounds)(ctx, lower, upper, state)

        @wp.func
        def visit_leaf(ctx: Context, tree: DeviceTree, node: int, begin: int,
                       end: int, tag: bool, state: wp.ref[State]):
            wp.static(visit)(ctx, tree, begin, end, state)
    else:
        @wp.func
        def test(ctx: Context, tree: DeviceTree, node: int, lower: wp.vec3,
                 upper: wp.vec3, state: wp.ref[State]):
            return wp.static(bounds)(ctx, tree, node, lower, upper, state)

        @wp.func
        def visit_leaf(ctx: Context, tree: DeviceTree, node: int, begin: int,
                       end: int, tag: bool, state: wp.ref[State]):
            wp.static(visit)(ctx, tree, node, begin, end, tag, state)

    return CallbackFunctions(test, visit_leaf)


@lru_cache(maxsize=None)
def children_function(Context, State, packed: bool, child_indices, read_node, test) -> wp.Function:
    if packed:
        @wp.func
        def children(tree: DeviceTree, ctx: Context, state: wp.ref[State],
                     node: int, current: wp.uint64):
            left, right = wp.static(child_indices)(tree, node, current)
            ll, lu, lmeta = wp.static(read_node)(tree, left)
            rl, ru, rmeta = wp.static(read_node)(tree, right)
            a, accept_a, tag_a = wp.static(test)(ctx, tree, left, ll, lu, state)
            b, accept_b, tag_b = wp.static(test)(ctx, tree, right, rl, ru, state)
            return left, right, lmeta, rmeta, a, b, accept_a, accept_b, tag_a, tag_b
    else:
        @wp.func
        def children(tree: DeviceTree, ctx: Context, state: wp.ref[State],
                     node: int, current: wp.uint64):
            left, right = wp.static(child_indices)(tree, node, current)
            # Load both children before either bound callback.
            ll, lu, lmeta = wp.static(read_node)(tree, left)
            rl, ru, rmeta = wp.static(read_node)(tree, right)
            a, accept_a, tag_a = wp.static(test)(ctx, tree, left, ll, lu, state)
            b, accept_b, tag_b = wp.static(test)(ctx, tree, right, rl, ru, state)
            if a != a or b != b:
                state.failed = True
            accept_a = accept_a and a <= state.limit
            accept_b = accept_b and b <= state.limit
            return left, right, lmeta, rmeta, a, b, accept_a, accept_b, tag_a, tag_b

    return children
