"""Compose a flat or grouped traversal from selected device operations."""
from functools import lru_cache

import warp as wp

from ..._warp_tree import DeviceTree, STACK_CAPACITY
from ..io.nodes import node_functions
from ..native import entry_index, entry_score, entry_tag
from .callbacks import callback_functions, children_function
from .deferred import deferred_functions, restore_function


def make_walk(Context, State, bounds, visit, *, stack_capacity=STACK_CAPACITY,
              near_first=True, packed=False, packed_nodes=False, test_root=True,
              resume=None, group_leaves=False):
    """Packed callbacks visit ranges; indexed callbacks receive node and tag."""
    if not 1 <= stack_capacity <= STACK_CAPACITY:
        raise ValueError("invalid stack capacity")
    if packed and resume is not None:
        raise ValueError("packed callbacks cannot resume indexed bounds")
    return _make_walk(Context, State, bounds, visit, stack_capacity,
                      bool(near_first), bool(packed), bool(packed or packed_nodes),
                      bool(test_root), resume, bool(group_leaves))


@lru_cache(maxsize=None)
def _make_walk(Context, State, bounds, visit, stack_capacity, near_first, packed,
               packed_nodes, test_root, resume, group_leaves):
    nodes = node_functions(packed_nodes)
    callbacks = callback_functions(Context, State, bounds, visit, packed)
    mode = "packed" if packed else "cached" if resume is None else "recomputed"
    deferred = deferred_functions(stack_capacity, mode)
    Stack = deferred.stack_type
    read_node = nodes.read
    is_leaf = nodes.is_leaf
    leaf_range = nodes.leaf_range
    metadata = nodes.metadata
    root_test = callbacks.test
    visit_leaf = callbacks.visit_leaf
    children = children_function(Context, State, packed, nodes.child_indices, read_node, callbacks.test)
    save = deferred.save
    pop = deferred.pop
    restore = (None if mode == "cached" else
               restore_function(Context, State, stack_capacity, mode, packed_nodes, resume))

    if group_leaves:
        @wp.func
        def walk(tree: DeviceTree, ctx: Context, state: wp.ref[State]):
            if tree.node_ray_lower.shape[0] == 0:
                return
            node, size = int(0), int(0)
            lower, upper, current = wp.static(read_node)(tree, 0)
            tag = bool(False)
            if wp.static(test_root):
                score, eligible, tag = wp.static(root_test)(ctx, tree, node, lower, upper, state)
                if wp.static(not packed):
                    if score != score:
                        state.failed = True
                    if state.failed or state.stop or score > state.limit:
                        return
                if not eligible:
                    return
            stack = Stack()
            while True:
                while True:
                    if wp.static(is_leaf)(tree, node, current):
                        break
                    left, right, lmeta, rmeta, a, b, accept_a, accept_b, tag_a, tag_b = wp.static(children)(tree, ctx, state, node, current)
                    if wp.static(not packed):
                        if state.stop:
                            return
                    if not (accept_a or accept_b):
                        if wp.static(packed_nodes):
                            # Dead ends reconverge before visiting leaves on CUDA.
                            current = wp.uint64(0)
                        break
                    swap = not accept_a
                    if wp.static(near_first):
                        swap = swap or (accept_b and b < a)
                    if accept_a and accept_b:
                        if size == wp.static(stack_capacity):
                            state.failed = True
                            return
                        wp.static(save)(stack, size, swap, left, right, lmeta, rmeta, a, b, tag_a, tag_b)
                        size += 1
                    current = wp.where(swap, rmeta, lmeta)
                    node, tag = wp.where(swap, right, left), wp.where(swap, tag_b, tag_a)
                if wp.static(is_leaf)(tree, node, current):
                    begin, end = wp.static(leaf_range)(tree, node, current)
                    wp.static(visit_leaf)(ctx, tree, node, begin, end, tag, state)
                    if state.failed or state.stop:
                        return
                if wp.static(mode == "cached"):
                    # Score before decoding; retain the walker-local pop loop
                    # so grouped CUDA lanes reconverge at the original break.
                    found = bool(False)
                    while size > 0:
                        size -= 1
                        entry = wp.static(pop)(stack, size)
                        score = entry_score(entry)
                        if score > state.limit:
                            continue
                        node, tag = entry_index(wp.uint32(entry)), entry_tag(wp.uint32(entry))
                        if score <= state.limit:
                            if wp.static(packed_nodes):
                                current = wp.static(metadata)(tree, node)
                            found = True
                            break
                    if not found:
                        return
                else:
                    found, size, node, tag, current = wp.static(restore)(tree, ctx, state, stack, size, node, tag, current)
                    if not found:
                        return
    else:
        @wp.func
        def walk(tree: DeviceTree, ctx: Context, state: wp.ref[State]):
            if tree.node_ray_lower.shape[0] == 0:
                return
            node, size = int(0), int(0)
            lower, upper, current = wp.static(read_node)(tree, 0)
            tag = bool(False)
            if wp.static(test_root):
                score, eligible, tag = wp.static(root_test)(ctx, tree, node, lower, upper, state)
                if wp.static(not packed):
                    if score != score:
                        state.failed = True
                    if state.failed or state.stop or score > state.limit:
                        return
                if not eligible:
                    return
            stack = Stack()
            while True:
                if wp.static(is_leaf)(tree, node, current):
                    begin, end = wp.static(leaf_range)(tree, node, current)
                    wp.static(visit_leaf)(ctx, tree, node, begin, end, tag, state)
                    if state.failed or state.stop:
                        return
                else:
                    left, right, lmeta, rmeta, a, b, accept_a, accept_b, tag_a, tag_b = wp.static(children)(tree, ctx, state, node, current)
                    if wp.static(not packed):
                        if state.stop:
                            return
                    if accept_a and accept_b:
                        if size == wp.static(stack_capacity):
                            state.failed = True
                            return
                        swap = bool(False)
                        if wp.static(near_first):
                            swap = b < a
                        wp.static(save)(stack, size, swap, left, right, lmeta, rmeta, a, b, tag_a, tag_b)
                        size += 1
                        current = wp.where(swap, rmeta, lmeta)
                        node, tag = wp.where(swap, right, left), wp.where(swap, tag_b, tag_a)
                        continue
                    if accept_a:
                        node, current, tag = left, lmeta, tag_a
                        continue
                    if accept_b:
                        node, current, tag = right, rmeta, tag_b
                        continue
                if wp.static(mode == "cached"):
                    found = bool(False)
                    while size > 0:
                        size -= 1
                        entry = wp.static(pop)(stack, size)
                        score = entry_score(entry)
                        if score > state.limit:
                            continue
                        node, tag = entry_index(wp.uint32(entry)), entry_tag(wp.uint32(entry))
                        if score <= state.limit:
                            if wp.static(packed_nodes):
                                current = wp.static(metadata)(tree, node)
                            found = True
                            break
                    if not found:
                        return
                else:
                    found, size, node, tag, current = wp.static(restore)(tree, ctx, state, stack, size, node, tag, current)
                    if not found:
                        return

    return walk
