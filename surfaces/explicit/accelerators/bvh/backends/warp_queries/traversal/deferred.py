"""Host-selected deferred stack encodings for BVH traversal."""
from dataclasses import dataclass
from functools import lru_cache

import warp as wp

from ..._warp_tree import DeviceTree
from ..io.nodes import node_functions
from ..native import cache_entry, entry_index, entry_tag, pack_entry, stack_functions


@dataclass(frozen=True)
class DeferredFunctions:
    stack_type: type
    push: wp.Function
    pop: wp.Function
    save: wp.Function
    decode: wp.Function | None


@lru_cache(maxsize=None)
def deferred_functions(capacity: int, mode: str) -> DeferredFunctions:
    """Modes are packed metadata, indexed cached score, indexed recomputed score."""
    if mode not in ("packed", "cached", "recomputed"):
        raise ValueError(f"invalid deferred mode {mode!r}")
    Stack, push, pop = stack_functions(capacity, mode != "recomputed")

    if mode == "packed":
        @wp.func
        def save(stack: wp.ref[Stack], size: int, swap: bool, left: int, right: int,
                 lmeta: wp.uint64, rmeta: wp.uint64, a: float, b: float, tag_a: bool, tag_b: bool):
            wp.static(push)(stack, size, wp.where(swap, lmeta, rmeta))

        return DeferredFunctions(Stack, push, pop, save, None)

    if mode == "cached":
        @wp.func
        def save(stack: wp.ref[Stack], size: int, swap: bool, left: int, right: int,
                 lmeta: wp.uint64, rmeta: wp.uint64, a: float, b: float, tag_a: bool, tag_b: bool):
            deferred = wp.where(swap, pack_entry(left, tag_a), pack_entry(right, tag_b))
            wp.static(push)(stack, size, cache_entry(deferred, wp.where(swap, a, b)))

        return DeferredFunctions(Stack, push, pop, save, None)

    else:
        @wp.func
        def save(stack: wp.ref[Stack], size: int, swap: bool, left: int, right: int,
                 lmeta: wp.uint64, rmeta: wp.uint64, a: float, b: float, tag_a: bool, tag_b: bool):
            wp.static(push)(stack, size, wp.where(swap, pack_entry(left, tag_a), pack_entry(right, tag_b)))

        @wp.func
        def decode(entry: wp.uint32):
            return entry_index(entry), entry_tag(entry)

    return DeferredFunctions(Stack, push, pop, save, decode)


@lru_cache(maxsize=None)
def restore_function(Context, State, capacity: int, mode: str,
                     packed_nodes: bool = False, resume=None) -> wp.Function:
    """Pop until a viable node is found, preserving each mode's load order."""
    if mode == "packed":
        if resume is not None:
            raise ValueError("packed deferred entries cannot recompute bounds")
        Stack, pop = deferred_functions(capacity, mode).stack_type, deferred_functions(capacity, mode).pop

        @wp.func
        def restore(tree: DeviceTree, ctx: Context, state: wp.ref[State],
                    stack: wp.ref[Stack], size: int, node: int, tag: bool, current: wp.uint64):
            if size == 0:
                return False, size, node, tag, current
            size -= 1
            current = wp.static(pop)(stack, size)
            return True, size, node, tag, current

        return restore

    if mode == "recomputed":
        if resume is None:
            raise ValueError("recomputed entries require a resume callback")
        bundle = deferred_functions(capacity, mode)
        Stack, pop, decode = bundle.stack_type, bundle.pop, bundle.decode
        read_node = node_functions(packed_nodes).read

        @wp.func
        def restore(tree: DeviceTree, ctx: Context, state: wp.ref[State],
                    stack: wp.ref[Stack], size: int, node: int, tag: bool, current: wp.uint64):
            while size > 0:
                size -= 1
                entry = wp.static(pop)(stack, size)
                node, tag = wp.static(decode)(entry)
                lower, upper, current = wp.static(read_node)(tree, node)
                score = wp.static(resume)(ctx, tree, node, lower, upper)
                if score != score:
                    state.failed = True
                    return False, size, node, tag, current
                if score <= state.limit:
                    return True, size, node, tag, current
            return False, size, node, tag, current

        return restore

    raise ValueError(f"invalid deferred mode {mode!r}")
