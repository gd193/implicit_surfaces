"""Shared ordered BVH traversal; callbacks specialize at compilation."""
import ctypes
from functools import lru_cache
from typing import Any

import warp as wp

from ._warp_tree import DeviceTree, STACK_CAPACITY


_PREAMBLE = r"""
namespace bvh {
CUDA_CALLABLE inline float quiet_nan() {
#if defined(__CUDA_ARCH__)
    return __int_as_float(0x7fc00000);
#else
    const uint32_t bits = 0x7fc00000u;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
#endif
}
template<typename T, int N> union Stack {
    T data[N];
    CUDA_CALLABLE Stack() {}
};
}
#undef NAN
#define NAN (::bvh::quiet_nan())
""" + "\n".join(f"namespace bvh {{ using Stack_{bits}_{n} = Stack<uint{bits}_t,{n}>; }}"
                 for bits in (32, 64) for n in range(1, STACK_CAPACITY+1))
# A union copies storage without reading unwritten slots; its first store activates the array.
# Warp's NAN macro divides doubles; its vector constructors also zero-fill stacks.
QUERY_BUILD_OPTIONS = wp.ModuleBuildOptions(extra_cpu_preamble=_PREAMBLE, extra_cuda_preamble=_PREAMBLE)


@lru_cache(maxsize=None)
def stack_functions(capacity, wide):
    Scalar, Storage, bits = (wp.uint64, ctypes.c_uint64, 64) if wide else (wp.uint32, ctypes.c_uint32, 32)
    Stack = type(f"Stack_{bits}_{capacity}", (ctypes.Structure,),
                 {"_fields_": [("storage", Storage*capacity)]})
    wp.build_experimental.add_native_type(Stack, native_name=f"bvh::Stack_{bits}_{capacity}")

    @wp.func_native("stack.data[index] = value;")
    def push(stack: wp.ref[Stack], index: int, value: Scalar): ...

    @wp.func_native("return stack.data[index];")
    def pop(stack: wp.ref[Stack], index: int) -> Scalar: ...

    return Stack, push, pop


@wp.func_native("return values.data[index];")
def load_array(values: Any, index: int):
    return values[index]


@wp.func_native("values.data[index] = value;")
def store_array(values: Any, index: Any, value: Any): ...

# Packed layout adapted from NVIDIA Warp bvh.h; Apache-2.0.
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
@wp.func_native(r"""
    #if defined(__CUDA_ARCH__)
    const float4 value=__ldg(reinterpret_cast<const float4*>(values.data+index));
    return wp::vec4(value.x,value.y,value.z,value.w);
    #else
    return values.data[index];
    #endif
""")
def load_node(values: wp.array(dtype=wp.vec4), index: int) -> wp.vec4: ...


@wp.func_native(r"""
    uint32_t a,b;
    memcpy(&a,&lower.c[3],4); memcpy(&b,&upper.c[3],4);
    return uint64_t(a) | (uint64_t(b)<<32);
""")
def pack_node(lower: wp.vec4, upper: wp.vec4) -> wp.uint64: ...


@wp.func
def load_metadata(lower: wp.array(dtype=wp.vec4), upper: wp.array(dtype=wp.vec4), index: int):
    return pack_node(load_array(lower, index), load_array(upper, index))


@wp.func_native("return (node & 0x80000000ull) != 0;")
def node_is_leaf(node: wp.uint64) -> bool: ...


@wp.func_native("return static_cast<int>(node & 0x7fffffffull);")
def node_lower(node: wp.uint64) -> int: ...


@wp.func_native("return static_cast<int>((node >> 32) & 0x7fffffffull);")
def node_upper(node: wp.uint64) -> int: ...


@wp.func_native("return uint32_t(index) | (uint32_t(tag)<<31);")
def pack_entry(index: int, tag: bool) -> wp.uint32: ...


@wp.func_native("return static_cast<int>(entry & 0x7fffffffull);")
def entry_index(entry: wp.uint32) -> int: ...


@wp.func_native("return (entry & 0x80000000ull) != 0;")
def entry_tag(entry: wp.uint32) -> bool: ...


@wp.func_native("uint32_t bits; memcpy(&bits,&score,4); return (uint64_t(bits)<<32) | entry;")
def cache_entry(entry: wp.uint32, score: float) -> wp.uint64: ...


@wp.func_native("uint32_t bits=uint32_t(entry>>32); float value; memcpy(&value,&bits,4); return value;")
def entry_score(entry: wp.uint64) -> float: ...


@lru_cache(maxsize=None)
def make_walk(Context, State, bounds, visit, *, stack_capacity=STACK_CAPACITY,
              near_first=True, packed=False, packed_nodes=False, test_root=True, resume=None, group_leaves=False):
    """Packed queries visit ranges; indexed queries also receive node metadata."""
    if not 1 <= stack_capacity <= STACK_CAPACITY:
        raise ValueError("invalid stack capacity")
    packed_nodes = packed or packed_nodes
    Stack, push, pop = stack_functions(stack_capacity, packed or resume is None)

    @wp.func
    def read_node(tree: DeviceTree, node: int):
        if wp.static(packed_nodes):
            lo, hi = load_node(tree.node_ray_lower, node), load_node(tree.node_ray_upper, node)
            return wp.vec3(lo[0], lo[1], lo[2]), wp.vec3(hi[0], hi[1], hi[2]), pack_node(lo, hi)
        else:
            return load_array(tree.node_lower, node), load_array(tree.node_upper, node), wp.uint64(0)

    @wp.func
    def test(ctx: Context, tree: DeviceTree, node: int, lower: wp.vec3, upper: wp.vec3, state: wp.ref[State]):
        if wp.static(packed):
            return wp.static(bounds)(ctx, lower, upper, state)
        else:
            return wp.static(bounds)(ctx, tree, node, lower, upper, state)

    @wp.func
    def children(tree: DeviceTree, ctx: Context, state: wp.ref[State], node: int, current: wp.uint64):
        left, right = int(0), int(0)
        if wp.static(packed_nodes):
            left, right = node_lower(current), node_upper(current)
        else:
            left, right = load_array(tree.node_left, node), load_array(tree.node_right, node)
        # Issue both child loads before callback arithmetic.
        ll, lu, lmeta = read_node(tree, left)
        rl, ru, rmeta = read_node(tree, right)
        a, accept_a, tag_a = test(ctx, tree, left, ll, lu, state)
        b, accept_b, tag_b = test(ctx, tree, right, rl, ru, state)
        if wp.static(not packed):
            if a != a or b != b:
                state.failed = True
            accept_a = accept_a and a <= state.limit
            accept_b = accept_b and b <= state.limit
        return left, right, lmeta, rmeta, a, b, accept_a, accept_b, tag_a, tag_b

    @wp.func
    def store_deferred(stack: wp.ref[Stack], size: int, swap: bool, left: int, right: int,
                       lmeta: wp.uint64, rmeta: wp.uint64, a: float, b: float, tag_a: bool, tag_b: bool):
        if wp.static(packed):
            wp.static(push)(stack, size, wp.where(swap, lmeta, rmeta))
        else:
            deferred = wp.where(swap, pack_entry(left, tag_a), pack_entry(right, tag_b))
            if wp.static(resume is None):
                wp.static(push)(stack, size, cache_entry(deferred, wp.where(swap, a, b)))
            else:
                wp.static(push)(stack, size, deferred)

    @wp.func
    def walk(tree: DeviceTree, ctx: Context, state: wp.ref[State]):
        if tree.node_ray_lower.shape[0] == 0:
            return
        node, size = int(0), int(0)
        lower, upper, current = read_node(tree, 0)
        tag = bool(False)
        if wp.static(test_root):
            score, eligible, tag = test(ctx, tree, node, lower, upper, state)
            if wp.static(not packed):
                if score != score:
                    state.failed = True
                if state.failed or state.stop or score > state.limit:
                    return
            if not eligible:
                return
        stack = Stack()
        while True:
            if wp.static(group_leaves):
                while True:
                    if wp.static(packed_nodes):
                        if node_is_leaf(current):
                            break
                    else:
                        if load_array(tree.node_left, node) < 0:
                            break
                    left, right, lmeta, rmeta, a, b, accept_a, accept_b, tag_a, tag_b = children(tree, ctx, state, node, current)
                    if wp.static(not packed):
                        if state.stop:
                            return
                    if not (accept_a or accept_b):
                        if wp.static(packed_nodes):
                            # Clear dead ends so CUDA reconverges descent before visiting leaves.
                            current = wp.uint64(0)
                        break
                    swap = not accept_a
                    if wp.static(near_first):
                        swap = swap or (accept_b and b < a)
                    if accept_a and accept_b:
                        if size == wp.static(stack_capacity):
                            state.failed = True
                            return
                        store_deferred(stack, size, swap, left, right, lmeta, rmeta, a, b, tag_a, tag_b)
                        size += 1
                    current = wp.where(swap, rmeta, lmeta)
                    node, tag = wp.where(swap, right, left), wp.where(swap, tag_b, tag_a)
            leaf = bool(False)
            if wp.static(packed_nodes):
                leaf = node_is_leaf(current)
            else:
                leaf = load_array(tree.node_left, node) < 0
            if leaf:
                if wp.static(packed_nodes):
                    begin, end = node_lower(current), node_upper(current)
                else:
                    begin = load_array(tree.node_leaf_offset, node)
                    end = begin+load_array(tree.node_leaf_count, node)
                if wp.static(packed):
                    wp.static(visit)(ctx, tree, begin, end, state)
                else:
                    wp.static(visit)(ctx, tree, node, begin, end, tag, state)
                if state.failed or state.stop:
                    return
            elif wp.static(not group_leaves):
                left, right, lmeta, rmeta, a, b, accept_a, accept_b, tag_a, tag_b = children(tree, ctx, state, node, current)
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
                    store_deferred(stack, size, swap, left, right, lmeta, rmeta, a, b, tag_a, tag_b)
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
            if wp.static(packed):
                if size == 0:
                    return
                size -= 1
                current = wp.static(pop)(stack, size)
            else:
                found = bool(False)
                while size > 0:
                    size -= 1
                    entry = wp.static(pop)(stack, size)
                    score = float(0.0)
                    if wp.static(resume is None):
                        score = entry_score(entry)
                        if score > state.limit:
                            continue
                    node, tag = entry_index(wp.uint32(entry)), entry_tag(wp.uint32(entry))
                    if wp.static(resume is not None):
                        lower, upper, current = read_node(tree, node)
                        score = wp.static(resume)(ctx, tree, node, lower, upper)
                        if score != score:
                            state.failed = True
                            return
                    if score <= state.limit:
                        if wp.static(packed_nodes and resume is None):
                            current = load_metadata(tree.node_ray_lower, tree.node_ray_upper, node)
                        found = True
                        break
                if not found:
                    return

    return walk
