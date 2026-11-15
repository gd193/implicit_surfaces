"""Native BVH ABI and build configuration shared by query kernels."""
import ctypes
from functools import lru_cache
from pathlib import Path

import warp as wp

from .._warp_tree import STACK_CAPACITY


_NATIVE_DIR = Path(__file__).resolve().parents[1] / "native"
_PREAMBLE = '#include "query.h"\n' + "\n".join(
    f"namespace bvh {{ using Stack_{bits}_{n} = Stack<uint{bits}_t,{n}>; }}"
    for bits in (32, 64) for n in range(1, STACK_CAPACITY + 1)
)
QUERY_BUILD_OPTIONS = wp.ModuleBuildOptions(
    extra_cpu_include_dirs=[_NATIVE_DIR], extra_cuda_include_dirs=[_NATIVE_DIR],
    extra_cpu_preamble=_PREAMBLE, extra_cuda_preamble=_PREAMBLE,
    extra_build_dependencies=sorted(_NATIVE_DIR.glob("*.h")),
)


@lru_cache(maxsize=None)
def stack_functions(capacity, wide):
    Scalar, Storage, bits = (wp.uint64, ctypes.c_uint64, 64) if wide else (wp.uint32, ctypes.c_uint32, 32)
    Stack = type(f"Stack_{bits}_{capacity}", (ctypes.Structure,),
                 {"_fields_": [("storage", Storage * capacity)]})
    wp.build_experimental.add_native_type(Stack, native_name=f"bvh::Stack_{bits}_{capacity}")

    @wp.func_native("bvh::push(stack, index, value);")
    def push(stack: wp.ref[Stack], index: int, value: Scalar): ...

    @wp.func_native("return bvh::pop(stack, index);")
    def pop(stack: wp.ref[Stack], index: int) -> Scalar: ...

    return Stack, push, pop


@wp.func_native("return bvh::pack_node(lower, upper);")
def pack_node(lower: wp.vec4, upper: wp.vec4) -> wp.uint64: ...


@wp.func_native("return bvh::node_is_leaf(node);")
def node_is_leaf(node: wp.uint64) -> bool: ...


@wp.func_native("return bvh::node_lower(node);")
def node_lower(node: wp.uint64) -> int: ...


@wp.func_native("return bvh::node_upper(node);")
def node_upper(node: wp.uint64) -> int: ...


@wp.func_native("return bvh::pack_entry(index, tag);")
def pack_entry(index: int, tag: bool) -> wp.uint32: ...


@wp.func_native("return bvh::entry_index(entry);")
def entry_index(entry: wp.uint32) -> int: ...


@wp.func_native("return bvh::entry_tag(entry);")
def entry_tag(entry: wp.uint32) -> bool: ...


@wp.func_native("return bvh::cache_entry(entry, score);")
def cache_entry(entry: wp.uint32, score: float) -> wp.uint64: ...


@wp.func_native("return bvh::entry_score(entry);")
def entry_score(entry: wp.uint64) -> float: ...
