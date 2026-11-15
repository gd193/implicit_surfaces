"""Static kNN candidate storage and its matching native ABI."""
import ctypes
from dataclasses import dataclass
from functools import lru_cache

import warp as wp

from ..native import _PREAMBLE, _NATIVE_DIR


REGISTER_LIMIT = 64


@dataclass(frozen=True)
class CandidateFunctions:
    storage: type
    insert: wp.Function
    add: wp.Function
    limit: wp.Function
    sort: wp.Function
    score: wp.Function
    item_id: wp.Function
    count: wp.Function
    slot: wp.Function | None


@lru_cache(maxsize=None)
def candidate_functions(k: int, track_slots: bool = False) -> CandidateFunctions:
    fields = ([('scores', ctypes.c_float*k), ('ids', ctypes.c_int32*k)] if k <= REGISTER_LIMIT
              else [('entries', ctypes.c_uint64*k)])
    if track_slots:
        fields = [('slots', ctypes.c_int32*k), *fields]
    name = f"KnnCandidates_{k}" + ("_points" if track_slots else "")
    storage = type(name, (ctypes.Structure,),
                   {"_fields_": [*fields, ("size", ctypes.c_int32)]})
    wp.build_experimental.add_native_type(storage, native_name=f"bvh::{name}")

    if track_slots:
        @wp.func_native("bvh::knn_insert(candidates, score, index, slot);")
        def insert(candidates: wp.ref[storage], score: float, index: int, slot: int): ...

        @wp.func
        def add(candidates: wp.ref[storage], score: float, index: int, slot: int):
            wp.static(insert)(candidates, score, index, slot)
    else:
        @wp.func_native("bvh::knn_insert(candidates, score, index);")
        def insert(candidates: wp.ref[storage], score: float, index: int): ...

        @wp.func
        def add(candidates: wp.ref[storage], score: float, index: int, slot: int):
            wp.static(insert)(candidates, score, index)

    @wp.func_native("return bvh::knn_limit(candidates, initial);")
    def limit(candidates: wp.ref[storage], initial: float) -> float: ...

    @wp.func_native("bvh::knn_sort(candidates);")
    def sort(candidates: wp.ref[storage]): ...

    @wp.func_native("return bvh::knn_score(candidates, index);")
    def score(candidates: wp.ref[storage], index: int) -> float: ...

    @wp.func_native("return bvh::knn_id(candidates, index);")
    def item_id(candidates: wp.ref[storage], index: int) -> int: ...

    @wp.func_native("return candidates.size;")
    def count(candidates: wp.ref[storage]) -> int: ...

    slot = None
    if track_slots:
        @wp.func_native("return candidates.slots[index];")
        def slot(candidates: wp.ref[storage], index: int) -> int: ...

    return CandidateFunctions(storage, insert, add, limit, sort, score, item_id, count, slot)


def configure_candidate_module(module, candidates: CandidateFunctions, k: int,
                               track_slots: bool) -> None:
    """Register the exact C++ representation used by the Warp struct."""
    small = "true" if k <= REGISTER_LIMIT else "false"
    tracked = "true" if track_slots else "false"
    name = candidates.storage.__name__
    preamble = (_PREAMBLE + '\n#include "knn_candidates.h"\n'
                + f"namespace bvh {{ using {name} = KnnCandidates<{k}, {small}, {tracked}>; }}\n"
                + f"static_assert(sizeof(bvh::{name}) == {ctypes.sizeof(candidates.storage)});\n"
                + f"static_assert(alignof(bvh::{name}) == {ctypes.alignment(candidates.storage)});\n")
    options = wp.ModuleBuildOptions(
        extra_cpu_include_dirs=[_NATIVE_DIR], extra_cuda_include_dirs=[_NATIVE_DIR],
        extra_cpu_preamble=preamble, extra_cuda_preamble=preamble,
        extra_build_dependencies=sorted(_NATIVE_DIR.glob("*.h")))
    wp.set_module_options({"extra_build_options": options,
                           "max_unroll": max(16, min(k, REGISTER_LIMIT))}, module=module)
