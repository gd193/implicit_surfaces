"""Public, statically bound traversal over the shared BVH walker."""
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import warp as wp

from ..._warp_tree import STACK_CAPACITY
from ..contracts import make_struct
from ..launch import array_kernel
from .scheduling import make_walk


def make_traversal_state(name: str, extra_fields: dict[str, Any] | None = None):
    """Create a Warp state with walker fields and caller-owned typed fields."""
    fields = dict(extra_fields or {})
    reserved = {"limit", "failed", "stop"}
    if reserved.intersection(fields):
        raise ValueError("limit, failed and stop are reserved traversal fields")
    return make_struct(name, {"limit": wp.float32, "failed": wp.bool,
                              "stop": wp.bool, **fields})


@dataclass(frozen=True)
class TraversalPolicy:
    """A device walk plus native dependencies for caller-owned kernels."""
    device: wp.Function
    state_type: Any
    _device_kernel: wp.Kernel


@lru_cache(maxsize=None)
def make_traversal(Context, State, bounds, visit, *, stack_capacity=STACK_CAPACITY,
                   near_first=True, packed_nodes=True, cache_bounds=True,
                   group_leaves=False, test_root=True, resume=None):
    """Bind indexed bounds/leaf callbacks into one fused Warp walk.

    ``State`` has ``limit``, ``failed`` and ``stop`` fields. The caller owns
    initialization and reads failure/stop after ``device(tree, ctx, state)``.
    Scores and ``limit`` share units; bounds must conservatively underbound
    eligible primitives. A bound equal to limit remains eligible.
    """
    fields = getattr(State, "vars", {})
    for name, dtype in (("limit", wp.float32), ("failed", wp.bool), ("stop", wp.bool)):
        if name not in fields or fields[name].type != dtype:
            raise TypeError(f"traversal state requires {name!r} with dtype {dtype}")
    for callback in (bounds, visit, resume):
        if callback is not None and not isinstance(callback, wp.Function):
            raise TypeError("traversal callbacks must be decorated Warp functions")
    if not cache_bounds and resume is None:
        raise ValueError("resume is required when cache_bounds=False")
    if cache_bounds and resume is not None:
        raise ValueError("resume requires cache_bounds=False")
    walk = make_walk(Context, State, bounds, visit,
                     stack_capacity=stack_capacity, near_first=near_first,
                     packed_nodes=packed_nodes, test_root=test_root,
                     resume=resume, group_leaves=group_leaves)
    # configure_query_kernel reads the build options from a query kernel.
    # A never-launched carrier keeps this contract without changing launch.py.
    dependency = array_kernel("traversal_dependency", [("unused", wp.array(dtype=wp.int32))],
                              ["return"], {})
    return TraversalPolicy(walk, State, dependency)
