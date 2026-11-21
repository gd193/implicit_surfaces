"""Host selection of device traversal operations."""
import pytest
import warp as wp

from sengi.bvh.backends._warp_tree import DeviceTree
from sengi.bvh.backends.warp_queries.io.nodes import node_functions
from sengi.bvh.backends.warp_queries.traversal.deferred import deferred_functions
from sengi.bvh.backends.warp_queries.traversal.scheduling import make_walk


@wp.struct
class Context:
    point: wp.vec3


@wp.struct
class State:
    limit: float
    failed: bool
    stop: bool


@wp.func
def bounds(ctx: Context, lower: wp.vec3, upper: wp.vec3, state: wp.ref[State]):
    return 0.0, True, False


@wp.func
def visit(ctx: Context, tree: DeviceTree, begin: int, end: int, state: wp.ref[State]):
    return


def test_host_bundle_cache_and_distinct_stack_layouts():
    assert node_functions(True) is node_functions(True)
    assert node_functions(False) is node_functions(False)
    assert node_functions(True).read is not node_functions(False).read

    packed = deferred_functions(8, "packed")
    cached = deferred_functions(8, "cached")
    recomputed = deferred_functions(8, "recomputed")
    assert packed is deferred_functions(8, "packed")
    assert packed.stack_type is cached.stack_type
    assert recomputed.stack_type is not cached.stack_type
    assert packed.decode is cached.decode is None
    assert cached.decode is not recomputed.decode


def test_equivalent_packed_selections_share_walk():
    implicit = make_walk(Context, State, bounds, visit, stack_capacity=8, packed=True)
    explicit = make_walk(Context, State, bounds, visit, stack_capacity=8,
                         packed=True, packed_nodes=True)
    assert implicit is explicit
    assert make_walk(Context, State, bounds, visit, stack_capacity=8,
                     packed=True, group_leaves=True) is not implicit


def test_invalid_combinations_rejected_on_host():
    with pytest.raises(ValueError, match="stack capacity"):
        make_walk(Context, State, bounds, visit, stack_capacity=0)
    with pytest.raises(ValueError, match="cannot resume"):
        make_walk(Context, State, bounds, visit, packed=True, resume=bounds)
    with pytest.raises(ValueError, match="deferred mode"):
        deferred_functions(8, "unknown")
