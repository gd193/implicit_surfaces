"""Host-resolved query stages retain the public CPU result contract."""

import jax
import jax.numpy as jnp
import numpy as np

from surfaces.explicit.accelerators.bvh.warp import (
    RayOps, Segments, build_tree, make_classify, make_overlap, make_ray_query,
)
from tests.typing.warp_api.query_extension import Cursor, next_hit
from tests.typing.warp_api.real_extension import ExternalArrays, ExternalPoints


def test_ray_stage_selection_and_canonical_identity():
    primitive = ExternalPoints()
    stateful = RayOps(state_type=Cursor, next_hit=next_hit)
    all_hits = make_ray_query(primitive, "all", ops=stateful)
    assert all_hits is make_ray_query(primitive, "all", ops=stateful)
    assert all_hits.describe()["components"] == {
        "mode": "all", "hit_evaluation": "stateful", "output": "times_and_faces",
    }
    assert make_ray_query(Segments(2), "any").describe()["components"] == {
        "mode": "any", "hit_evaluation": "single", "output": "count_and_status",
    }


def test_stateful_all_capacity_and_exclusion():
    primitive = ExternalPoints()
    tree, data = build_tree(primitive, ExternalArrays(jnp.array([[0., 0., 0.]])))
    query = jax.jit(make_ray_query(primitive, "all", ops=RayOps(state_type=Cursor, next_hit=next_hit)),
                    static_argnames=("max_hits",))
    origin, direction = jnp.array([[-1., 0., 0.]]), jnp.array([[1., 0., 0.]])
    times, faces, count, status = query(tree, data, origin, direction, max_hits=1)
    np.testing.assert_array_equal(times, [[1.]])
    np.testing.assert_array_equal(faces, [[0]])
    np.testing.assert_array_equal(count, [2])
    np.testing.assert_array_equal(status, [-2])
    excluded = jax.jit(make_ray_query(primitive, "all", ops=RayOps(state_type=Cursor, next_hit=next_hit), exclude=True),
                       static_argnames=("max_hits",))
    times, faces, count, status = excluded(tree, data, origin, direction, exclude_face=0, max_hits=1)
    np.testing.assert_array_equal(faces, [[-1]])
    np.testing.assert_array_equal(count, [0])
    np.testing.assert_array_equal(status, [0])


def test_small_query_composition_descriptions():
    primitive = Segments(2)
    assert make_overlap(primitive).describe()["components"] == {"intersection": "primitive_aabb"}
    assert make_classify(primitive).describe()["components"] == {"event": "policy"}
