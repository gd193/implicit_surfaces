"""Each backend agrees with one reference; oracle tests cover individual correctness."""
import jax
import numpy as np
import pytest

from surfaces.explicit.accelerators import list_backends
from ._harness import build_accelerator, skip_if_primitive_math_missing


_BACKENDS = list_backends("bvh")
_PAIRS = [(_BACKENDS[0], name) for name in _BACKENDS[1:]]


@pytest.mark.parametrize("a,b", _PAIRS, ids=[f"{a}|{b}" for a, b in _PAIRS])
@pytest.mark.parametrize("query", ["closest_point", "first_hit", "aabb_overlap"])
def test_bvh_backends_agree(tri_fixture_sphere_320, a, b, query):
    fixture = tri_fixture_sphere_320
    callback = {"closest_point": "closest_point_on_primitive", "first_hit": "ray_primitive_intersect",
                "aabb_overlap": "primitive_aabb"}[query]
    skip_if_primitive_math_missing(callback, fixture.primitives)
    args = {"closest_point": (fixture.query_points,),
            "first_hit": (fixture.query_ray_origins, fixture.query_ray_dirs, fixture.query_ray_max_t),
            "aabb_overlap": (fixture.query_aabb_lo, fixture.query_aabb_hi)}[query]
    outputs = [jax.vmap(getattr(build_accelerator("bvh", name, fixture.primitives), query))(*args)
               for name in (a, b)]
    if query == "closest_point":
        np.testing.assert_allclose(outputs[0][1], outputs[1][1], rtol=1e-4, atol=1e-5)
    elif query == "first_hit":
        np.testing.assert_array_equal(outputs[0][2], outputs[1][2])
        mask = np.asarray(outputs[0][2])
        np.testing.assert_allclose(np.asarray(outputs[0][0])[mask], np.asarray(outputs[1][0])[mask], rtol=1e-4, atol=1e-5)
    else:
        np.testing.assert_array_equal(*outputs)
