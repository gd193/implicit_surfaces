"""External structural primitives and typed callbacks use the public API."""
from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import warp as wp

from sengi.bvh.warp import (
    Angular, Euclidean, NearestCallbacks, Points, Primitive,
    as_warp_function, build_tree, make_nearest, warp_func,
)


from tests.typing.warp_api.real_extension import (
    ExternalArrays, ExternalData, ExternalPoints, external_bound,
    external_bounds, external_evaluate, external_load,
)


@pytest.mark.parametrize("descriptor", [False, True])
def test_external_primitive_callback_parity_and_canonical_cache(descriptor):
    primitive = ExternalPoints()
    if descriptor:
        primitive = Primitive(
            dimension=3, point_type=wp.vec3, data_type=ExternalData,
            load=external_load, bounds_of_item=external_bounds,
            host_bounds=primitive.bounds, host_pack=primitive.pack,
        )
        with pytest.raises(FrozenInstanceError):
            primitive.dimension = 2
    policy = NearestCallbacks(external_evaluate, external_bound)
    explicit = make_nearest(primitive, policy)
    sugar = make_nearest(primitive, evaluate=external_evaluate, lower_bound=external_bound)
    assert explicit is sugar
    assert explicit.kernel is sugar.kernel
    assert policy.bind(primitive).node_bound is external_bound
    with pytest.raises(FrozenInstanceError):
        policy.evaluate = external_evaluate
    positions = jnp.array([[0., 1., 0.], [5., 0., 0.], [2., 2., 2.]], dtype=jnp.float32)
    queries = jnp.array([[[0., 0., 0.], [5., 2., 0.], [2., 2., 5.]]], dtype=jnp.float32)
    tree, data = build_tree(primitive, ExternalArrays(positions))
    builtin = Points(3)
    builtin_tree, builtin_data = build_tree(builtin, builtin.data(positions))
    run = jax.jit(explicit)
    assert run.lower(tree, data, queries).as_text().count("stablehlo.custom_call") == 1
    actual = run(tree, data, queries)
    expected = jax.jit(make_nearest(builtin, Euclidean()))(builtin_tree, builtin_data, queries)
    np.testing.assert_allclose(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], np.square(expected[1]))
    np.testing.assert_array_equal(actual[2], expected[2])
    np.testing.assert_array_equal(actual[3], expected[3])
    np.testing.assert_allclose(actual[1], [[1., 4., 9.]])


@pytest.mark.parametrize("policy,message", [(Euclidean(), "Euclidean"), (Angular(), "endpoint")])
def test_external_primitive_reports_missing_capabilities(policy, message):
    with pytest.raises(TypeError, match=message):
        make_nearest(ExternalPoints(), policy)


def test_typed_bridge_keeps_warp_function_identity(monkeypatch):
    seen = []
    original = wp.func

    def record(function):
        result = original(function)
        seen.append(result)
        return result

    monkeypatch.setattr(wp, "func", record)

    @warp_func
    def identity(value: float) -> float:
        return value

    assert len(seen) == 1
    assert identity is seen[0]
    assert isinstance(identity, wp.Function)
    assert as_warp_function(identity) is identity
    with pytest.raises(TypeError, match="Warp Function"):
        as_warp_function(lambda value: value)


def test_typed_builtin_evaluator_compiles_and_uses_raw_scores():
    from tests.typing.warp_api.real_extension import builtin_evaluate

    primitive = Points(3)
    tree, data = build_tree(primitive, primitive.data([[0., 0., 0.], [8., 0., 0.]]))
    callbacks = NearestCallbacks(builtin_evaluate, external_bound)
    explicit = make_nearest(primitive, callbacks)
    assert explicit.kernel is make_nearest(
        primitive, evaluate=builtin_evaluate, lower_bound=external_bound).kernel
    point, score, index, status = jax.jit(explicit)(tree, data, jnp.array([[2., 0., 0.], [8., 3., 0.]]))
    np.testing.assert_allclose(point, [[0., 0., 0.], [8., 0., 0.]])
    np.testing.assert_allclose(score, [4., 9.])
    np.testing.assert_array_equal(index, [0, 1])
    np.testing.assert_array_equal(status, [1, 1])
