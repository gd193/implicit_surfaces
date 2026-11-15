"""Public typed policies compile unchanged as real CPU Warp extensions."""
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from surfaces.explicit.accelerators.bvh.warp import (
    ClassificationOps, MeasureOps, RayOps, RegionOps, build_tree,
    make_boundary_sample, make_classify, make_ray_query,
)
from tests.typing.warp_api.real_extension import ExternalArrays, ExternalPoints
from tests.typing.warp_api.query_extension import (
    Cursor, ExternalMetadata, MetadataArrays, classification_event, intersect,
    metadata_valid, next_hit, region_contains, region_overlaps, region_valid, sample,
)


def _tree():
    primitive = ExternalPoints()
    return primitive, *build_tree(primitive, ExternalArrays(jnp.array([[0., 0., 0.]])))


def _fused(query, *args, **kwargs):
    compiled = jax.jit(query, static_argnames=tuple(kwargs))
    assert compiled.lower(*args, **kwargs).as_text().count("stablehlo.custom_call") == 1
    return compiled(*args, **kwargs)


@pytest.mark.parametrize("mode", ["first", "any", "count", "all"])
@pytest.mark.parametrize("stateful", [False, True])
def test_external_ray_policy_modes_compile_and_fuse(mode, stateful):
    primitive, tree, data = _tree()
    ops = RayOps(state_type=Cursor, next_hit=next_hit) if stateful else RayOps(intersect=intersect)
    query = make_ray_query(primitive, mode, ops=ops)
    result = _fused(query, tree, data, jnp.array([[-1., 0., 0.]]),
                    jnp.array([[1., 0., 0.]]), **({"max_hits": 3} if mode == "all" else {}))
    count = 2 if stateful else 1
    if mode == "first":
        np.testing.assert_allclose(result[0], [1.])
        np.testing.assert_array_equal(result[1], [0])
        np.testing.assert_array_equal(result[2], [True])
    elif mode == "any":
        np.testing.assert_array_equal(result[0], [True])
    elif mode == "count":
        np.testing.assert_array_equal(result[0], [count])
    else:
        np.testing.assert_allclose(result[0][0, :count], [1., 2.] if stateful else [1.])
        np.testing.assert_array_equal(result[1][0, :count], 0)
        np.testing.assert_array_equal(result[2], [count])
    np.testing.assert_array_equal(result[-1], [1])


def test_external_classification_policy_compiles_and_fuses():
    primitive, tree, data = _tree()
    policy = ClassificationOps(ExternalMetadata, metadata_valid, classification_event)
    query = make_classify(primitive, policy, metadata_host_type=MetadataArrays)
    assert query is make_classify(primitive, policy)
    assert query.kernel is make_classify(primitive, policy).kernel
    metadata = MetadataArrays(jnp.array([1], dtype=jnp.int32))
    sign, status = _fused(query, tree, data, metadata, jnp.array([[-1., 0., 0.], [1., 0., 0.]]))
    np.testing.assert_array_equal(sign, [-1., 1.])
    np.testing.assert_array_equal(status, [1, 1])
    sign, status = jax.jit(query)(tree, data, MetadataArrays(jnp.array([0], dtype=jnp.int32)), jnp.zeros((1, 3)))
    assert np.isnan(sign).all()
    np.testing.assert_array_equal(status, [-1])


def test_external_sample_region_policies_compile_and_fuse():
    import warp as wp
    primitive, tree, data = _tree()
    measure = MeasureOps(sample)
    region = RegionOps(wp.float32, (), region_valid, region_overlaps, region_contains)
    query = make_boundary_sample(primitive, measure, region)
    SamplingArrays = namedtuple("SamplingArrays", "primitive_weight node_weight")
    metadata = SamplingArrays(jnp.ones((1,)), jnp.ones(tree.node_left.shape))
    point, normal, pdf, index, status = _fused(query, tree, data, metadata,
        jnp.array([1., -1.]), 0., jnp.full((2, 3), .5))
    np.testing.assert_array_equal(point[0], [0., 0., 0.])
    np.testing.assert_array_equal(normal[0], [0., 1., 0.])
    np.testing.assert_array_equal(pdf, [1., 0.])
    np.testing.assert_array_equal(index, [0, -1])
    np.testing.assert_array_equal(status, [1, -1])


def test_external_robin_policy_compiles_and_fuses():
    import warp as wp
    from surfaces.explicit.accelerators.bvh.warp import (
        BoundaryCore, RobinData, RobinOps, SilhouetteMeta, make_robin_radius,
    )
    from surfaces.explicit.accelerators.bvh.backends.warp_queries.queries.boundary import prepare_robin_coefficients
    from tests.typing.warp_api.query_extension import (
        ExternalRobinMetric, radius_to_score, robin_feature, robin_node, robin_primitive,
        robin_valid, score_to_radius,
    )
    primitive, tree, data = _tree()
    policy = RobinOps(wp.vec3, (3,), robin_valid, robin_node, robin_primitive, robin_feature,
        core_type=BoundaryCore, feature_type=SilhouetteMeta,
        to_score=radius_to_score, to_distance=score_to_radius)
    # No boundary features: only the external point contributes a radius.
    # These are the actual metadata schemas expected by the fused kernel.
    def empty_payload(schema):
        fields = {}
        for name, field in schema.vars.items():
            dtype = field.type.dtype
            components = {wp.vec2: 2, wp.vec3: 3}.get(dtype, 0)
            shape = (tree.node_left.shape[0], components) if components else (tree.node_left.shape[0],) * field.type.ndim
            fields[name] = jnp.zeros(shape,
                                    dtype=jnp.int32 if dtype is wp.int32 else jnp.float32)
        return namedtuple(schema.key + "Arrays", fields)(**fields)
    metadata = RobinData(empty_payload(BoundaryCore), empty_payload(SilhouetteMeta),
        prepare_robin_coefficients(tree, jnp.inf))
    radius, status = _fused(make_robin_radius(primitive, policy), tree, data, metadata,
                           jnp.array([[0., 2., 0.]]), 10.)
    np.testing.assert_allclose(radius, [2.])
    np.testing.assert_array_equal(status, [1])
    metric_result = _fused(make_robin_radius(primitive, metric=ExternalRobinMetric()),
                           tree, data, metadata, jnp.array([[0., 2., 0.]]), 10.)
    np.testing.assert_array_equal(metric_result[0], radius)
    np.testing.assert_array_equal(metric_result[1], status)


def test_external_silhouette_and_star_adapter_compile():
    from surfaces.explicit.accelerators.bvh.warp import (
        NearestCallbacks, build_boundary, make_silhouette, make_star_radius,
    )
    from tests.typing.warp_api.query_extension import external_adapter
    from tests.typing.warp_api.real_extension import external_bound, external_evaluate
    primitive = ExternalPoints()
    tree, data, boundary = build_boundary(primitive,
        ExternalArrays(jnp.array([[0., 0., 0.]])), adapter=external_adapter)
    custom = make_silhouette(primitive, NearestCallbacks(external_evaluate, external_bound),
                             adapter=external_adapter)
    point, score, index, status = _fused(custom, tree, data, boundary, jnp.array([[0., 2., 0.]]))
    np.testing.assert_array_equal(point, [[0., 0., 0.]])
    np.testing.assert_allclose(score, [4.])
    np.testing.assert_array_equal(index, [0])
    np.testing.assert_array_equal(status, [1])
    # Star composes a silhouette kernel with ordinary host JAX operations.
    star = jax.jit(make_star_radius(primitive, adapter=external_adapter))
    radius, status = star(tree, data, boundary, jnp.array([[0., 2., 0.]]), 10.)
    np.testing.assert_allclose(radius, [2.])
    np.testing.assert_array_equal(status, [1])


def test_builtin_payload_uses_same_external_ray_contract():
    from surfaces.explicit.accelerators.bvh.warp import Points
    from tests.typing.warp_api.query_extension import builtin_intersect
    primitive = Points(3)
    tree, data = build_tree(primitive, primitive.data([[2., 0., 0.]]))
    result = _fused(make_ray_query(primitive, ops=RayOps(intersect=builtin_intersect)),
                    tree, data, jnp.zeros((1, 3)), jnp.array([[1., 0., 0.]]))
    np.testing.assert_allclose(result[0], [2.])
    np.testing.assert_array_equal(result[1], [0])
    np.testing.assert_array_equal(result[2], [True])
    np.testing.assert_array_equal(result[3], [1])


def test_external_overlap_preserves_primitive_aabb_semantics_and_fuses():
    from surfaces.explicit.accelerators.bvh.warp import make_overlap
    primitive, tree, data = _tree()
    hit, status = _fused(make_overlap(primitive), tree, data,
        jnp.array([[-1., -1., -1.], [2., 2., 2.]]),
        jnp.array([[1., 1., 1.], [3., 3., 3.]]))
    np.testing.assert_array_equal(hit, [True, False])
    np.testing.assert_array_equal(status, [1, 0])
