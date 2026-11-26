"""The procedural field and Hermite stages must see identical grid coordinates."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import warp as wp

from meshExtractor import _dual_common as common, _dual_warp as topology
from meshExtractor._field import _FieldEvaluator

wp.set_module_options(wp.get_module_options(topology.grid_position.module), module=__name__)


@wp.kernel
def _coordinates(ids: wp.array(dtype=int), ny: int, nz: int,
                 lower: wp.vec3, spacing: wp.vec3, out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    out[i] = topology.grid_position(topology.dense_cell(ids[i], ny, nz), lower, spacing)


@pytest.mark.parametrize('platform', ['cpu', 'gpu'])
def test_dense_field_and_hermite_coordinate_consistency(platform):
    try:
        device = jax.devices(platform)[0]
    except RuntimeError:
        pytest.skip(f'JAX {platform} unavailable')
    nodes = (16, 18, 14)
    lower = tuple(wp.vec3(-1., -1.2, -1.4))
    spacing = tuple(wp.vec3(.14, .14705883, .19999999))
    with jax.default_device(device), wp.ScopedDevice(wp.device_from_jax(device)):
        ids = np.arange(np.prod(nodes), dtype=np.int32)
        indices = wp.array(ids, dtype=int)
        points = wp.empty(len(ids), dtype=wp.vec3)
        wp.launch(_coordinates, len(ids), [indices, nodes[1], nodes[2],
                                         wp.vec3(*lower), wp.vec3(*spacing), points])
        generated = jax.jit(common._grid_points, static_argnums=(1, 2, 3))(
            jnp.asarray(ids), nodes, lower, spacing)
        np.testing.assert_array_equal(points.numpy(), np.asarray(generated))
        field = lambda radius, p: jnp.linalg.norm(p - jnp.array([.023, -.031, .017])) - radius
        evaluator = _FieldEvaluator(field, 256)
        params = jnp.float32(.61)
        sampler = common._make_dense_sampler(evaluator, nodes, lower, spacing)
        np.testing.assert_array_equal(np.asarray(sampler(params)), evaluator(params, points).numpy())

        if platform == 'cpu':
            # Static iotas and different loop shapes must not change CPU rounding.
            for batch_size in (1, 256, len(ids) + 1):
                for axis in range(3):
                    evaluator = _FieldEvaluator(lambda params, p, axis=axis: p[axis], batch_size)
                    sampler = common._make_dense_sampler(evaluator, nodes, lower, spacing)
                    np.testing.assert_array_equal(np.asarray(sampler(())), points.numpy()[:, axis])
