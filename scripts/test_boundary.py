"""Boundary-condition contracts: partitions, trees, data, and WoS adaptation.

Run as is and again with JAX_ENABLE_X64=true; float cases cover every dtype
the current mode allows, including explicit f32 geometry under x64.
"""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from boundaryConditions import (MIXED, BoundaryConditions, BoundaryKind as BK,
                                FunctionBoundaryData, HalfSpacePartition, TreePartition,
                                VoxelTreePartition)
from sphereWalker.integrator import WoSIntegrator
from sphereWalker.types import WalkerTermination as WT
from surfaces.implicit.lipschitz import sdfImplicit

X64 = jax.config.read('jax_enable_x64')
FLOATS = (jnp.float32, jnp.float64) if X64 else (jnp.float32,)


def plane_grid(n, dim, lower=-1., upper=1.):
    """Labels of the half-space x0 >= 0.25 sampled at cell centers. With these
    bounds the plane lies on a cell face, so the tree matches it exactly."""
    centers = lower + (np.arange(n) + .5) * (upper - lower) / n
    return np.broadcast_to((centers >= .25).astype(np.int32)[(...,) + (None,) * (dim - 1)],
                           (n,) * dim).copy()


def reference_node_count(labels):
    """Nodes in the fully collapsed 2**dim tree, by plain recursion."""
    if labels.size == 1 or np.all(labels == labels.flat[0]):
        return 1
    half = labels.shape[0] // 2
    blocks = [labels[tuple(slice(b * half, (b + 1) * half) for b in bits)]
              for bits in np.ndindex(*(2,) * labels.ndim)]
    return 1 + sum(reference_node_count(b) for b in blocks)


class BinarySplitPartition(TreePartition):
    """Test-only tree: one split on axis 0 at a stored plane."""

    def _encode_point(self, x):
        lower, upper = self.arrays.lower.astype(x.dtype), self.arrays.upper.astype(x.dtype)
        return jnp.all((x >= lower) & (x < upper)), x[0] >= self.arrays.split.astype(x.dtype)

    def _point_child(self, node, key, depth):
        return key.astype(jnp.int32)

    def _box_label(self, lower, upper):
        lo, hi = self.arrays.lower.astype(lower.dtype), self.arrays.upper.astype(lower.dtype)
        plane = self.arrays.split.astype(lower.dtype)
        inside = jnp.all((lower >= lo) & (upper < hi))
        outside = jnp.any((upper < lo) | (lower >= hi))
        side = lower[0] >= plane
        node, _ = self._descend(side, jnp.int32(1))
        label = jnp.where((side == (upper[0] >= plane)) & self._is_leaf(node),
                          self.arrays.uniform[node], jnp.int32(MIXED))
        return inside, outside, label

    @classmethod
    def from_grid(cls, labels, lower, upper, default_region=0, n_regions=None):
        raise NotImplementedError


class PartitionTests(unittest.TestCase):

    def test_voxel_points_match_reference_and_own_planes(self):
        for dim in (2, 3):
            tree = VoxelTreePartition.from_grid(plane_grid(8, dim), -np.ones(dim), np.ones(dim))
            ref = HalfSpacePartition(np.eye(dim)[0], -.25)
            for ftype in FLOATS:
                x = jax.random.uniform(jax.random.PRNGKey(dim), (512, dim), ftype, -.999, .999)
                got = jax.jit(jax.vmap(lambda p: tree.classify_point(None, p)))(x)
                self.assertEqual(got.dtype, jnp.int32)
                np.testing.assert_array_equal(got, jax.vmap(lambda p: ref.classify_point(None, p))(x))
                # Split planes belong to the upper child; the root's upper face
                # is outside the half-open root and gets the default region.
                on_plane = jnp.zeros(dim, ftype).at[0].set(.25)
                self.assertEqual(int(tree.classify_point(None, on_plane)), 1)
                fine_plane = jnp.zeros(dim, ftype).at[0].set(.5)
                self.assertEqual(int(tree.classify_point(None, fine_plane)), 1)
                upper_face = jnp.zeros(dim, ftype).at[0].set(1.)
                self.assertEqual(int(tree.classify_point(None, upper_face)), 0)
                self.assertEqual(int(tree.classify_point(None, jnp.full(dim, jnp.nan, ftype))), 0)

    def test_collapse_preserves_queries(self):
        uniform = VoxelTreePartition.from_grid(np.full((16, 16), 2), [0., 0.], [1., 1.])
        self.assertEqual(uniform.arrays.children.shape[0], 1)
        labels = plane_grid(16, 2, 0., 1.)
        tree = VoxelTreePartition.from_grid(labels, [0., 0.], [1., 1.])
        self.assertLess(tree.arrays.children.shape[0], 16 * 16)
        centers = (np.stack(np.meshgrid(np.arange(16), np.arange(16), indexing='ij'), -1) + .5) / 16
        got = jax.vmap(lambda p: tree.classify_point(None, p))(jnp.asarray(centers.reshape(-1, 2)))
        np.testing.assert_array_equal(got.reshape(16, 16), labels)

    def test_boxes_are_conservative(self):
        tree = VoxelTreePartition.from_grid(plane_grid(8, 2), [-1., -1.], [1., 1.])
        ref = HalfSpacePartition([1., 0.], -.25)
        for ftype in FLOATS:
            box = lambda part, lo, hi: int(part.classify_box(None, jnp.asarray(lo, ftype),
                                                             jnp.asarray(hi, ftype)))
            for part in (tree, ref):
                self.assertEqual(box(part, [-.9, -.9], [-.1, .9]), 0)
                self.assertEqual(box(part, [.6, .1], [.9, .4]), 1)
                self.assertEqual(box(part, [.2, 0.], [.3, .1]), MIXED)
                # A box whose upper corner touches the plane contains a region-1 point.
                self.assertEqual(box(part, [0., 0.], [.25, .1]), MIXED)
                self.assertEqual(box(part, [.1, 0.], [0., .1]), MIXED)  # inverted
            # Fixed-cost tree query: at the level where this box covers two
            # cells per axis, one cell contains the plane, so it stays MIXED.
            self.assertEqual(box(ref, [.3, -.5], [.9, .5]), 1)
            self.assertEqual(box(tree, [.3, -.5], [.9, .5]), MIXED)
            # Outside the root: default region; straddling the root: MIXED
            # unless the whole root is uniform with the default label.
            self.assertEqual(box(tree, [2., 2.], [3., 3.]), 0)
            self.assertEqual(box(tree, [.5, .5], [1.5, .9]), MIXED)
            self.assertEqual(box(tree, [-1.5, -.5], [-.5, .5]), 0)
            ones = VoxelTreePartition.from_grid(np.ones((4, 4), np.int32), [-1., -1.], [1., 1.],
                                                default_region=1)
            self.assertEqual(box(ones, [-2., -2.], [2., 2.]), 1)

            # Random boxes: any certified label must hold at sampled points.
            key = jax.random.PRNGKey(3)
            corners = jax.random.uniform(key, (256, 2, 2), ftype, -1.2, 1.2)
            lo, hi = jnp.min(corners, 1), jnp.max(corners, 1)
            labels = jax.jit(jax.vmap(lambda a, b: tree.classify_box(None, a, b)))(lo, hi)
            self.assertEqual(labels.dtype, jnp.int32)
            t = jax.random.uniform(jax.random.PRNGKey(4), (64, 2), ftype)
            for a, b, label in zip(lo, hi, labels):
                if label == MIXED:
                    continue
                pts = jnp.concatenate([a + t * (b - a), jnp.stack([a, b])])
                np.testing.assert_array_equal(
                    jax.vmap(lambda p: tree.classify_point(None, p))(pts), int(label))
            self.assertTrue(np.any(labels != MIXED) and np.any(labels == MIXED))

    def test_random_grids_build_valid_trees(self):
        """JAX construction: valid storage, minimal node count, exact lookup."""
        rng = np.random.default_rng(0)
        for dim, n, block in ((2, 32, 4), (3, 16, 2), (1, 64, 8), (2, 1, 1)):
            coarse = rng.integers(0, 3, (n // block,) * dim)
            labels = coarse
            for axis in range(dim):
                labels = np.repeat(labels, block, axis=axis)
            if n > 1:
                labels[(0,) * dim] = 4  # one odd cell forces a full-depth path
            tree = VoxelTreePartition.from_grid(labels, np.zeros(dim), np.ones(dim))
            a = tree.arrays
            # Host validation of the device-built arrays (raises if invalid).
            TreePartition._make_arrays(np.asarray(a.children), np.asarray(a.uniform),
                                       np.zeros(dim), np.ones(dim))
            self.assertEqual(len(a.uniform), reference_node_count(labels))
            self.assertEqual((a.children.dtype, a.uniform.dtype), (jnp.int32, jnp.int32))
            grid = np.stack(np.meshgrid(*[np.arange(n)] * dim, indexing='ij'), -1)
            centers = jnp.asarray((grid.reshape(-1, dim) + .5) / n)
            got = jax.jit(jax.vmap(lambda p: tree.classify_point(None, p)))(centers)
            np.testing.assert_array_equal(np.asarray(got).reshape(labels.shape), labels)
            # Certified boxes hold at sampled points.
            corners = rng.uniform(-.1, 1.1, (128, 2, dim))
            lo, hi = jnp.asarray(corners.min(1)), jnp.asarray(corners.max(1))
            boxes = jax.jit(jax.vmap(lambda u, v: tree.classify_box(None, u, v)))(lo, hi)
            t = jnp.asarray(rng.uniform(size=(32, dim)))
            for u, v, label in zip(lo, hi, boxes):
                if label != MIXED:
                    pts = jnp.concatenate([u + t * (v - u), jnp.stack([u, v])])
                    np.testing.assert_array_equal(
                        jax.vmap(lambda p: tree.classify_point(None, p))(pts), int(label))

    @unittest.skipUnless(any(d.platform == 'gpu' for d in jax.devices()),
                         'set JAX_PLATFORMS=cuda to run on the GPU')
    def test_builds_and_queries_on_gpu(self):
        gpu = jax.devices('gpu')[0]
        labels = jax.device_put(jnp.asarray(plane_grid(64, 3, 0., 1.)), gpu)
        tree = VoxelTreePartition.from_grid(labels, np.zeros(3), np.ones(3))
        self.assertEqual(tree.arrays.children.devices(), {gpu})
        x = jax.device_put(jax.random.uniform(jax.random.PRNGKey(0), (4096, 3)), gpu)
        got = jax.jit(jax.vmap(lambda p: tree.classify_point(None, p)))(x)
        self.assertEqual(got.devices(), {gpu})
        np.testing.assert_array_equal(got, (np.asarray(x)[:, 0] >= .25).astype(np.int32))

    def test_custom_tree_reuses_traversal(self):
        arrays = TreePartition._make_arrays(
            np.array([[1, 2], [-1, -1], [-1, -1]]), np.array([MIXED, 3, 5]),
            np.array([0., 0.]), np.array([1., 1.]), split=jnp.asarray(.5))
        tree = BinarySplitPartition(arrays, dim=2, n_regions=6, default_region=0, max_depth=1)
        point = lambda *x: int(tree.classify_point(None, jnp.array(x)))
        self.assertEqual((point(.2, .5), point(.5, .5), point(1., .5)), (3, 5, 0))
        box = lambda lo, hi: int(tree.classify_box(None, jnp.array(lo), jnp.array(hi)))
        self.assertEqual((box([.1, .1], [.4, .4]), box([.4, .1], [.6, .4])), (3, MIXED))

    def test_partition_is_a_jit_argument(self):
        tree = VoxelTreePartition.from_grid(plane_grid(8, 3), -np.ones(3), np.ones(3))
        query = jax.jit(lambda part, x: jax.vmap(lambda p: part.classify_point(None, p))(x))
        x = jax.random.uniform(jax.random.PRNGKey(0), (32, 3), minval=-1., maxval=1.)
        np.testing.assert_array_equal(query(tree, x),
                                      jax.vmap(lambda p: tree.classify_point(None, p))(x))
        leaves = jax.tree.leaves(tree)
        self.assertTrue(all(leaf.dtype in (jnp.int32, jnp.float32, jnp.float64) for leaf in leaves))
        self.assertEqual(jax.tree.leaves(tree)[0].dtype, jnp.int32)

    def test_invalid_grids(self):
        with self.assertRaises(ValueError):
            VoxelTreePartition.from_grid(np.zeros((3, 3), np.int32), [0., 0.], [1., 1.])
        with self.assertRaises(ValueError):
            VoxelTreePartition.from_grid(np.zeros((4, 4)), [0., 0.], [1., 1.])
        with self.assertRaises(ValueError):
            VoxelTreePartition.from_grid(np.zeros((4, 4), np.int32), [0., 0.], [1., 0.])
        with self.assertRaises(ValueError):
            VoxelTreePartition.from_grid(np.full((4, 4), 2), [0., 0.], [1., 1.], n_regions=2)


class DataTests(unittest.TestCase):

    def test_kinds_values_and_dtypes(self):
        values = {'v': jnp.asarray([2., -3.], jnp.float64 if X64 else jnp.float32)}
        lookup = FunctionBoundaryData([BK.DIRICHLET, BK.NEUMANN], lambda p, r, x, n: p['v'][r])
        per_region = FunctionBoundaryData.from_region_fns(
            [BK.DIRICHLET, BK.NEUMANN], [lambda p, x, n: x @ x, lambda p, x, n: 7])
        self.assertEqual(lookup.kinds().dtype, jnp.int32)
        for ftype in FLOATS:
            x, n = jnp.asarray([.5, 1.], ftype), jnp.asarray([1., 0.], ftype)
            for region, expected in ((0, (1, 0, 2)), (1, (0, 1, -3))):
                out = jax.jit(lookup.evaluate)(values, jnp.int32(region), x, n)
                np.testing.assert_allclose(out, expected)
                self.assertTrue(all(v.dtype == ftype and v.shape == () for v in out))
            np.testing.assert_allclose(per_region.evaluate(None, 0, x, n).c, 1.25)
            out = per_region.evaluate(None, jnp.int32(1), x, n)
            self.assertEqual((float(out.c), out.c.dtype), (7., ftype))
            # Data must be differentiable in x near the boundary.
            _, dc = jax.jvp(lambda y: per_region.evaluate(None, 0, y, n).c, (x,), (n,))
            np.testing.assert_allclose(dc, 1.)
            self.assertEqual(dc.dtype, ftype)
        with self.assertRaises(ValueError):
            FunctionBoundaryData([2], lambda p, r, x, n: 0.)
        with self.assertRaises(ValueError):
            FunctionBoundaryData([0], lambda p, r, x, n: x).evaluate(None, 0, jnp.zeros(2), jnp.zeros(2))

    def test_vector_values(self):
        data = FunctionBoundaryData([BK.DIRICHLET, BK.NEUMANN],
                                    lambda p, r, x, n: x * (r + 1.), value_shape=(2,))
        per_region = FunctionBoundaryData.from_region_fns(
            [BK.DIRICHLET, BK.DIRICHLET], [lambda p, x, n: x, lambda p, x, n: -x], (2,))
        self.assertEqual(data.value_shape, (2,))
        for ftype in FLOATS:
            x, n = jnp.asarray([.5, 1.], ftype), jnp.asarray([1., 0.], ftype)
            out = jax.jit(data.evaluate)(None, jnp.int32(1), x, n)
            np.testing.assert_allclose(out.c, [1., 2.])
            self.assertEqual((out.a.shape, out.b.shape, out.c.shape, out.c.dtype),
                             ((), (), (2,), ftype))
            np.testing.assert_allclose(per_region.evaluate(None, 1, x, n).c, [-.5, -1.])
            bc = BoundaryConditions(HalfSpacePartition([1., 0.], 0.), data)
            g = bc.dirichlet_fn(None, None)
            np.testing.assert_allclose(g(-x), [-.5, -1.])          # Dirichlet region
            self.assertTrue(np.all(np.isnan(g(x))) and g(x).shape == (2,))  # Neumann
        # Shapes must match exactly; scalars are not broadcast to vectors.
        for fn, shape in ((lambda p, r, x, n: 1., (2,)), (lambda p, r, x, n: x, ())):
            with self.assertRaises(ValueError):
                FunctionBoundaryData([0], fn, shape).evaluate(None, 0, jnp.zeros(2), jnp.zeros(2))
        with self.assertRaises(ValueError):
            FunctionBoundaryData([0], lambda p, r, x, n: x, value_shape=(0,))

    def test_container_checks_regions(self):
        with self.assertRaises(ValueError):
            BoundaryConditions(HalfSpacePartition([1., 0.], 0.),
                               FunctionBoundaryData([0], lambda p, r, x, n: 0.))


class WoSAdapterTests(unittest.TestCase):

    def test_split_disk(self):
        """Unit disk, u = 1 on x0 >= 0 and 0 elsewhere: u(0) = 1/2."""
        disk = sdfImplicit(fn=lambda p, x: jnp.linalg.norm(x) - 1., dim=2)
        labels = np.repeat((np.arange(4) >= 2)[:, None], 4, axis=1).astype(np.int32)
        tree = VoxelTreePartition.from_grid(labels, [-2., -2.], [2., 2.])
        data = FunctionBoundaryData.from_region_fns(
            [BK.DIRICHLET, BK.DIRICHLET], [lambda p, x, n: 0., lambda p, x, n: 1.])
        options = dict(init_trust_region_size=1., n_samples=4000, steps_between_refill=16,
                       max_active_states=4000, jit_buck_size=1)
        key = jax.random.PRNGKey(0)
        for ftype in FLOATS:
            bc = BoundaryConditions(tree, data)
            solver = WoSIntegrator(disk.empty_sphere, bc.dirichlet_fn(None, None),
                                   lambda x: jnp.asarray(0., x.dtype), dim=2, eps=1e-3,
                                   max_steps=200)
            out = solver.integrate_batch(None, jnp.zeros((1, 2), ftype), key, **options)
            np.testing.assert_allclose(out.w[0, 0], .5, atol=.03)
            self.assertEqual(out.w.dtype, ftype)
        # Vector field: harmonic data g(x) = x gives u(p) = p componentwise.
        identity = BoundaryConditions(tree, FunctionBoundaryData(
            [BK.DIRICHLET, BK.DIRICHLET], lambda p, r, x, n: x, value_shape=(2,)))
        solver = WoSIntegrator(disk.empty_sphere, identity.dirichlet_fn(None, None),
                               lambda x: jnp.asarray(0., x.dtype), dim=2, max_steps=200)
        point = jnp.array([[.3, -.2]])
        out = solver.integrate_batch(None, point, key, **options)
        self.assertEqual(out.w.shape, (1, 2))
        np.testing.assert_allclose(out.w, point, atol=.03)
        # Plain WoS cannot use Neumann data: those walks are explicit failures.
        mixed = BoundaryConditions(tree, FunctionBoundaryData.from_region_fns(
            [BK.DIRICHLET, BK.NEUMANN], [lambda p, x, n: 0., lambda p, x, n: 1.]))
        solver = WoSIntegrator(disk.empty_sphere, mixed.dirichlet_fn(None, None),
                               lambda x: jnp.asarray(0., x.dtype), dim=2, max_steps=200)
        details = solver.integrate_batch(None, jnp.zeros((1, 2)), key, return_details=True,
                                         **{**options, 'n_samples': 400})
        invalid = int(details.termination_counts[0, WT.INVALID])
        self.assertGreater(invalid, 100)
        self.assertEqual(invalid + int(details.successful_samples[0]), 400)
        np.testing.assert_allclose(details.contribution_sum[0], 0.)


if __name__ == '__main__':
    unittest.main()
