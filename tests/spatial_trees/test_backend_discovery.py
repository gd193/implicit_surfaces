"""Only supported Warp queries register; host construction is independent."""
from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

from surfaces.explicit.accelerators import get_backend, list_backends


_REMOVED_BACKENDS = (
    "bvh.lbvh.jax", "bvh.sah.jax", "bvh.snch.jax",
    "kd.left_balanced.jax", "kd.sah.jax", "octree.morton_explicit.jax",
)
_REMOVED_QUERIES = (
    "closest_point", "first_hit", "aabb_overlap", "distance_bounds",
    "closest_silhouette", "_closest_point_single", "_first_hit_single",
    "_aabb_overlap_single", "_distance_bounds_single", "_closest_silhouette_single",
)


def test_only_supported_query_backends_register():
    expected = (["bvh.sah.warp", "bvh.snch.warp"]
                if importlib.util.find_spec("warp") is not None else [])
    assert list_backends() == expected
    assert list_backends("bvh") == expected
    assert list_backends(runtime="warp") == expected
    assert list_backends(runtime="jax") == []
    assert list_backends("kd") == list_backends("octree") == []


@pytest.mark.parametrize("name", _REMOVED_BACKENDS)
def test_legacy_builder_modules_do_not_restore_queries(name):
    flavor, algorithm, _ = name.split(".")
    module = importlib.import_module(
        f"surfaces.explicit.accelerators.{flavor}.backends.{algorithm}_jax")
    assert callable(module.build)
    for query in _REMOVED_QUERIES:
        assert not hasattr(module, query), f"legacy traversal restored: {name}.{query}"
    for value in vars(module).values():
        if isinstance(value, type) and value.__module__ == module.__name__:
            assert not set(_REMOVED_QUERIES).intersection(vars(value))
    with pytest.raises(KeyError, match="not registered"):
        get_backend(name)
    assert name not in list_backends()


def test_builders_work_without_query_backends_when_warp_is_unavailable():
    # Isolate imports and simulate a missing Warp installation.
    code = r'''
import importlib.abc
import importlib.util
import sys

real_find_spec = importlib.util.find_spec
def find_spec_without_warp(name, package=None):
    if name == "warp":
        return None
    return real_find_spec(name, package)
importlib.util.find_spec = find_spec_without_warp

class BlockWarpImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "warp" or fullname.startswith("warp."):
            raise ModuleNotFoundError("Warp is blocked for this test")
        return None

sys.meta_path.insert(0, BlockWarpImports())
assert importlib.util.find_spec("warp") is None

import jax
import jax.numpy as jnp
import numpy as np
from surfaces.base import TriMesh
from surfaces.explicit.primitives import TriMeshPrimitives
from surfaces.explicit.accelerators import list_backends
from surfaces.explicit.accelerators.bvh import build_lbvh, build_sah, build_snch
from surfaces.explicit.accelerators.kd import build_left_balanced, build_sah as build_kd_sah
from surfaces.explicit.accelerators.octree import build_morton_explicit

assert jax.devices()[0].platform == "cpu"
assert list_backends() == []
primitives = TriMeshPrimitives(TriMesh(
    vertices=jnp.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=jnp.float32),
    faces=jnp.asarray([[0, 1, 2]], dtype=jnp.int32),
))
for build in (build_lbvh, build_sah, build_snch, build_left_balanced,
              build_kd_sah, build_morton_explicit):
    payload = build(primitives)
    assert int(payload.root) == 0
    np.testing.assert_array_equal(payload.prim_ids, [0])
    np.testing.assert_array_equal(payload.node_lower, [[0., 0., 0.]])
    np.testing.assert_array_equal(payload.node_upper, [[1., 1., 0.]])
assert "warp" not in sys.modules
'''
    env = dict(os.environ, JAX_PLATFORMS="cpu",
               XLA_PYTHON_CLIENT_PREALLOCATE="false")
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
