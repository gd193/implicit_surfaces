"""Backend discovery without Warp installed."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_jax_backends_register_when_warp_is_unavailable():
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
from surfaces.explicit.accelerators import list_backends

assert jax.devices()[0].platform == "cpu"
backends = set(list_backends("bvh"))
assert {"bvh.lbvh.jax", "bvh.sah.jax", "bvh.snch.jax"} <= backends
assert not any(name.endswith(".warp") for name in backends)
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
