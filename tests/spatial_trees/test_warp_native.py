"""External native source participates in Warp's compilation cache."""
from pathlib import Path

import numpy as np
import pytest

wp = pytest.importorskip("warp")


def test_header_edit_invalidates_loaded_cpu_kernel(tmp_path):
    from surfaces.explicit.accelerators.bvh.backends import _warp_traversal
    from surfaces.explicit.accelerators.bvh.warp import Euclidean, Points, make_nearest

    shipped = set(Path(_warp_traversal.__file__).with_name("native").glob("*.h"))
    configured = wp.get_module_options(make_nearest(Points(2), Euclidean()).kernel.module)["extra_build_options"]
    assert shipped and shipped <= set(map(Path, configured.extra_build_dependencies))
    header = tmp_path / "dependency.h"
    source = "namespace dependency_probe { CUDA_CALLABLE inline float value() { return %s.f; } }"
    header.write_text(source % 1)

    @wp.func_native("return dependency_probe::value();")
    def native_value() -> float: ...

    @wp.kernel(module="unique", enable_backward=False)
    def kernel(out: wp.array(dtype=float)):
        out[0] = native_value()

    options = configured.merged(wp.ModuleBuildOptions(
        extra_cpu_include_dirs=[tmp_path], extra_cpu_preamble='#include "dependency.h"',
        extra_build_dependencies=[header]))
    out = wp.empty(1, dtype=float, device="cpu")
    hashes = []
    for value in (1, 2):
        header.write_text(source % value)
        wp.set_module_options({"extra_build_options": options}, module=kernel.module)
        wp.launch(kernel, dim=1, outputs=[out], device="cpu")
        np.testing.assert_array_equal(out.numpy(), [float(value)])
        hashes.append(kernel.module.get_module_hash())
    assert hashes[0] != hashes[1]
