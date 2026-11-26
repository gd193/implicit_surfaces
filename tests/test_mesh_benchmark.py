"""Small independent geometry checks and timing-accounting regression."""
import gzip
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

wp = pytest.importorskip("warp")
pytest.importorskip("warp.geometry")

from scripts.bench.mesh_extraction._cases import (  # noqa: E402
    FIXED_MC_BACKENDS, fixture, host_mesh, make_runner, validate_mesh,
)
from scripts.bench.mesh_extraction.benchmark import trace_samples  # noqa: E402
from scripts.bench.mesh_extraction._warp_mc import _extract_dense  # noqa: E402


@pytest.mark.parametrize("name,level,phase", [
    ("sphere", 0., "pipeline"), ("sphere", .12, "extraction"),
    ("torus", 0., "pipeline"), ("neural", 0., "extraction"),
    ("gyroid", .1, "pipeline"), ("sphere", 5., "extraction"),
])
def test_mesh_adapters_against_dense_reference(name, level, phase):
    wp.set_device(wp.device_from_jax(jax.devices()[0]))
    scene, cells = fixture(name, level), 16
    axis = jnp.linspace(-1., 1., cells+1)
    points = jnp.stack(jnp.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    values = np.array(scene(scene.params, points))
    if values.min() <= level <= values.max():
        reference, reference_faces, _, _ = marching_cubes(values, level, spacing=(2./cells,)*3, method="lorensen")
        reference -= 1.
    else:
        reference, reference_faces = np.empty((0, 3)), np.empty((0, 3), int)
    for backend in FIXED_MC_BACKENDS:
        run, _ = make_runner(backend, scene, cells, phase, 1024)
        vertices, faces = host_mesh(run())
        validate_mesh(vertices, faces, scene, cells)
        assert len(faces) == len(reference_faces), backend
        if not len(faces):
            assert len(vertices) == 0
            continue
        for source, target in ((vertices, reference), (reference, vertices)):
            assert cKDTree(target).query(source)[0].max() < 2e-5, backend
        assert len(vertices) == len(reference), backend
        if name != "gyroid":
            edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
            _, counts = np.unique(edges, axis=0, return_counts=True)
            assert np.all(counts == 2), backend


def test_adaptive_mc_pipeline_uses_requested_level_and_resolution():
    scene, cells = fixture("sphere", .2), 12
    run, details = make_runner("mc-adaptive", scene, cells, "pipeline", 1024)
    assert details["max_depth"] == 4
    assert details["tolerance"] == pytest.approx(2 * np.sqrt(3) / cells)
    assert not details["gradient_evaluation_in_timed_region"]
    vertices, faces = host_mesh(run())
    metrics = validate_mesh(vertices, faces, scene, cells, tolerance=details["tolerance"])
    assert metrics["triangles"] > 0
    assert metrics["max_field_residual"] < .05
    with pytest.raises(ValueError, match="pipeline"):
        make_runner("mc-adaptive", scene, cells, "extraction", 1024)


def test_trace_excludes_host_gaps_and_accounts_for_copies(tmp_path):
    events = [{"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "/device:GPU:0"}},
              {"ph": "X", "pid": 2, "name": "mc_sample", "ts": 100, "dur": 1000},
              {"ph": "X", "pid": 2, "name": "host_work", "ts": 110, "dur": 800}]
    for name, ts, duration, args in [("kernel", 200, 30, {"kernel_details": ""}),
                                     ("Memcpy D2H", 900, 5, {}), ("Memset", 950, 2, {}),
                                     ("outside", 1200, 900, {"kernel_details": ""})]:
        events.append({"ph": "X", "pid": 1, "name": name, "ts": ts, "dur": duration, "args": args})
    path = tmp_path / "trace.json.gz"
    with gzip.open(path, "wt") as stream:
        json.dump({"traceEvents": events}, stream)
    samples, kernels = trace_samples(path)
    assert samples == [{"kernel_ms": .03, "memcpy_ms": .005, "memset_ms": .002, "kernels": 1}]
    assert kernels == {"kernel": 30}
    with gzip.open(path, "wt") as stream:
        json.dump({"traceEvents": events[:3]}, stream)
    with pytest.raises(RuntimeError, match="no kernels"):
        trace_samples(path)
    samples, kernels = trace_samples(path, allow_no_kernels=True)
    assert samples[0]["kernel_ms"] == 0 and not kernels


@pytest.mark.parametrize("shape", [(3, 4, 5), (9, 7, 6)])
def test_dense_workspace_alias_matches_upstream(shape):
    wp.set_device(wp.device_from_jax(jax.devices()[0]))
    values = np.random.default_rng(42).normal(size=shape).astype(np.float32)
    values[0] = 0.  # Equal-to-level corners and boundary edges.
    data = wp.array(values)
    for level in (0., 5.):
        expected = wp.geometry.IsoSurfaceMarchingCubes.extract(data, level, lower=(-2., -1., 0.), upper=(1., 3., 2.))
        actual = _extract_dense(data, level, lower=(-2., -1., 0.), upper=(1., 3., 2.))
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(a.numpy(), b.numpy())
