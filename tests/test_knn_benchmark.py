"""Benchmark reporting must preserve failures before timing returns any samples."""

import json
import sys

import numpy as np
import pytest

from scripts.bench.spatial_trees import warp_knn_reference as benchmark


@pytest.mark.parametrize("failure_stage", ["prepare", "execute", "validate", None])
def test_warp_case_report_replaces_stale_results(tmp_path, monkeypatch, failure_stage):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "build.json").write_text(json.dumps({"cuda_root": str(tmp_path)}))
    output = tmp_path / "results"
    output.mkdir()
    report_path = output / "results.json"
    report_path.write_text(json.dumps({"results": [{"validation": "stale"}]}))
    monkeypatch.setattr(sys, "argv", [
        "benchmark", "--binaries", str(binaries), "--output", str(output),
        "--dimensions", "2", "--sizes", "1", "--queries", "1", "--k", "1",
        "--fixtures", "uniform", "--implementations", "ours",
    ])
    monkeypatch.setattr(benchmark.subprocess, "check_output", lambda *a, **kw: "mock metadata")
    monkeypatch.setattr(benchmark, "fixtures", lambda *a: (
        np.zeros((1, 2), np.float32), np.zeros((1, 2), np.float32),
    ))
    monkeypatch.setattr(benchmark, "prepare_ours", lambda *a: object())
    monkeypatch.setattr(benchmark, "ours", lambda *a, **kw: (
        (np.zeros((1, 1), np.float32), np.zeros((1, 1), np.int32),
         np.ones(1, np.int32), np.ones(1, np.int32)),
        [0.1], {"gpu": "mock"},
    ))

    error = AssertionError("deliberate invalid result")

    def fail(*args, **kwargs):
        raise error

    if failure_stage is not None:
        target = {"prepare": "prepare_ours", "execute": "ours", "validate": "validate"}
        monkeypatch.setattr(benchmark, target[failure_stage], fail)
        with pytest.raises(AssertionError) as caught:
            benchmark.main()
        assert caught.value is error
    else:
        benchmark.main()

    rows = json.loads(report_path.read_text())["results"]
    assert len(rows) == 1
    row = rows[0]
    assert (row["implementation"], row["dimension"], row["fixture"], row["k"],
            row["n"], row["q"]) == ("ours", 2, "uniform", 1, 1, 1)
    assert set(row["fixture_sha256"]) == {"data.f32", "queries.f32"}
    if failure_stage is not None:
        assert row["validation"] == "FAILED"
        assert row["error"] == "AssertionError: deliberate invalid result"
    else:
        assert row["validation"] == "passed"
        assert row["samples_ms"] == [0.1]
        assert set(row["output_sha256"]) == {"scores", "ids", "count", "status"}
