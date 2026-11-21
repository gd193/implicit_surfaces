"""Independent point-kNN comparisons; warmed resident CUDA graph/event timings."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from importlib.resources import files
import subprocess
import numpy as np


def fixtures(d, n, q, kind, seed=619):
    rng = np.random.default_rng(seed)

    def sample(m):
        x = rng.uniform(-1, 1, (m, d))
        if kind == "clustered":
            centers = np.random.default_rng(seed + 1).uniform(-1, 1, (8, d))
            x = centers[rng.integers(8, size=m)] + rng.normal(0, 0.035, (m, d))
        elif kind == "surface":
            x = rng.normal(size=(m, d))
            x /= np.linalg.norm(x, axis=1)[:, None]
        return x.astype(np.float32)

    return sample(n), sample(q)


def validate(
    data, queries, scores, ids, counts, status, k, cap, samples=64, strict_ids=False
):
    """Float64 brute force, score tolerance and tie-aware membership, no NxQ buffer."""
    assert scores.shape == ids.shape == (len(queries), k)
    assert np.all(counts == np.sum(ids >= 0, axis=1))
    assert np.all(status == (counts > 0))
    assert np.array_equal(ids >= 0, np.arange(k)[None, :] < counts[:, None])
    assert np.all(ids[ids >= 0] < len(data))
    assert np.all(np.isposinf(scores[ids < 0])) and np.all(ids[ids < 0] == -1)
    assert np.isfinite(scores[ids >= 0]).all() and np.all(scores >= 0)
    assert np.all(scores[:, :-1] <= scores[:, 1:])
    for row in np.unique(
        np.linspace(0, len(queries) - 1, min(samples, len(queries)), dtype=int)
    ):
        delta = data.astype(np.float64) - queries[row].astype(np.float64)
        all_scores = np.sum(delta * delta, axis=1)
        order = np.lexsort((np.arange(len(data)), all_scores))
        order = order[all_scores[order] <= cap][:k]
        n = len(order)
        assert counts[row] == n, (row, counts[row], n)
        selected = ids[row, :n]
        assert len(np.unique(selected)) == n and np.all(
            (selected >= 0) & (selected < len(data))
        )
        np.testing.assert_allclose(
            scores[row, :n], all_scores[order], rtol=3e-6, atol=1e-12
        )
        np.testing.assert_allclose(
            scores[row, :n], all_scores[selected], rtol=3e-6, atol=1e-12
        )
        if strict_ids:
            # Deterministic ordering applies to emitted float32 scores, with ID tie break.
            assert list(zip(scores[row, :n], selected)) == sorted(
                zip(scores[row, :n], selected)
            )
        assert np.all(all_scores[selected] <= cap + 3e-6*cap + 1e-12)


def prepare_ours(data, leaf):
    from sengi.bvh.warp import Points, build_tree
    from sengi.protocol import Config

    primitive = Points(data.shape[1])
    tree, geometry = build_tree(
        primitive, primitive.data(data), Config(leaf_size_threshold=leaf)
    )
    return primitive, tree, geometry


def ours(
    data,
    queries,
    k,
    cap,
    block,
    leaf,
    launches,
    repeats,
    validate_samples,
    *,
    prepared=None,
    cache_bounds=True,
    packed_nodes=True,
    group_leaves=None,
):
    import jax

    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("CUDA GPU required for benchmark timing")
    import jax.numpy as jnp
    import warp as wp
    from sengi.bvh.warp import Euclidean, make_knn
    from scripts.bench.spatial_trees._warp_timing import _make_case, _measure

    primitive, tree, geometry = prepared or prepare_ours(data, leaf)
    ops = Euclidean().bind(primitive)
    # Raw callback path returns squared scores, matching both external kernels.
    call = make_knn(
        primitive,
        k,
        evaluate=ops.closest,
        lower_bound=ops.node_bound,
        block_dim=block,
        cache_bounds=cache_bounds,
        packed_nodes=packed_nodes,
        group_leaves=group_leaves,
    )
    n = len(queries)
    inputs = [jnp.asarray(queries), jnp.full((n,), cap, jnp.float32), *tree, *geometry]
    device = wp.device_from_jax(jax.devices()[0])
    case = _make_case(
        "knn",
        "ours",
        call.kernel,
        inputs,
        n,
        device,
        block,
        ((n, k), (n, k), (n,), (n,)),
    )

    def check_result(case, result):
        validate(data, queries, *result, k, cap, validate_samples, strict_ids=True)

    _measure([case], n, device, launches, repeats, validate=check_result)
    return (
        tuple(x.numpy() for x in case.outputs),
        case.samples_ms,
        dict(
            gpu=device.name,
            uuid=device.uuid,
            module_hash=call.kernel.module.get_module_hash(block).hex(),
            kernel=wp.get_cuda_kernel_properties(
                call.kernel, device=device, block_dim=block
            ),
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binaries", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dimensions", type=int, nargs="+", choices=(2, 3), default=[2, 3])
    p.add_argument("--k", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64, 128])
    p.add_argument("--sizes", type=int, nargs="+", default=[1024, 65536, 1048576])
    p.add_argument("--queries", type=int, nargs="+", default=[1024, 32768])
    p.add_argument(
        "--fixtures",
        nargs="+",
        choices=("uniform", "clustered", "surface"),
        default=["uniform", "clustered", "surface"],
    )
    p.add_argument(
        "--implementations",
        nargs="+",
        choices=(
            "ours",
            "cubql",
            "cudakdtree_fixed",
            "cudakdtree_heap",
            "cudakdtree_fixed_stack",
            "cudakdtree_heap_stack",
        ),
        default=["ours", "cubql", "cudakdtree_fixed", "cudakdtree_heap"],
    )
    p.add_argument("--block-dim", type=int, default=64)
    p.add_argument("--leaf-size", type=int, default=4)
    p.add_argument(
        "--no-cache-bounds",
        action="store_true",
        help="Recompute deferred Warp node bounds with a smaller traversal stack",
    )
    p.add_argument("--node-access", choices=("packed", "indexed"), default="packed")
    p.add_argument("--scheduling", choices=("auto", "flat", "grouped"), default="auto")
    p.add_argument("--launches", type=int, default=30)
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--validate-samples", type=int, default=64)
    p.add_argument("--cap-squared", type=float, default=float("inf"))
    a = p.parse_args()
    if (
        min(
            *a.k,
            *a.sizes,
            *a.queries,
            a.launches,
            a.repeats,
            a.validate_samples,
            a.leaf_size,
        )
        < 1
    ):
        p.error("counts must be positive")
    if not 1 <= a.block_dim <= 1024 or np.isnan(a.cap_squared) or a.cap_squared < 0:
        p.error("invalid block or cap")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    build = json.loads((a.binaries / "build.json").read_text())
    env = os.environ.copy()
    cuda = Path(build["cuda_root"])
    env["LD_LIBRARY_PATH"] = (
        ":".join(
            map(str, [cuda / "lib", cuda / "lib64", cuda / "targets/x86_64-linux/lib"])
        )
        + ":"
        + env.get("LD_LIBRARY_PATH", "")
    )
    report = dict(
        build=build,
        settings=vars(a).copy(),
        results=[],
        scope="Resident CUDA graphs/events; squared scores; heap sorting and output normalization included; build/JIT/transfers excluded.",
    )
    report["settings"] = {
        key: str(v) if isinstance(v, Path) else v
        for key, v in report["settings"].items()
    }
    import importlib.metadata

    report["versions"] = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "jax", "jaxlib", "warp-lang")
    }
    report["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    report["source_sha256"] = {
        str(f): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in [
            Path(__file__),
            Path(__file__).with_name("build_knn_references.py"),
            *Path(__file__).with_name("references").glob("knn_*"),
        ]
    }
    repo = Path(__file__).resolve().parents[3]
    report["repo"] = dict(
        head=subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        status=subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"], text=True
        ),
    )
    backend = files("sengi") / "bvh/backends"
    for f in [
        *backend.glob("*.py"),
        *(backend / "warp_queries").rglob("*.py"),
        *(backend / "native").glob("*.h"),
        files("sengi") / "protocol.py",
    ]:
        report["source_sha256"][str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
    report["device"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
        text=True,
    )
    for d in a.dimensions:
        for n in a.sizes:
            for q in a.queries:
                for kind in a.fixtures:
                    folder = a.output / f"{kind}_d{d}_n{n}_q{q}"
                    folder.mkdir(exist_ok=True)
                    data, queries = fixtures(d, n, q, kind)
                    data.tofile(folder / "data.f32")
                    queries.tofile(folder / "queries.f32")
                    identity = {
                        f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                        for f in folder.glob("*.f32")
                    }
                    prepared = None
                    for k in a.k:
                        for implementation in a.implementations:
                            prefix = folder / f"{implementation}_k{k}"
                            row = dict(
                                implementation=implementation,
                                dimension=d,
                                n=n,
                                q=q,
                                fixture=kind,
                                k=k,
                                fixture_sha256=identity,
                                output=str(prefix),
                            )
                            try:
                                if implementation == "ours":
                                    if prepared is None:
                                        prepared = prepare_ours(data, a.leaf_size)
                                    result, times, properties = ours(
                                        data,
                                        queries,
                                        k,
                                        a.cap_squared,
                                        a.block_dim,
                                        a.leaf_size,
                                        a.launches,
                                        a.repeats,
                                        a.validate_samples,
                                        prepared=prepared,
                                        cache_bounds=not a.no_cache_bounds,
                                        packed_nodes=a.node_access == "packed",
                                        group_leaves={"auto": None, "flat": False, "grouped": True}[a.scheduling],
                                    )
                                    for suffix, value in zip(
                                        ("scores", "ids", "count", "status"), result
                                    ):
                                        value.tofile(str(prefix) + "." + suffix)
                                else:
                                    binary = (
                                        a.binaries / f"{implementation}_d{d}_k{k}"
                                    ).resolve()
                                    assert (
                                        hashlib.sha256(binary.read_bytes()).hexdigest()
                                        == build["binaries"][binary.name]
                                    )
                                    command = list(
                                        map(
                                            str,
                                            [
                                                binary,
                                                folder,
                                                a.block_dim,
                                                a.launches,
                                                a.repeats,
                                                prefix,
                                                a.cap_squared,
                                                a.leaf_size,
                                            ],
                                        )
                                    )
                                    with open(str(prefix) + ".log", "w") as log:
                                        subprocess.run(
                                            command,
                                            check=True,
                                            env=env,
                                            stdout=log,
                                            stderr=subprocess.STDOUT,
                                        )
                                    result = tuple(
                                        np.fromfile(
                                            str(prefix) + "." + suffix, dtype
                                        ).reshape(shape)
                                        for suffix, dtype, shape in [
                                            ("scores", np.float32, (q, k)),
                                            ("ids", np.int32, (q, k)),
                                            ("count", np.int32, (q,)),
                                            ("status", np.int32, (q,)),
                                        ]
                                    )
                                    times = np.loadtxt(
                                        str(prefix) + ".ms", ndmin=1
                                    ).tolist()
                                    properties = json.loads(
                                        Path(str(prefix) + ".json").read_text()
                                    )
                                output_hashes = {
                                    suffix: hashlib.sha256(
                                        Path(str(prefix) + "." + suffix).read_bytes()
                                    ).hexdigest()
                                    for suffix in ("scores", "ids", "count", "status")
                                }
                                row.update(
                                    output_sha256=output_hashes,
                                    samples_ms=times,
                                    properties=properties,
                                )
                                validate(
                                    data,
                                    queries,
                                    *result,
                                    k,
                                    a.cap_squared,
                                    a.validate_samples,
                                    strict_ids=implementation == "ours",
                                )
                                row["validation"] = "passed"
                            except Exception as error:
                                row["validation"] = "FAILED"
                                row["error"] = f"{type(error).__name__}: {error}"
                                report["results"].append(row)
                                (a.output / "results.json").write_text(
                                    json.dumps(report, indent=2) + "\n"
                                )
                                raise
                            report["results"].append(row)
                            (a.output / "results.json").write_text(
                                json.dumps(report, indent=2) + "\n"
                            )
                            print(
                                kind,
                                d,
                                n,
                                q,
                                k,
                                implementation,
                                float(np.median(times)),
                                flush=True,
                            )


if __name__ == "__main__":
    main()
