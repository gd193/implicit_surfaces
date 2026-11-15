"""Compile pinned independent point-kNN references (never launches GPU work)."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

PINS = {
    "cubql": "d2e62a39b83c1b8ca72c266703fdc0dfdcb67952",
    "cudakdtree": "37ce4c377170d13670632143b976192d5b69ed03",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("cubql", "cudakdtree", "cuda", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument(
        "--traversals", nargs="+", choices=("cct", "stack"), default=["cct", "stack"]
    )
    p.add_argument("--arch", default="86")
    p.add_argument("--dimensions", nargs="+", type=int, choices=(2, 3), default=[2, 3])
    p.add_argument("--k", nargs="+", type=int, default=[1, 4, 8, 16, 32, 64, 128])
    a = p.parse_args()
    if min(a.k) < 1:
        p.error("k must be positive")
    a.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().with_name("references")
    refs = {}
    for key, pin in PINS.items():
        root = getattr(a, key).resolve()
        head = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True
        )
        if head != pin or dirty:
            p.error(f"{key}: expected clean pinned {pin}, got {head}: {dirty}")
        refs[key] = dict(path=str(root), commit=head, dirty=dirty)
    commands = []
    completed = {}
    manifest = dict(
        references=refs,
        cuda_root=str(a.cuda.resolve()),
        commands=commands,
        nvcc=subprocess.check_output(
            [str(a.cuda / "bin/nvcc"), "--version"], text=True
        ),
        completed=completed,
        sources={
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in source.glob("knn_*")
        },
    )
    for d in a.dimensions:
        for k in a.k:
            for name in ["cubql"] + [
                f"cudakdtree_{mode}" + ("_stack" if traversal == "stack" else "")
                for traversal in a.traversals
                for mode in ("fixed", "heap")
            ]:
                cubql = name == "cubql"
                root = Path(refs["cubql" if cubql else "cudakdtree"]["path"])
                cmd = [
                    str(a.cuda / "bin/nvcc"),
                    "-ccbin",
                    "/usr/bin/g++",
                    "-std=c++17",
                    "-O3",
                    "-lineinfo",
                    "-arch=sm_" + a.arch,
                    "-I" + str(root),
                    "-DBENCH_DIM=" + str(d),
                    "-DBENCH_K=" + str(k),
                ]
                if cubql:
                    cmd += [
                        "-DCUBQL_HAVE_CUDA=1",
                        "-DCUBQL_INSTANTIATE_T=float",
                        "-DCUBQL_INSTANTIATE_D=" + str(d),
                        str(source / "knn_cubql.cu"),
                        str(root / "cuBQL/builder/cuda/instantiate_builders.cu"),
                    ]
                else:
                    cmd += [
                        "-DBENCH_HEAP=" + str(int("heap" in name)),
                        "-DBENCH_STACK=" + str(int(name.endswith("_stack"))),
                        str(source / "knn_cudakdtree.cu"),
                    ]
                cmd += ["-o", str(a.output / f"{name}_d{d}_k{k}")]
                commands.append(cmd)
                before = {
                    f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                    for f in source.glob("knn_*")
                }
                (a.output / "build.json").write_text(
                    json.dumps(manifest, indent=2) + "\n"
                )
                print("Building", cmd[-1], flush=True)
                subprocess.run(cmd, check=True)
                after = {
                    f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                    for f in source.glob("knn_*")
                }
                if before != after:
                    raise RuntimeError("source changed during compilation; rebuild")
                completed[Path(cmd[-1]).name] = dict(command=cmd, sources=before)
    manifest["binaries"] = {
        name: hashlib.sha256((a.output / name).read_bytes()).hexdigest()
        for name in completed
    }
    (a.output / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
