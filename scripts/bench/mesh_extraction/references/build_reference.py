"""Build pinned VTK-m CUDA references into an explicit artifact directory."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

COMMIT = "2cd5a5e168124787192f4f857e7e7df58793a67a"
REPO = "https://gitlab.kitware.com/vtk/vtk-m.git"
SUCCESSOR_REPO = "https://github.com/Viskores/viskores.git"
SUCCESSOR_COMMIT = "2f90c079cd4e191f0ea17c37971e611616d1f531"


def source_hashes():
    """Hash all shipped reference Python, CUDA and CMake sources."""
    root = Path(__file__).resolve().parent
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and (path.suffix in (".py", ".cu") or path.name == "CMakeLists.txt")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", type=Path, default=Path(sys.prefix))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-compiler", default="g++")
    parser.add_argument("--arch", default="86")
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    code = Path(__file__).resolve().parent
    initial_hashes = source_hashes()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source, build, adapter_build = (
        root / "vtk-m",
        root / "vtkm-build",
        root / "native-build",
    )
    commands = []

    def run(command):
        commands.append([str(x) for x in command])
        subprocess.run(commands[-1], check=True, cwd=root)

    if not source.exists():
        run(["git", "clone", REPO, source])
        run(["git", "-C", source, "checkout", COMMIT])
    current = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if current != COMMIT:
        raise RuntimeError(f"Expected VTK-m {COMMIT}, found {current}")

    # CUDA13 CCCL requires C++17 and explicit formerly transitive includes.
    path = source / "CMakeLists.txt"
    path.write_text(
        path.read_text().replace("set(CMAKE_CXX_STANDARD 14)", "set(CMAKE_CXX_STANDARD 17)")
    )
    path = source / "vtkm/exec/cuda/internal/ExecutionPolicy.h"
    content = path.read_text()
    if "#include <thrust/pair.h>" not in content:
        path.write_text(
            content.replace(
                "#include <thrust/execution_policy.h>",
                "#include <thrust/pair.h>\n#include <thrust/distance.h>\n#include <thrust/execution_policy.h>",
            )
        )

    # Extract the CUDA13 allocation API backport from pinned successor source.
    successor = root / "viskores"
    if not successor.exists():
        run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                SUCCESSOR_REPO,
                successor,
            ]
        )

    def successor_text(path):
        return subprocess.check_output(
            ["git", "-C", str(successor), "show", SUCCESSOR_COMMIT + ":" + path],
            text=True,
        )

    path = source / "vtkm/cont/cuda/internal/CudaAllocator.cu"
    content = path.read_text()
    begin, end = (
        "void CudaAllocator::PrepareForControl",
        "void CudaAllocator::Initialize",
    )
    allocator = successor_text("viskores/cont/cuda/internal/CudaAllocator.cu")
    allocator = allocator.replace("viskores", "vtkm").replace("VISKORES", "VTKM")
    path.write_text(
        content[: content.index(begin)]
        + allocator[allocator.index(begin) : allocator.index(end)]
        + content[content.index(end) :]
    )
    (source / "VISKORES_LICENSE.txt").write_text(successor_text("LICENSE.txt"))
    host_compiler = shutil.which(args.host_compiler)
    if not host_compiler:
        parser.error(f"host compiler not found: {args.host_compiler}")
    common = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CUDA_COMPILER=" + str(args.cuda / "bin/nvcc"),
        "-DCMAKE_CUDA_HOST_COMPILER=" + host_compiler,
        "-DCMAKE_CUDA_ARCHITECTURES=" + args.arch,
        "-DCMAKE_CUDA_STANDARD=17",
    ]
    run(
        [
            "cmake",
            "-S",
            source,
            "-B",
            build,
            *common,
            "-DCMAKE_CUDA_FLAGS=-std=c++17",
            "-DVTKm_ENABLE_CUDA=ON",
            "-DVTKm_ENABLE_RENDERING=OFF",
            "-DVTKm_ENABLE_TESTING=OFF",
            "-DVTKm_ENABLE_EXAMPLES=OFF",
            "-DVTKm_ENABLE_DEVELOPER_FLAGS=OFF",
            "-DVTKm_USE_64BIT_IDS=OFF",
        ]
    )
    run(["cmake", "--build", build, "--target", "vtkm_worklet", "-j", str(args.jobs)])
    run(
        [
            "cmake",
            "-S",
            code / "adapter",
            "-B",
            adapter_build,
            *common,
            "-DVTKm_DIR=" + str(build / "lib/cmake/vtkm-2.3"),
            "-DVTKm_SOURCE_DIR=" + str(source),
            "-DVTKm_BINARY_DIR=" + str(build / "include"),
        ]
    )
    run(["cmake", "--build", adapter_build, "-j", str(args.jobs)])
    if source_hashes() != initial_hashes:
        raise RuntimeError("reference sources changed during the build")
    library = adapter_build / "libvtkm_mc_reference.so"
    patch = subprocess.check_output(["git", "-C", str(source), "diff"], text=True)
    (root / "upstream-build-compatibility.patch").write_text(patch)
    manifest = {
        "upstream": REPO,
        "commit": COMMIT,
        "commands": commands,
        "license": "VTK-m License Version 2.3; see vtk-m/LICENSE.txt",
        "source_changes": patch,
        "source_sha256": initial_hashes,
        "cuda13_backport_source": {
            "repository": SUCCESSOR_REPO,
            "commit": SUCCESSOR_COMMIT,
        },
        "library": str(library),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "grid_copy": False,
        "merge_duplicate_points": True,
        "float_precision": "FP32",
        "indices": "int32",
    }
    (root / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
