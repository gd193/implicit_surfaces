"""Build benchmark executables against existing cuBQL and FCPW/Slang-RHI builds."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fcpw", "fcpw-build", "cubql", "cuda", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--arch", default="86")
    args = parser.parse_args()
    paths = {k: v.resolve() for k, v in vars(args).items() if isinstance(v, Path)}
    source = Path(__file__).resolve().with_name("references")
    output = paths["output"]
    output.mkdir(parents=True, exist_ok=True)
    fcpw, build, cuda = (paths[k] for k in ("fcpw", "fcpw_build", "cuda"))
    rhi = build/"source/deps/slang-rhi"
    slang = build/"_deps/slang-src"
    includes = [fcpw/"include", fcpw/"deps/eigen", fcpw/"deps/slang-rhi/include",
                fcpw/"deps/slang-rhi/src", rhi/"include", slang/"include", cuda/"targets/x86_64-linux/include"]
    commands = [["g++", "-O3", "-DNDEBUG", "-std=c++20", "-DFCPW_USE_GPU", '-DSLANG_USER_CONFIG="slang-user-config.h"',
                 *("-I"+str(p) for p in includes), str(source/"fcpw_queries.cpp"), str(source/"fcpw_queue.cpp"),
                 str(rhi/"libslang-rhi.a"), str(slang/"lib/libslang-compiler.so"),
                 str(rhi/"libslang-rhi-vma.a"), str(rhi/"libslang-rhi-resources.a"),
                 "-lcuda", "-ldl", "-Wl,-rpath,"+str(slang/"lib"), "-o", str(output/"fcpw_queries")]]
    for dimension in (2, 3):
        flags = [str(cuda/"bin/nvcc"), "-ccbin", "/usr/bin/g++", "-std=c++17", "-O3", "-lineinfo", "-arch=sm_"+args.arch,
                 "-I"+str(paths["cubql"]), "-DCUBQL_HAVE_CUDA=1", "-DBENCH_DIM="+str(dimension),
                 "-DCUBQL_INSTANTIATE_T=float", "-DCUBQL_INSTANTIATE_D="+str(dimension)]
        commands += [flags + [str(source/"cubql_points.cu"), str(paths["cubql"]/"cuBQL/builder/cuda/instantiate_builders.cu"),
                              "-o", str(output/f"cubql_points{dimension}")],
                     flags + ["--ptx", str(source/"cubql_points.cu"), "-o", str(output/f"cubql_points{dimension}.ptx")]]
    manifest = {"commands": commands, "cuda_root": str(cuda), "references": {name: {
        "path": str(paths[name]), "commit": subprocess.check_output(["git", "-C", str(paths[name]), "rev-parse", "HEAD"], text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "-C", str(paths[name]), "status", "--porcelain", "--untracked-files=no"], text=True)),
    } for name in ("fcpw", "cubql")}}
    for command in commands:
        print("Building", command[-1], flush=True)
        subprocess.run(command, check=True)
    (output/"build.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
