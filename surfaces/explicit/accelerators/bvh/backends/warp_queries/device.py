"""Register native query dependencies on a caller-owned Warp kernel."""
from typing import Any, Protocol

import warp as wp


class DeviceQuery(Protocol):
    @property
    def _device_kernel(self) -> Any: ...


def configure_query_kernel(kernel: Any, *queries: DeviceQuery) -> Any:
    """Merge query build requirements before compiling a custom Warp kernel.

    Call once with every query used by the kernel, including queries called
    indirectly through another Warp function. Repeated registration is safe.
    Existing include paths, dependencies, preambles and compiler flags survive.
    This performs host setup only: no compilation, launch or data transfer.
    """
    if not isinstance(kernel, wp.Kernel):
        raise TypeError("kernel must be a decorated Warp kernel")
    if not queries:
        raise ValueError("provide at least one query specialization")
    options = wp.get_module_options(kernel.module)
    build = options.get("extra_build_options") or wp.ModuleBuildOptions()
    unroll = options.get("max_unroll", 16)
    for query in queries:
        source = wp.get_module_options(query._device_kernel.module)
        required = source["extra_build_options"]
        # Avoid repeating a registered preamble, but still merge dependency
        # paths: their contents participate in Warp's compilation cache.
        addition = wp.ModuleBuildOptions(
            extra_cpu_include_dirs=required.extra_cpu_include_dirs,
            extra_cuda_include_dirs=required.extra_cuda_include_dirs,
            extra_cpu_preamble=("" if required.extra_cpu_preamble in build.extra_cpu_preamble
                                else required.extra_cpu_preamble),
            extra_cuda_preamble=("" if required.extra_cuda_preamble in build.extra_cuda_preamble
                                 else required.extra_cuda_preamble),
            extra_build_dependencies=required.extra_build_dependencies,
        )
        build = build.merged(addition)
        unroll = max(unroll, source.get("max_unroll", 16))
    wp.set_module_options({"extra_build_options": build, "max_unroll": unroll}, module=kernel.module)
    return kernel
