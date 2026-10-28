"""Optional exact analytic volume queries for the fused pressure source.

This capability mirrors BoxBoundary and ExteriorTorusBoundary containment.
It performs no ray solving or distance-bound estimation. Unknown geometries,
including subclasses with different semantics, retain the generic JAX path.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import warp as wp


@dataclass(frozen=True)
class VolumeGeometry:
    kind: str
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    center: tuple[float, float, float] = (0., 0., 0.)
    major: float = 0.
    minor: float = 0.
    level_set: float = 0.


def volume_geometry(geometry) -> VolumeGeometry | None:
    """Resolve the optional capability once, before tracing sample work."""
    from examples.fluids.pressure_prototype.boundary_sampling import BoxBoundary, ExteriorTorusBoundary
    # Sphere containment's three-component reduction can choose a different
    # floating-point order for different XLA shapes. Keep its JAX mask path.
    kinds = {BoxBoundary: 'box', ExteriorTorusBoundary: 'torus'}
    kind = kinds.get(type(geometry))
    if kind is None:
        return None
    def values(x):
        return tuple(float(v) for v in np.asarray(x))

    if kind == 'box':
        return VolumeGeometry(kind, values(geometry.lower), values(geometry.upper))
    return VolumeGeometry(kind, values(geometry.lower), values(geometry.upper),
                          values(geometry.center), geometry.major, geometry.minor,
                          float(geometry.surface.level_set))


@lru_cache(maxsize=32)
def warp_volume_contains(dtype, geometry: VolumeGeometry):
    scalar = wp.float32 if dtype == np.dtype('float32') else wp.float64
    vector = wp.vec3 if scalar == wp.float32 else wp.vec3d
    lower, upper, center = (wp.constant(vector(*v)) for v in
                            (geometry.lower, geometry.upper, geometry.center))
    major = wp.constant(scalar(geometry.major))
    minor = wp.constant(scalar(geometry.minor))
    level = wp.constant(scalar(geometry.level_set))
    safe_norm_eps = wp.constant(scalar(1e-12))
    torus = geometry.kind == 'torus'

    # XLA's CUDA f32 containment fusion lowers sqrt to sqrt.approx.f32.
    # Warp's ordinary sqrt uses sqrt.rn.f32; that changes strict signs for
    # samples within an ulp of the obstacle, even with identical positions.
    # Keep this instruction local to masks; pressure/interpolation arithmetic
    # and the f64 correctly rounded sqrt retain their original operations.
    @wp.func_native('''
        float result;
        asm("sqrt.approx.f32 %0, %1;" : "=f"(result) : "f"(x));
        return result;
        ''' if scalar == wp.float32 else "return sqrt(x);")
    def mask_sqrt(x: scalar) -> scalar:
        ...

    @wp.func
    def contains(x: vector) -> wp.bool:
        inside_box = (x[0] > lower[0] and x[0] < upper[0]
                      and x[1] > lower[1] and x[1] < upper[1]
                      and x[2] > lower[2] and x[2] < upper[2])
        if wp.static(torus):
            p = x-center
            radial = mask_sqrt(p[0]*p[0]+p[2]*p[2]+safe_norm_eps)-major
            value = mask_sqrt(radial*radial+p[1]*p[1]+safe_norm_eps)-minor-level
            return inside_box and value > scalar(0.)
        return inside_box

    return contains
