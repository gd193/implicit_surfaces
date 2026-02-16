from drjit.auto import Array2f, Array3f, PCG32
import drjit as dr

def sample_uniform_1sphere(rng: PCG32) -> Array2f:
    theta = rng.next_float32() * dr.pi * 2.
    return Array2f([dr.cos(theta), dr.sin(theta)])



# def sample_uniform_2sphere(rng: PCG32) -> Array3f:
#     """
#     Sample a point uniformly on the surface of a 2-sphere.
#     Wrapper around mitsuba's square_to_uniform_sphere.
#     """
#     return Array3f(square_to_uniform_sphere(Array2f([rng.next_float32(), rng.next_float32()])))

def sample_uniform_2sphere(rng: PCG32) -> Array3f:
    """
    Sample a point uniformly on the surface of a 2-sphere.
    Taken from Mitsuba https://github.com/mitsuba-renderer/mitsuba3/blob/master/include/mitsuba/core/warp.h square_to_uniform_sphere
    """
    z = dr.fma(2.0, rng.next_float32(), -1.0)
    r = dr.safe_sqrt(dr.fma(-z, z, 1.0))
    phi = rng.next_float32() * dr.pi * 2.0
    return Array3f([r * dr.cos(phi), r * dr.sin(phi), z])
