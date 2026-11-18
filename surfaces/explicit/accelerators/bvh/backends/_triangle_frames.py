"""Host preparation for the built-in Euclidean triangle evaluator."""

import numpy as np


# The record holds four doubles (the plane normal and its inverse squared
# norm), six floats (the transverse axis and two local side vectors), the
# cyclic permutation, and one padding word. Keep the native record aligned
# without requiring JAX's optional float64 mode.
TRIANGLE_FRAME_WORDS = 16


def pack_triangle_frames(triangles):
    points = np.asarray(triangles, dtype=np.float64)
    rows = np.arange(len(points))
    edges = np.roll(points, -1, axis=1) - points
    squared = np.sum(edges * edges, axis=2)
    longest = squared.argmax(axis=1)
    base = edges[rows, longest]
    bc = edges[rows, (longest + 1) % 3]
    ca = edges[rows, (longest + 2) % 3]

    # Use the shorter non-base edge in the cross product. This retains the
    # altitude when two endpoints are close relative to the base length.
    shorter = np.where((squared[rows, (longest + 1) % 3] <
                        squared[rows, (longest + 2) % 3])[:, None], bc, -ca)
    normal = np.cross(base, shorter)
    height = np.sqrt(np.sum(normal * normal, axis=1))
    length2 = squared[rows, longest]
    # Local coordinates are in squared world units, and half-plane products
    # are fourth order. Mark out-of-range and poorly conditioned frames once.
    usable = ((length2 >= 1.e-16) & (length2 <= 1.e16) &
              (height >= 1.e-6 * length2))
    unit_normal = np.divide(normal, height[:, None], out=np.zeros_like(normal),
                            where=usable[:, None])
    v = np.cross(unit_normal, base)
    # Power-of-two scaling keeps the original plane coefficients exact.
    _, exponent = np.frexp(np.abs(normal).max(axis=1))
    normal = np.ldexp(normal, -exponent[:, None])
    normal2 = np.sum(normal * normal, axis=1)
    inverse_normal2 = np.divide(1., normal2, out=np.zeros_like(normal2), where=normal2 > 0.)
    cx_a = np.sum(base * -ca, axis=1)
    cx_b = np.sum(base * bc, axis=1)
    geometry = np.column_stack((v, cx_a, cx_b, height))
    geometry[~usable] = 0.
    frame = np.empty((len(points), TRIANGLE_FRAME_WORDS), dtype=np.uint32)
    frame[:, :6] = normal.view(np.uint32)
    frame[:, 6:8] = inverse_normal2[:, None].view(np.uint32)
    frame[:, 8:14] = geometry.astype(np.float32).view(np.uint32)
    frame[:, 14] = np.where(usable, longest, 3)
    frame[:, 15] = 0
    return frame
