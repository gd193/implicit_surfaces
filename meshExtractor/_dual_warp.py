"""Shared dual-contouring kernels."""
from pathlib import Path

import warp as wp
from ._dual_qef import solve_qef


_NATIVE_DIR = Path(__file__).with_name("native").resolve()
_NATIVE_OPTIONS = wp.ModuleBuildOptions(
    extra_cpu_include_dirs=[_NATIVE_DIR], extra_cuda_include_dirs=[_NATIVE_DIR],
    extra_cpu_preamble='#include "dual_grid.h"', extra_cuda_preamble='#include "dual_grid.h"',
    extra_build_dependencies=[_NATIVE_DIR / "dual_grid.h"],
)
wp.set_module_options({"extra_build_options": _NATIVE_OPTIONS, "enable_backward": False}, module=__name__)


@wp.func
def corner(k: int):
    return wp.vec3i((k ^ (k >> 1)) & 1, (k >> 1) & 1, k >> 2)


@wp.func
def edge_a(e: int):
    a = int(0)
    if e < 4:
        if e == 1: a = 3
        elif e == 2: a = 4
        elif e == 3: a = 7
    elif e < 8:
        if e == 5: a = 1
        elif e == 6: a = 4
        elif e == 7: a = 5
    else:
        a = e - 8
    return a


@wp.func
def edge_b(e: int):
    b = int(0)
    if e < 4:
        if e == 0: b = 1
        elif e == 1: b = 2
        elif e == 2: b = 5
        else: b = 6
    elif e < 8:
        if e == 4: b = 3
        elif e == 5: b = 2
        elif e == 6: b = 7
        else: b = 6
    else: b = e - 4
    return b


@wp.func
def dense_cell(row: int, ny: int, nz: int):
    return wp.vec3i(row // (ny * nz), (row // nz) % ny, row % nz)


vec8i = wp.types.vector(length=8, dtype=int)


@wp.func
def mark_crossings(corners: vec8i, values: wp.array(dtype=float), level: float, edges: wp.array(dtype=int)):
    signs = vec8i(0)
    for k in range(8):
        signs[k] = int(values[corners[k]] < level)
    found = int(0)
    for e in range(12):
        ca = edge_a(e)
        cb = edge_b(e)
        if signs[ca] != signs[cb]:
            wp.atomic_max(edges, 3*corners[ca] + e//4, 1)
            found = 1
    return found


@wp.func_native("return dual_grid::position(c, lower, spacing);")
def grid_position(c: wp.vec3i, lower: wp.vec3, spacing: wp.vec3) -> wp.vec3:
    ...


@wp.func
def point_position(points: wp.array(dtype=wp.vec3), index: int, ny: int, nz: int,
                   lower: wp.vec3, spacing: wp.vec3):
    if ny > 0:
        p = dense_cell(index, ny, nz)
        return grid_position(p, lower, spacing)
    return points[index]


@wp.func
def emit_edge(j: int, cc: wp.array2d(dtype=int), values: wp.array(dtype=float),
               points: wp.array(dtype=wp.vec3), level: float,
               positions: wp.array(dtype=wp.vec3), endpoints: wp.array2d(dtype=int),
               ids: wp.array2d(dtype=int), cell_edges: wp.array2d(dtype=int),
               face_scan: wp.array(dtype=int), ny: int, nz: int, lower: wp.vec3, spacing: wp.vec3):
    owner = int(2147483647)
    complete = int(1)
    for q in range(4):
        candidate = ids[j, q]
        if candidate >= 0:
            owner = wp.min(owner, candidate)
        else:
            complete = 0
    face_scan[j] = complete
    for e in range(12):
        if cell_edges[owner, e] == j:
            a = cc[owner, edge_a(e)]
            b = cc[owner, edge_b(e)]
            t = wp.clamp((level-values[a])/(values[b]-values[a]), 0.0, 1.0)
            pa = point_position(points, a, ny, nz, lower, spacing)
            pb = point_position(points, b, ny, nz, lower, spacing)
            positions[j] = pa + t*(pb-pa)
            endpoints[j, 0] = a
            endpoints[j, 1] = b


@wp.func
def assign_incidence(i: int, cc: wp.array2d(dtype=int), values: wp.array(dtype=float), level: float,
              scan: wp.array(dtype=int),
              ids: wp.array2d(dtype=int), cell_edges: wp.array2d(dtype=int)):
    for e in range(12):
        cell_edges[i, e] = -1
        a = int(cc[i, edge_a(e)])
        b = int(cc[i, edge_b(e)])
        if (values[a] < level) != (values[b] < level):
            j = scan[3*a+e//4]-1
            cell_edges[i, e] = j
            c = corner(edge_a(e))
            axis = e//4
            u, v = c[(axis+1)%3], c[(axis+2)%3]
            q = int(0)
            if u == 1 and v == 0: q = 1
            elif u == 1 and v == 1: q = 2
            elif u == 0 and v == 1: q = 3
            ids[j, q] = i


@wp.func
def diagonal_quality(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3):
    n0 = wp.cross(b-a, c-a)
    n1 = wp.cross(c-a, d-a)
    scale = wp.length(n0)*wp.length(n1)
    quality = float(-2.0)
    if wp.isfinite(scale) and scale > 0.0:
        quality = wp.dot(n0, n1)/scale
    return quality


@wp.func
def write_quad(a: int, b: int, c: int, d: int, reverse: bool, k: int,
               vertices: wp.array(dtype=wp.vec3), faces: wp.array(dtype=int)):
    if reverse:
        b, d = d, b
    # Preserve the quad cycle while avoiding a concave or folded diagonal.
    if diagonal_quality(vertices[b], vertices[c], vertices[d], vertices[a]) > diagonal_quality(vertices[a], vertices[b], vertices[c], vertices[d]):
        old_a = a
        a = b
        b = c
        c = d
        d = old_a
    faces[k] = a
    faces[k+1] = b
    faces[k+2] = c
    faces[k+3] = a
    faces[k+4] = c
    faces[k+5] = d


@wp.func
def place_vertex(i: int, c: wp.vec3i, cell_edges: wp.array2d(dtype=int),
                 positions: wp.array(dtype=wp.vec3), normals: wp.array(dtype=wp.vec3),
                 lower: wp.vec3, spacing: wp.vec3):
    """Place an active cell; capacity/overflow guards belong to the caller."""
    count = int(0)
    centroid = wp.vec3(0.0)
    A = wp.mat33(0.0)
    b = wp.vec3(0.0)
    origin = lower + wp.cw_mul(wp.vec3(float(c[0]), float(c[1]), float(c[2])), spacing)
    for e in range(12):
        j = cell_edges[i, e]
        if j >= 0:
            p = positions[j] - origin
            centroid += p
            count += 1
            normal = normals[j]
            length = wp.length(normal)
            if wp.isfinite(length) and length > 1.0e-20:
                normal = normal/length
                A += wp.outer(normal, normal)
                b += normal*wp.dot(normal, p)
    vertex = origin
    if count > 0:
        centroid = centroid/float(count)
        vertex = origin + solve_qef(A, b, centroid, wp.vec3(0.0), spacing)

    return vertex
