"""Shared dual-contouring kernels and dynamically allocated workspace."""
from dataclasses import dataclass
from pathlib import Path

import warp as wp
from warp._src import utils
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


@wp.func
def cell_corner(cc: wp.array2d(dtype=int), row: int, k: int, ny: int, nz: int):
    if ny > 0:
        p = dense_cell(row, ny, nz) + corner(k)
        return (p[0] * (ny + 1) + p[1]) * (nz + 1) + p[2]
    return cc[row, k]


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


@wp.kernel
def mark(cc: wp.array2d(dtype=int), values: wp.array(dtype=float), level: float,
         edges: wp.array(dtype=int), active: wp.array(dtype=int), ny: int, nz: int):
    i = wp.tid()
    corners = vec8i(0)
    for k in range(8):
        corners[k] = cell_corner(cc, i, k, ny, nz)
    active[i] = mark_crossings(corners, values, level, edges)


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



@wp.kernel
def emit_edges(cc: wp.array2d(dtype=int), values: wp.array(dtype=float),
               points: wp.array(dtype=wp.vec3), level: float,
               positions: wp.array(dtype=wp.vec3), endpoints: wp.array2d(dtype=int),
               ids: wp.array2d(dtype=int), cell_edges: wp.array2d(dtype=int),
               face_scan: wp.array(dtype=int), ny: int, nz: int, lower: wp.vec3, spacing: wp.vec3):
    j = wp.tid()
    emit_edge(j, cc, values, points, level, positions, endpoints, ids, cell_edges, face_scan,
              ny, nz, lower, spacing)


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



@wp.kernel
def incidence(cc: wp.array2d(dtype=int), values: wp.array(dtype=float), level: float,
              scan: wp.array(dtype=int),
              ids: wp.array2d(dtype=int), cell_edges: wp.array2d(dtype=int)):
    i = wp.tid()
    assign_incidence(i, cc, values, level, scan, ids, cell_edges)


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


@wp.kernel
def emit_faces(ids: wp.array2d(dtype=int), endpoints: wp.array2d(dtype=int),
               values: wp.array(dtype=float), level: float, scan: wp.array(dtype=int),
               vertices: wp.array(dtype=wp.vec3), faces: wp.array(dtype=int)):
    i = wp.tid()
    if ids[i, 0] >= 0 and ids[i, 1] >= 0 and ids[i, 2] >= 0 and ids[i, 3] >= 0:
        a = int(ids[i, 0])
        b = int(ids[i, 1])
        c = int(ids[i, 2])
        d = int(ids[i, 3])
        write_quad(a, b, c, d, values[endpoints[i, 0]] >= level,
                   6*(scan[i]-1), vertices, faces)


@wp.kernel
def compact(cells: wp.array(dtype=wp.vec3i), cc: wp.array2d(dtype=int),
            scan: wp.array(dtype=int), out: wp.array(dtype=wp.vec3i),
            out_cc: wp.array2d(dtype=int), ny: int, nz: int):
    i = wp.tid()
    previous = int(0)
    if i > 0:
        previous = scan[i-1]
    if scan[i] > previous:
        j = scan[i]-1
        if ny > 0:
            out[j] = dense_cell(i, ny, nz)
        else:
            out[j] = cells[i]
        for k in range(8):
            out_cc[j, k] = cell_corner(cc, i, k, ny, nz)



@wp.kernel(enable_backward=False)
def compact_children(cells: wp.array(dtype=wp.vec3i), scan: wp.array(dtype=int),
                      out: wp.array(dtype=wp.vec3i), final: bool):
    i = wp.tid()
    previous = int(0)
    if i > 0:
        previous = scan[i-1]
    if scan[i] > previous:
        j = scan[i]-1
        c = cells[i]
        if final:
            out[j] = c
        else:
            for k in range(8):
                out[8*j+k] = 2*c + wp.vec3i(k & 1, (k >> 1) & 1, k >> 2)


@dataclass
class Workspace:
    positions: wp.array
    edge_endpoints: wp.array
    cells: wp.array
    cell_edges: wp.array
    incident: wp.array
    corner_values: wp.array
    lower: wp.vec3
    spacing: wp.vec3
    level: float
    face_scan: wp.array
    face_count: int


@wp.kernel
def count_tails(a: wp.array(dtype=int), b: wp.array(dtype=int), counts: wp.array(dtype=int)):
    if a.shape[0] > 0:
        counts[0] = a[a.shape[0]-1]
    else:
        counts[0] = 0
    if b.shape[0] > 0:
        counts[1] = b[b.shape[0]-1]
    else:
        counts[1] = 0



def prepare(cells, cell_corners, corner_positions, corner_values, lower, spacing, level):
    device = corner_values.device
    ny, nz = 0, 0
    if isinstance(cells, tuple):
        nx, ny, nz = cells
        n = nx * ny * nz
        cells = None
    else:
        n = cells.shape[0]
    m = corner_values.shape[0]
    if max(3*m, 12*n) > 2147483647:
        raise ValueError("dual contouring exceeds int32 indexing")
    edge_scan = wp.zeros(3*m, dtype=int, device=device)
    active = wp.empty(n, dtype=int, device=device)
    if n:
        wp.launch(mark, n, [cell_corners, corner_values, level, edge_scan, active, ny, nz], device=device)
    for array in (edge_scan, active):
        if array.size:
            utils.array_scan(array, array, inclusive=True)
    counts = wp.empty(2, dtype=int, device=device)
    wp.launch(count_tails, 1, [edge_scan, active, counts], device=device)
    ne, nv = map(int, counts.numpy())
    compact_cells = wp.empty(nv, dtype=wp.vec3i, device=device)
    compact_corners = wp.empty((nv, 8), dtype=int, device=device)
    if n:
        wp.launch(compact, n, [cells, cell_corners, active, compact_cells, compact_corners, ny, nz], device=device)
    cells, cell_corners, n = compact_cells, compact_corners, nv
    positions = wp.empty(ne, dtype=wp.vec3, device=device)
    endpoints = wp.empty((ne, 2), dtype=int, device=device)
    incident = wp.full((ne, 4), -1, dtype=int, device=device)
    cell_edges = wp.empty((n, 12), dtype=int, device=device)
    face_scan = wp.empty(ne, dtype=int, device=device)
    if n:
        wp.launch(incidence, n, [cell_corners, corner_values, level, edge_scan,
                               incident, cell_edges], device=device)
    if ne:
        wp.launch(emit_edges, ne, [cell_corners, corner_values, corner_positions, level,
                                 positions, endpoints, incident, cell_edges, face_scan,
                                 ny + 1 if ny else 0, nz + 1 if ny else 0, lower, spacing], device=device)
    nf = 0
    if ne:
        utils.array_scan(face_scan, face_scan, inclusive=True)
        nf = int(face_scan[-1:].numpy()[0])
    if 6*nf > 2147483647:
        raise ValueError("dual contouring exceeds int32 face indexing")
    return Workspace(positions, endpoints, cells, cell_edges, incident,
                     corner_values, lower, spacing, level, face_scan, nf)


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


@wp.kernel
def place(cells: wp.array(dtype=wp.vec3i), cell_edges: wp.array2d(dtype=int),
          positions: wp.array(dtype=wp.vec3),
          normals: wp.array(dtype=wp.vec3), lower: wp.vec3, spacing: wp.vec3,
          vertices: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    vertices[i] = place_vertex(i, cells[i], cell_edges, positions, normals, lower, spacing)


def finish(w: Workspace, normals):
    device = w.cells.device
    vertices = wp.empty(w.cells.size, dtype=wp.vec3, device=device)
    if w.cells.size:
        wp.launch(place, w.cells.size,
                  [w.cells, w.cell_edges, w.positions, normals,
                   w.lower, w.spacing, vertices], device=device)
    faces = wp.empty(6 * w.face_count, dtype=int, device=device)
    if w.face_count:
        wp.launch(emit_faces, w.positions.size,
                  [w.incident, w.edge_endpoints, w.corner_values, w.level, w.face_scan, vertices, faces], device=device)
    return vertices, faces
