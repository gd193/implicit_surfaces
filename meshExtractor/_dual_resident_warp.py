"""Capacity kernels: only valid prefixes may be read; shared exact DC math."""

import warp as wp
from meshExtractor import _dual_warp as shared

wp.set_module_options({"extra_build_options": shared._NATIVE_OPTIONS})


@wp.kernel(enable_backward=False)
def classify(values: wp.array(dtype=float), nx: int, ny: int, nz: int, level: float,
             edge_flags: wp.array(dtype=int), cell_flags: wp.array(dtype=int)):
    i = wp.tid()
    if i < edge_flags.shape[0]:
        node = i // 3
        axis = i % 3
        p = shared.dense_cell(node, ny + 1, nz + 1)
        valid = False
        delta = int(1)
        if axis == 0:
            valid = p[0] < nx
            delta = (ny + 1) * (nz + 1)
        elif axis == 1:
            valid = p[1] < ny
            delta = nz + 1
        else:
            valid = p[2] < nz
        edge_flags[i] = 0
        if valid:
            edge_flags[i] = int((values[node] < level) != (values[node + delta] < level))
    if i < cell_flags.shape[0]:
        p = shared.dense_cell(i, ny, nz)
        first = False
        active = False
        for k in range(8):
            c = p + shared.corner(k)
            node = (c[0] * (ny + 1) + c[1]) * (nz + 1) + c[2]
            sign = values[node] < level
            if k == 0:
                first = sign
            elif sign != first:
                active = True
        cell_flags[i] = int(active)


@wp.kernel(enable_backward=False)
def compact(scan: wp.array(dtype=int), counts: wp.array(dtype=int), ny: int, nz: int,
            cells: wp.array(dtype=wp.vec3i), cc: wp.array2d(dtype=int)):
    i = wp.tid()
    previous = int(0)
    if i > 0:
        previous = scan[i - 1]
    if counts[3] == 0 and scan[i] > previous:
        j = scan[i] - 1
        if j < cells.shape[0]:
            p = shared.dense_cell(i, ny, nz)
            cells[j] = p
            for k in range(8):
                c = p + shared.corner(k)
                cc[j, k] = (c[0] * (ny + 1) + c[1]) * (nz + 1) + c[2]


@wp.kernel(enable_backward=False)
def incidence(cc: wp.array2d(dtype=int), values: wp.array(dtype=float), level: float, scan: wp.array(dtype=int),
              counts: wp.array(dtype=int), ids: wp.array2d(dtype=int), cell_edges: wp.array2d(dtype=int)):
    i = wp.tid()
    if counts[3] == 0 and i < counts[0]:
        shared.assign_incidence(i, cc, values, level, scan, ids, cell_edges)


@wp.kernel(enable_backward=False)
def edges(cc: wp.array2d(dtype=int), values: wp.array(dtype=float), points: wp.array(dtype=wp.vec3), level: float,
          ids: wp.array2d(dtype=int), cell_edges: wp.array2d(dtype=int), counts: wp.array(dtype=int), ny: int,
          nz: int, ox: float, oy: float, oz: float, sx: float, sy: float, sz: float,
          positions: wp.array(dtype=wp.vec3), endpoints: wp.array2d(dtype=int), flags: wp.array(dtype=int)):
    i = wp.tid()
    flags[i] = 0
    if counts[3] == 0 and i < counts[1]:
        shared.emit_edge(
            i,
            cc,
            values,
            points,
            level,
            positions,
            endpoints,
            ids,
            cell_edges,
            flags,
            ny,
            nz,
            wp.vec3(ox, oy, oz),
            wp.vec3(sx, sy, sz),
        )


@wp.kernel(enable_backward=False)
def place(cells: wp.array(dtype=wp.vec3i), cell_edges: wp.array2d(dtype=int), positions: wp.array(dtype=wp.vec3),
          normals: wp.array(dtype=wp.vec3), counts: wp.array(dtype=int), ox: float, oy: float, oz: float,
          sx: float, sy: float, sz: float, vertices: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    if counts[3] == 0 and i < counts[0]:
        vertices[i] = shared.place_vertex(
            i, cells[i], cell_edges, positions, normals, wp.vec3(ox, oy, oz), wp.vec3(sx, sy, sz)
        )


@wp.kernel(enable_backward=False)
def faces(ids: wp.array2d(dtype=int), endpoints: wp.array2d(dtype=int), values: wp.array(dtype=float),
          level: float, scan: wp.array(dtype=int), vertices: wp.array(dtype=wp.vec3), counts: wp.array(dtype=int),
          result: wp.array(dtype=int)):
    i = wp.tid()
    if counts[3] == 0 and i < counts[1]:
        if ids[i, 0] >= 0 and ids[i, 1] >= 0 and ids[i, 2] >= 0 and ids[i, 3] >= 0:
            shared.write_quad(
                ids[i, 0],
                ids[i, 1],
                ids[i, 2],
                ids[i, 3],
                values[endpoints[i, 0]] >= level,
                6 * (scan[i] - 1),
                vertices,
                result,
            )


@wp.kernel(enable_backward=False)
def classify_sparse(corner_map: wp.array(dtype=int), values: wp.array(dtype=float), upstream: wp.array(dtype=int),
                    level: float, edge_flags: wp.array(dtype=int), cell_flags: wp.array(dtype=int)):
    i = wp.tid()
    cell_flags[i] = 0
    if upstream[3] == 0 and i < upstream[5] // 8:
        corners = shared.vec8i(0)
        for k in range(8):
            corners[k] = corner_map[8 * i + k]
        cell_flags[i] = shared.mark_crossings(corners, values, level, edge_flags)


@wp.kernel(enable_backward=False)
def compact_sparse(candidates: wp.array(dtype=wp.vec3i), corner_map: wp.array(dtype=int),
                   scan: wp.array(dtype=int), counts: wp.array(dtype=int), cells: wp.array(dtype=wp.vec3i),
                   cc: wp.array2d(dtype=int)):
    i = wp.tid()
    previous = int(0)
    if i > 0:
        previous = scan[i - 1]
    if counts[3] == 0 and scan[i] > previous:
        j = scan[i] - 1
        if j < cells.shape[0]:
            cells[j] = candidates[i]
            for k in range(8):
                cc[j, k] = corner_map[8 * i + k]


@wp.kernel(enable_backward=False)
def decode_points(codes: wp.array(dtype=wp.int64), count: wp.array(dtype=int), stride: wp.int64, ox: float,
                  oy: float, oz: float, sx: float, sy: float, sz: float, points: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    point = wp.vec3(0.0)
    if i < count[0]:
        code = codes[i]
        x = code // (stride * stride)
        rem = code - x * stride * stride
        y = rem // stride
        z = rem - y * stride
        point = wp.vec3(ox, oy, oz) + wp.cw_mul(
            wp.vec3(sx, sy, sz), wp.vec3(float(wp.int32(x)), float(wp.int32(y)), float(wp.int32(z)))
        )
    points[i] = point
