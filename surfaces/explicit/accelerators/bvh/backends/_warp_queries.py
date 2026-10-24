"""Closest-point and overlap kernels; root=0, triangles stored in leaf order."""
from __future__ import annotations

import warp as wp

from ._warp_geometry import (
    boxes_overlap,
    closest_triangle,
    finite3,
    point_box_distance_sq,
)
from ._warp_specs import STACK_CAPACITY


_STACK64 = wp.types.vector(length=STACK_CAPACITY, dtype=wp.int32)
_STACK_LIMIT = wp.constant(STACK_CAPACITY)


@wp.kernel
def closest_point_kernel(
    query: wp.array(dtype=wp.vec3),
    node_lower: wp.array(dtype=wp.vec3),
    node_upper: wp.array(dtype=wp.vec3),
    node_left: wp.array(dtype=wp.int32),
    node_right: wp.array(dtype=wp.int32),
    node_leaf_offset: wp.array(dtype=wp.int32),
    node_leaf_count: wp.array(dtype=wp.int32),
    prim_ids: wp.array(dtype=wp.int32),
    tri_v0: wp.array(dtype=wp.vec3),
    tri_e1: wp.array(dtype=wp.vec3),
    tri_e2: wp.array(dtype=wp.vec3),
    out_point: wp.array(dtype=wp.vec3),
    out_distance: wp.array(dtype=wp.float32),
    out_primitive_id: wp.array(dtype=wp.int32),
    out_status: wp.array(dtype=wp.int32),
):
    qi = wp.tid()
    p = query[qi]
    out_point[qi] = wp.vec3(0.0, 0.0, 0.0)
    out_distance[qi] = wp.float32(wp.inf)
    out_primitive_id[qi] = wp.int32(-1)
    out_status[qi] = wp.int32(0)
    if not finite3(p):
        out_status[qi] = wp.int32(-1)
        return
    if wp.len(node_left) == 0:
        return

    stack = _STACK64()
    sp = wp.int32(0)
    node = wp.int32(0)
    best_d2 = wp.float32(3.4028234e38)
    best_point = wp.vec3(0.0, 0.0, 0.0)
    best_id = wp.int32(-1)
    failed = wp.bool(False)
    while node >= 0 or sp > 0:
        if node < 0:
            sp = sp - 1
            node = stack[sp]
        lower = node_lower[node]
        upper = node_upper[node]
        if point_box_distance_sq(p, lower, upper) <= best_d2:
            count = node_leaf_count[node]
            if count > 0:
                offset = node_leaf_offset[node]
                for j in range(count):
                    k = offset + j
                    a = tri_v0[k]
                    # Cull triangles before projection/edge tests.
                    b = a + tri_e1[k]
                    c = a + tri_e2[k]
                    tri_lo = wp.vec3(wp.min(a[0], wp.min(b[0], c[0])),
                                     wp.min(a[1], wp.min(b[1], c[1])),
                                     wp.min(a[2], wp.min(b[2], c[2])))
                    tri_hi = wp.vec3(wp.max(a[0], wp.max(b[0], c[0])),
                                     wp.max(a[1], wp.max(b[1], c[1])),
                                     wp.max(a[2], wp.max(b[2], c[2])))
                    # Preserve distance ties after float32 vertex reconstruction.
                    cull_limit = best_d2 * 1.000002 + 1.0e-12 * wp.dot(p, p)
                    if point_box_distance_sq(p, tri_lo, tri_hi) <= cull_limit:
                        q = closest_triangle(p, a, tri_e1[k], tri_e2[k])
                        delta = p - q
                        d2 = wp.dot(delta, delta)
                        pid = prim_ids[k]
                        if d2 < best_d2 or (d2 == best_d2 and (best_id < 0 or pid < best_id)):
                            best_d2 = d2
                            best_point = q
                            best_id = pid
                node = wp.int32(-1)
            else:
                left = node_left[node]
                right = node_right[node]
                dl = point_box_distance_sq(p, node_lower[left], node_upper[left])
                dr = point_box_distance_sq(p, node_lower[right], node_upper[right])
                visit_left = dl <= best_d2
                visit_right = dr <= best_d2
                if visit_left and visit_right:
                    near = left
                    far = right
                    if dr < dl:
                        near = right
                        far = left
                    if sp >= _STACK_LIMIT:
                        failed = True
                        node = wp.int32(-1)
                        sp = wp.int32(0)
                    else:
                        stack[sp] = far
                        sp = sp + 1
                        node = near
                elif visit_left:
                    node = left
                elif visit_right:
                    node = right
                else:
                    node = wp.int32(-1)
        else:
            node = wp.int32(-1)

    if failed:
        out_status[qi] = wp.int32(-1)
    elif best_id >= 0:
        out_point[qi] = best_point
        out_distance[qi] = wp.sqrt(best_d2)
        out_primitive_id[qi] = best_id
        out_status[qi] = wp.int32(1)


@wp.kernel
def aabb_overlap_kernel(
    query_lower: wp.array(dtype=wp.vec3),
    query_upper: wp.array(dtype=wp.vec3),
    node_lower: wp.array(dtype=wp.vec3),
    node_upper: wp.array(dtype=wp.vec3),
    node_left: wp.array(dtype=wp.int32),
    node_right: wp.array(dtype=wp.int32),
    node_leaf_offset: wp.array(dtype=wp.int32),
    node_leaf_count: wp.array(dtype=wp.int32),
    prim_ids: wp.array(dtype=wp.int32),
    tri_v0: wp.array(dtype=wp.vec3),
    tri_e1: wp.array(dtype=wp.vec3),
    tri_e2: wp.array(dtype=wp.vec3),
    out_overlap: wp.array(dtype=wp.int32),
    out_status: wp.array(dtype=wp.int32),
):
    qi = wp.tid()
    qlo = query_lower[qi]
    qhi = query_upper[qi]
    out_overlap[qi] = wp.int32(0)
    out_status[qi] = wp.int32(0)
    if (not finite3(qlo) or not finite3(qhi)
            or qlo[0] > qhi[0] or qlo[1] > qhi[1] or qlo[2] > qhi[2]):
        out_status[qi] = wp.int32(-1)
        return
    if wp.len(node_left) == 0:
        return
    if not boxes_overlap(qlo, qhi, node_lower[0], node_upper[0]):
        return

    stack = _STACK64()
    sp = wp.int32(0)
    node = wp.int32(0)
    found = wp.bool(False)
    failed = wp.bool(False)
    while (node >= 0 or sp > 0) and not found and not failed:
        if node < 0:
            sp = sp - 1
            node = stack[sp]
        if boxes_overlap(qlo, qhi, node_lower[node], node_upper[node]):
            count = node_leaf_count[node]
            if count > 0:
                offset = node_leaf_offset[node]
                for j in range(count):
                    k = offset + j
                    a = tri_v0[k]
                    b = a + tri_e1[k]
                    c = a + tri_e2[k]
                    tri_lo = wp.min(a, wp.min(b, c))
                    tri_hi = wp.max(a, wp.max(b, c))
                    if boxes_overlap(qlo, qhi, tri_lo, tri_hi):
                        found = True
                node = wp.int32(-1)
            else:
                left = node_left[node]
                right = node_right[node]
                hl = boxes_overlap(qlo, qhi, node_lower[left], node_upper[left])
                hr = boxes_overlap(qlo, qhi, node_lower[right], node_upper[right])
                if hl and hr:
                    if sp >= _STACK_LIMIT:
                        failed = True
                    else:
                        stack[sp] = right
                        sp = sp + 1
                        node = left
                elif hl:
                    node = left
                elif hr:
                    node = right
                else:
                    node = wp.int32(-1)
        else:
            node = wp.int32(-1)

    if failed:
        out_status[qi] = wp.int32(-1)
    elif found:
        out_overlap[qi] = wp.int32(1)
        out_status[qi] = wp.int32(1)
