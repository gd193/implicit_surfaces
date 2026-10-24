"""Sphere-guided area sampling; three U[0,1) variates.
PDF is unconditional; retain null events (status 0) in averages."""
from __future__ import annotations

import warp as wp

from ._warp_geometry import finite3, point_box_distance_sq


@wp.func
def triangle_box_overlaps_sphere(
    point: wp.vec3, radius2: float, a: wp.vec3, e1: wp.vec3, e2: wp.vec3,
) -> bool:
    b = a + e1
    c = a + e2
    lower = wp.min(wp.min(a, b), c)
    upper = wp.max(wp.max(a, b), c)
    return point_box_distance_sq(point, lower, upper) <= radius2


@wp.kernel
def sample_boundary_kernel(
    points: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.float32),
    randoms: wp.array(dtype=wp.vec3),
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
    tri_normal: wp.array(dtype=wp.vec3),
    tri_area: wp.array(dtype=wp.float32),
    node_area: wp.array(dtype=wp.float32),
    out_point: wp.array(dtype=wp.vec3),
    out_normal: wp.array(dtype=wp.vec3),
    out_pdf: wp.array(dtype=wp.float32),
    out_id: wp.array(dtype=wp.int32),
    out_status: wp.array(dtype=wp.int32),
):
    q = wp.tid()
    point = points[q]
    radius = radii[q]
    rnd = randoms[q]
    out_point[q] = wp.vec3(0.0)
    out_normal[q] = wp.vec3(0.0)
    out_pdf[q] = 0.0
    out_id[q] = -1
    out_status[q] = 0
    if (not finite3(point) or wp.isnan(radius) or radius < 0.0 or not finite3(rnd)
            or rnd[0] < 0.0 or rnd[0] >= 1.0 or rnd[1] < 0.0 or rnd[1] >= 1.0
            or rnd[2] < 0.0 or rnd[2] >= 1.0):
        out_status[q] = -1
        return
    if node_left.shape[0] == 0:
        return
    radius2 = radius * radius
    if node_area[0] <= 0.0 or point_box_distance_sq(point, node_lower[0], node_upper[0]) > radius2:
        return

    node = int(0)
    u = rnd[0]
    probability = float(1.0)
    while node_leaf_count[node] == 0:
        left = node_left[node]
        right = node_right[node]
        left_weight = float(0.0)
        right_weight = float(0.0)
        if point_box_distance_sq(point, node_lower[left], node_upper[left]) <= radius2:
            left_weight = node_area[left]
        if point_box_distance_sq(point, node_lower[right], node_upper[right]) <= radius2:
            right_weight = node_area[right]
        total = left_weight + right_weight
        if total <= 0.0:
            return
        choose_left = left_weight / total
        if right_weight <= 0.0 or (left_weight > 0.0 and u < choose_left):
            probability *= choose_left
            u /= choose_left
            node = left
        else:
            probability *= 1.0 - choose_left
            u = (u - choose_left) / (1.0 - choose_left)
            node = right
        u = wp.clamp(u, 0.0, 0.99999994)

    offset = node_leaf_offset[node]
    end = offset + node_leaf_count[node]
    total_area = float(0.0)
    last_eligible = int(-1)
    for k in range(offset, end):
        if tri_area[k] > 0.0 and triangle_box_overlaps_sphere(point, radius2, tri_v0[k], tri_e1[k], tri_e2[k]):
            total_area += tri_area[k]
            last_eligible = k
    if total_area <= 0.0:
        return
    threshold = u * total_area
    accumulated = float(0.0)
    selected = last_eligible
    for k in range(offset, end):
        if tri_area[k] > 0.0 and triangle_box_overlaps_sphere(point, radius2, tri_v0[k], tri_e1[k], tri_e2[k]):
            accumulated += tri_area[k]
            if threshold < accumulated:
                selected = k
                break
    root_u = wp.sqrt(rnd[1])
    b1 = root_u * (1.0 - rnd[2])
    b2 = root_u * rnd[2]
    sampled = tri_v0[selected] + b1 * tri_e1[selected] + b2 * tri_e2[selected]
    delta = sampled - point
    if wp.dot(delta, delta) > radius2:
        return
    out_point[q] = sampled
    out_normal[q] = tri_normal[selected]
    out_pdf[q] = probability / total_area
    out_id[q] = prim_ids[selected]
    out_status[q] = 1
