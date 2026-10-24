"""Closest silhouette; cone radii enclose whole edges, open edges use n1=-n0."""
from __future__ import annotations

import warp as wp

from ._warp_specs import STACK_CAPACITY

from ._warp_geometry import finite3, point_box_distance_sq

_STACK = wp.types.vector(length=STACK_CAPACITY, dtype=wp.int32)
_STACK_LIMIT = wp.constant(STACK_CAPACITY)


@wp.func
def cone_may_have_silhouette(
    point: wp.vec3, lower: wp.vec3, upper: wp.vec3,
    axis: wp.vec3, half_angle: float, radius: float,
) -> bool:
    if half_angle < 0.0:
        return False
    if half_angle >= 1.5707953:
        return True
    view = 0.5 * (lower + upper) - point
    distance2 = wp.dot(view, view)
    # Never cone-prune near contacts.
    if distance2 <= radius * radius * 1.000002:
        return True
    inverse_distance = 1.0 / wp.sqrt(distance2)
    sin_beta = wp.min(radius * inverse_distance, 1.0)
    sin_alpha = wp.sin(half_angle)
    cos_alpha = wp.cos(half_angle)
    if sin_beta >= cos_alpha:
        return True
    # sin(alpha+beta), avoiding inverse trig.
    limit = sin_alpha * wp.sqrt(wp.max(0.0, 1.0 - sin_beta * sin_beta)) + cos_alpha * sin_beta
    alignment = wp.abs(wp.dot(axis, view) * inverse_distance)
    return alignment <= limit + 2.0e-6


@wp.kernel
def closest_silhouette_kernel(
    points: wp.array(dtype=wp.vec3),
    max_distance: wp.array(dtype=wp.float32),
    node_lower: wp.array(dtype=wp.vec3),
    node_upper: wp.array(dtype=wp.vec3),
    node_left: wp.array(dtype=wp.int32),
    node_right: wp.array(dtype=wp.int32),
    node_leaf_count: wp.array(dtype=wp.int32),
    node_silhouette_offset: wp.array(dtype=wp.int32),
    node_silhouette_count: wp.array(dtype=wp.int32),
    silhouette_refs: wp.array(dtype=wp.int32),
    silhouette_v0: wp.array(dtype=wp.vec3),
    silhouette_v1: wp.array(dtype=wp.vec3),
    silhouette_n0: wp.array(dtype=wp.vec3),
    silhouette_n1: wp.array(dtype=wp.vec3),
    node_cone_axis: wp.array(dtype=wp.vec3),
    node_cone_half_angle: wp.array(dtype=wp.float32),
    node_cone_radius: wp.array(dtype=wp.float32),
    out_point: wp.array(dtype=wp.vec3),
    out_distance: wp.array(dtype=wp.float32),
    out_edge_id: wp.array(dtype=wp.int32),
    out_status: wp.array(dtype=wp.int32),
):
    q = wp.tid()
    point = points[q]
    cap = max_distance[q]
    out_point[q] = wp.vec3(0.0)
    out_distance[q] = wp.float32(wp.inf)
    out_edge_id[q] = -1
    out_status[q] = 0
    if not finite3(point) or wp.isnan(cap) or cap < 0.0:
        out_status[q] = -1
        return
    if node_left.shape[0] == 0:
        return

    best2 = cap * cap
    best_point = wp.vec3(0.0)
    best_edge = int(-1)
    stack = _STACK()
    size = int(0)
    node = int(0)
    failed = bool(False)
    while node >= 0 or size > 0:
        if node < 0:
            size -= 1
            node = stack[size]
        box_distance = point_box_distance_sq(point, node_lower[node], node_upper[node])
        if box_distance > best2:
            node = -1
            continue
        if not cone_may_have_silhouette(
            point, node_lower[node], node_upper[node], node_cone_axis[node],
            node_cone_half_angle[node], node_cone_radius[node],
        ):
            node = -1
            continue
        if node_leaf_count[node] > 0:
            end = node_silhouette_offset[node] + node_silhouette_count[node]
            for position in range(node_silhouette_offset[node], end):
                edge = silhouette_refs[position]
                a = silhouette_v0[edge]
                ab = silhouette_v1[edge] - a
                view = point - a
                # Retain tangencies; n1=-n0 makes open edges unconditional candidates.
                side0 = wp.dot(view, silhouette_n0[edge])
                side1 = wp.dot(view, silhouette_n1[edge])
                if (side0 <= 0.0 and side1 >= 0.0) or (side0 >= 0.0 and side1 <= 0.0):
                    length2 = wp.dot(ab, ab)
                    t = float(0.0)
                    if length2 > 0.0:
                        t = wp.clamp(wp.dot(view, ab) / length2, 0.0, 1.0)
                    candidate = a + t * ab
                    delta = candidate - point
                    distance2 = wp.dot(delta, delta)
                    if distance2 < best2 or (distance2 == best2 and (best_edge < 0 or edge < best_edge)):
                        best2 = distance2
                        best_point = candidate
                        best_edge = edge
            node = -1
        else:
            left = node_left[node]
            right = node_right[node]
            left_distance = point_box_distance_sq(point, node_lower[left], node_upper[left])
            right_distance = point_box_distance_sq(point, node_lower[right], node_upper[right])
            near = left
            far = right
            near_distance = left_distance
            far_distance = right_distance
            if right_distance < left_distance:
                near = right
                far = left
                near_distance = right_distance
                far_distance = left_distance
            if near_distance > best2:
                node = -1
            elif far_distance <= best2:
                if size >= _STACK_LIMIT:
                    failed = True
                    break
                stack[size] = far
                size += 1
                node = near
            else:
                node = near
    if failed:
        out_status[q] = -1
    elif best_edge >= 0:
        out_point[q] = best_point
        out_distance[q] = wp.sqrt(best2)
        out_edge_id[q] = best_edge
        out_status[q] = 1
