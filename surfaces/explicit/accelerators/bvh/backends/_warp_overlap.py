"""Triangle-AABB overlap traversal."""
from __future__ import annotations

import warp as wp

from ._warp_geometry import boxes_overlap, finite3
from ._warp_specs import STACK_CAPACITY


_STACK64 = wp.types.vector(length=STACK_CAPACITY, dtype=wp.int32)
_STACK_LIMIT = wp.constant(STACK_CAPACITY)


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
