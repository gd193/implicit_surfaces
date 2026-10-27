"""Triangle Robin bound arithmetic."""

_NATIVE_ROBIN_RADIUS_SQUARED = r"""
    const auto radius_bound_sq = [](float h, float mu, float distance2) {
        const float scale = mu * distance2;
        const float denominator = scale == INFINITY ? 1.0f : (scale - h) / scale;
        return distance2 / (denominator * denominator) * 0.999996f;
    };
"""


_NATIVE_ROBIN_NODE = r"""
    const float box2 = box_distance_sq(lower, upper);
    float priority = box2;
    const auto result = [&](bool take, bool omit) {
        score = priority; bound = best2; accepted = take; skip = omit;
    };
    if (box2 > best2) return result(false, false);
    const wp::vec2 mu = node_robin_bounds.data[node];
    if (mu[0] == INFINITY) return result(true, true);
    const wp::vec2 sc = node_cone_sin_cos.data[node];
    if (sc[0] < 0.0f) return result(mu[1] > 0.0f, true);
    if (sc[1] <= 0.0f) return result(true, false);
    const wp::vec3 view = 0.5f * (lower + upper) - p;
    const wp::vec3 extent = 0.5f * (upper - lower);
    const wp::vec3 axis = node_cone_axis.data[node];
    const float projection = wp::dot(axis, view);
    const float support = wp::dot(wp::abs(axis), extent);
    const float gap = wp::abs(projection) - support;
    if (!(gap > 0.0f)) return result(true, false);
    const float radius = node_cone_radius.data[node];
    const wp::vec3 perpendicular = view - projection * axis;
    const float slack = (mu[1] == 0.0f ? 2.0e-6f : 4.0e-6f)
                      * (wp::length(view) + radius);
    // Lower bound on |normal dot (surface point - query)|.
    const float h_min = gap * sc[1] - sc[0] * (wp::length(perpendicular) + radius) - slack;
    if (!(h_min > 0.0f) || !(h_min < INFINITY)) return result(true, false);
    if (mu[1] == 0.0f) return result(false, false);
    const wp::vec3 far = wp::max(wp::abs(lower - p), wp::abs(upper - p));
    const float far2 = wp::dot(far, far);
    const float scale = mu[1] * far2;
    if (scale <= h_min) return result(false, false);
    const float distance2 = wp::clamp(3.0f * (h_min / mu[1]), box2, far2);
    const float optimal_scale = mu[1] * distance2;
    if (optimal_scale > h_min && optimal_scale < INFINITY) {
        priority = radius_bound_sq(h_min, mu[1], distance2);
        if (priority > best2) return result(false, false);
    }
    const float upper_scale = mu[0] * wp::sqrt(box2);
    if (upper_scale > 1.0f) {
        const float denominator = upper_scale == INFINITY ? 1.0f
            : (upper_scale - 1.0f) / upper_scale;
        best2 = wp::min(best2, far2 / (denominator * denominator) * 1.000008f);
    }
    return result(true, true);
"""


_NATIVE_ROBIN_TRIANGLE = r"""
    const float mu = tri_robin_bounds.data[k][1];
    if (mu == 0.0f) continue;
    const wp::vec3 a = tri_v0.data[k];
    float h = 0.0f, q = 0.0f, farthest2 = INFINITY;
    if (mu < INFINITY) {
        const wp::vec3 normal = tri_normal.data[k];
        const wp::vec3 view = p - a;
        const float normal_length = wp::length(normal);
        h = normal_length > 0.0f ? wp::abs(wp::dot(view, normal)) / normal_length : 0.0f;
        if (!(h < INFINITY)) { failed = true; break; }
        q = mu * h;
        const float error = 4.0e-6f * (wp::abs(view[0]) + wp::abs(view[1]) + wp::abs(view[2]));
        if (h > error) {
            const float lower_h = h - error;
            if (lower_h * lower_h > best2 || 6.75f * (lower_h / mu) * 0.999996f > best2)
                continue;
            if (q < 3.0f) {
                const wp::vec3 b = a + tri_e1.data[k];
                const wp::vec3 c = a + tri_e2.data[k];
                const float da2 = wp::dot(view, view);
                const float db2 = wp::dot(b - p, b - p);
                const float dc2 = wp::dot(c - p, c - p);
                farthest2 = wp::max(da2, wp::max(db2, dc2));
                const float scale = mu * farthest2;
                if (farthest2 < INFINITY) {
                    if (scale <= lower_h) continue;
                    // The minimum is at r*=sqrt(3h/mu), or an interval endpoint.
                    if (3.0f * (lower_h / mu) >= farthest2 && scale > h && scale < INFINITY) {
                        best2 = wp::min(best2, radius_bound_sq(h, mu, farthest2));
                        continue;
                    }
                    const float near_vertex2 = wp::min(da2, wp::min(db2, dc2));
                    if (near_vertex2 * 1.000004f <= 3.0f * (lower_h / mu)
                        && 3.0f * ((h + error) / mu) <= farthest2 * 0.999996f) {
                        best2 = wp::min(best2, 6.75f * (h / mu) * 0.999996f);
                        continue;
                    }
                }
            }
        }
    }
    const wp::vec3 e1 = tri_e1.data[k];
    const wp::vec3 e2 = tri_e2.data[k];
    const wp::vec3 delta = closest_triangle(a, e1, e2) - p;
    const float closest2 = wp::dot(delta, delta);
    if (!(closest2 < INFINITY)) { failed = true; break; }
    if (closest2 > best2) continue;
    if (mu == INFINITY || closest2 == 0.0f || h == 0.0f || q == 0.0f || q == INFINITY) {
        best2 = closest2;
        continue;
    }
    float distance2 = closest2;
    if (q < 3.0f) {
        if (farthest2 == INFINITY) {
            const wp::vec3 da = a - p;
            const wp::vec3 db = a + e1 - p;
            const wp::vec3 dc = a + e2 - p;
            farthest2 = wp::max(wp::dot(da, da),
                wp::max(wp::dot(db, db), wp::dot(dc, dc)));
        }
        if (!(farthest2 < INFINITY)) { failed = true; break; }
        distance2 = wp::clamp(3.0f * (h / mu), closest2, farthest2);
    }
    const float scale = mu * distance2;
    if (scale > h) {
        best2 = wp::min(best2, wp::max(closest2, radius_bound_sq(h, mu, distance2)));
    }
"""
