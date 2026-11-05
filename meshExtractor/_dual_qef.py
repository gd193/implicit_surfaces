"""Small box-constrained, rank-aware Hermite QEF for Warp dual contouring."""
import warp as wp


@wp.func
def solve_qef(A: wp.mat33, b: wp.vec3, centroid: wp.vec3,
              lower: wp.vec3, upper: wp.vec3) -> wp.vec3:
    # All coordinates are relative to the cell origin. Enumerate the free,
    # lower and upper state of each axis; each face solves a reduced PSD QEF.
    best = wp.vec3(0.0)
    for axis in range(3):
        best[axis] = wp.clamp(centroid[axis], lower[axis], upper[axis])
    residual = b - A * centroid
    delta = best - centroid
    best_error = wp.dot(delta, A * delta) - 2.0 * wp.dot(residual, delta)
    best_distance = wp.dot(delta, delta)
    for state in range(27):
        code = state
        free = wp.vec3(0.0)
        anchor = centroid
        for axis in range(3):
            mode = code % 3
            code = code // 3
            if mode == 0:
                free[axis] = 1.0
            elif mode == 1:
                anchor[axis] = lower[axis]
            else:
                anchor[axis] = upper[axis]
        reduced = wp.mat33(0.0)
        rhs = b - A * anchor
        for row in range(3):
            rhs[row] = rhs[row] * free[row]
            for col in range(3):
                reduced[row, col] = A[row, col] * free[row] * free[col]
        candidate = anchor
        free_count = free[0] + free[1] + free[2]
        if free_count <= 1.0:
            for axis in range(3):
                if free[axis] > 0.0 and A[axis, axis] > 1.0e-20:
                    candidate[axis] += rhs[axis] / A[axis, axis]
        else:
            vectors, values = wp.eig3(reduced)
            scale = wp.max(wp.max(values[0], values[1]), values[2])
            for axis in range(3):
                if values[axis] > wp.max(1.0e-6 * scale, 1.0e-20):
                    direction = wp.vec3(vectors[0, axis], vectors[1, axis], vectors[2, axis])
                    candidate += direction * (wp.dot(direction, rhs) / values[axis])
        valid = True
        for axis in range(3):
            if candidate[axis] < lower[axis] - 1.0e-6 * (upper[axis] - lower[axis]):
                valid = False
            if candidate[axis] > upper[axis] + 1.0e-6 * (upper[axis] - lower[axis]):
                valid = False
        if valid:
            for axis in range(3):
                candidate[axis] = wp.clamp(candidate[axis], lower[axis], upper[axis])
            if state == 0:
                return candidate
            delta = candidate - centroid
            error = wp.dot(delta, A * delta) - 2.0 * wp.dot(residual, delta)
            distance = wp.dot(delta, delta)
            if error < best_error or (error == best_error and distance < best_distance):
                best = candidate
                best_error = error
                best_distance = distance
    return best
