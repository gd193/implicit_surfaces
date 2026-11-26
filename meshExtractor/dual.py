"""Adaptive dual contouring of three-dimensional signed surfaces."""
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import lsq_linear

from meshExtractor._adaptive import _batched, make_adaptive_extractor


def _fit_vertex(points, normals, lower, upper):
    """Box-constrained Hermite least squares, centered for rank-deficient fits."""
    center = points.mean(axis=0)
    lengths = np.linalg.norm(normals, axis=1)
    good = np.isfinite(lengths) & (lengths > 0)
    if not good.any():
        return np.clip(center, lower, upper)
    matrix = normals[good] / lengths[good, None]
    rhs = np.einsum("ij,ij->i", matrix, points[good] - center)
    # Weak centering selects a stable position along unconstrained directions.
    matrix = np.vstack((matrix, np.eye(3) * 1e-4))
    rhs = np.r_[rhs, np.zeros(3)]
    fixed = upper == lower
    offset = lower - center
    free = ~fixed
    offset[free] = lsq_linear(matrix[:, free], rhs - matrix[:, fixed] @ offset[fixed],
                              bounds=((lower - center)[free], (upper - center)[free]),
                              method="bvls").x
    return np.clip(center + offset, lower, upper)


def _stitch_ring(edge, wedges, vertices, faces, points, midpoint, slab_cells):
    """Walk incident leaves in contour order, including clipped boundary chains."""
    incoming = {segment for _, segment in wedges.values()}
    starts = list(wedges.keys() - incoming)
    while wedges:
        open_chain = bool(starts)
        start = starts.pop() if starts else next(iter(wedges))
        segment = start
        polygon = []
        if open_chain:
            polygon.append(midpoint(start))
        local_fan = open_chain
        while segment in wedges:
            cell, segment = wedges.pop(segment)
            polygon.append(cell)
            if segment in wedges or segment == start:
                neighbor = wedges[segment][0] if segment in wedges else polygon[0]
                if cell in slab_cells or neighbor in slab_cells:
                    polygon.append(midpoint(segment))
                    local_fan = True
            elif open_chain:
                polygon.append(midpoint(segment))
        if len(polygon) < 3:
            continue
        if local_fan:
            # Split at the common crossing to keep large-cell triangles
            # inside their certified slab and terminate open domain edges.
            pivot = len(vertices)
            vertices.append(points[edge])
            pairs = list(zip(polygon, polygon[1:]))
            if not open_chain:
                pairs.append((polygon[-1], polygon[0]))
            faces.extend((pivot, a, b) for a, b in pairs)
        else:
            # Three incident leaves give a transition triangle. Four
            # incident leaves give a quad; choose its shorter diagonal.
            if len(polygon) == 4:
                a, b, c, d = (vertices[i] for i in polygon)
                if np.linalg.norm(b - d) < np.linalg.norm(a - c):
                    polygon = polygon[1:] + polygon[:1]
            faces.extend((polygon[0], a, b) for a, b in zip(polygon[1:], polygon[2:]))


def _mesh_leaves(leaves, samples, contours, world):
    edges = list(samples)
    indices = {edge: i for i, edge in enumerate(edges)}
    data = np.asarray(list(samples.values()), float).reshape(-1, 6)
    points, normals = data[:, :3], data[:, 3:]
    vertices, faces, invalid = [], [], []
    rings = defaultdict(dict)
    slab_cells = set()
    cell_leaves = []

    for leaf, links in zip(leaves, contours, strict=True):
        origin, width, slab = leaf
        # A second contour loop would be pinched onto the same dual vertex.
        loop = []
        if links:
            first = next(iter(links))
            edge = first
            while True:
                loop.append(edge)
                edge, _ = links[edge]
                if edge == first:
                    break
        ids = [indices[e] for e in loop]
        if len(loop) != len(links) or len(np.unique(points[ids], axis=0)) < 3:
            invalid.append((origin, width))
            continue
        lower, upper = world(np.array([origin, np.asarray(origin) + width])).astype(float)
        cell = len(vertices)
        cell_leaves.append((origin, width))
        if slab is not None:
            axis, bottom, top = slab
            lower[axis], upper[axis] = bottom, top
            slab_cells.add(cell)
        vertices.append(_fit_vertex(points[ids], normals[ids], lower, upper))
        # Include the face tile in the key: adjacent tiles can pair the
        # same two crossings when a subdivided edge crosses the surface twice.
        segments = [(min(a, b), max(a, b), links[edge][1])
                    for edge, a, b in zip(loop, ids, ids[1:] + ids[:1], strict=True)]
        for i, here in enumerate(ids):
            rings[here][segments[i]] = (cell, segments[i - 1])

    boundary_ids = {}

    def midpoint(segment):
        if segment not in boundary_ids:
            boundary_ids[segment] = len(vertices)
            vertices.append(points[list(segment[:2])].mean(axis=0))
        return boundary_ids[segment]

    for edge, wedges in rings.items():
        _stitch_ring(edge, wedges, vertices, faces, points, midpoint, slab_cells)

    vertices = np.asarray(vertices, np.float32).reshape(-1, 3)
    faces = np.asarray(faces, np.int32).reshape(-1, 3)
    a, b, c = vertices[faces].astype(float).transpose(1, 0, 2)
    faces = faces[np.any(np.cross(b - a, c - a) != 0, axis=1)]
    used, inverse = np.unique(faces.ravel(), return_inverse=True)
    invalid.extend(cell_leaves[i] for i in set(range(len(cell_leaves))).difference(used))
    # At exact grid ties distinct cells and boundary crossings can meet at
    # identical float32 positions. Weld these endpoints after area filtering.
    unique, welded = np.unique(vertices[used], axis=0, return_inverse=True)
    return unique, welded[inverse].astype(np.int32).reshape(-1, 3), invalid


def make_dual_contouring(surface, bounds, *, tolerance, max_depth=8):
    """Return extract(params) yielding an adaptive TriMesh and unresolved boxes.

    Hermite normals use spatial_derivative when supplied and JAX autodiff
    of signed_value otherwise. Tolerance and bounds are in world units.
    """
    derivative = getattr(surface, "spatial_derivative", None)
    if derivative is None:
        derivative = jax.grad(surface.signed_value, argnums=1)

    def hermite(params, a, b):
        field = surface.signed_value
        fa, fb = field(params, a), field(params, b)

        def bisect(_, bracket):
            lo, hi, flo, fhi = bracket
            mid = .5 * lo + .5 * hi
            fm = field(params, mid)
            left = (fm < 0) == (flo < 0)
            return (jnp.where(left, mid, lo), jnp.where(left, hi, mid),
                    jnp.where(left, fm, flo), jnp.where(left, fhi, fm))

        lo, hi, flo, fhi = jax.lax.fori_loop(0, 16, bisect, (a, b, fa, fb))
        t = flo / (flo - fhi)
        point = jnp.where(flo == 0, lo, jnp.where(fhi == 0, hi, lo + t * (hi - lo)))
        normal = derivative(params, point)
        return jnp.concatenate((point, normal))

    evaluate = _batched(hermite, 2)

    def sample_edges(params, endpoints, fractions):
        samples = evaluate(params, endpoints[:, 0], endpoints[:, 1]).astype(float)
        if samples.shape != (len(endpoints), 6) or not np.isfinite(samples[:, :3]).all():
            raise ValueError("Hermite sampling requires finite crossings and a 3D spatial derivative")
        return samples

    return make_adaptive_extractor(surface, bounds, tolerance, max_depth, sample_edges, _mesh_leaves)
