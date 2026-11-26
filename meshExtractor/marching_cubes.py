"""Adaptive marching cubes for continuous, signed implicit surfaces."""
from bisect import bisect_left, bisect_right
from collections import defaultdict
from itertools import product
from operator import index

import jax
import numpy as np

from surfaces.meshes import TriMesh
from meshExtractor.results import AdaptiveMeshResult


_CORNERS = np.array(list(product((0, 1), repeat=3)), dtype=np.int32)


def _batched(query, input_count):
    """Reuse compiled batch shapes across different octree frontiers."""
    evaluate = jax.jit(jax.vmap(query, (None,) + (0,) * input_count))

    def run(params, *arrays):
        chunks = []
        for start in range(0, len(arrays[0]), 256):
            count = min(len(arrays[0]) - start, 256)
            capacity = 1 << (count - 1).bit_length()
            ids = start + np.minimum(np.arange(capacity), count - 1)
            chunks.append(np.asarray(evaluate(params, *(a[ids] for a in arrays)))[:count])
        return np.concatenate(chunks)
    return run


def _corners(origin, width):
    return origin[:, None, :] + width * _CORNERS


def _face_children(face):
    axis, origin, width = face
    half = width // 2
    u, v = (axis + 1) % 3, (axis + 2) % 3
    for i, j in product((0, half), repeat=2):
        point = list(origin)
        point[u] += i
        point[v] += j
        yield axis, tuple(point), half


def _tile_corners(face):
    axis, origin, width = face
    u, v = (axis + 1) % 3, (axis + 2) % 3
    corners = []
    for i, j in ((0, 0), (width, 0), (width, width), (0, width)):
        p = list(origin)
        p[u] += i
        p[v] += j
        corners.append(tuple(p))
    return corners


def _shared_tiles(leaves, root_width):
    """Partition each shared face using its finest incident dyadic tiles."""
    split_faces = set()
    leaf_faces = []
    for origin, width, _ in leaves:
        faces = []
        for axis, side in product(range(3), (0, 1)):
            p = list(origin)
            p[axis] += side * width
            faces.append(((axis, tuple(p), width), side))
            size = width * 2
            while size <= root_width:
                for d in range(3):
                    if d != axis:
                        p[d] = p[d] // size * size
                split_faces.add((axis, tuple(p), size))
                size *= 2
        leaf_faces.append(faces)

    def tiles(face):
        if face in split_faces:
            for child in _face_children(face):
                yield from tiles(child)
        else:
            yield face

    return [[(tile, side) for face, side in faces for tile in tiles(face)]
            for faces in leaf_faces]


def _face_segments(leaf_tiles, world, values):
    """Identify each crossing by its lattice edge, including split edges."""
    tile_points = {tile: _tile_corners(tile) for faces in leaf_tiles for tile, _ in faces}
    lines = defaultdict(set)
    for points in tile_points.values():
        for point in points:
            for axis in range(3):
                line = (axis, point[(axis + 1) % 3], point[(axis + 2) % 3])
                lines[line].add(point[axis])
    lines = {key: sorted(nodes) for key, nodes in lines.items()}

    # Include coordinates introduced while tiling coarse faces.
    point_keys = sorted({p for points in tile_points.values() for p in points})
    signed = dict(zip(point_keys, values(np.asarray(point_keys)), strict=True))
    positions = {}
    segments = {}
    for tile, points in tile_points.items():
        boundary = []
        for a, b in zip(points, points[1:] + points[:1], strict=True):
            axis = next(k for k in range(3) if a[k] != b[k])
            nodes = lines[axis, a[(axis + 1) % 3], a[(axis + 2) % 3]]
            lo, hi = sorted((a[axis], b[axis]))
            steps = nodes[bisect_left(nodes, lo):bisect_right(nodes, hi)]
            for s in (steps if a[axis] < b[axis] else steps[::-1])[:-1]:
                p = list(a)
                p[axis] = s
                boundary.append(tuple(p))
        crossings = []
        for a, b in zip(boundary, boundary[1:] + boundary[:1], strict=True):
            if (signed[a] < 0) == (signed[b] < 0):
                continue
            edge = tuple(sorted((a, b)))
            if edge not in positions:
                p, q = edge
                t = float(signed[p]) / (float(signed[p]) - float(signed[q]))
                start, end = world(np.asarray(edge)).astype(float)
                positions[edge] = start + t * (end - start)
            crossings.append((edge, signed[b] < 0))
        # Deterministic ambiguity rule: connect across each negative boundary run.
        segments[tile] = [(edge, crossings[(i + 1) % len(crossings)][0])
                          for i, (edge, entering) in enumerate(crossings) if entering]

    return segments, positions


def _mesh_leaves(leaves, root_width, world, values):
    leaf_tiles = _shared_tiles(leaves, root_width)
    segments, positions = _face_segments(leaf_tiles, world, values)
    # A true root on a crossing edge belongs to every incident certified slab.
    # Constrain the shared linear interpolant once for all incident leaves.
    links_by_leaf = []
    for (_, _, slab), faces in zip(leaves, leaf_tiles, strict=True):
        links = {}
        for tile, side in faces:
            for a, b in segments[tile]:
                if slab is not None:
                    axis, lo, hi = slab
                    for key in (a, b):
                        positions[key][axis] = np.clip(positions[key][axis], lo, hi)
                if not side:
                    a, b = b, a
                links[a] = b
        links_by_leaf.append(links)

    vertices, triangles, owners = [], [], []
    vertex_ids = {}

    def vertex_id(point):
        key = tuple(np.asarray(point, np.float32))
        if key not in vertex_ids:
            vertex_ids[key] = len(vertices)
            vertices.append(key)
        return vertex_ids[key]

    for owner, links in enumerate(links_by_leaf):
        while links:
            start = next(iter(links))
            loop = [start]
            edge = links.pop(start)
            while edge != start:
                loop.append(edge)
                edge = links.pop(edge)
            ids = [vertex_id(positions[key]) for key in loop]
            if len(set(ids)) < 3:
                continue
            if len(ids) == 3:
                triangles.append(ids)
                owners.append(owner)
                continue
            if len(ids) == 4:
                a, b, c, d = (np.asarray(vertices[i], float) for i in ids)
                # A consistent diagonal retains all four boundary segments.
                if np.dot(np.cross(b - a, c - a), np.cross(c - a, d - a)) > 0:
                    triangles.extend((ids[:3], (ids[0], ids[2], ids[3])))
                    owners.extend((owner, owner))
                    continue
            center = np.mean([vertices[i] for i in ids], axis=0, dtype=float)
            middle = vertex_id(center)
            fan = [(a, b, middle) for a, b in zip(ids, ids[1:] + ids[:1], strict=True)]
            triangles.extend(fan)
            owners.extend([owner] * len(fan))

    vertices = np.asarray(vertices, np.float32).reshape(-1, 3)
    faces = np.asarray(triangles, np.int32).reshape(-1, 3)
    a, b, c = vertices[faces].astype(float).transpose(1, 0, 2)
    keep = np.any(np.cross(b - a, c - a) != 0, axis=1)
    faces = faces[keep]
    meshed = set(np.asarray(owners, int)[keep])
    collapsed = [leaf[:2] for i, leaf in enumerate(leaves) if i not in meshed]
    used, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    return (vertices[used],
            inverse.reshape(-1, 3).astype(np.int32), collapsed)


def _certify_slabs(params, classify, lo, hi, samples, tolerance, accept):
    """Try thin slabs normal to each axis on cells spanning the full footprint.

    Strict, opposite signs in the two caps put every root inside the slab.
    Continuity gives a root above every footprint point. The patch spans that
    same footprint and its clipped vertices keep every triangle in the slab.
    """
    slabs = {}
    for axis in range(3):
        below = samples[:, _CORNERS[:, axis] == 0]
        above = samples[:, _CORNERS[:, axis] == 1]
        graph = ((below.max(1) < 0) & (above.min(1) > 0) |
                 (above.max(1) < 0) & (below.min(1) > 0))
        ids = np.flatnonzero(graph & ~accept)
        if not len(ids):
            continue
        low, high = lo[ids], hi[ids]
        f, g = below[ids].astype(float), above[ids].astype(float)
        span = high[:, axis, None].astype(float) - low[:, axis, None]
        heights = low[:, axis, None] + span * f / (f - g)
        mid = (heights.min(1) + heights.max(1)) * 0.5
        # Round slab endpoints inward to retain the stated tolerance.
        bottom = np.maximum(low[:, axis], np.nextafter(
            (mid - tolerance / 2).astype(np.float32), np.float32(np.inf)))
        top = np.minimum(high[:, axis], np.nextafter(
            (mid + tolerance / 2).astype(np.float32), np.float32(-np.inf)))
        valid = (bottom <= heights.min(1)) & (heights.max(1) <= top)
        ids, bottom, top = ids[valid], bottom[valid], top[valid]
        if not len(ids):
            continue
        a = np.repeat(lo[ids, None], 2, axis=1)
        b = np.repeat(hi[ids, None], 2, axis=1)
        b[:, 0, axis], a[:, 1, axis] = bottom, top
        signs = classify(params, lo[ids], hi[ids], a, b)
        for k in np.flatnonzero(signs[:, 0] * signs[:, 1] == -1):
            i = ids[k]
            accept[i] = True
            slabs[i] = (axis, bottom[k], top[k])
    return slabs


def make_marching_cubes(surface, bounds, *, tolerance, max_depth=8):
    """Build an eager extract(params) for a continuous surface in a 3D box.

    Returns an AdaptiveMeshResult with a TriMesh and unresolved boxes.
    tolerance is a world-space, two-sided distance bound within accepted leaves.
    Unresolved leaves contribute their AABB instead of triangles. Point-only
    surfaces need signed_value(params, point); box classifiers are optional.
    """
    lower, upper = np.asarray(bounds, np.float32)
    max_depth = index(max_depth)
    if (lower.shape != (3,) or upper.shape != (3,) or
            not np.isfinite([lower, upper]).all() or np.any(upper <= lower)):
        raise ValueError("bounds must be finite increasing 3D corners")
    if not 0 <= max_depth <= 24 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("max_depth must be 0..24 and tolerance finite and positive")
    root_width = 1 << max_depth

    def world(nodes):
        t = np.asarray(nodes) / root_width
        return ((1 - t) * lower + t * upper).astype(np.float32)

    point_query = _batched(lambda params, p: surface.signed_value(params, p), 1)
    box_query = getattr(surface, "classify_box", None)
    group_query = getattr(surface, "classify_subcubes", None)
    classify_root = jax.jit(box_query) if box_query is not None else None
    if group_query and getattr(surface, "use_classify_subcubes", True):
        classify = _batched(lambda params, lo, hi, a, b: group_query(params, lo, hi, a, b), 4)
    elif box_query:
        classify_boxes = _batched(box_query, 2)

        def classify(params, parent_lo, parent_hi, a, b):
            signs = classify_boxes(params, a.reshape(-1, 3), b.reshape(-1, 3))
            return signs.reshape(a.shape[:-1])
    else:
        classify = None

    def extract(params):
        devices = {a.device for a in jax.tree.leaves(params) if isinstance(a, jax.Array)}
        if len(devices) > 1:
            raise ValueError("parameters must share one device")
        device = next(iter(devices), None)
        leaves, unresolved = [], []
        cache = {}

        def values(points):
            keys = list(map(tuple, points))
            missing = sorted(set(keys).difference(cache))
            if missing:
                result = point_query(params, world(np.asarray(missing)))
                if result.shape != (len(missing),) or not np.isfinite(result).all():
                    raise ValueError("signed_value must return a finite scalar at each point")
                cache.update(zip(missing, result, strict=True))
            return np.array([cache[key] for key in keys])

        def children_to_visit(parents, width):
            half = width // 2
            children = _corners(parents, half)
            if classify is None or not len(parents):
                return children.reshape(-1, 3)
            signs = classify(params, world(parents), world(parents + width),
                             world(children), world(children + half))
            return children[signs == 0]

        def visit(origins, width):
            while len(origins):
                lo, hi = world(origins), world(origins + width)
                if np.any(hi <= lo):
                    raise ValueError("max_depth exceeds float32 coordinate precision for these bounds")
                samples = values(_corners(origins, width).reshape(-1, 3)).reshape(-1, 8)
                crossing = (samples.min(axis=1) < 0) & (samples.max(axis=1) >= 0)
                accept = crossing & (np.linalg.norm(hi.astype(float) - lo, axis=1) <= tolerance)
                slabs = {}
                if classify is not None:
                    slabs = _certify_slabs(params, classify, lo, hi, samples, tolerance, accept)
                leaves.extend((tuple(origins[i]), width, slabs.get(i)) for i in np.flatnonzero(accept))
                pending = origins[~accept]
                if width == 1:
                    unresolved.extend((p, width) for p in pending)
                    break
                origins = children_to_visit(pending, width)
                width //= 2

        if classify_root is None or int(classify_root(params, lower, upper)) == 0:
            visit(np.zeros((1, 3), np.int32), root_width)

        while True:
            vertices, faces, collapsed = _mesh_leaves(leaves, root_width, world, values)
            refine = {(p, w) for p, w in collapsed if w > 1}
            if not refine:
                unresolved.extend(collapsed)
                break
            # Exact corner contacts can collapse after shared vertices are welded.
            # Revisit just those cells, then rebuild the affected face partition.
            leaves = [leaf for leaf in leaves if leaf[:2] not in refine]
            by_width = defaultdict(list)
            for p, w in sorted(refine):
                by_width[w].append(p)
            for w, origins in by_width.items():
                visit(children_to_visit(np.asarray(origins), w), w // 2)

        boxes = np.asarray([[world(p), world(np.asarray(p) + w)] for p, w in unresolved],
                           np.float32).reshape(-1, 2, 3)
        vertices, faces, boxes = jax.device_put((vertices, faces, boxes), device)
        return AdaptiveMeshResult(TriMesh(vertices, faces), boxes)

    return extract
