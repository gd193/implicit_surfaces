"""Supported builtin combinations and shared correctness/timing fixtures."""
from functools import lru_cache
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import trimesh

from surfaces.explicit.accelerators.protocol import Config
from surfaces.explicit.accelerators.bvh.warp import (
    Angular, Euclidean, Points, Segments, SilhouetteEdges, Triangles,
    build_boundary, build_tree, make_nearest, make_overlap, make_ray_query,
    make_silhouette, make_star_radius, make_classify, make_boundary_sample,
    make_robin_radius, prepare_boundary_sample, prepare_robin,
)

PRIMITIVES = dict(points2=Points(2), points3=Points(3), segments2=Segments(2),
                  segments3=Segments(3), triangles=Triangles(), edges=SilhouetteEdges())
METRICS = {"euclidean": Euclidean(), "angular": Angular()}
RAYS = ("first", "any", "count", "all")
BOUNDARIES = ("segments2", "triangles")


class QueryCase(NamedTuple):
    primitive: str
    query: str
    metric: str = "none"

    @property
    def id(self):
        return "/".join(self)


CASES = tuple(
    [QueryCase(p, "nearest", m) for p, primitive in PRIMITIVES.items() for m in METRICS
     if m == "euclidean" or primitive.segment_endpoints is not None]
    + [QueryCase(p, "overlap") for p in PRIMITIVES]
    + [QueryCase(p, q) for p, primitive in PRIMITIVES.items() if primitive.intersect is not None for q in RAYS]
    + [QueryCase(p, "silhouette", m) for p in BOUNDARIES for m in METRICS]
    + [QueryCase(p, q, m) for p in BOUNDARIES for q, m in
       (("star", "euclidean"), ("robin", "euclidean"), ("classify", "none"), ("sample", "none"))]
)


def query_factory(case):
    primitive, metric = PRIMITIVES[case.primitive], METRICS.get(case.metric)
    if case.query == "nearest":
        return make_nearest(primitive, metric)
    if case.query == "silhouette":
        return make_silhouette(primitive, metric)
    if case.query in RAYS:
        return make_ray_query(primitive, case.query)
    return {"overlap": make_overlap, "star": make_star_radius, "robin": make_robin_radius,
            "classify": make_classify, "sample": make_boundary_sample}[case.query](primitive)


class Scene(NamedTuple):
    tree: object
    geometry: object
    boundary: object
    raw: object


@lru_cache(maxsize=None)
def scene(name, size=16):
    primitive = PRIMITIVES[name]
    rng = np.random.default_rng(73)
    if name == "triangles":
        mesh = trimesh.creation.box(extents=[2., 2., 2.])
        while len(mesh.faces) < size:
            mesh = mesh.subdivide()
        raw = primitive.indexed(mesh.vertices, mesh.faces)
    elif name == "segments2":
        corners = np.array([[-1., -1], [1., -1], [1., 1], [-1., 1]], np.float32)
        t = np.arange(max(1, size//4))/max(1, size//4)
        vertices = np.concatenate([a+t[:, None]*(b-a) for a, b in zip(corners, np.roll(corners, -1, axis=0))])
        raw = primitive.indexed(vertices, np.column_stack((np.arange(len(vertices)), np.roll(np.arange(len(vertices)), -1))))
    else:
        a = rng.normal(size=(size, primitive.dimension)).astype(np.float32)
        if name.startswith("points"):
            raw = primitive.data(a)
        else:
            b = a+.5*rng.normal(size=a.shape)
            raw = (primitive.data(a, b, rng.normal(size=a.shape), rng.normal(size=a.shape))
                   if name == "edges" else primitive.data(a, b))
    if name in BOUNDARIES:
        tree, geometry, boundary = build_boundary(primitive, raw, Config(leaf_size_threshold=4))
    else:
        tree, geometry = build_tree(primitive, raw, Config(leaf_size_threshold=4))
        boundary = None
    return Scene(tree, geometry, boundary, raw)


def arguments(case, geometry, n, radius=.75):
    dim = PRIMITIVES[case.primitive].dimension
    rng = np.random.default_rng(404+n)
    points, directions = rng.normal(size=(2, n, dim)).astype(np.float32)
    points *= 1.5
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    queries = np.stack((points, directions), axis=1) if case.metric == "angular" else points
    points, directions, queries = map(jnp.asarray, (points, directions, queries))
    caps = jnp.full((n,), radius, jnp.float32)
    tree, data, boundary, _ = geometry
    if case.query in RAYS:
        kwargs = dict(max_t=jnp.full((n,), 10., jnp.float32))
        if case.query in ("count", "all"):
            kwargs.update(min_t=jnp.where(jnp.arange(n) % 2 == 0, -jnp.inf, 0.))
        if case.query == "all":
            kwargs["max_hits"] = 4
        return (tree, data, points, directions), kwargs
    if case.query == "nearest":
        return (tree, data, queries, caps), {}
    if case.query == "overlap":
        return (tree, data, points-radius, points+radius), {}
    if case.query == "classify":
        return (tree, data, boundary, points), {}
    if case.query in ("silhouette", "star"):
        return (tree, data, boundary, queries, caps), {}
    if case.query == "robin":
        return (tree, data, prepare_robin(tree, boundary, 1.), points, caps), {}
    randoms = jnp.asarray(rng.random((n, 3)), jnp.float32)
    return (tree, data, prepare_boundary_sample(tree, boundary), points,
            jnp.full_like(caps, jnp.inf), randoms), {}
