"""Deterministic BVH benchmark fixtures."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import trimesh

from surfaces.base import TriMesh
from surfaces.explicit.primitives import TriMeshPrimitives


def _mesh(subdivisions=3):
    tm = trimesh.creation.icosphere(subdivisions=subdivisions)
    vertices = np.asarray(tm.vertices, np.float32)
    faces = np.asarray(tm.faces, np.int32)
    p = TriMesh(vertices=jnp.asarray(vertices), faces=jnp.asarray(faces))
    return TriMeshPrimitives(p), len(faces)


def _queries(n, seed=404, point_distribution="gaussian"):
    rng = np.random.default_rng(seed + n)
    points = rng.normal(size=(n, 3)).astype(np.float32) * 1.5
    origins = rng.normal(size=(n, 3)).astype(np.float32) * 2.0
    targets = rng.normal(size=(n, 3)).astype(np.float32)
    directions = targets - origins
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-30)
    # Preserve identical rays across point distributions.
    if point_distribution != "gaussian":
        point_rng = np.random.default_rng(seed + n + 100000)
        unit = point_rng.normal(size=(n, 3)).astype(np.float32)
        unit /= np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-30)
        if point_distribution == "shell":
            radius = point_rng.uniform(.95, 1.05, size=(n, 1))
            points = (unit * radius).astype(np.float32)
        elif point_distribution == "interior":
            radius = .95 * np.cbrt(point_rng.random((n, 1)))
            points = (unit * radius).astype(np.float32)
        elif point_distribution == "center":
            points = np.zeros((n, 3), np.float32)
        else:
            raise ValueError(f"unknown point distribution: {point_distribution}")
    return jnp.asarray(points), jnp.asarray(origins), jnp.asarray(directions), jnp.full((n,), 10., jnp.float32)
