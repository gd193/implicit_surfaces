"""Build/query interface for mesh accelerators."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Generic, Tuple, Type, TypeVar

import jax.numpy as jnp

from surfaces.explicit.primitives import Primitives


P = TypeVar("P", bound=tuple)


@dataclass(frozen=True)
class Config:
    """Static build settings."""
    max_stack_depth: int = 32
    leaf_size_threshold: int = 8


class Backend(abc.ABC, Generic[P]):
    """Stateless backend; payloads are JAX pytrees."""

    name: str
    flavor: str
    algorithm: str
    runtime: str

    payload_type: Type[P]
    is_differentiable: bool = False
    has_distance_bounds: bool = False
    has_silhouette_query: bool = False
    has_ray_interval: bool = False
    has_boundary_sampling: bool = False
    has_all_hits: bool = False
    has_point_classification: bool = False
    has_robin_query: bool = False

    @abc.abstractmethod
    def build(self, primitives: Primitives, config: Config) -> P:
        """Build a payload."""

    @abc.abstractmethod
    def closest_point(self, payload: P, x: jnp.ndarray
                      ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """(point, distance, primitive_id)."""

    @abc.abstractmethod
    def first_hit(self, payload: P, ray_origin: jnp.ndarray,
                  ray_direction: jnp.ndarray, max_t: jnp.ndarray
                  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """(t, primitive_id, hit)."""

    @abc.abstractmethod
    def aabb_overlap(self, payload: P,
                     lower: jnp.ndarray, upper: jnp.ndarray) -> jnp.ndarray:
        """Whether any primitive AABB overlaps [lower, upper]."""

    def distance_bounds(self, payload: P,
                        lower: jnp.ndarray, upper: jnp.ndarray
                        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """(minimum, maximum) distance over the query AABB."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement distance_bounds"
        )

    def classify_point(self, payload: P, x: jnp.ndarray) -> jnp.ndarray:
        """Sign: -1 inside, 0 on boundary, +1 outside, NaN on failure."""
        raise NotImplementedError("point classification is unavailable")

    def closest_silhouette(self, payload: P, x: jnp.ndarray
                           ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """(point, distance, edge_id); requires silhouette capability."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement closest_silhouette"
        )
