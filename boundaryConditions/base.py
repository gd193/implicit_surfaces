"""Boundary-condition contracts: which condition applies where, and its data.

A `BoundaryPartition` labels space with integer region ids. A `BoundaryData`
maps a region to coefficients of  a u + b du/dn_out = c,  where n_out is the
outward unit normal supplied by the geometry. Partitions do not approximate
the implicit surface; they only say which boundary patch a point belongs to.

Dtype contract (holds with and without JAX_ENABLE_X64):
  * Integers (region ids, kinds, MIXED) are int32.
  * Floats follow the query position: ftype = jnp.result_type(x, float32),
    so explicit f32 geometry stays f32 under x64.
"""
import abc
from enum import IntEnum
from typing import Any, Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp


class BoundaryKind(IntEnum):
    DIRICHLET = 0
    NEUMANN = 1
    # ROBIN = 2 is reserved for  du/dn + mu u = h  -> (mu, 1, h). Not supported.


# Region-id sentinel returned by classify_box: "no uniform label established".
# Region ids are int32 >= 0. MIXED is never passed to BoundaryData.evaluate.
MIXED = -1


class BoundaryValues(NamedTuple):
    """Coefficients of  a u + b du/dn_out = c, in the query ftype.

    a and b are scalars: a region's kind applies to every component. c has the
    data's value_shape: () for a scalar field, e.g. (m,) for a vector field.

    Dirichlet u = g:          (1, 0, g)
    Neumann   du/dn_out = h:  (0, 1, h)   h is a normal derivative, not -k du/dn.
    """
    a: jax.Array
    b: jax.Array
    c: jax.Array


def float_dtype(*xs) -> jnp.dtype:
    """Working float dtype for queries at xs (never narrower than float32)."""
    return jnp.result_type(*xs, jnp.float32)


def storage_float_dtype() -> jnp.dtype:
    """Widest float JAX currently allows: f64 under x64, otherwise f32."""
    return jax.dtypes.canonicalize_dtype(jnp.float64)


def _check_count(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class BoundaryPartition(abc.ABC):
    """Assigns every point exactly one region id.

    Queries are pure, single-point (batch with vmap), and jit-compatible.
    `params` keeps the signature symmetric with surfaces; fixed partitions
    ignore it.
    """

    def __init__(self, dim: int, n_regions: int):
        self.dim = _check_count("dim", dim)
        self.n_regions = _check_count("n_regions", n_regions)

    @abc.abstractmethod
    def classify_point(self, params: Any, x: jax.Array) -> jax.Array:
        """int32 scalar region id of x in [0, n_regions). Never MIXED.

        Points on internal split planes have a deterministic owner for a given
        float dtype. Points within rounding of a plane may be owned differently
        in f32 and f64.
        """
        raise NotImplementedError

    def classify_box(self, params: Any, lower: jax.Array, upper: jax.Array) -> jax.Array:
        """int32 region id if every point of the closed box [lower, upper] has it.

        Otherwise MIXED, meaning a uniform label could not be established, not
        that the box necessarily contains several regions or the surface.
        Certification holds in exact arithmetic: implementations widen the box
        by their rounding error. Default: trivially conservative.
        """
        return jnp.asarray(MIXED, jnp.int32)


class BoundaryData(abc.ABC):
    """Per-region boundary-condition coefficients.

    Contract for `evaluate`:
      * region_id is a valid id (never MIXED); kinds are fixed during a solve.
      * x is a boundary point, but evaluate must be defined and differentiable
        in x on a neighborhood of the boundary: differential WoS takes the
        normal derivative of the Dirichlet data (ImplicitSurface.
        dirichlet_boundary_differential). Region lookup happens once at the
        boundary point and is held fixed under that derivative.
      * outward_normal is a unit vector from the geometry; data that does not
        depend on it ignores it.
      * Returns coefficients in float_dtype(x): scalar a, b and c of shape
        value_shape, fixed at construction. Pure; jit/vmap safe.
      * Trainable values live in the explicit params, separate from geometry.
    """

    def __init__(self, n_regions: int, value_shape: Tuple[int, ...] = ()):
        self.n_regions = _check_count("n_regions", n_regions)
        shape = tuple(value_shape)
        if not all(isinstance(s, int) and not isinstance(s, bool) and s > 0 for s in shape):
            raise ValueError("value_shape must be a tuple of positive integers")
        self.value_shape = shape

    @abc.abstractmethod
    def kinds(self) -> jax.Array:
        """(n_regions,) int32 BoundaryKind per region."""
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, params: Any, region_id: jax.Array, x: jax.Array,
                 outward_normal: jax.Array) -> BoundaryValues:
        raise NotImplementedError


class BoundaryConditions:
    """Pairs a partition with the data for its regions."""

    def __init__(self, partition: BoundaryPartition, data: BoundaryData):
        if partition.n_regions != data.n_regions:
            raise ValueError(f"partition has {partition.n_regions} regions, "
                             f"data has {data.n_regions}")
        self.partition = partition
        self.data = data
        self.dim = partition.dim

    def evaluate(self, partition_params: Any, data_params: Any, x: jax.Array,
                 outward_normal: jax.Array) -> Tuple[jax.Array, BoundaryValues]:
        region = self.partition.classify_point(partition_params, x)
        return region, self.data.evaluate(data_params, region, x, outward_normal)

    def dirichlet_fn(self, partition_params: Any, data_params: Any,
                     normal_fn: Callable[[jax.Array], jax.Array] | None = None
                     ) -> Callable[[jax.Array], jax.Array]:
        """Adapt to the integrators' `boundary_condition_fn(x) -> g`.

        g has the data's value_shape: scalar for scalar fields, a vector for
        vector fields.

        A walk ending in a non-Dirichlet region returns NaN, which the WoS
        accumulator records as INVALID; plain WoS cannot use Neumann data.
        Without normal_fn the normal is NaN, so normal-dependent data also
        fails explicitly instead of silently using a wrong normal.
        """
        def g(x):
            ftype = float_dtype(x)
            x = jnp.asarray(x, ftype)
            normal = (jnp.full((self.dim,), jnp.nan, ftype) if normal_fn is None
                      else jnp.asarray(normal_fn(x), ftype))
            region, values = self.evaluate(partition_params, data_params, x, normal)
            dirichlet = self.data.kinds()[region] == BoundaryKind.DIRICHLET
            return jnp.where(dirichlet, values.c, jnp.full(self.data.value_shape, jnp.nan, ftype))
        return g
