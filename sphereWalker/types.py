from typing import NamedTuple, Any
from enum import IntEnum
import jax.numpy as jnp

class WalkerTermination(IntEnum):
    ACTIVE = 0
    BOUNDARY_HIT = 1
    MAX_STEPS = 2
    INVALID = 3


class WalkerState(NamedTuple):
    position: jnp.ndarray # shape (dim, )
    nSteps: jnp.ndarray  # shape ()
    w: jnp.ndarray       # shape ()
    trust_region_size: jnp.ndarray # shape ()
    rng_key: jnp.ndarray # PRNG key
    done_flag: jnp.ndarray # phase completion, not necessarily a valid sample
    status: jnp.ndarray

class DifferentialWalkerState(NamedTuple):
    position: jnp.ndarray # shape (dim, )
    boundary_hit_position: jnp.ndarray # shape (dim, ) # position where boundary was hit in phase 0
    nSteps: jnp.ndarray # shape ()
    w: jnp.ndarray # shape ()
    trust_region_size: jnp.ndarray # shape ()
    rng_key: jnp.ndarray # PRNG key
    done_flag: jnp.ndarray # phase completion, not necessarily a valid sample
    status: jnp.ndarray
    offset_length: jnp.ndarray # accepted inward offset; zero before transition
    phase: jnp.ndarray # shape ()

class WalkerOutput(NamedTuple):
    position: jnp.ndarray # shape (dim, )
    nSteps: jnp.ndarray  # shape ()
    w: jnp.ndarray       # shape ()

class WalkerDiagnostics(NamedTuple):
    """Partial sums, never a success-only average. Counts index WalkerTermination.

    nSteps is the mean total step count over all requested samples, including
    failures. A differential sample succeeds only after both boundary hits.
    """
    position: jnp.ndarray
    contribution_sum: Any
    successful_samples: jnp.ndarray
    termination_counts: jnp.ndarray  # (N, 4), ACTIVE column is zero
    nSteps: jnp.ndarray

    @property
    def complete(self):
        return jnp.sum(self.termination_counts[:, 2:], axis=-1) == 0
