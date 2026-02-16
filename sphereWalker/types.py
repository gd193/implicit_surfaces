from typing import NamedTuple
import jax.numpy as jnp

class WalkerState(NamedTuple):
    position: jnp.ndarray # shape (dim, )
    nSteps: jnp.ndarray  # shape ()
    w: jnp.ndarray       # shape ()
    trust_region_size: jnp.ndarray # shape ()
    rng_key: jnp.ndarray # PRNG key
    done_flag: jnp.ndarray # shape (), boolean flag indicating if the walker is done

class DifferentialWalkerState(NamedTuple):
    position: jnp.ndarray # shape (dim, )
    boundary_hit_position: jnp.ndarray # shape (dim, ) # position where boundary was hit in phase 0
    nSteps: jnp.ndarray # shape ()
    w: jnp.ndarray # shape ()
    trust_region_size: jnp.ndarray # shape ()
    rng_key: jnp.ndarray # PRNG key
    done_flag: jnp.ndarray # shape (), boolean flag indicating if the walker is done
    phase: jnp.ndarray # shape ()

class WalkerOutput(NamedTuple):
    position: jnp.ndarray # shape (dim, )
    nSteps: jnp.ndarray  # shape ()
    w: jnp.ndarray       # shape ()