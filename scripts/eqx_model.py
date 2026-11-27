"""Load dense MLPs stored as alternating weight and bias arrays in an Equinox stream."""
from pathlib import Path

import numpy as np


def read_eqx_stream(path: str | Path) -> list[np.ndarray]:
    arrs = []
    with open(path, "rb") as f:
        while f.peek(1):
            arrs.append(np.lib.format.read_array(f))
    return arrs


def eqx_to_layers(eqx_path: str | Path) -> list[tuple[np.ndarray, np.ndarray]]:
    arrs = read_eqx_stream(eqx_path)
    if len(arrs) % 2 != 0:
        raise ValueError(f"{eqx_path}: expected even number of arrays, got {len(arrs)}")
    layers = []
    for i in range(0, len(arrs), 2):
        W, b = arrs[i], arrs[i + 1]
        if W.ndim != 2 or b.ndim != 1 or W.shape[0] != b.shape[0]:
            raise ValueError(f"{eqx_path}: bad layer shape at index {i}: W={W.shape}, b={b.shape}")
        layers.append((W, b))
    if not layers:
        raise ValueError(f"{eqx_path}: no dense layers found")
    return layers


def load_eqx_mlp(path, activation_name, dtype):
    """Build field(params, x) -> f(x) for surfaces (params unused); expose weights for metadata."""
    import jax
    import jax.numpy as jnp

    layers = [(jnp.asarray(w, dtype), jnp.asarray(b, dtype)) for w, b in eqx_to_layers(path)]
    activation = getattr(jax.nn, activation_name)

    def field(_params, x):
        for weight, bias in layers[:-1]:
            x = activation(weight @ x + bias)
        weight, bias = layers[-1]
        return weight @ x + bias

    return field, [weight for weight, _ in layers]
