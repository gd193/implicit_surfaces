"""Respect device settings; default to JAX's first available device."""
from __future__ import annotations

import os

# Avoid reserving most GPU memory.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def pytest_configure(config):  # noqa: D401
    import jax

    devices = jax.devices()
    if not devices:
        raise RuntimeError("JAX reported no available devices")
    default_device = devices[0]
    jax.config.update("jax_default_device", default_device)
    config._jax_devices = devices
    config._jax_default_device = default_device


def pytest_report_header(config):
    device = getattr(config, "_jax_default_device", None)
    if device is None:
        return None
    count = len(getattr(config, "_jax_devices", (device,)))
    return f"jax: default device {device} ({count} visible)"
