"""Mesh accelerator backends."""
from importlib.util import find_spec as _find_spec

from . import lbvh_jax  # noqa: F401
from . import sah_jax  # noqa: F401
from . import snch_jax  # noqa: F401

# Warp is optional. Discovery in a JAX-only environment must not advertise
# backends that cannot run there, nor initialize CUDA while importing.
if _find_spec("warp") is not None:
    from . import sah_warp  # noqa: F401
    from . import snch_warp  # noqa: F401
