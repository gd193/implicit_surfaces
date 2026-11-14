"""Warp query backend discovery; host builders remain independently usable."""
from importlib.util import find_spec as _find_spec

# Warp is optional. Discovery without Warp must not advertise
# backends that cannot run there, nor initialize CUDA while importing.
if _find_spec("warp") is not None:
    from . import sah_warp  # noqa: F401
    from . import snch_warp  # noqa: F401
