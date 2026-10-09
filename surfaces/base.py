"""Shared surface protocol base class.

The implicit-surface implementations inherit from this class so callers can
use common capability flags without depending on a particular surface family.
Concrete query methods are defined by the implicit/explicit subclasses.
"""


class Surface:
    """Minimal common protocol for geometric surface implementations."""

    is_differentiable_wrt_params: bool = True
    needs_grad_for_normals: bool = True
