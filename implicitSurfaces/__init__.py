"""Back-compat shim — implicit surfaces moved to `surfaces.implicit`.

This package re-exports the relocated modules under their old names so
existing imports keep working. New code should import directly from
`surfaces.implicit`. Plan to remove this shim in a follow-up release.
"""
from surfaces.implicit.base import ImplicitSurface, implicitSurface  # noqa: F401
