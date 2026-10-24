"""Register and select accelerator backends."""
from __future__ import annotations

from typing import Dict, List, Type

from .protocol import Backend


_REGISTRY: Dict[str, Type[Backend]] = {}


_REQUIRED_IDENTITY_ATTRS = ("name", "flavor", "algorithm", "runtime",
                            "payload_type")


def register_backend(cls: Type[Backend]) -> Type[Backend]:
    """Register a backend after checking its identity."""
    missing = [a for a in _REQUIRED_IDENTITY_ATTRS
               if not getattr(cls, a, None)]
    if missing:
        raise ValueError(
            f"{cls.__name__} missing required Backend attribute(s): "
            f"{', '.join(missing)}"
        )
    expected_name = f"{cls.flavor}.{cls.algorithm}.{cls.runtime}"
    if cls.name != expected_name:
        raise ValueError(
            f"{cls.__name__}.name = {cls.name!r} does not match "
            f"the required '<flavor>.<algorithm>.<runtime>' convention "
            f"(expected {expected_name!r})"
        )
    if cls.name in _REGISTRY:
        raise ValueError(
            f"backend name {cls.name!r} already registered "
            f"by {_REGISTRY[cls.name].__name__}"
        )
    _REGISTRY[cls.name] = cls
    return cls


def get_backend(name: str) -> Type[Backend]:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(
            f"backend {name!r} not registered. Available: {available}"
        )
    return _REGISTRY[name]


def list_backends(flavor: str | None = None,
                  runtime: str | None = None) -> List[str]:
    out = []
    for name, cls in _REGISTRY.items():
        if flavor is not None and cls.flavor != flavor:
            continue
        if runtime is not None and cls.runtime != runtime:
            continue
        out.append(name)
    return sorted(out)
