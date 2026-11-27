"""Shared argument validation."""
from numbers import Integral


def check_int(name: str, value, *, minimum: int = 1) -> int:
    """Return value as int; reject bools, non-integers and values below minimum."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        kind = {0: 'a nonnegative integer', 1: 'a positive integer'}.get(
            minimum, f'an integer >= {minimum}')
        raise ValueError(f'{name} must be {kind}')
    return int(value)
