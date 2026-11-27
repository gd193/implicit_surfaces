"""GPU memory statistics for benchmark records."""
import jax


def gpu_memory_bytes(stat: str = "bytes_in_use") -> int:
    """`stat` from the first device's memory_stats(); -1 when unavailable (e.g. CPU)."""
    try:
        return int(jax.devices()[0].memory_stats()[stat])
    except Exception:
        return -1
