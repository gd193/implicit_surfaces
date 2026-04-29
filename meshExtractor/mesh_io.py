"""Minimal mesh writers (OBJ, PLY)."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np


def write_obj(path: Union[str, Path], vertices, triangles) -> None:
    """Write vertices [V, 3] and triangles [T, 3] (0-indexed) to a Wavefront OBJ."""
    v = np.asarray(vertices)
    t = np.asarray(triangles)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must be [V, 3], got {v.shape}")
    if t.ndim != 2 or t.shape[1] != 3:
        raise ValueError(f"triangles must be [T, 3], got {t.shape}")

    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in v]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in t]
    Path(path).write_text("\n".join(lines) + "\n")


def write_ply(path: Union[str, Path], vertices, triangles) -> None:
    """Write a binary-free ASCII PLY."""
    v = np.asarray(vertices)
    t = np.asarray(triangles)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must be [V, 3], got {v.shape}")
    if t.ndim != 2 or t.shape[1] != 3:
        raise ValueError(f"triangles must be [T, 3], got {t.shape}")

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {v.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {t.shape[0]}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    body = [f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in v]
    body += [f"3 {a} {b} {c}" for a, b, c in t]
    Path(path).write_text("\n".join(header + body) + "\n")
