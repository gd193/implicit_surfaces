"""Device-pointer adapters to independent VTK-m CUDA contour worklets."""

from __future__ import annotations
import ctypes as C
import os
from pathlib import Path
import numpy as np


class ReferenceMesh:
    def __init__(self, library, handle):
        self._library, self._handle = library, handle
        vp, fp, nv, nt = C.c_void_p(), C.c_void_p(), C.c_int(), C.c_int()
        if library.vtkm_mc_info(handle, C.byref(vp), C.byref(fp), C.byref(nv), C.byref(nt)):
            self.close()
            raise RuntimeError(library.vtkm_mc_error().decode())
        self.vertices_ptr, self.faces_ptr = vp.value, fp.value
        self.vertex_count, self.triangle_count = nv.value, nt.value

    def numpy(self):
        vertices = np.empty((self.vertex_count, 3), np.float32)
        faces = np.empty((self.triangle_count, 3), np.int32)
        if self._library.vtkm_mc_copy(self._handle, vertices.ctypes.data, faces.ctypes.data):
            raise RuntimeError(self._library.vtkm_mc_error().decode())
        return vertices, faces

    def edges(self):
        edges = np.empty((self.vertex_count, 2), np.int32)
        if self._library.vtkm_mc_copy_edges(self._handle, edges.ctypes.data):
            raise RuntimeError(self._library.vtkm_mc_error().decode())
        return np.sort(edges, axis=1)

    def close(self):
        if self._handle:
            self._library.vtkm_mc_free(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


class VTKmReference:
    def __init__(self, library=None):
        library = library or os.environ.get("IMPLICIT_SURFACES_VTKM_LIBRARY")
        if library is None:
            raise ValueError("provide a library path or IMPLICIT_SURFACES_VTKM_LIBRARY")
        self.library_path = Path(library).resolve()
        self.lib = C.CDLL(str(self.library_path))
        self.lib.vtkm_mc_error.restype = C.c_char_p
        self.lib.vtkm_mc_run.restype = C.c_void_p
        self.lib.vtkm_mc_run.argtypes = [
            C.c_void_p,
            C.c_int,
            C.c_int,
            C.c_int,
            C.c_float,
            C.c_int,
            C.c_int,
            C.c_void_p,
            C.c_void_p,
        ]
        self.lib.vtkm_mc_info.argtypes = [
            C.c_void_p,
            C.POINTER(C.c_void_p),
            C.POINTER(C.c_void_p),
            C.POINTER(C.c_int),
            C.POINTER(C.c_int),
        ]
        self.lib.vtkm_mc_free.argtypes = [C.c_void_p]
        self.lib.vtkm_mc_copy_edges.argtypes = [C.c_void_p, C.c_void_p]
        self.lib.vtkm_mc_copy.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p]

    def extract(
        self,
        grid,
        threshold=0.0,
        lower=None,
        upper=None,
        *,
        synchronize_input=True,
        backend="marching_cells",
        capture_edges=False,
    ):
        import warp as wp

        if grid.dtype != wp.float32 or len(grid.shape) != 3 or min(grid.shape) < 2:
            raise ValueError("grid must have FP32 dtype and three dimensions of size >=2")
        if str(grid.device) != "cuda:0" or not grid.is_contiguous:
            raise ValueError("grid must be C-contiguous and resident on cuda:0")
        if np.prod(grid.shape, dtype=np.int64) > np.iinfo(np.int32).max:
            raise ValueError("VTK-m reference uses int32 grid indices")
        if backend not in ("marching_cells", "flying_edges"):
            raise ValueError("backend must be marching_cells or flying_edges")
        shape = grid.shape[::-1]  # VTK-m stores x contiguously: [z, y, x].
        lower = np.asarray((0, 0, 0) if lower is None else lower, dtype=np.float32)
        upper = np.asarray(np.asarray(shape) - 1 if upper is None else upper, dtype=np.float32)
        spacing = np.asarray(
            (upper - lower) / (np.asarray(shape, dtype=np.float32) - 1),
            dtype=np.float32,
        )
        if (
            lower.shape != (3,)
            or upper.shape != (3,)
            or not np.isfinite(lower).all()
            or not np.isfinite(spacing).all()
            or np.any(spacing <= 0)
        ):
            raise ValueError("bounds must be finite increasing 3D corners")
        if synchronize_input:
            wp.synchronize_device(grid.device)
        handle = self.lib.vtkm_mc_run(
            grid.ptr,
            *shape,
            threshold,
            int(backend == "flying_edges"),
            int(capture_edges),
            lower.ctypes.data,
            spacing.ctypes.data,
        )
        if not handle:
            raise RuntimeError(self.lib.vtkm_mc_error().decode())
        return ReferenceMesh(self.lib, handle)
