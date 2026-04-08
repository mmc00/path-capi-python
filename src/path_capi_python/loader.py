from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path


class PATHLibraryError(RuntimeError):
    """Raised when PATH shared libraries cannot be loaded."""


@dataclass(frozen=True)
class PATHRuntime:
    """Loaded PATH runtime handles."""

    path: ctypes.CDLL
    lusol: ctypes.CDLL | None = None


class PATHLoader:
    """Thin loader for PATH C API shared libraries."""

    PATH_LIB_ENV = "PATH_CAPI_LIBPATH"
    LUSOL_LIB_ENV = "PATH_CAPI_LIBLUSOL"

    def __init__(self, path_lib: str | os.PathLike[str], lusol_lib: str | os.PathLike[str] | None = None):
        self.path_lib = Path(path_lib).expanduser().resolve()
        self.lusol_lib = Path(lusol_lib).expanduser().resolve() if lusol_lib else None

    @classmethod
    def from_environment(cls) -> "PATHLoader":
        path_lib = os.environ.get(cls.PATH_LIB_ENV)
        if not path_lib:
            raise PATHLibraryError(f"Missing required environment variable: {cls.PATH_LIB_ENV}")
        return cls(
            path_lib=path_lib,
            lusol_lib=os.environ.get(cls.LUSOL_LIB_ENV),
        )

    def load(self) -> PATHRuntime:
        if not self.path_lib.exists():
            raise PATHLibraryError(f"PATH library not found: {self.path_lib}")

        old_dyld = os.environ.get("DYLD_LIBRARY_PATH")
        lib_dir = str(self.path_lib.parent)
        if old_dyld:
            if lib_dir not in old_dyld.split(":"):
                os.environ["DYLD_LIBRARY_PATH"] = f"{lib_dir}:{old_dyld}"
        else:
            os.environ["DYLD_LIBRARY_PATH"] = lib_dir

        lusol_handle: ctypes.CDLL | None = None
        if self.lusol_lib:
            if not self.lusol_lib.exists():
                raise PATHLibraryError(f"LUSOL library not found: {self.lusol_lib}")
            lusol_handle = ctypes.CDLL(str(self.lusol_lib))

        path_handle = ctypes.CDLL(str(self.path_lib))
        path_handle.Path_Version.restype = ctypes.c_char_p
        path_handle.Path_CheckLicense.argtypes = [ctypes.c_int, ctypes.c_int]
        path_handle.Path_CheckLicense.restype = ctypes.c_int

        return PATHRuntime(path=path_handle, lusol=lusol_handle)

    @staticmethod
    def version(runtime: PATHRuntime) -> str:
        raw = runtime.path.Path_Version()
        if raw is None:
            raise PATHLibraryError("Path_Version returned null")
        return raw.decode()

    @staticmethod
    def check_license(runtime: PATHRuntime, n_vars: int, nnz: int) -> bool:
        return bool(runtime.path.Path_CheckLicense(int(n_vars), int(nnz)))
