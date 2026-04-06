from __future__ import annotations

import os
from pathlib import Path

import pytest

from path_capi_python import PATHLoader


def _env_or_none(name: str) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value else None


@pytest.mark.skipif(
    _env_or_none("PATH_CAPI_LIBPATH") is None,
    reason="Set PATH_CAPI_LIBPATH to run PATH runtime tests.",
)
def test_loader_can_report_version() -> None:
    path_lib = _env_or_none("PATH_CAPI_LIBPATH")
    lusol_lib = _env_or_none("PATH_CAPI_LIBLUSOL")

    loader = PATHLoader(path_lib=Path(path_lib), lusol_lib=Path(lusol_lib) if lusol_lib else None)
    runtime = loader.load()

    version = loader.version(runtime)
    assert version.lower().startswith("path")


@pytest.mark.skipif(
    _env_or_none("PATH_CAPI_LIBPATH") is None,
    reason="Set PATH_CAPI_LIBPATH to run PATH runtime tests.",
)
def test_loader_license_check_small_problem() -> None:
    path_lib = _env_or_none("PATH_CAPI_LIBPATH")
    lusol_lib = _env_or_none("PATH_CAPI_LIBLUSOL")

    loader = PATHLoader(path_lib=Path(path_lib), lusol_lib=Path(lusol_lib) if lusol_lib else None)
    runtime = loader.load()

    assert loader.check_license(runtime, n_vars=10, nnz=10) is True
