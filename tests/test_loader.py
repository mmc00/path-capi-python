from __future__ import annotations

import os
from pathlib import Path

import pytest

from path_capi_python import PATHLibraryError, PATHLoader


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


def test_loader_from_environment_reads_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path_lib = tmp_path / "libpath.dylib"
    lusol_lib = tmp_path / "liblusol.dylib"
    path_lib.write_text("")
    lusol_lib.write_text("")

    monkeypatch.setenv("PATH_CAPI_LIBPATH", str(path_lib))
    monkeypatch.setenv("PATH_CAPI_LIBLUSOL", str(lusol_lib))

    loader = PATHLoader.from_environment()

    assert loader.path_lib == path_lib.resolve()
    assert loader.lusol_lib == lusol_lib.resolve()


def test_loader_from_environment_requires_path_library(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PATH_CAPI_LIBPATH", raising=False)
    monkeypatch.delenv("PATH_CAPI_LIBLUSOL", raising=False)

    with pytest.raises(PATHLibraryError, match="PATH_CAPI_LIBPATH"):
        PATHLoader.from_environment()
