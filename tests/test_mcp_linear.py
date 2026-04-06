from __future__ import annotations

import os
from pathlib import Path

import pytest

from path_capi_python import PATHLoader, solve_linear_mcp


def _env_or_none(name: str) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value else None


@pytest.mark.skipif(
    _env_or_none("PATH_CAPI_LIBPATH") is None,
    reason="Set PATH_CAPI_LIBPATH to run PATH MCP solve tests.",
)
def test_solve_linear_mcp_reference_case() -> None:
    path_lib = _env_or_none("PATH_CAPI_LIBPATH")
    lusol_lib = _env_or_none("PATH_CAPI_LIBLUSOL")

    loader = PATHLoader(path_lib=Path(path_lib), lusol_lib=Path(lusol_lib) if lusol_lib else None)
    runtime = loader.load()

    M = [
        [0.0, 0.0, -1.0, -1.0],
        [0.0, 0.0, 1.0, -2.0],
        [1.0, -1.0, 2.0, -2.0],
        [1.0, 2.0, -2.0, 4.0],
    ]
    q = [2.0, 2.0, -2.0, -6.0]
    lb = [0.0, 0.0, 0.0, 0.0]
    ub = [1.0e20, 1.0e20, 1.0e20, 1.0e20]
    x0 = [0.0, 0.0, 0.0, 0.0]

    result = solve_linear_mcp(runtime, M, q, lb, ub, x0, output=False)

    assert result.termination_code == 1
    assert result.residual < 1.0e-10

    expected = [2.8, 0.0, 0.8, 1.2]
    for got, exp in zip(result.x, expected):
        assert abs(got - exp) < 1.0e-8
