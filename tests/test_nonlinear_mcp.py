from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from path_capi_python import JacobianStructure, solve_nonlinear_mcp
from tests._fake_path_runtime import FakePath


def test_solve_nonlinear_mcp_callback_wiring() -> None:
    fake_path = FakePath()
    runtime = SimpleNamespace(path=fake_path)

    structure = JacobianStructure.from_column_rows([[1, 2], [1]])

    result = solve_nonlinear_mcp(
        runtime,
        n=2,
        lb=[0.0, 0.0],
        ub=[10.0, 10.0],
        x0=[0.0, 0.0],
        callback_f=lambda x: [x[0] ** 2 + x[1] - 3.0, x[0] + 2.0 * x[1] - 2.0],
        callback_jac=lambda x: [2.0 * x[0], 1.0, 1.0],
        jacobian_structure=structure,
        output=False,
    )

    assert result.termination_code == 1
    assert result.x == [1.0, 2.0]
    assert result.residual == 1.25e-9
    assert result.major_iterations == 3
    assert result.minor_iterations == 4
    assert result.function_evaluations == 5
    assert result.jacobian_evaluations == 2
    assert result.callback_profile.function_calls == 1
    assert result.callback_profile.jacobian_calls == 1
    assert result.callback_profile.jacobian_function_reuse_calls == 1
    assert result.callback_profile.function_time_sec >= 0.0
    assert result.callback_profile.jacobian_time_sec >= 0.0

    assert fake_path.last_output_option == b"output no"
    assert fake_path.last_function_values == [-0.25, 0.5]
    assert fake_path.last_jacobian_values == [3.0, 1.0, 1.0]
    assert fake_path.last_row_indices == [1, 2, 1]
    assert fake_path.last_col_starts == [1, 3]
    assert fake_path.last_col_lengths == [2, 1]


def test_solve_nonlinear_mcp_progress_history_and_env_options(monkeypatch, tmp_path: Path) -> None:
    fake_path = FakePath()
    runtime = SimpleNamespace(path=fake_path)
    structure = JacobianStructure.from_column_rows([[1, 2], [1]])
    progress_path = tmp_path / "progress.json"
    history_path = tmp_path / "progress.jsonl"

    monkeypatch.setenv("PATH_CAPI_PROGRESS_FILE", str(progress_path))
    monkeypatch.setenv("PATH_CAPI_PROGRESS_HISTORY_FILE", str(history_path))
    monkeypatch.setenv("PATH_CAPI_PROGRESS_INTERVAL_SEC", "0")
    monkeypatch.setenv("PATH_CAPI_OPTIONS", "crash_method none;minor_iteration_limit 25")

    solve_nonlinear_mcp(
        runtime,
        n=2,
        lb=[0.0, 0.0],
        ub=[10.0, 10.0],
        x0=[0.0, 0.0],
        callback_f=lambda x: [x[0] ** 2 + x[1] - 3.0, x[0] + 2.0 * x[1] - 2.0],
        callback_jac=lambda x: [2.0 * x[0], 1.0, 1.0],
        jacobian_structure=structure,
        output=False,
    )

    payload = json.loads(progress_path.read_text())
    assert payload["finished"] is True
    assert payload["termination_code"] == 1

    history_rows = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    assert len(history_rows) >= 2
    assert history_rows[-1]["finished"] is True
    assert history_rows[-1]["termination_code"] == 1

    assert b"output no" in fake_path.last_options
    assert b"crash_method none" in fake_path.last_options
    assert b"minor_iteration_limit 25" in fake_path.last_options
