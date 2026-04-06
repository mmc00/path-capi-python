from __future__ import annotations

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

    assert fake_path.last_output_option == b"output no"
    assert fake_path.last_function_values == [-0.25, 0.5]
    assert fake_path.last_jacobian_values == [3.0, 1.0, 1.0]
    assert fake_path.last_row_indices == [1, 2, 1]
    assert fake_path.last_col_starts == [1, 3]
    assert fake_path.last_col_lengths == [2, 1]
