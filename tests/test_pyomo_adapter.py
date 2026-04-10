from __future__ import annotations

from types import SimpleNamespace

import pytest


pytest.importorskip("pyomo")

from pyomo.environ import ConcreteModel, Var

from path_capi_python import PyomoMCPAdapter
from tests._fake_path_runtime import FakePath


def test_build_callbacks_linear_controlled_case() -> None:
    model = ConcreteModel()
    model.x1 = Var(bounds=(0.0, None), initialize=0.0)
    model.x2 = Var(bounds=(0.0, None), initialize=0.0)

    # F1 = 2 - x1 - x2
    # F2 = -1 + 2*x1 + 0.5*x2
    expressions = [
        2.0 - model.x1 - model.x2,
        -1.0 + 2.0 * model.x1 + 0.5 * model.x2,
    ]

    adapter = PyomoMCPAdapter()
    data = adapter.build_callbacks(model, expressions=expressions, variables=[model.x1, model.x2])

    assert data.variable_names == ["x1", "x2"]
    assert data.q == [2.0, -1.0]
    assert data.M == [[-1.0, -1.0], [2.0, 0.5]]
    assert data.lb == [0.0, 0.0]
    assert data.ub == [1.0e20, 1.0e20]
    assert data.x0 == [0.0, 0.0]


def test_build_nonlinear_callbacks() -> None:
    model = ConcreteModel()
    model.x1 = Var(bounds=(0.0, None), initialize=1.5)
    model.x2 = Var(bounds=(0.0, 4.0), initialize=0.5)

    expressions = [
        model.x1**2 - 3.0,
        model.x1 + 2.0 * model.x2 - 2.0,
    ]

    adapter = PyomoMCPAdapter()
    data = adapter.build_nonlinear_callbacks(model, expressions=expressions, variables=[model.x1, model.x2])

    assert data.variable_names == ["x1", "x2"]
    assert data.lb == [0.0, 0.0]
    assert data.ub == [1.0e20, 4.0]
    assert data.x0 == [1.5, 0.5]
    assert data.jacobian_structure.col_starts == [1, 3]
    assert data.jacobian_structure.col_lengths == [2, 1]
    assert data.jacobian_structure.row_indices == [1, 2, 2]

    assert data.callback_f([1.5, 0.5]) == [-0.75, 0.5]
    assert data.callback_jac([1.5, 0.5]) == [3.0, 1.0, 2.0]


def test_build_nonlinear_callbacks_reverse_numeric_mode() -> None:
    model = ConcreteModel()
    model.x1 = Var(bounds=(0.0, None), initialize=1.5)
    model.x2 = Var(bounds=(0.0, 4.0), initialize=0.5)

    expressions = [
        model.x1**2 - 3.0,
        model.x1 + 2.0 * model.x2 - 2.0,
    ]

    adapter = PyomoMCPAdapter()
    data = adapter.build_nonlinear_callbacks(
        model,
        expressions=expressions,
        variables=[model.x1, model.x2],
        jacobian_eval_mode="reverse_numeric",
    )

    assert data.jacobian_eval_mode == "reverse_numeric"
    assert data.jacobian_structure.col_starts == [1, 3]
    assert data.jacobian_structure.col_lengths == [2, 1]
    assert data.jacobian_structure.row_indices == [1, 2, 2]
    assert data.callback_jac([1.5, 0.5]) == pytest.approx([3.0, 1.0, 2.0])


def test_solve_nonlinear_from_expressions() -> None:
    model = ConcreteModel()
    model.x1 = Var(bounds=(0.0, None), initialize=1.5)
    model.x2 = Var(bounds=(0.0, 4.0), initialize=0.5)

    expressions = [
        model.x1**2 - 3.0,
        model.x1 + 2.0 * model.x2 - 2.0,
    ]

    fake_path = FakePath()
    runtime = SimpleNamespace(path=fake_path)

    adapter = PyomoMCPAdapter()
    result = adapter.solve_nonlinear(
        runtime,
        model,
        expressions=expressions,
        variables=[model.x1, model.x2],
        output=False,
    )

    assert result.termination_code == 1
    assert result.x == [1.0, 2.0]
    assert result.residual == 1.25e-9
    assert model.x1.value == pytest.approx(1.0)
    assert model.x2.value == pytest.approx(2.0)
    assert fake_path.last_function_values == [-0.75, 0.5]
    assert fake_path.last_jacobian_values == [3.0, 1.0, 2.0]
