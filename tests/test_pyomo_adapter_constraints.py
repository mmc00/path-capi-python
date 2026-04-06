from __future__ import annotations

from types import SimpleNamespace

import pytest


pytest.importorskip("pyomo")

from pyomo.environ import ConcreteModel, Constraint, Var

from path_capi_python import PyomoMCPAdapter
from tests._fake_path_runtime import FakePath


def test_build_from_equality_constraints_linear_case() -> None:
    model = ConcreteModel()
    model.x1 = Var(bounds=(0.0, None), initialize=0.0)
    model.x2 = Var(bounds=(0.0, None), initialize=0.0)

    model.c1 = Constraint(expr=2.0 - model.x1 - model.x2 == 0.0)
    model.c2 = Constraint(expr=-1.0 + 2.0 * model.x1 + 0.5 * model.x2 == 0.0)

    adapter = PyomoMCPAdapter()
    data = adapter.build_from_equality_constraints(model, constraints=[model.c1, model.c2], variables=[model.x1, model.x2])

    assert data.variable_names == ["x1", "x2"]
    assert data.q == [2.0, -1.0]
    assert data.M == [[-1.0, -1.0], [2.0, 0.5]]


def test_build_nonlinear_from_equality_constraints() -> None:
    model = ConcreteModel()
    model.x1 = Var(bounds=(0.0, None), initialize=1.5)
    model.x2 = Var(bounds=(0.0, 4.0), initialize=0.5)

    model.c1 = Constraint(expr=model.x1**2 - 3.0 == 0.0)
    model.c2 = Constraint(expr=model.x1 + 2.0 * model.x2 - 2.0 == 0.0)

    adapter = PyomoMCPAdapter()
    data = adapter.build_nonlinear_from_equality_constraints(
        model,
        constraints=[model.c1, model.c2],
        variables=[model.x1, model.x2],
    )

    assert data.variable_names == ["x1", "x2"]
    assert data.jacobian_structure.col_starts == [1, 3]
    assert data.jacobian_structure.col_lengths == [2, 1]
    assert data.jacobian_structure.row_indices == [1, 2, 2]
    assert data.callback_f([1.5, 0.5]) == [-0.75, 0.5]
    assert data.callback_jac([1.5, 0.5]) == [3.0, 1.0, 2.0]


def test_solve_nonlinear_from_equality_constraints() -> None:
    model = ConcreteModel()
    model.x1 = Var(bounds=(0.0, None), initialize=1.5)
    model.x2 = Var(bounds=(0.0, 4.0), initialize=0.5)

    model.c1 = Constraint(expr=model.x1**2 - 3.0 == 0.0)
    model.c2 = Constraint(expr=model.x1 + 2.0 * model.x2 - 2.0 == 0.0)

    fake_path = FakePath()
    runtime = SimpleNamespace(path=fake_path)

    adapter = PyomoMCPAdapter()
    result = adapter.solve_nonlinear_from_equality_constraints(
        runtime,
        model,
        constraints=[model.c1, model.c2],
        variables=[model.x1, model.x2],
        output=False,
    )

    assert result.termination_code == 1
    assert result.x == [1.0, 2.0]
    assert result.residual == 1.25e-9
    assert fake_path.last_function_values == [-0.75, 0.5]
    assert fake_path.last_jacobian_values == [3.0, 1.0, 2.0]
