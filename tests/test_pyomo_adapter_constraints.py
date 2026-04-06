from __future__ import annotations

import pytest


pytest.importorskip("pyomo")

from pyomo.environ import ConcreteModel, Constraint, Var

from path_capi_python import PyomoMCPAdapter


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
