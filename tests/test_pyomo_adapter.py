from __future__ import annotations

import pytest


pytest.importorskip("pyomo")

from pyomo.environ import ConcreteModel, Var

from path_capi_python import PyomoMCPAdapter


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
