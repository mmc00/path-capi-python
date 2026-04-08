from __future__ import annotations

from types import SimpleNamespace

import pytest


pytest.importorskip("pyomo")

from pyomo.environ import ConcreteModel, Constraint, Var
from pyomo.opt import SolverFactory
from pyomo.opt.results.solver import SolverStatus, TerminationCondition

import path_capi_python  # noqa: F401 - ensure SolverFactory registration
from tests._fake_path_runtime import FakePath


def test_solver_factory_path_capi_bridge_solves_model_and_returns_results() -> None:
    model = ConcreteModel()
    model.x1 = Var(bounds=(0.0, None), initialize=1.5)
    model.x2 = Var(bounds=(0.0, 4.0), initialize=0.5)

    model.c1 = Constraint(expr=model.x1**2 - 3.0 == 0.0)
    model.c2 = Constraint(expr=model.x1 + 2.0 * model.x2 - 2.0 == 0.0)

    solver = SolverFactory("path_capi_bridge")
    assert solver.available()

    runtime = SimpleNamespace(path=FakePath())
    results = solver.solve(model, runtime=runtime, output=False)

    assert results.solver[0].status == SolverStatus.ok
    assert results.solver[0].termination_condition == TerminationCondition.optimal
    assert results.solver[0].return_code == 1
    assert results.problem[0].number_of_variables == 2
    assert results.problem[0].number_of_constraints == 2
    assert results.problem[0].number_of_nonzeros == 3
    assert model.x1.value == pytest.approx(1.0)
    assert model.x2.value == pytest.approx(2.0)


def test_solver_factory_path_capi_bridge_uses_explicit_variable_and_constraint_order() -> None:
    model = ConcreteModel()
    model.x = Var(bounds=(0.0, None), initialize=0.5)
    model.y = Var(bounds=(0.0, None), initialize=1.5)

    model.eq1 = Constraint(expr=model.y + 2.0 * model.x - 2.0 == 0.0)
    model.eq2 = Constraint(expr=model.y**2 - 3.0 == 0.0)

    solver = SolverFactory("path_capi_bridge")
    runtime = SimpleNamespace(path=FakePath())
    results = solver.solve(
        model,
        runtime=runtime,
        constraints=[model.eq1, model.eq2],
        variables=[model.y, model.x],
        output=False,
    )

    assert results.solver[0].termination_condition == TerminationCondition.optimal
    assert model.y.value == pytest.approx(1.0)
    assert model.x.value == pytest.approx(2.0)
