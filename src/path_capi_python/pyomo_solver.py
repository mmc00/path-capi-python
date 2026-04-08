from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Sequence

from pyomo.opt.base.solvers import OptSolver, SolverFactory
from pyomo.opt.results.problem import ProblemSense
from pyomo.opt.results.results_ import SolverResults
from pyomo.opt.results.solver import SolverStatus, TerminationCondition

from .loader import PATHLoader, PATHRuntime
from .pyomo_adapter import NonlinearCallbackData, PyomoMCPAdapter


def _termination_condition(code: int) -> TerminationCondition:
    if code == 1:
        return TerminationCondition.optimal
    return TerminationCondition.unknown


@SolverFactory.register("path_capi_bridge", doc="PATH C API bridge for Pyomo equality-constraint MCP models")
class PATHCAPIBridgeSolver(OptSolver):
    def __init__(self, **kwds):
        kwds["type"] = "path_capi_bridge"
        super().__init__(**kwds)
        self._model: Any | None = None
        self._runtime: PATHRuntime | None = None
        self._constraints: Sequence[Any] | None = None
        self._variables: Sequence[Any] | None = None
        self._output = True
        self._adapter = PyomoMCPAdapter()
        self._callback_data: NonlinearCallbackData | None = None
        self._solve_result = None
        self.results: SolverResults | None = None

    def available(self, exception_flag: bool = True) -> bool:
        return True

    def license_is_valid(self) -> bool:
        try:
            runtime = self._runtime or PATHLoader.from_environment().load()
        except Exception:
            return False
        return PATHLoader.check_license(runtime, n_vars=10, nnz=10)

    def warm_start_capable(self) -> bool:
        return True

    def _presolve(self, *args, **kwds):
        if len(args) != 1:
            raise ValueError("path_capi_bridge expects exactly one Pyomo model instance")

        self._model = args[0]
        self._runtime = kwds.pop("runtime", None)
        path_lib = kwds.pop("path_lib", None)
        lusol_lib = kwds.pop("lusol_lib", None)
        self._constraints = kwds.pop("constraints", None)
        self._variables = kwds.pop("variables", None)
        self._output = bool(kwds.pop("output", getattr(self.options, "output", True)))

        if self._runtime is None:
            if path_lib is not None:
                self._runtime = PATHLoader(path_lib=path_lib, lusol_lib=lusol_lib).load()
            else:
                self._runtime = PATHLoader.from_environment().load()

        super()._presolve(*args, **kwds)

    def _apply_solver(self):
        if self._model is None or self._runtime is None:
            raise RuntimeError("Solver state not initialized")

        constraints = self._constraints
        if constraints is None:
            from pyomo.environ import Constraint

            constraints = list(self._model.component_data_objects(Constraint, active=True, descend_into=True))

        self._callback_data = self._adapter.build_nonlinear_from_equality_constraints(
            self._model,
            constraints=constraints,
            variables=self._variables,
        )
        self._solve_result = self._adapter.solve_nonlinear_from_equality_constraints(
            self._runtime,
            self._model,
            constraints=constraints,
            variables=self._variables,
            output=self._output,
        )

        return SimpleNamespace(rc=0, log="")

    def _postsolve(self):
        if self._callback_data is None or self._solve_result is None or self._model is None:
            raise RuntimeError("Solver did not run")

        results = SolverResults()
        solver = results.solver.add()
        problem = results.problem.add()

        term = _termination_condition(self._solve_result.termination_code)
        solver.name = self.name
        solver.status = SolverStatus.ok if self._solve_result.termination_code == 1 else SolverStatus.warning
        solver.return_code = self._solve_result.termination_code
        solver.termination_condition = term
        solver.termination_message = f"PATH termination code {self._solve_result.termination_code}"
        solver.message = solver.termination_message
        solver.wallclock_time = None

        problem.name = getattr(self._model, "name", None)
        problem.number_of_objectives = 0
        problem.number_of_constraints = len(self._callback_data.variable_names)
        problem.number_of_variables = len(self._callback_data.variable_names)
        problem.number_of_nonzeros = self._callback_data.jacobian_structure.nnz
        problem.number_of_continuous_variables = len(self._callback_data.variable_names)
        problem.number_of_binary_variables = 0
        problem.number_of_integer_variables = 0
        problem.sense = ProblemSense.unknown

        self.results = results
        return results
