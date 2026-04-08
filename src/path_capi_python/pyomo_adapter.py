from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .loader import PATHRuntime
from .mcp import JacobianStructure, NonlinearMCPResult, solve_nonlinear_mcp


@dataclass(frozen=True)
class PyomoModelSummary:
    n_variables: int
    n_constraints: int
    n_complementarity_constraints: int


@dataclass(frozen=True)
class LinearCallbackData:
    variable_names: list[str]
    M: list[list[float]]
    q: list[float]
    lb: list[float]
    ub: list[float]
    x0: list[float]


@dataclass(frozen=True)
class NonlinearCallbackData:
    variable_names: list[str]
    lb: list[float]
    ub: list[float]
    x0: list[float]
    jacobian_structure: JacobianStructure
    callback_f: Callable[[Sequence[float]], list[float]]
    callback_jac: Callable[[Sequence[float]], list[float]]


class PyomoMCPAdapter:
    """Initial scaffold for adapting Pyomo MCP models to PATH C API callbacks.

    Current status:
    - Provides model structure inspection helpers.
    - Placeholder for callback construction (F/J, bounds, starts).
    """

    def summarize_model(self, model: Any) -> PyomoModelSummary:
        from pyomo.environ import Constraint, Var
        try:
            from pyomo.mpec import Complementarity
        except Exception:
            Complementarity = tuple()  # pragma: no cover

        n_vars = sum(1 for _ in model.component_data_objects(Var, active=True, descend_into=True))
        n_cons = sum(1 for _ in model.component_data_objects(Constraint, active=True, descend_into=True))

        if Complementarity:
            n_comp = sum(
                1 for _ in model.component_data_objects(Complementarity, active=True, descend_into=True)
            )
        else:
            n_comp = 0

        return PyomoModelSummary(
            n_variables=n_vars,
            n_constraints=n_cons,
            n_complementarity_constraints=n_comp,
        )

    def build_callbacks(
        self,
        model: Any,
        *,
        expressions: Sequence[Any],
        variables: Sequence[Any] | None = None,
    ) -> LinearCallbackData:
        """Build linear PATH callback data from Pyomo expressions.

        This method targets a controlled linear MCP workflow where the user
        provides one expression per variable in the chosen ordering:

            F_i(x) = q_i + sum_j M_ij * x_j

        Args:
            model: Pyomo model instance.
            expressions: Sequence of linear Pyomo expressions for F(x).
            variables: Optional ordered variable list. If omitted, uses active
                scalar variable components in deterministic name order.

        Returns:
            LinearCallbackData ready for `solve_linear_mcp`.
        """
        from pyomo.environ import Var, value
        from pyomo.repn.standard_repn import generate_standard_repn

        if variables is None:
            variables = sorted(
                model.component_data_objects(Var, active=True, descend_into=True),
                key=lambda v: v.name,
            )

        var_list = list(variables)
        if not var_list:
            raise ValueError("No variables provided/found for callback construction")

        n = len(var_list)
        expr_list = list(expressions)
        if len(expr_list) != n:
            raise ValueError(
                "Expected one expression per variable. "
                f"Got {len(expr_list)} expressions for {n} variables."
            )

        var_to_pos = {var.name: idx for idx, var in enumerate(var_list)}
        M = [[0.0 for _ in range(n)] for _ in range(n)]
        q = [0.0 for _ in range(n)]

        for i, expr in enumerate(expr_list):
            repn = generate_standard_repn(expr, compute_values=True)
            if repn.is_nonlinear():
                raise ValueError(f"Expression {i} is nonlinear; linear adapter cannot handle it")

            q[i] = float(repn.constant or 0.0)
            for coef, var in zip(repn.linear_coefs or [], repn.linear_vars or []):
                vname = var.name
                if vname not in var_to_pos:
                    raise ValueError(
                        f"Variable {var.name} in expression {i} is outside provided variable order"
                    )
                j = var_to_pos[vname]
                M[i][j] += float(coef)

        lb: list[float] = []
        ub: list[float] = []
        x0: list[float] = []
        inf = 1.0e20

        for var in var_list:
            lo = value(var.lb) if var.has_lb() else -inf
            up = value(var.ub) if var.has_ub() else inf
            st = value(var) if var.value is not None else 0.0
            lb.append(float(lo))
            ub.append(float(up))
            x0.append(float(st))

        return LinearCallbackData(
            variable_names=[v.name for v in var_list],
            M=M,
            q=q,
            lb=lb,
            ub=ub,
            x0=x0,
        )

    def build_nonlinear_callbacks(
        self,
        model: Any,
        *,
        expressions: Sequence[Any],
        variables: Sequence[Any] | None = None,
    ) -> NonlinearCallbackData:
        """Build nonlinear residual and Jacobian callbacks from Pyomo expressions."""
        from pyomo.environ import Var, value
        from pyomo.core.expr.calculus.derivatives import Modes, differentiate

        def _is_zero_derivative(expr: Any) -> bool:
            if isinstance(expr, (int, float)):
                return float(expr) == 0.0
            if hasattr(expr, "is_constant") and expr.is_constant():
                return float(value(expr)) == 0.0
            return False

        if variables is None:
            variables = sorted(
                model.component_data_objects(Var, active=True, descend_into=True),
                key=lambda v: v.name,
            )

        var_list = list(variables)
        if not var_list:
            raise ValueError("No variables provided/found for callback construction")

        expr_list = list(expressions)
        n = len(var_list)
        if len(expr_list) != n:
            raise ValueError(
                "Expected one expression per variable. "
                f"Got {len(expr_list)} expressions for {n} variables."
            )

        inf = 1.0e20
        lb: list[float] = []
        ub: list[float] = []
        x0: list[float] = []

        for var in var_list:
            lo = value(var.lb) if var.has_lb() else -inf
            up = value(var.ub) if var.has_ub() else inf
            st = value(var) if var.value is not None else 0.0
            lb.append(float(lo))
            ub.append(float(up))
            x0.append(float(st))

        jacobian_exprs: list[tuple[int, Any]] = []
        column_rows: list[list[int]] = [[] for _ in range(n)]

        for j, var in enumerate(var_list):
            for i, expr in enumerate(expr_list):
                derivative = differentiate(expr, wrt=var, mode=Modes.reverse_symbolic)
                if _is_zero_derivative(derivative):
                    continue
                column_rows[j].append(i + 1)
                jacobian_exprs.append((i, derivative))

        structure = JacobianStructure.from_column_rows(column_rows)

        def _assign_values(x: Sequence[float]) -> None:
            if len(x) != n:
                raise ValueError(f"Expected {n} values, got {len(x)}")
            for var, val in zip(var_list, x):
                var.set_value(float(val), skip_validation=True)

        def _callback_f(x: Sequence[float]) -> list[float]:
            _assign_values(x)
            return [float(value(expr)) for expr in expr_list]

        def _callback_jac(x: Sequence[float]) -> list[float]:
            _assign_values(x)
            return [float(value(derivative)) for _i, derivative in jacobian_exprs]

        return NonlinearCallbackData(
            variable_names=[v.name for v in var_list],
            lb=lb,
            ub=ub,
            x0=x0,
            jacobian_structure=structure,
            callback_f=_callback_f,
            callback_jac=_callback_jac,
        )

    def build_from_equality_constraints(
        self,
        model: Any,
        *,
        constraints: Sequence[Any],
        variables: Sequence[Any] | None = None,
    ) -> LinearCallbackData:
        """Build linear callback data from Pyomo equality constraints.

        Each constraint is converted to a residual expression of the form:

            F_i(x) = body_i - rhs_i = 0

        where rhs_i is the equality target (lower == upper).
        """
        from pyomo.environ import value
        from pyomo.repn.standard_repn import generate_standard_repn

        con_list = list(constraints)
        if not con_list:
            raise ValueError("No constraints provided")

        expressions: list[Any] = []
        inferred_vars_by_name: dict[str, Any] = {}

        for con in con_list:
            if not con.active:
                continue
            if not con.equality:
                raise ValueError(f"Constraint {con.name} is not an equality")

            rhs = value(con.lower)
            expr = con.body - rhs
            expressions.append(expr)

            if variables is None:
                repn = generate_standard_repn(expr, compute_values=False)
                if repn.is_nonlinear():
                    raise ValueError(f"Constraint {con.name} is nonlinear")
                for var in repn.linear_vars or []:
                    inferred_vars_by_name[var.name] = var

        if variables is None:
            ordered_vars = [inferred_vars_by_name[name] for name in sorted(inferred_vars_by_name.keys())]
        else:
            ordered_vars = list(variables)

        return self.build_callbacks(model, expressions=expressions, variables=ordered_vars)

    def build_nonlinear_from_equality_constraints(
        self,
        model: Any,
        *,
        constraints: Sequence[Any],
        variables: Sequence[Any] | None = None,
    ) -> NonlinearCallbackData:
        """Build nonlinear callbacks from Pyomo equality constraints."""
        from pyomo.environ import value
        from pyomo.repn.standard_repn import generate_standard_repn

        con_list = list(constraints)
        if not con_list:
            raise ValueError("No constraints provided")

        expressions: list[Any] = []
        inferred_vars_by_name: dict[str, Any] = {}

        for con in con_list:
            if not con.active:
                continue
            if not con.equality:
                raise ValueError(f"Constraint {con.name} is not an equality")

            rhs = value(con.lower)
            expr = con.body - rhs
            expressions.append(expr)

            if variables is None:
                repn = generate_standard_repn(expr, compute_values=False)
                for var in repn.linear_vars or []:
                    inferred_vars_by_name[var.name] = var
                for var in getattr(repn, "nonlinear_vars", None) or []:
                    inferred_vars_by_name[var.name] = var

        if variables is None:
            ordered_vars = [inferred_vars_by_name[name] for name in sorted(inferred_vars_by_name.keys())]
        else:
            ordered_vars = list(variables)

        return self.build_nonlinear_callbacks(model, expressions=expressions, variables=ordered_vars)

    def solve_nonlinear(
        self,
        runtime: PATHRuntime,
        model: Any,
        *,
        expressions: Sequence[Any],
        variables: Sequence[Any] | None = None,
        output: bool = True,
    ) -> NonlinearMCPResult:
        """Build nonlinear callbacks from expressions and solve with PATH."""
        data = self.build_nonlinear_callbacks(model, expressions=expressions, variables=variables)
        result = solve_nonlinear_mcp(
            runtime,
            n=len(data.variable_names),
            lb=data.lb,
            ub=data.ub,
            x0=data.x0,
            callback_f=data.callback_f,
            callback_jac=data.callback_jac,
            jacobian_structure=data.jacobian_structure,
            output=output,
        )
        self._write_solution(model, result.x, variables=variables)
        return result

    def solve_nonlinear_from_equality_constraints(
        self,
        runtime: PATHRuntime,
        model: Any,
        *,
        constraints: Sequence[Any],
        variables: Sequence[Any] | None = None,
        output: bool = True,
    ) -> NonlinearMCPResult:
        """Build nonlinear callbacks from equality constraints and solve with PATH."""
        data = self.build_nonlinear_from_equality_constraints(
            model,
            constraints=constraints,
            variables=variables,
        )
        result = solve_nonlinear_mcp(
            runtime,
            n=len(data.variable_names),
            lb=data.lb,
            ub=data.ub,
            x0=data.x0,
            callback_f=data.callback_f,
            callback_jac=data.callback_jac,
            jacobian_structure=data.jacobian_structure,
            output=output,
        )
        self._write_solution(model, result.x, variables=variables)
        return result

    def _write_solution(
        self,
        model: Any,
        values: Sequence[float],
        *,
        variables: Sequence[Any] | None = None,
    ) -> None:
        from pyomo.environ import Var

        if variables is None:
            variables = sorted(
                model.component_data_objects(Var, active=True, descend_into=True),
                key=lambda v: v.name,
            )

        var_list = list(variables)
        if len(var_list) != len(values):
            raise ValueError(
                f"Expected {len(var_list)} solution values for Pyomo write-back, got {len(values)}"
            )

        for var, value in zip(var_list, values):
            var.set_value(float(value), skip_validation=True)
