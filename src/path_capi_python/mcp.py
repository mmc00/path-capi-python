from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Callable, Sequence

from .loader import PATHRuntime


CB_PROBLEM_SIZE = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
CB_BOUNDS = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
)
CB_FUNCTION = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
)
CB_JACOBIAN = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_double),
)


class MCPInterface(ctypes.Structure):
    _fields_ = [
        ("interface_data", ctypes.c_void_p),
        ("problem_size", CB_PROBLEM_SIZE),
        ("bounds", CB_BOUNDS),
        ("function_evaluation", CB_FUNCTION),
        ("jacobian_evaluation", CB_JACOBIAN),
        ("hessian_evaluation", ctypes.c_void_p),
        ("start", ctypes.c_void_p),
        ("finish", ctypes.c_void_p),
        ("variable_name", ctypes.c_void_p),
        ("constraint_name", ctypes.c_void_p),
        ("basis", ctypes.c_void_p),
    ]


class Information(ctypes.Structure):
    _fields_ = [
        ("residual", ctypes.c_double),
        ("distance", ctypes.c_double),
        ("steplength", ctypes.c_double),
        ("total_time", ctypes.c_double),
        ("basis_time", ctypes.c_double),
        ("maximum_distance", ctypes.c_double),
        ("major_iterations", ctypes.c_int),
        ("minor_iterations", ctypes.c_int),
        ("crash_iterations", ctypes.c_int),
        ("function_evaluations", ctypes.c_int),
        ("jacobian_evaluations", ctypes.c_int),
        ("gradient_steps", ctypes.c_int),
        ("restarts", ctypes.c_int),
        ("generate_output", ctypes.c_int),
        ("generated_output", ctypes.c_int),
        ("forward", ctypes.c_int),
        ("backtrace", ctypes.c_int),
        ("gradient", ctypes.c_int),
        ("use_start", ctypes.c_int),
        ("use_basics", ctypes.c_int),
        ("used_start", ctypes.c_int),
        ("used_basics", ctypes.c_int),
    ]


@dataclass(frozen=True)
class JacobianStructure:
    """Sparse Jacobian structure in PATH's column-compressed callback format."""

    col_starts: list[int]
    col_lengths: list[int]
    row_indices: list[int]

    @property
    def nnz(self) -> int:
        return len(self.row_indices)

    @classmethod
    def from_column_rows(cls, column_rows: Sequence[Sequence[int]]) -> "JacobianStructure":
        col_starts: list[int] = []
        col_lengths: list[int] = []
        row_indices: list[int] = []
        cursor = 1

        for rows in column_rows:
            col_starts.append(cursor)
            col_lengths.append(len(rows))
            for row in rows:
                if row < 1:
                    raise ValueError("Jacobian row indices must be 1-based positive integers")
                row_indices.append(int(row))
                cursor += 1

        return cls(
            col_starts=col_starts,
            col_lengths=col_lengths,
            row_indices=row_indices,
        )


@dataclass
class LinearMCPResult:
    termination_code: int
    x: list[float]
    residual: float
    major_iterations: int
    minor_iterations: int


@dataclass
class NonlinearMCPResult:
    termination_code: int
    x: list[float]
    residual: float
    major_iterations: int
    minor_iterations: int
    function_evaluations: int
    jacobian_evaluations: int


def _configure_path_functions(path: object) -> None:
    path.Options_Create.restype = ctypes.c_void_p
    path.Options_Destroy.argtypes = [ctypes.c_void_p]
    path.Path_AddOptions.argtypes = [ctypes.c_void_p]
    path.Options_Default.argtypes = [ctypes.c_void_p]
    path.Options_Set.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    path.MCP_Create.argtypes = [ctypes.c_int, ctypes.c_int]
    path.MCP_Create.restype = ctypes.c_void_p
    path.MCP_Destroy.argtypes = [ctypes.c_void_p]
    path.MCP_SetInterface.argtypes = [ctypes.c_void_p, ctypes.POINTER(MCPInterface)]
    path.MCP_Jacobian_Structure_Constant.argtypes = [ctypes.c_void_p, ctypes.c_int]
    path.MCP_Jacobian_Data_Contiguous.argtypes = [ctypes.c_void_p, ctypes.c_int]
    path.Path_Solve.argtypes = [ctypes.c_void_p, ctypes.POINTER(Information)]
    path.Path_Solve.restype = ctypes.c_int
    path.MCP_GetX.argtypes = [ctypes.c_void_p]
    path.MCP_GetX.restype = ctypes.POINTER(ctypes.c_double)


def solve_nonlinear_mcp(
    runtime: PATHRuntime,
    *,
    n: int,
    lb: Sequence[float],
    ub: Sequence[float],
    x0: Sequence[float],
    callback_f: Callable[[Sequence[float]], Sequence[float]],
    callback_jac: Callable[[Sequence[float]], Sequence[float]],
    jacobian_structure: JacobianStructure,
    output: bool = True,
) -> NonlinearMCPResult:
    if n <= 0:
        raise ValueError("n must be positive")
    if not (len(lb) == n and len(ub) == n and len(x0) == n):
        raise ValueError("lb, ub, and x0 must have length n")
    if len(jacobian_structure.col_starts) != n or len(jacobian_structure.col_lengths) != n:
        raise ValueError("jacobian_structure must define one start and length per variable")

    c_lb = [float(value) for value in lb]
    c_ub = [float(value) for value in ub]
    c_x0 = [float(value) for value in x0]
    nnz = jacobian_structure.nnz

    def _read_x(size: int, x_ptr: ctypes.POINTER(ctypes.c_double)) -> list[float]:
        return [float(x_ptr[i]) for i in range(size)]

    @CB_PROBLEM_SIZE
    def _problem_size(_id, size_ptr, nnz_ptr):
        size_ptr[0] = n
        nnz_ptr[0] = nnz

    @CB_BOUNDS
    def _bounds(_id, size, x_ptr, l_ptr, u_ptr):
        for i in range(size):
            x_ptr[i] = c_x0[i]
            l_ptr[i] = c_lb[i]
            u_ptr[i] = c_ub[i]

    @CB_FUNCTION
    def _function(_id, size, x_ptr, f_ptr):
        values = list(callback_f(_read_x(size, x_ptr)))
        if len(values) != size:
            raise ValueError(f"callback_f returned {len(values)} values, expected {size}")
        for i, value in enumerate(values):
            f_ptr[i] = float(value)
        return 0

    @CB_JACOBIAN
    def _jacobian(_id, size, x_ptr, wantf, f_ptr, nnz_ptr, col_ptr, len_ptr, row_ptr, data_ptr):
        if wantf > 0:
            _function(_id, size, x_ptr, f_ptr)

        values = list(callback_jac(_read_x(size, x_ptr)))
        if len(values) != nnz:
            raise ValueError(f"callback_jac returned {len(values)} values, expected {nnz}")

        nnz_ptr[0] = nnz
        for j in range(size):
            col_ptr[j] = jacobian_structure.col_starts[j]
            len_ptr[j] = jacobian_structure.col_lengths[j]
        for k, row in enumerate(jacobian_structure.row_indices):
            row_ptr[k] = row
            data_ptr[k] = float(values[k])
        return 0

    callbacks = (_problem_size, _bounds, _function, _jacobian)

    path = runtime.path
    _configure_path_functions(path)

    options = path.Options_Create()
    if not options:
        raise RuntimeError("Options_Create returned null")
    path.Path_AddOptions(options)
    path.Options_Default(options)
    path.Options_Set(options, b"output yes" if output else b"output no")

    mcp = path.MCP_Create(n, max(nnz, 1))
    if not mcp:
        path.Options_Destroy(options)
        raise RuntimeError("MCP_Create returned null")

    interface = MCPInterface(
        interface_data=None,
        problem_size=_problem_size,
        bounds=_bounds,
        function_evaluation=_function,
        jacobian_evaluation=_jacobian,
        hessian_evaluation=None,
        start=None,
        finish=None,
        variable_name=None,
        constraint_name=None,
        basis=None,
    )
    path.MCP_SetInterface(mcp, ctypes.byref(interface))
    path.MCP_Jacobian_Structure_Constant(mcp, 1)
    path.MCP_Jacobian_Data_Contiguous(mcp, 1)

    info = Information()
    info.generate_output = 1 if output else 0
    info.use_start = 1
    info.use_basics = 0

    term = int(path.Path_Solve(mcp, ctypes.byref(info)))
    x_ptr = path.MCP_GetX(mcp)
    x_sol = [float(x_ptr[i]) for i in range(n)]

    path.MCP_Destroy(mcp)
    path.Options_Destroy(options)

    _ = callbacks

    return NonlinearMCPResult(
        termination_code=term,
        x=x_sol,
        residual=float(info.residual),
        major_iterations=int(info.major_iterations),
        minor_iterations=int(info.minor_iterations),
        function_evaluations=int(info.function_evaluations),
        jacobian_evaluations=int(info.jacobian_evaluations),
    )


def solve_linear_mcp(
    runtime: PATHRuntime,
    M: Sequence[Sequence[float]],
    q: Sequence[float],
    lb: Sequence[float],
    ub: Sequence[float],
    x0: Sequence[float],
    *,
    output: bool = True,
) -> LinearMCPResult:
    n = len(q)
    if not (len(M) == n and all(len(row) == n for row in M)):
        raise ValueError("M must be square with size len(q)")
    if not (len(lb) == n and len(ub) == n and len(x0) == n):
        raise ValueError("lb, ub, and x0 must have length len(q)")

    entries: list[float] = []
    column_rows: list[list[int]] = [[] for _ in range(n)]
    for j in range(n):
        for i in range(n):
            value = float(M[i][j])
            if value != 0.0:
                column_rows[j].append(i + 1)
                entries.append(value)

    result = solve_nonlinear_mcp(
        runtime,
        n=n,
        lb=lb,
        ub=ub,
        x0=x0,
        callback_f=lambda x: [
            float(q[i]) + sum(float(M[i][j]) * float(x[j]) for j in range(n))
            for i in range(n)
        ],
        callback_jac=lambda _x: entries,
        jacobian_structure=JacobianStructure.from_column_rows(column_rows),
        output=output,
    )

    return LinearMCPResult(
        termination_code=result.termination_code,
        x=result.x,
        residual=result.residual,
        major_iterations=result.major_iterations,
        minor_iterations=result.minor_iterations,
    )
