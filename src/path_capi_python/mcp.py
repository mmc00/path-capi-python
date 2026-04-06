from __future__ import annotations

import ctypes
from dataclasses import dataclass

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


@dataclass
class LinearMCPResult:
    termination_code: int
    x: list[float]
    residual: float
    major_iterations: int
    minor_iterations: int


def solve_linear_mcp(
    runtime: PATHRuntime,
    M: list[list[float]],
    q: list[float],
    lb: list[float],
    ub: list[float],
    x0: list[float],
    *,
    output: bool = True,
) -> LinearMCPResult:
    n = len(q)
    if not (len(M) == n and all(len(row) == n for row in M)):
        raise ValueError("M must be square with size len(q)")
    if not (len(lb) == n and len(ub) == n and len(x0) == n):
        raise ValueError("lb, ub, and x0 must have length len(q)")

    entries: list[tuple[int, int, float]] = []
    col_starts: list[int] = []
    col_lens: list[int] = []
    cursor = 1
    for j in range(n):
        col_starts.append(cursor)
        count = 0
        for i in range(n):
            val = float(M[i][j])
            if val != 0.0:
                entries.append((i + 1, j + 1, val))
                count += 1
                cursor += 1
        col_lens.append(count)
    nnz = len(entries)

    c_lb = (ctypes.c_double * n)(*map(float, lb))
    c_ub = (ctypes.c_double * n)(*map(float, ub))
    c_x0 = (ctypes.c_double * n)(*map(float, x0))
    c_q = (ctypes.c_double * n)(*map(float, q))

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
        for i in range(size):
            total = c_q[i]
            for j in range(size):
                total += M[i][j] * x_ptr[j]
            f_ptr[i] = total
        return 0

    @CB_JACOBIAN
    def _jacobian(_id, size, x_ptr, wantf, f_ptr, nnz_ptr, col_ptr, len_ptr, row_ptr, data_ptr):
        if wantf > 0:
            _function(_id, size, x_ptr, f_ptr)
        nnz_ptr[0] = nnz
        for j in range(size):
            col_ptr[j] = col_starts[j]
            len_ptr[j] = col_lens[j]
        for k, (row_i, _col_j, val) in enumerate(entries):
            row_ptr[k] = row_i
            data_ptr[k] = val
        return 0

    # Keep callback references alive for the full solve call.
    callbacks = (_problem_size, _bounds, _function, _jacobian)

    path = runtime.path
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

    return LinearMCPResult(
        termination_code=term,
        x=x_sol,
        residual=float(info.residual),
        major_iterations=int(info.major_iterations),
        minor_iterations=int(info.minor_iterations),
    )
