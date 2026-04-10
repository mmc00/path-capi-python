from __future__ import annotations

import ctypes
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
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
    callback_profile: "CallbackProfile"


@dataclass
class CallbackProfile:
    function_calls: int = 0
    function_time_sec: float = 0.0
    jacobian_calls: int = 0
    jacobian_time_sec: float = 0.0
    jacobian_function_reuse_calls: int = 0

    @property
    def total_callback_time_sec(self) -> float:
        return self.function_time_sec + self.jacobian_time_sec


@dataclass
class ProgressSnapshot:
    started_at: float
    last_emit_at: float
    interval_sec: float
    path: Path | None
    history_path: Path | None = None
    latest_x: list[float] | None = None
    latest_function_inf_norm: float | None = None
    latest_function_l2_norm: float | None = None
    latest_jacobian_inf_norm: float | None = None
    latest_stage: str | None = None

    def maybe_write(self, callback_profile: "CallbackProfile") -> None:
        if self.path is None:
            return
        now = time.time()
        if (now - self.last_emit_at) < self.interval_sec:
            return
        self.last_emit_at = now
        self._write(callback_profile, finished=False)

    def write_final(self, callback_profile: "CallbackProfile", *, finished: bool, termination_code: int | None) -> None:
        if self.path is None:
            return
        self.last_emit_at = time.time()
        self._write(callback_profile, finished=finished, termination_code=termination_code)

    def _write(
        self,
        callback_profile: "CallbackProfile",
        *,
        finished: bool,
        termination_code: int | None = None,
    ) -> None:
        if self.path is None:
            return
        payload = {
            "started_at_epoch": self.started_at,
            "updated_at_epoch": time.time(),
            "elapsed_sec": time.time() - self.started_at,
            "finished": finished,
            "termination_code": termination_code,
            "stage": self.latest_stage,
            "function_calls": callback_profile.function_calls,
            "function_time_sec": callback_profile.function_time_sec,
            "jacobian_calls": callback_profile.jacobian_calls,
            "jacobian_time_sec": callback_profile.jacobian_time_sec,
            "jacobian_function_reuse_calls": callback_profile.jacobian_function_reuse_calls,
            "total_callback_time_sec": callback_profile.total_callback_time_sec,
            "latest_function_inf_norm": self.latest_function_inf_norm,
            "latest_function_l2_norm": self.latest_function_l2_norm,
            "latest_jacobian_inf_norm": self.latest_jacobian_inf_norm,
            "latest_x_inf_norm": max(abs(v) for v in self.latest_x) if self.latest_x else None,
            "latest_x": self.latest_x,
        }
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
            tmp_path.replace(self.path)
        if self.history_path is not None:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with self.history_path.open("a", encoding="utf-8") as history_file:
                history_file.write(json.dumps(payload, sort_keys=True))
                history_file.write("\n")


def _iter_path_option_strings(raw_value: str) -> list[bytes]:
    option_lines: list[bytes] = []
    for line in raw_value.replace(";", "\n").splitlines():
        option = line.strip()
        if option:
            option_lines.append(option.encode("utf-8"))
    return option_lines


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
    callback_profile = CallbackProfile()
    progress_path_env = os.environ.get("PATH_CAPI_PROGRESS_FILE")
    progress_history_path_env = os.environ.get("PATH_CAPI_PROGRESS_HISTORY_FILE")
    progress_interval_env = os.environ.get("PATH_CAPI_PROGRESS_INTERVAL_SEC", "5")
    try:
        progress_interval_sec = max(float(progress_interval_env), 0.25)
    except ValueError:
        progress_interval_sec = 5.0
    progress = ProgressSnapshot(
        started_at=time.time(),
        last_emit_at=0.0,
        interval_sec=progress_interval_sec,
        path=Path(progress_path_env).expanduser() if progress_path_env else None,
        history_path=Path(progress_history_path_env).expanduser() if progress_history_path_env else None,
    )

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
        x_values = _read_x(size, x_ptr)
        started_at = time.perf_counter()
        values = list(callback_f(x_values))
        callback_profile.function_calls += 1
        callback_profile.function_time_sec += time.perf_counter() - started_at
        if len(values) != size:
            raise ValueError(f"callback_f returned {len(values)} values, expected {size}")
        for i, value in enumerate(values):
            f_ptr[i] = float(value)
        progress.latest_stage = "function"
        progress.latest_x = x_values
        progress.latest_function_inf_norm = max(abs(float(v)) for v in values) if values else 0.0
        progress.latest_function_l2_norm = sum(float(v) * float(v) for v in values) ** 0.5 if values else 0.0
        progress.maybe_write(callback_profile)
        return 0

    @CB_JACOBIAN
    def _jacobian(_id, size, x_ptr, wantf, f_ptr, nnz_ptr, col_ptr, len_ptr, row_ptr, data_ptr):
        started_at = time.perf_counter()
        if wantf > 0:
            callback_profile.jacobian_function_reuse_calls += 1
            _function(_id, size, x_ptr, f_ptr)
        x_values = _read_x(size, x_ptr)
        values = list(callback_jac(x_values))
        callback_profile.jacobian_calls += 1
        callback_profile.jacobian_time_sec += time.perf_counter() - started_at
        if len(values) != nnz:
            raise ValueError(f"callback_jac returned {len(values)} values, expected {nnz}")

        nnz_ptr[0] = nnz
        for j in range(size):
            col_ptr[j] = jacobian_structure.col_starts[j]
            len_ptr[j] = jacobian_structure.col_lengths[j]
        for k, row in enumerate(jacobian_structure.row_indices):
            row_ptr[k] = row
            data_ptr[k] = float(values[k])
        progress.latest_stage = "jacobian"
        progress.latest_x = x_values
        progress.latest_jacobian_inf_norm = max(abs(float(v)) for v in values) if values else 0.0
        progress.maybe_write(callback_profile)
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
    for option in _iter_path_option_strings(os.environ.get("PATH_CAPI_OPTIONS", "")):
        path.Options_Set(options, option)

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
    progress.latest_x = x_sol
    progress.write_final(callback_profile, finished=True, termination_code=term)

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
        callback_profile=callback_profile,
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
