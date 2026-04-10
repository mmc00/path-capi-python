from __future__ import annotations

import ctypes


class FakeCFunc:
    def __init__(self, impl):
        self.impl = impl
        self.argtypes = None
        self.restype = None

    def __call__(self, *args, **kwargs):
        return self.impl(*args, **kwargs)


class FakeMCP:
    def __init__(self, n: int, nnz_capacity: int):
        self.n = n
        self.nnz_capacity = nnz_capacity
        self.interface = None
        self.x_buffer = (ctypes.c_double * n)()


class FakePath:
    def __init__(self):
        self.Options_Create = FakeCFunc(self._options_create)
        self.Options_Destroy = FakeCFunc(self._options_destroy)
        self.Path_AddOptions = FakeCFunc(self._path_add_options)
        self.Options_Default = FakeCFunc(self._options_default)
        self.Options_Set = FakeCFunc(self._options_set)
        self.MCP_Create = FakeCFunc(self._mcp_create)
        self.MCP_Destroy = FakeCFunc(self._mcp_destroy)
        self.MCP_SetInterface = FakeCFunc(self._mcp_set_interface)
        self.MCP_Jacobian_Structure_Constant = FakeCFunc(self._noop)
        self.MCP_Jacobian_Data_Contiguous = FakeCFunc(self._noop)
        self.Path_Solve = FakeCFunc(self._path_solve)
        self.MCP_GetX = FakeCFunc(self._mcp_get_x)

        self.last_output_option: bytes | None = None
        self.last_options: list[bytes] = []
        self.last_function_values: list[float] | None = None
        self.last_jacobian_values: list[float] | None = None
        self.last_row_indices: list[int] | None = None
        self.last_col_starts: list[int] | None = None
        self.last_col_lengths: list[int] | None = None

    def _options_create(self):
        return object()

    def _options_destroy(self, _options):
        return None

    def _path_add_options(self, _options):
        return None

    def _options_default(self, _options):
        return None

    def _options_set(self, _options, setting):
        self.last_output_option = setting
        self.last_options.append(setting)
        return None

    def _mcp_create(self, n: int, nnz_capacity: int):
        return FakeMCP(n, nnz_capacity)

    def _mcp_destroy(self, _mcp):
        return None

    def _mcp_set_interface(self, mcp, interface_ptr):
        mcp.interface = interface_ptr._obj
        return None

    def _noop(self, *_args):
        return None

    def _path_solve(self, mcp, info_ptr):
        n_ptr = ctypes.pointer(ctypes.c_int())
        nnz_ptr = ctypes.pointer(ctypes.c_int())
        mcp.interface.problem_size(None, n_ptr, nnz_ptr)

        x0 = (ctypes.c_double * mcp.n)()
        lb = (ctypes.c_double * mcp.n)()
        ub = (ctypes.c_double * mcp.n)()
        mcp.interface.bounds(None, mcp.n, x0, lb, ub)

        trial_x = (ctypes.c_double * mcp.n)(1.5, 0.5)
        f = (ctypes.c_double * mcp.n)()
        col = (ctypes.c_int * mcp.n)()
        lengths = (ctypes.c_int * mcp.n)()
        rows = (ctypes.c_int * max(nnz_ptr[0], 1))()
        data = (ctypes.c_double * max(nnz_ptr[0], 1))()

        mcp.interface.jacobian_evaluation(
            None,
            mcp.n,
            trial_x,
            1,
            f,
            nnz_ptr,
            col,
            lengths,
            rows,
            data,
        )

        self.last_function_values = [float(f[i]) for i in range(mcp.n)]
        self.last_jacobian_values = [float(data[i]) for i in range(nnz_ptr[0])]
        self.last_row_indices = [int(rows[i]) for i in range(nnz_ptr[0])]
        self.last_col_starts = [int(col[i]) for i in range(mcp.n)]
        self.last_col_lengths = [int(lengths[i]) for i in range(mcp.n)]

        solution = [1.0, 2.0]
        for i, value in enumerate(solution):
            mcp.x_buffer[i] = value

        info = info_ptr._obj
        info.residual = 1.25e-9
        info.major_iterations = 3
        info.minor_iterations = 4
        info.function_evaluations = 5
        info.jacobian_evaluations = 2
        return 1

    def _mcp_get_x(self, mcp):
        return mcp.x_buffer
