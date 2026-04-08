"""PATH C API wrapper package."""

from .loader import PATHLoader, PATHLibraryError, PATHRuntime
from .mcp import JacobianStructure, LinearMCPResult, NonlinearMCPResult, solve_linear_mcp, solve_nonlinear_mcp
from .pyomo_adapter import LinearCallbackData, NonlinearCallbackData, PyomoMCPAdapter, PyomoModelSummary

try:  # pragma: no cover - optional Pyomo registration
    from .pyomo_solver import PATHCAPIBridgeSolver
except Exception:  # pragma: no cover
    PATHCAPIBridgeSolver = None

__all__ = [
    "PATHLoader",
    "PATHLibraryError",
    "PATHRuntime",
    "JacobianStructure",
    "LinearMCPResult",
    "NonlinearMCPResult",
    "solve_linear_mcp",
    "solve_nonlinear_mcp",
    "PyomoMCPAdapter",
    "PyomoModelSummary",
    "LinearCallbackData",
    "NonlinearCallbackData",
    "PATHCAPIBridgeSolver",
]
