from __future__ import annotations

import os
from pathlib import Path

from path_capi_python import PATHLoader
from path_capi_python.mcp import solve_linear_mcp


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    path_lib = Path(_required_env("PATH_CAPI_LIBPATH"))
    lusol_lib = Path(_required_env("PATH_CAPI_LIBLUSOL"))

    loader = PATHLoader(path_lib=path_lib, lusol_lib=lusol_lib)
    runtime = loader.load()

    print("PATH version:", loader.version(runtime))
    print("License check (10,10):", loader.check_license(runtime, 10, 10))

    # Same small LCP used in PATHSolver.jl docs.
    M = [
        [0.0, 0.0, -1.0, -1.0],
        [0.0, 0.0, 1.0, -2.0],
        [1.0, -1.0, 2.0, -2.0],
        [1.0, 2.0, -2.0, 4.0],
    ]
    q = [2.0, 2.0, -2.0, -6.0]
    lb = [0.0, 0.0, 0.0, 0.0]
    ub = [1.0e20, 1.0e20, 1.0e20, 1.0e20]
    x0 = [0.0, 0.0, 0.0, 0.0]

    result = solve_linear_mcp(runtime, M, q, lb, ub, x0, output=False)

    print("Termination code:", result.termination_code)
    print("Residual:", f"{result.residual:.3e}")
    print("Iterations (major, minor):", result.major_iterations, result.minor_iterations)
    print("x:", [round(v, 8) for v in result.x])


if __name__ == "__main__":
    main()
