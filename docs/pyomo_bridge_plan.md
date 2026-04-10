# Plan: PATH C API Bridge + Pyomo Connector

## Goal
Provide a reliable PATH C API runtime on Apple Silicon and a Python bridge for MCP solves, plus a Pyomo-facing adapter that can later become a Pyomo solver plugin.

## Plan A: Python Bridge Using libpath (C API)

### A1. Acquire PATH shared libraries (arm64)
- Source: PATHSolver.jl artifacts for macOS aarch64.
- Expected files: `libpath.dylib` and `liblusol.dylib`.
- Store in a local folder (example): `notes/tmp/path_capi_artifacts/`.

### A2. Configure runtime loader
- Set environment variables:
  - `PATH_CAPI_LIBPATH=/absolute/path/to/libpath.dylib`
  - `PATH_CAPI_LIBLUSOL=/absolute/path/to/liblusol.dylib`
- Use `PATHLoader` to load and validate:
  - `PATHLoader.load()`
  - `PATHLoader.version()`

### A3. Validate C API runtime
- Run a minimal MCP (linear or nonlinear) with `solve_nonlinear_mcp`.
- Confirm:
  - license check passes (or returns limited size)
  - PATH returns a termination code

### A4. Add a small runtime check script
- New script (optional): `examples/check_runtime.py` to print PATH version and license status.

## Plan B: Pyomo Connector (Bridge -> Pyomo)

### B1. Non-native adapter (fast path)
- Use `PyomoMCPAdapter` to build callbacks from Pyomo expressions/constraints.
- Solve with `solve_nonlinear_mcp` and write results back to model variables.
- This is a separate workflow (not `SolverFactory`).

### B2. Pyomo solver plugin (native path)
- Create a new Pyomo solver plugin (e.g., `path_capi_bridge`).
- Required steps:
  1) Extract ordered variables and equality constraints.
  2) Build `F(x)` and sparse Jacobian via `PyomoMCPAdapter`.
  3) Call PATH C API (`solve_nonlinear_mcp`).
  4) Map solution back to Pyomo variables and return a Pyomo results object.

### B3. Tests and diagnostics
- Add tests for:
  - callback construction correctness
  - Jacobian structure consistency
  - basic solve pipeline on a small MCP

## Notes
- Pyomo native PATH integration uses `pathampl` (ASL). This bridge is separate.
- The bridge is viable for Apple Silicon because we can use arm64 `libpath`.
- License handling must be documented (env var or explicit API call).
