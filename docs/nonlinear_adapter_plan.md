# Plan: Nonlinear MCP Adapter for PATH-CAPI

## Objective
Implement a nonlinear MCP adapter that can solve a full, nonlinear complementarity system using PATH-CAPI callbacks, not just linear M, q form.

## Scope
- Add a new solver entry point that accepts nonlinear residual and Jacobian callbacks.
- Provide a minimal test MCP and validation harness.
- Keep the existing linear MCP path intact.

## Phases and Tasks

### Phase 1: Repository Audit and Design (1-2 days)
1) Locate current linear MCP entry points
   - Identify the module that exports solve_linear_mcp and any existing adapter utilities.
2) Confirm PATH-CAPI capabilities for nonlinear MCP
   - Review PATH-CAPI API for callback signatures (residual and Jacobian).
   - Capture any required sparsity structure formats and memory ownership rules.
3) Define the new API surface
   - Proposed function name: solve_nonlinear_mcp
   - Inputs: n, lb, ub, x0, callback_f, callback_jac, structure or pattern
   - Outputs: solution vector, residual, termination code, iterations

Deliverable: short design note (this document) updated with API signature and callback spec.

#### Implemented API signature

```python
solve_nonlinear_mcp(
    runtime,
    *,
    n: int,
    lb: Sequence[float],
    ub: Sequence[float],
    x0: Sequence[float],
    callback_f: Callable[[Sequence[float]], Sequence[float]],
    callback_jac: Callable[[Sequence[float]], Sequence[float]],
    jacobian_structure: JacobianStructure,
    output: bool = True,
) -> NonlinearMCPResult
```

#### Callback contract

- `callback_f(x)` returns the residual vector `F(x)` with length `n`.
- `callback_jac(x)` returns the Jacobian nonzeros in the fixed order defined by `jacobian_structure`.
- `jacobian_structure` uses PATH's callback layout:
  - `col_starts[j]`: 1-based offset of column `j`
  - `col_lengths[j]`: nonzero count in column `j`
  - `row_indices[k]`: 1-based row index for nonzero `k`
- The Jacobian sparsity is treated as constant for the solve call.
- `solve_linear_mcp` remains available and now reuses this nonlinear callback path internally.

### Phase 2: Core Nonlinear Adapter Implementation (2-4 days)
4) Create a new module or extend existing adapter
   - Implement a small CFFI/ctypes wrapper if needed for callbacks.
   - Map Python callbacks to PATH-CAPI function pointers.
5) Add support for Jacobian sparsity
   - Accept a fixed sparsity pattern or allow recomputation each call.
   - Implement a small helper to convert Python sparse data to PATH format.
6) Implement error handling and status mapping
   - Normalize PATH return codes to a unified result object.
   - Ensure error messages include PATH code and context.

Deliverable: solve_nonlinear_mcp available in the Python API with unit-tested callback wiring.

### Phase 3: Validation and Minimal Test MCP (1-2 days)
7) Build a minimal nonlinear MCP example
   - Example: 2x2 complementarity with known solution.
8) Add automated tests
   - Validate solution accuracy and residuals.
   - Verify that the Jacobian is used (finite difference fallback only for debugging).
9) Benchmark behavior against linear mode
   - Ensure no regression to solve_linear_mcp.

Deliverable: passing test suite for nonlinear mode.

### Phase 4: Integration Hooks (Optional, 2-3 days)
10) Add a thin adapter interface for Pyomo or generic modeling tools
    - Allow passing a model object that can evaluate residuals and Jacobian.
11) Provide a helper to build MCP from a sparse Jacobian structure
    - Useful for large CGE models.

Deliverable: optional helper layer; not required for core functionality.

### Phase 5: Documentation and Release (1 day)
12) Write docs for solve_nonlinear_mcp
    - Usage examples with callbacks and sparsity.
13) Update README or docs index

Deliverable: docs page and release note.

## Risks and Mitigations
- Callback performance overhead: use preallocated buffers and avoid Python allocations in hot loops.
- Jacobian correctness: require explicit Jacobian; allow finite difference only for debugging.
- PATH-CAPI ABI mismatch: pin PATH version and include a runtime validation check.

## Acceptance Criteria
- solve_nonlinear_mcp solves a known nonlinear MCP example with correct termination and residuals.
- No regressions to solve_linear_mcp.
- Clear documentation with example usage and expected inputs.

## Status
- Core nonlinear callback-based solver entry point implemented.
- Fixed sparsity helper (`JacobianStructure`) implemented.
- Unit test harness added for nonlinear callback wiring without requiring a live PATH library.
- Existing linear solver path preserved through the shared callback infrastructure.

## Suggested File Locations
- docs/nonlinear_adapter_plan.md (this document)
- src/path_capi_python/nonlinear_mcp.py (proposed new module)
- tests/test_nonlinear_mcp.py
