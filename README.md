# path-capi-python

Python wrapper for the PATH C API with Apple Silicon support for solving MCP and LCP models, with optional adapters for Pyomo workflows.

## Overview

`path-capi-python` provides a thin, explicit bridge from Python to the PATH C API.
The project is designed to:

- Load PATH shared libraries on modern platforms (including Apple Silicon).
- Expose core C API operations in a Pythonic interface.
- Support MCP/LCP solve workflows with deterministic diagnostics.
- Offer an optional adapter layer for Pyomo-centered workflows.

## Scope

This repository wraps the PATH C API. It does not reimplement the PATH solver.

## PATH Licensing

This project is MIT-licensed for the wrapper code only.
The underlying PATH solver is distributed under its own terms.
Users are responsible for obtaining and complying with PATH licensing requirements.

## Initial Roadmap

- C API loader and runtime checks.
- Minimal MCP solve example with callbacks.
- Structured status and residual reporting.
- Optional Pyomo adapter prototype.

## Current Status

- PATH shared library loader implemented (`Path_Version`, `Path_CheckLicense`).
- Minimal linear MCP solve implemented through C callbacks.
- Example script available at `examples/minimal_mcp.py`.
- Pyomo adapter scaffold started (`PyomoMCPAdapter`) for model introspection.

## Running Tests

Set library paths to your local PATH shared libraries before running tests:

```bash
PATH_CAPI_LIBPATH=/absolute/path/to/libpath50.silicon.dylib \
PATH_CAPI_LIBLUSOL=/absolute/path/to/liblusol.silicon.dylib \
PYTHONPATH=src python3 -m pytest -q
```

## Development Notes

- Target platform priority: macOS arm64.
- Keep the low-level C API layer explicit and well-tested.
- Build higher-level adapters only after the C layer is stable.
