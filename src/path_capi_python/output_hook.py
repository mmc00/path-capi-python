"""Output capture hook for PATH C-API.

PATH writes diagnostic messages through an ``Output_Interface`` callback
mechanism rather than the C runtime's stdout. When the wrapper does not
install an Output_Interface, PATH's default sink absorbs the messages —
the user sees nothing about factorization method, license check, basis
state, or load failures.

This module provides:

* ``OutputInterface`` — the ctypes struct PATH expects
* ``install_output_hook`` — installs a Python callback that streams
  PATH's messages to stderr (or a user-provided sink)
* mode flags PATH uses to tag messages (Log / Status / Listing)

The callbacks must be kept alive for the lifetime of the PATH solve;
``install_output_hook`` returns a handle the caller stores in scope.
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


# PATH Output mode flags (from PATH 5.x convention)
OUTPUT_LOG = 1
OUTPUT_STATUS = 2
OUTPUT_LISTING = 4

_MODE_NAMES = {
    OUTPUT_LOG: "log",
    OUTPUT_STATUS: "status",
    OUTPUT_LISTING: "listing",
}


# void print(void *data, int mode, const char *buf)
CB_OUTPUT_PRINT = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p
)
# void flush(void *data, int mode)
CB_OUTPUT_FLUSH = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int)


class OutputInterface(ctypes.Structure):
    """Mirror of PATH's ``Output_Interface`` struct."""

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("print", CB_OUTPUT_PRINT),
        ("flush", CB_OUTPUT_FLUSH),
    ]


@dataclass
class OutputHookHandle:
    """Anchor object that keeps callback references alive.

    PATH stores raw C function pointers into the OutputInterface struct.
    If the Python callbacks are garbage-collected while PATH still
    holds those pointers, the next message crashes the interpreter.
    Hold onto this handle for the entire scope of the PATH solve.
    """

    interface: OutputInterface
    _print_cb: ctypes._CFuncPtr
    _flush_cb: ctypes._CFuncPtr
    captured: List[Tuple[int, str]] = field(default_factory=list)


def _format_mode_prefix(mode: int) -> str:
    name = _MODE_NAMES.get(mode, f"mode={mode}")
    return f"[PATH {name}]"


def install_output_hook(
    runtime,
    *,
    sink: Optional[Callable[[int, str], None]] = None,
    echo_to_stderr: bool = True,
    capture: bool = True,
) -> OutputHookHandle:
    """Install an output callback on PATH's runtime.

    Parameters
    ----------
    runtime : PATHRuntime
        Loaded PATH runtime (from PATHLoader.load()).
    sink : callable, optional
        If provided, called as ``sink(mode, text)`` for every message.
    echo_to_stderr : bool, default True
        If True, messages also print to sys.stderr with a mode prefix.
    capture : bool, default True
        If True, messages are appended to ``handle.captured`` for
        post-mortem inspection.

    Returns
    -------
    OutputHookHandle
        Anchor that must remain in scope until after Path_Solve returns.
    """

    path = runtime.path

    # The Output_* functions may not be in older path52 builds. Probe.
    if not hasattr(path, "Output_SetInterface"):
        raise RuntimeError(
            "PATH library does not export Output_SetInterface; cannot install hook"
        )

    path.Output_SetInterface.argtypes = [ctypes.POINTER(OutputInterface)]
    path.Output_SetInterface.restype = None

    # The struct lives at module/handle scope (handle keeps it alive).
    interface = OutputInterface()
    interface.data = None

    captured: List[Tuple[int, str]] = []

    @CB_OUTPUT_PRINT
    def _print(_data: int, mode: int, buf: bytes) -> None:
        # buf is a const char* — may be NULL on flush-like calls
        if not buf:
            return
        try:
            text = buf.decode("utf-8", errors="replace")
        except Exception:
            text = repr(buf)
        if capture:
            captured.append((mode, text))
        if echo_to_stderr:
            try:
                sys.stderr.write(f"{_format_mode_prefix(mode)} {text}")
                if not text.endswith("\n"):
                    sys.stderr.write("\n")
                sys.stderr.flush()
            except Exception:
                pass
        if sink is not None:
            try:
                sink(mode, text)
            except Exception as exc:
                # Never let a sink raise into PATH's callback
                sys.stderr.write(f"[PATH output sink error: {exc!r}]\n")

    @CB_OUTPUT_FLUSH
    def _flush(_data: int, mode: int) -> None:
        if echo_to_stderr:
            try:
                sys.stderr.flush()
            except Exception:
                pass

    interface.print = _print
    interface.flush = _flush

    path.Output_SetInterface(ctypes.byref(interface))

    return OutputHookHandle(
        interface=interface,
        _print_cb=_print,
        _flush_cb=_flush,
        captured=captured,
    )


def uninstall_output_hook(runtime) -> None:
    """Reset PATH's output to its compiled-in default.

    Optional — primarily useful in long-running processes that want to
    drop callback references after a solve.
    """
    path = runtime.path
    if hasattr(path, "Output_Default"):
        path.Output_Default.argtypes = []
        path.Output_Default.restype = None
        path.Output_Default()
