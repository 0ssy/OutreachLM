from __future__ import annotations

from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import threading
from typing import Iterator

import psutil


@contextmanager
def pin_process_to_first_cores(core_count: int) -> Iterator[bool]:
    if core_count <= 0:
        raise ValueError("core_count must be > 0")
    process = psutil.Process()
    if not hasattr(process, "cpu_affinity"):
        yield False
        return

    original = process.cpu_affinity()
    target = original[:core_count]
    if len(target) < core_count:
        target = list(range(core_count))
    pinned = False
    try:
        process.cpu_affinity(target)
        pinned = True
        yield pinned
    finally:
        process.cpu_affinity(original)


THREAD_SET_INFORMATION = 0x0020
THREAD_QUERY_INFORMATION = 0x0040
THREAD_QUERY_LIMITED_INFORMATION = 0x0800


_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_kernel32.OpenThread.restype = wintypes.HANDLE
_kernel32.SetThreadAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
_kernel32.SetThreadAffinityMask.restype = ctypes.c_size_t
_kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
_kernel32.CloseHandle.restype = wintypes.BOOL


def _set_thread_affinity(thread_id: int, mask: int) -> bool:
    access = THREAD_SET_INFORMATION | THREAD_QUERY_INFORMATION | THREAD_QUERY_LIMITED_INFORMATION
    handle = _kernel32.OpenThread(access, False, thread_id)
    if not handle:
        return False
    try:
        previous_mask = _kernel32.SetThreadAffinityMask(handle, ctypes.c_size_t(mask))
        return bool(previous_mask)
    finally:
        _kernel32.CloseHandle(handle)


def pin_current_thread_to_core(core_index: int) -> bool:
    if core_index < 0:
        raise ValueError("core_index must be >= 0")
    thread_id = threading.get_native_id()
    mask = 1 << core_index
    try:
        return _set_thread_affinity(thread_id, mask)
    except Exception:
        return False


@contextmanager
def pin_current_thread(core_index: int) -> Iterator[bool]:
    pinned = pin_current_thread_to_core(core_index)
    try:
        yield pinned
    finally:
        # Resetting to all cores reliably from Python without the prior mask is platform-fragile.
        # Process-level affinity context in callers controls outer restoration.
        return
