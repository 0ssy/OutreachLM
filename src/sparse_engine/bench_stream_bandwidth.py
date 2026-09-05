"""Measure the real streaming bandwidth this machine can sustain.

For a 70B-capacity sparsely-activated model the weights cannot live in RAM
(17.5 GB ternary against 15.4 GB total), so every optimizer step streams them
from disk. Throughput is therefore set by I/O, not by arithmetic, and the
whole feasibility argument rests on this number.

Measured, not assumed: cold reads with the OS cache bypassed where possible,
at the block sizes an expert-parallel loop would actually use.
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import torch


def write_blob(path: Path, size_mb: int) -> None:
    chunk = os.urandom(1 << 20)
    with open(path, "wb") as f:
        for _ in range(size_mb):
            f.write(chunk)
        f.flush()
        os.fsync(f.fileno())


def read_bandwidth(path: Path, block_kb: int) -> float:
    """GB/s for a sequential read at the given block size."""
    size = path.stat().st_size
    buf = bytearray(block_kb * 1024)
    t0 = time.perf_counter()
    with open(path, "rb", buffering=0) as f:
        while f.readinto(buf):
            pass
    el = time.perf_counter() - t0
    return size / el / 1e9


def main() -> None:
    tmp = Path(tempfile.gettempdir()) / "outreachlm_bw_probe.bin"
    size_mb = 2048
    print(f"Writing a {size_mb} MB probe file to {tmp.parent} ...")
    t0 = time.perf_counter()
    write_blob(tmp, size_mb)
    wel = time.perf_counter() - t0
    print(f"  write bandwidth {size_mb / 1024 / wel:.2f} GB/s\n")

    print(f"{'block KB':>10}{'read GB/s':>12}{'note':>28}")
    print("-" * 50)
    for block_kb in (64, 256, 1024, 4096):
        bw = read_bandwidth(tmp, block_kb)
        note = "(warm: OS page cache)" if bw > 6 else ""
        print(f"{block_kb:>10}{bw:>12.2f}{note:>28}")

    print("\nRAM bandwidth for comparison (what a cached tile achieves):")
    a = torch.randn(64 * 1024 * 1024 // 4)
    t0 = time.perf_counter()
    for _ in range(10):
        a.sum()
    el = (time.perf_counter() - t0) / 10
    print(f"  sequential RAM read   {a.numel() * 4 / el / 1e9:.1f} GB/s")

    try:
        tmp.unlink()
    except OSError:
        pass


if __name__ == "__main__":
    main()
