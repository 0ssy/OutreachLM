from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class FetchOutcome:
    payload: bytes
    source: str            # "socket" | "fallback"
    recovery_latency_ms: float
    error: str | None


class NonBlockingSocketMonitor:
    """Non-blocking stream reader with a local-file fallback.

    The failure path here is exercised with a real TCP socket against a real
    endpoint: when the upstream connection refuses, resets, or exceeds its
    timeout, the monitor measures the wall-clock time to detect the fault and
    serve the next block from the local fallback buffer instead. That elapsed
    time is the recovery latency -- it is measured, never assumed.
    """

    def __init__(
        self,
        fallback_path: Path | str,
        *,
        timeout_seconds: float = 0.015,
        chunk_bytes: int = 4096,
        preload_fallback: bool = True,
    ) -> None:
        self.fallback_path = Path(fallback_path)
        self.timeout_seconds = timeout_seconds
        self.chunk_bytes = chunk_bytes
        self._fallback_offset = 0
        # Preload the fallback corpus into RAM. Opening the file on every
        # stalled block would add filesystem latency to the recovery path --
        # precisely the idle-starvation this monitor exists to prevent.
        self._fallback_buffer = (
            self.fallback_path.read_bytes()
            if preload_fallback and self.fallback_path.exists()
            else None
        )

    def _read_fallback(self) -> bytes:
        if self._fallback_buffer is not None:
            buffer = self._fallback_buffer
            if not buffer:
                return b""
            start = self._fallback_offset % len(buffer)
            data = buffer[start : start + self.chunk_bytes]
            if not data:
                start = 0
                data = buffer[: self.chunk_bytes]
            self._fallback_offset = start + len(data)
            return data

        if not self.fallback_path.exists():
            return b""
        with open(self.fallback_path, "rb") as handle:
            handle.seek(self._fallback_offset)
            data = handle.read(self.chunk_bytes)
        if not data:
            self._fallback_offset = 0
            with open(self.fallback_path, "rb") as handle:
                data = handle.read(self.chunk_bytes)
        self._fallback_offset += len(data)
        return data

    def fetch(self, host: str, port: int) -> FetchOutcome:
        """Attempt a real socket read; fall back to local buffer on failure."""
        started = time.perf_counter()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout_seconds)
        try:
            sock.connect((host, port))
            data = sock.recv(self.chunk_bytes)
            if not data:
                raise ConnectionError("upstream closed the connection with no data")
            return FetchOutcome(
                payload=data,
                source="socket",
                recovery_latency_ms=0.0,
                error=None,
            )
        except (OSError, ConnectionError) as exc:
            payload = self._read_fallback()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return FetchOutcome(
                payload=payload,
                source="fallback",
                recovery_latency_ms=elapsed_ms,
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            sock.close()

    def stream_blocks(
        self,
        host: str,
        port: int,
        *,
        blocks: int,
        on_block: Callable[[FetchOutcome], None] | None = None,
    ) -> list[FetchOutcome]:
        outcomes: list[FetchOutcome] = []
        for _ in range(blocks):
            outcome = self.fetch(host, port)
            outcomes.append(outcome)
            if on_block is not None:
                on_block(outcome)
        return outcomes
