"""Training telemetry client -- runs on the TRAINING node (laptop 2).

Design contract: recording a metric must never block, slow, or crash training.
The trainer calls `record()` and moves on; everything else (buffering, network
IO, reconnection, failure) happens on a background thread behind that call.

If the monitoring node is unreachable -- asleep, off the network, firewalled --
metrics are still written to a local JSONL file, so a run is never lost just
because nobody was watching it.
"""
from __future__ import annotations

import json
import queue
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TelemetryStats:
    recorded: int = 0
    sent: int = 0
    dropped: int = 0
    reconnects: int = 0
    last_error: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "recorded": self.recorded,
            "sent": self.sent,
            "dropped": self.dropped,
            "reconnects": self.reconnects,
            "last_error": self.last_error,
        }


class TrainingTelemetry:
    """Non-blocking metric shipper with a durable local mirror.

    Usage on the training node::

        telemetry = TrainingTelemetry(
            run_id="r1-5m-realtext",
            collector_host="192.168.1.69",
            local_mirror="experiments/phase_m/results/local_metrics.jsonl",
        )
        telemetry.start()
        ...
        telemetry.record(step=step, loss=loss, lr=lr)   # never blocks
        ...
        telemetry.close()
    """

    def __init__(
        self,
        *,
        run_id: str,
        collector_host: str,
        collector_port: int = 51799,
        local_mirror: Path | str | None = None,
        queue_size: int = 4096,
        connect_timeout: float = 2.0,
        node_name: str | None = None,
    ) -> None:
        self.run_id = run_id
        self.collector_host = collector_host
        self.collector_port = collector_port
        self.node_name = node_name or socket.gethostname()
        self.local_mirror = Path(local_mirror) if local_mirror else None
        self.connect_timeout = connect_timeout

        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._sock: socket.socket | None = None
        self._mirror_handle = None
        self.stats = TelemetryStats()
        self._lock = threading.Lock()

    # -- public interface --------------------------------------------------
    def start(self) -> "TrainingTelemetry":
        if self.local_mirror is not None:
            self.local_mirror.parent.mkdir(parents=True, exist_ok=True)
            self._mirror_handle = open(self.local_mirror, "a", encoding="utf-8")
        self._thread = threading.Thread(target=self._pump, name="telemetry", daemon=True)
        self._thread.start()
        return self

    def record(self, **metrics: Any) -> None:
        """Queue one metric record. Never blocks; drops if the queue is full."""
        record = {
            "run_id": self.run_id,
            "node": self.node_name,
            "wall_clock": time.time(),
            **metrics,
        }
        with self._lock:
            self.stats.recorded += 1
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            with self._lock:
                self.stats.dropped += 1

    def close(self, *, drain_timeout: float = 5.0) -> dict[str, Any]:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=drain_timeout)
        self._teardown_socket()
        if self._mirror_handle is not None:
            self._mirror_handle.flush()
            self._mirror_handle.close()
            self._mirror_handle = None
        return self.stats.snapshot()

    def __enter__(self) -> "TrainingTelemetry":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- internals ---------------------------------------------------------
    def _teardown_socket(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _ensure_socket(self) -> bool:
        if self._sock is not None:
            return True
        try:
            sock = socket.create_connection(
                (self.collector_host, self.collector_port), timeout=self.connect_timeout
            )
            sock.settimeout(self.connect_timeout)
            self._sock = sock
            with self._lock:
                self.stats.reconnects += 1
            return True
        except OSError as exc:
            with self._lock:
                self.stats.last_error = f"{type(exc).__name__}: {exc}"
            return False

    def _pump(self) -> None:
        backoff = 0.5
        next_attempt = 0.0

        def _process(record: dict[str, Any]) -> None:
            nonlocal backoff, next_attempt
            line = json.dumps(record, separators=(",", ":")) + "\n"

            # Durable local mirror first, and unconditionally: the run's own
            # record must survive regardless of whether the monitoring node is
            # listening. This must never be gated behind network retry pacing.
            if self._mirror_handle is not None:
                self._mirror_handle.write(line)
                self._mirror_handle.flush()

            # Network send is rate-limited by a timestamp gate rather than by
            # sleeping here. Sleeping in this path would stall queue draining
            # and cause records to pile up (and be lost) whenever the
            # collector is down -- the exact opposite of the intent.
            now = time.monotonic()
            if now < next_attempt:
                return
            if self._ensure_socket():
                try:
                    self._sock.sendall(line.encode("utf-8"))  # type: ignore[union-attr]
                    with self._lock:
                        self.stats.sent += 1
                    backoff = 0.5
                    return
                except OSError as exc:
                    with self._lock:
                        self.stats.last_error = f"{type(exc).__name__}: {exc}"
                    self._teardown_socket()
            backoff = min(backoff * 2, 5.0)
            next_attempt = now + backoff

        stopping = False
        while not stopping:
            try:
                record = self._queue.get(timeout=0.25)
            except queue.Empty:
                if self._stop.is_set():
                    break
                continue
            if record is None:
                stopping = True
                break
            _process(record)

        # Drain whatever is still queued so a clean shutdown does not discard
        # records the trainer already handed over.
        while True:
            try:
                record = self._queue.get_nowait()
            except queue.Empty:
                break
            if record is None:
                continue
            _process(record)
