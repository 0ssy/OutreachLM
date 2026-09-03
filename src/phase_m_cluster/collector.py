"""Metrics collector -- runs on the MONITORING node (laptop 1, this machine).

Accepts newline-delimited JSON records from one or more training nodes over
TCP, appends them to a durable JSONL log, and maintains live per-run summary
state that a dashboard can read without re-parsing the whole file.
"""
from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunSummary:
    run_id: str
    node: str
    first_seen: float
    last_seen: float
    record_count: int = 0
    latest: dict[str, Any] = field(default_factory=dict)
    first_loss: float | None = None
    latest_loss: float | None = None
    min_loss: float | None = None
    latest_step: int | None = None

    def observe(self, record: dict[str, Any]) -> None:
        self.last_seen = record.get("wall_clock", time.time())
        self.record_count += 1
        self.latest = record

        step = record.get("step")
        if isinstance(step, (int, float)):
            self.latest_step = int(step)

        loss = record.get("loss")
        if isinstance(loss, (int, float)):
            loss = float(loss)
            if self.first_loss is None:
                self.first_loss = loss
            self.latest_loss = loss
            self.min_loss = loss if self.min_loss is None else min(self.min_loss, loss)

    def to_dict(self) -> dict[str, Any]:
        elapsed = max(self.last_seen - self.first_seen, 1e-9)
        return {
            "run_id": self.run_id,
            "node": self.node,
            "record_count": self.record_count,
            "latest_step": self.latest_step,
            "first_loss": self.first_loss,
            "latest_loss": self.latest_loss,
            "min_loss": self.min_loss,
            "loss_delta": (
                None
                if self.first_loss is None or self.latest_loss is None
                else self.first_loss - self.latest_loss
            ),
            "records_per_second": self.record_count / elapsed,
            "elapsed_seconds": elapsed,
            "seconds_since_last_record": max(0.0, time.time() - self.last_seen),
            "latest": self.latest,
        }


class MetricsCollector:
    """TCP listener that durably records training telemetry.

    The collector is intentionally passive: it never sends anything back to a
    training node and never applies backpressure, so a slow or stopped
    collector cannot influence training on the other machine.
    """

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 51799,
        log_path: Path | str = "experiments/phase_m/results/collected_metrics.jsonl",
        summary_path: Path | str = "experiments/phase_m/results/live_summary.json",
    ) -> None:
        self.host = host
        self.port = port
        self.log_path = Path(log_path)
        self.summary_path = Path(summary_path)
        self.runs: dict[str, RunSummary] = {}
        self._server: socket.socket | None = None
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._log_handle = None
        self.bound_port: int | None = None

    # -- public interface --------------------------------------------------
    def start(self) -> "MetricsCollector":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = open(self.log_path, "a", encoding="utf-8")

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(8)
        server.settimeout(0.5)
        self._server = server
        self.bound_port = server.getsockname()[1]

        thread = threading.Thread(target=self._accept_loop, name="collector", daemon=True)
        thread.start()
        self._threads.append(thread)
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
            self._server = None
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._threads.clear()
        if self._log_handle is not None:
            self._log_handle.flush()
            self._log_handle.close()
            self._log_handle = None

    def __enter__(self) -> "MetricsCollector":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "collector_port": self.bound_port,
                "active_runs": len(self.runs),
                "runs": [run.to_dict() for run in self.runs.values()],
            }

    def write_summary(self) -> Path:
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(
            json.dumps(self.summary(), indent=2), encoding="utf-8"
        )
        return self.summary_path

    # -- internals ---------------------------------------------------------
    def _accept_loop(self) -> None:
        while not self._stop.is_set() and self._server is not None:
            try:
                conn, _addr = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            thread = threading.Thread(
                target=self._client_loop, args=(conn,), daemon=True
            )
            thread.start()
            self._threads.append(thread)

    def _client_loop(self, conn: socket.socket) -> None:
        conn.settimeout(1.0)
        buffer = b""
        try:
            while not self._stop.is_set():
                try:
                    chunk = conn.recv(8192)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    raw, buffer = buffer.split(b"\n", 1)
                    if raw.strip():
                        self._ingest(raw)
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _ingest(self, raw: bytes) -> None:
        try:
            record = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return  # ignore malformed frames rather than killing the connection
        if not isinstance(record, dict):
            return

        run_id = str(record.get("run_id", "unknown"))
        node = str(record.get("node", "unknown"))

        with self._lock:
            if self._log_handle is not None:
                self._log_handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                self._log_handle.flush()
            summary = self.runs.get(run_id)
            if summary is None:
                now = record.get("wall_clock", time.time())
                summary = RunSummary(
                    run_id=run_id, node=node, first_seen=now, last_seen=now
                )
                self.runs[run_id] = summary
            summary.observe(record)
