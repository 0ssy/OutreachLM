from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase_m_cluster.collector import MetricsCollector
from src.phase_m_cluster.telemetry import TrainingTelemetry


def _wait_for(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_metrics_travel_from_training_node_to_collector(tmp_path: Path) -> None:
    collector = MetricsCollector(
        host="127.0.0.1",
        port=0,
        log_path=tmp_path / "collected.jsonl",
        summary_path=tmp_path / "summary.json",
    ).start()
    try:
        telemetry = TrainingTelemetry(
            run_id="r1-test",
            collector_host="127.0.0.1",
            collector_port=collector.bound_port,
            local_mirror=tmp_path / "mirror.jsonl",
        ).start()
        for step in range(5):
            telemetry.record(step=step, loss=10.0 - step, lr=0.001)
        stats = telemetry.close()

        assert stats["recorded"] == 5
        assert _wait_for(lambda: collector.summary()["active_runs"] == 1)

        run = collector.summary()["runs"][0]
        assert run["run_id"] == "r1-test"
        assert run["record_count"] == 5
        assert run["latest_step"] == 4
        assert run["min_loss"] == 6.0
        assert run["first_loss"] == 10.0
    finally:
        collector.stop()


def test_training_continues_when_collector_is_unreachable(tmp_path: Path) -> None:
    """The core safety property: no collector, no problem.

    `record()` must not raise, block, or lose the run's own history when the
    monitoring node is absent -- otherwise monitoring could take down a
    multi-day training run.
    """
    mirror = tmp_path / "mirror.jsonl"
    telemetry = TrainingTelemetry(
        run_id="r1-offline",
        collector_host="127.0.0.1",
        collector_port=9,  # discard port: nothing is listening
        local_mirror=mirror,
        connect_timeout=0.05,
    ).start()

    started = time.perf_counter()
    for step in range(20):
        telemetry.record(step=step, loss=1.0 / (step + 1))
    elapsed = time.perf_counter() - started
    stats = telemetry.close()

    # Recording is queue-only, so 20 records must be effectively instant even
    # though every send attempt is failing in the background.
    assert elapsed < 1.0
    assert stats["recorded"] == 20

    lines = [line for line in mirror.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 20
    assert json.loads(lines[0])["run_id"] == "r1-offline"


def test_record_never_blocks_when_queue_is_saturated(tmp_path: Path) -> None:
    telemetry = TrainingTelemetry(
        run_id="r1-flood",
        collector_host="127.0.0.1",
        collector_port=9,
        local_mirror=None,
        queue_size=8,
        connect_timeout=0.05,
    ).start()

    started = time.perf_counter()
    for step in range(2000):
        telemetry.record(step=step, loss=0.5)
    elapsed = time.perf_counter() - started
    stats = telemetry.close()

    assert elapsed < 2.0, "record() must stay non-blocking under saturation"
    assert stats["recorded"] == 2000
    assert stats["dropped"] > 0, "a bounded queue must shed load rather than grow"


def test_collector_survives_malformed_frames(tmp_path: Path) -> None:
    import socket

    collector = MetricsCollector(
        host="127.0.0.1",
        port=0,
        log_path=tmp_path / "collected.jsonl",
        summary_path=tmp_path / "summary.json",
    ).start()
    try:
        sock = socket.create_connection(("127.0.0.1", collector.bound_port), timeout=2.0)
        sock.sendall(b"this is not json\n")
        sock.sendall(b'{"run_id":"ok","node":"n","wall_clock":1.0,"step":1,"loss":2.0}\n')
        sock.close()

        assert _wait_for(lambda: collector.summary()["active_runs"] == 1)
        run = collector.summary()["runs"][0]
        assert run["run_id"] == "ok"
        assert run["record_count"] == 1
    finally:
        collector.stop()


def test_collector_writes_durable_log_and_summary(tmp_path: Path) -> None:
    log_path = tmp_path / "collected.jsonl"
    summary_path = tmp_path / "summary.json"
    collector = MetricsCollector(
        host="127.0.0.1", port=0, log_path=log_path, summary_path=summary_path
    ).start()
    try:
        telemetry = TrainingTelemetry(
            run_id="r1-durable",
            collector_host="127.0.0.1",
            collector_port=collector.bound_port,
        ).start()
        telemetry.record(step=1, loss=3.5, tokens_per_second=1978.0)
        telemetry.close()

        assert _wait_for(lambda: log_path.exists() and log_path.stat().st_size > 0)
        collector.write_summary()

        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert payload["runs"][0]["run_id"] == "r1-durable"
        assert payload["runs"][0]["latest"]["tokens_per_second"] == 1978.0
    finally:
        collector.stop()
