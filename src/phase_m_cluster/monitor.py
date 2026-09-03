"""Live training monitor -- run this on the MONITORING node (laptop 1).

    python -m src.phase_m_cluster.monitor

Listens for telemetry from the training node, writes every record durably to
JSONL, and prints a refreshing summary of each active run.
"""
from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase_m_cluster.collector import MetricsCollector

DEFAULT_PORT = 51799


def local_addresses() -> list[str]:
    addresses: list[str] = []
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            address = info[4][0]
            if address not in addresses and not address.startswith("127."):
                addresses.append(address)
    except OSError:
        pass
    return addresses


def _format_run(run: dict) -> str:
    step = run.get("latest_step")
    loss = run.get("latest_loss")
    best = run.get("min_loss")
    stale = run.get("seconds_since_last_record", 0.0)
    health = "LIVE" if stale < 30 else f"STALE {stale:.0f}s"
    parts = [
        f"  [{health}] {run['run_id']} @ {run['node']}",
        f"    step={step if step is not None else '-'}"
        f"  loss={loss:.4f}" if isinstance(loss, float) else "    step=-  loss=-",
    ]
    if isinstance(best, float):
        parts.append(f"    best_loss={best:.4f}  records={run['record_count']}")
    else:
        parts.append(f"    records={run['record_count']}")
    latest = run.get("latest", {})
    extras = {
        k: v
        for k, v in latest.items()
        if k not in {"run_id", "node", "wall_clock", "step", "loss"}
    }
    if extras:
        rendered = "  ".join(
            f"{k}={v:.6g}" if isinstance(v, (int, float)) else f"{k}={v}"
            for k, v in list(extras.items())[:6]
        )
        parts.append(f"    {rendered}")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="OutreachLM cluster training monitor.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--refresh", type=float, default=5.0)
    parser.add_argument(
        "--log",
        type=Path,
        default=ROOT / "experiments" / "phase_m" / "results" / "collected_metrics.jsonl",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "experiments" / "phase_m" / "results" / "live_summary.json",
    )
    args = parser.parse_args()

    collector = MetricsCollector(
        host=args.host, port=args.port, log_path=args.log, summary_path=args.summary
    ).start()

    print("=" * 68)
    print("  OutreachLM Training Monitor")
    print("=" * 68)
    print(f"  listening on {args.host}:{collector.bound_port}")
    for address in local_addresses():
        print(f"  training node should connect to: {address}:{collector.bound_port}")
    print(f"  durable log : {args.log}")
    print(f"  live summary: {args.summary}")
    print("  (Ctrl+C to stop; the training node is unaffected either way)")
    print("=" * 68, flush=True)

    try:
        while True:
            time.sleep(args.refresh)
            summary = collector.summary()
            collector.write_summary()
            stamp = time.strftime("%H:%M:%S")
            if not summary["runs"]:
                print(f"[{stamp}] waiting for training node...", flush=True)
                continue
            print(f"[{stamp}] {summary['active_runs']} run(s)", flush=True)
            for run in summary["runs"]:
                print(_format_run(run), flush=True)
    except KeyboardInterrupt:
        print("\nstopping monitor...")
    finally:
        collector.write_summary()
        collector.stop()
        print(f"final summary written to {args.summary}")


if __name__ == "__main__":
    main()
