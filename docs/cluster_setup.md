# Cluster Setup: Training Node + Monitoring Node

Two-laptop split:

| Role | Machine | Runs |
|---|---|---|
| **Monitoring node** | Laptop 1 (`192.168.1.69`) | `src.phase_m_cluster.monitor` |
| **Training node** | Laptop 2 | the training script + `TrainingTelemetry` |

## Design guarantee

Training never stalls because of monitoring. `TrainingTelemetry.record()` only
puts a record on a bounded in-memory queue; all network work happens on a
background thread. If the monitoring node is asleep, off the network, or
firewalled:

- `record()` still returns immediately (verified: 2000 records in <2s with no
  collector listening),
- metrics are still written to the training node's **local mirror file**, so
  the run's own history is never lost,
- the queue sheds load rather than growing without bound.

This is covered by `tests/test_cluster_telemetry.py`.

---

## Step 1 — Monitoring node (laptop 1, this machine)

Open the firewall port **once**, as Administrator:

```powershell
cd C:\Users\josep\Desktop\OutreachLM
.\scripts\allow_monitor_port.ps1
```

Without this, laptop 2 cannot connect. Windows Firewall blocks unsolicited
inbound TCP by default. The rule is scoped to **Private** networks only, so it
does not expose the port on public WiFi.

Then start the monitor:

```powershell
python -m src.phase_m_cluster.monitor
```

It prints the address laptop 2 should target and then refreshes a live summary:

```
[18:43:52] 1 run(s)
  [LIVE] r1-realtext @ LAPTOP2
    step=1250  loss=4.8213
    best_loss=4.8102  records=1250
    lr=0.0018  tokens_per_second=1978
```

Outputs:
- `experiments/phase_m/results/collected_metrics.jsonl` — every record, durable
- `experiments/phase_m/results/live_summary.json` — current state per run

## Step 2 — Training node (laptop 2)

Both machines need the repository. Confirm laptop 2 can reach laptop 1:

```powershell
Test-NetConnection 192.168.1.69 -Port 51799
```

`TcpTestSucceeded : True` means the path is open. If it is False, re-check
Step 1 and confirm both machines are on the same network.

Then wrap the training loop:

```python
from src.phase_m_cluster.telemetry import TrainingTelemetry

telemetry = TrainingTelemetry(
    run_id="r1-5m-realtext",
    collector_host="192.168.1.69",     # laptop 1
    collector_port=51799,
    local_mirror="experiments/phase_m/results/local_metrics.jsonl",
).start()

try:
    for step, (inputs, targets) in enumerate(batches):
        ...                                   # forward / backward / step
        if step % 25 == 0:                    # ~every 25 steps is plenty
            telemetry.record(
                step=step,
                loss=float(loss.item()),
                lr=current_lr,
                tokens_per_second=tokens_per_second,
            )
finally:
    print(telemetry.close())                  # flushes and reports send stats
```

`close()` returns `{"recorded", "sent", "dropped", "reconnects", "last_error"}`
so you can confirm afterwards how much actually reached the monitor.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Monitor stuck on "waiting for training node" | Firewall | Run Step 1 as Administrator |
| `TcpTestSucceeded : False` | Different networks / VPN | Put both on the same WiFi; disable VPN |
| `sent` is 0 but `recorded` is high | Wrong host or port | Check the IP the monitor printed |
| Run shows `STALE` | Training stopped or laptop slept | Check laptop 2; monitor keeps prior data |
| Laptop 1 IP changed | DHCP reassignment | Re-read the address the monitor prints |

Laptop 1's IP can change on reconnect. If it does, update `collector_host` on
laptop 2 — or reserve a static DHCP lease for laptop 1 on your router.

## Optional: laptop 1 also contributes compute

Not built yet, and worth being deliberate about. True data-parallel training
across both machines needs `torch.distributed` with the `gloo` backend, and it
synchronizes gradients every step — over WiFi that sync can easily cost more
time than the second machine's compute adds, because this model is small and
its gradients are large relative to the work per step.

Recommended sequence: get single-node training measured first, then decide
whether distributed is worth it based on the measured step time, rather than
assuming two nodes is automatically ~2x.
