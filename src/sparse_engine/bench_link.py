"""Measure the two-laptop link with the precision actually available.

A first version of this file used `ping` and reported 131 MB/s. That was an
artifact: Windows ping rounds RTT to 1 ms, and 65500 B / 1 ms = 131 MB/s
EXCEEDS the 144.4 Mbps (18.05 MB/s) negotiated link rate, so the number was
physically impossible. Recorded here because it is exactly the kind of
measurement that looks like data and is not.

What CAN be measured from this side alone:
    * TCP connect round-trip at microsecond resolution, against the one open
      port (445/SMB).
What CANNOT, without a cooperating listener on the far machine:
    * sustained stream throughput. It is therefore BOUNDED rather than
      measured: 144.4 Mbps negotiated, and 802.11 with both nodes on the same
      AP realistically delivers 35-75% of that depending on contention.

The design conclusion is insensitive to where in that range the truth lies:
even the optimistic bound is ~300x slower than this machine's disk, so any
scheme moving per-step activations across it fails either way.
"""
from __future__ import annotations

import socket
import statistics
import time

HOST = "192.168.1.69"
PORT = 445

LINK_MBPS = 144.4
NEGOTIATED_MBS = LINK_MBPS / 8            # 18.05 MB/s
EFFICIENCY_RANGE = (0.35, 0.75)           # contended .. clean


def connect_rtt(n: int = 25) -> list[float]:
    """TCP connect RTT in ms. The only precise number available from here."""
    out = []
    for _ in range(n):
        s = socket.socket()
        s.settimeout(3.0)
        t0 = time.perf_counter()
        try:
            s.connect((HOST, PORT))
            out.append((time.perf_counter() - t0) * 1000)
        except OSError:
            pass
        finally:
            s.close()
        time.sleep(0.01)
    return out


def usable_gbps() -> tuple[float, float]:
    lo = NEGOTIATED_MBS * EFFICIENCY_RANGE[0] / 1000
    hi = NEGOTIATED_MBS * EFFICIENCY_RANGE[1] / 1000
    return lo, hi


def main() -> None:
    print(f"Link to {HOST}:{PORT}\n")
    rtts = connect_rtt()
    if rtts:
        rtts.sort()
        print(f"  TCP connect RTT   median {statistics.median(rtts):.2f} ms,"
              f" min {rtts[0]:.2f}, p90 "
              f"{rtts[-max(1, len(rtts) // 10)]:.2f}")
    else:
        print("  no successful connects")

    lo, hi = usable_gbps()
    print(f"\n  negotiated link   {LINK_MBPS} Mbps = {NEGOTIATED_MBS:.1f} MB/s")
    print(f"  usable estimate   {lo * 1000:.1f} - {hi * 1000:.1f} MB/s"
          f"  ({lo:.4f} - {hi:.4f} GB/s)")

    print(f"\n{'resource':>22}{'GB/s':>10}{'vs link (hi)':>14}")
    print("-" * 46)
    for name, gbs in (("cross-laptop WiFi", hi),
                      ("local disk write", 0.26),
                      ("local disk read", 4.10),
                      ("local RAM", 10.6)):
        print(f"{name:>22}{gbs:>10.4f}{gbs / hi:>13.0f}x")

    print("\n  Consequence: per-step cross-machine traffic must be tens of MB,")
    print("  not GB. That rules out expert-parallel all-to-all and forces")
    print("  disjoint expert ownership with infrequent trunk sync.")


if __name__ == "__main__":
    main()
