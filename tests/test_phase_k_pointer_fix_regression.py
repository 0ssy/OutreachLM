from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FIXTURES = ROOT / "experiments" / "phase_k" / "fixtures" / "multi_hop"
if str(FIXTURES) not in sys.path:
    sys.path.insert(0, str(FIXTURES))

from experiments.phase_k.run_phase_k_empirical_fixed import _measure_depth as _measure_depth_fixed


def test_pointer_fix_recovers_full_accuracy_at_shallow_and_deep_hops() -> None:
    """Locks in the fix: unlike the unaugmented n-gram baseline (which measured
    ~0% argmax accuracy on unseen permutations at every depth), the pointer
    mechanism must resolve unseen chain-completion queries deterministically
    at both a shallow depth and a much deeper one."""
    shallow = _measure_depth_fixed(1)
    deep = _measure_depth_fixed(64)

    assert shallow["argmax_next_token_accuracy"] == 1.0
    assert deep["argmax_next_token_accuracy"] == 1.0
    assert shallow["pointer_utilization_rate"] == 1.0
    assert deep["pointer_utilization_rate"] == 1.0
