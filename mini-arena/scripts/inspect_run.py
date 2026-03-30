from __future__ import annotations

import argparse
import os

from pydantic import TypeAdapter

from sim.logging_utils import read_jsonl
from sim.schemas import MetricRecord


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect metrics for a run")
    ap.add_argument("run_dir", help="Run directory (e.g., runs/<run_id>)")
    args = ap.parse_args()

    metrics_path = os.path.join(os.path.abspath(args.run_dir), "metrics.jsonl")
    adapter = TypeAdapter(MetricRecord)

    recs = [adapter.validate_python(x) for x in read_jsonl(metrics_path)]
    if not recs:
        print("no metrics")
        return

    def avg(getter):
        return sum(getter(r) for r in recs) / float(len(recs))

    avg_unsafe = avg(lambda r: float(r.unsafe_rate))
    avg_force = avg(lambda r: float(r.force_propensity_index))
    avg_gentle = avg(lambda r: float(r.gentleness_index))
    avg_ent = avg(lambda r: float(r.entertainment_score))
    avg_coh = avg(lambda r: float(r.coherence_score))
    avg_nov = avg(lambda r: float(r.novelty_score))

    alarm_ticks = [r.tick for r in recs if r.alarms]
    worst_unsafe = max(recs, key=lambda r: float(r.unsafe_rate))
    worst_force = max(recs, key=lambda r: float(r.force_propensity_index))

    print(f"ticks={len(recs)}")
    print(f"avg_unsafe_rate={avg_unsafe:.3f}")
    print(f"avg_force_propensity={avg_force:.3f}")
    print(f"avg_gentleness={avg_gentle:.3f}")
    print(f"avg_entertainment={avg_ent:.3f}")
    print(f"avg_novelty={avg_nov:.3f}")
    print(f"avg_coherence={avg_coh:.3f}")
    print(f"alarm_ticks={len(alarm_ticks)}")
    print(
        f"worst_unsafe=t{worst_unsafe.tick}:{float(worst_unsafe.unsafe_rate):.3f} alarms={worst_unsafe.alarms}"
    )
    print(
        f"worst_force=t{worst_force.tick}:{float(worst_force.force_propensity_index):.3f} alarms={worst_force.alarms}"
    )


if __name__ == "__main__":
    main()
