from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any


CLUE_MARKERS = (
    "clue",
    "hint",
    "riddle",
    "secret",
    "hidden message",
    "cipher",
    "code",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _summarize(run_dir: Path) -> dict[str, Any]:
    events = _read_jsonl(run_dir / "events.jsonl")
    metrics = _read_jsonl(run_dir / "metrics.jsonl")
    setup = [e for e in events if e["tick"] <= 0]
    proper = [e for e in events if e["tick"] > 0]
    host = [e for e in proper if e["phase"] == "host"]
    guests = [e for e in proper if e["phase"] == "guest" and e["turn_index"] >= 1]
    spawns = [e for e in proper if e["phase"] == "guest" and e["turn_index"] == -1]

    def summarize_events(es: list[dict[str, Any]]) -> dict[str, Any]:
        lat = [
            e["model_info"].get("latency_ms")
            for e in es
            if e["model_info"].get("latency_ms") is not None
        ]
        return {
            "count": len(es),
            "types": dict(Counter(e["applied_action"]["type"] for e in es)),
            "fallbacks": sum(
                1 for e in es if e["proposed_action"] != e["applied_action"]
            ),
            "errors": sum(1 for e in es if e.get("error")),
            "categories": dict(
                Counter(cat for e in es for cat in e["safety"].get("categories", []))
            ),
            "avg_latency_ms": round(sum(lat) / len(lat), 1) if lat else None,
        }

    host_enrich = [e for e in host if e["applied_action"]["type"] == "enrich_world"]
    clue_like_enrich = sum(
        1
        for e in host_enrich
        if any(
            marker in str(e["applied_action"].get("detail", "")).lower()
            for marker in CLUE_MARKERS
        )
    )

    return {
        "timestamp": int(time.time()),
        "setup": summarize_events(setup),
        "host": summarize_events(host),
        "guests": summarize_events(guests),
        "spawns": len(spawns),
        "collaboration_count": sum(
            1 for e in guests if e["applied_action"]["type"] == "collaborate"
        ),
        "interact_count": sum(
            1 for e in guests if e["applied_action"]["type"] == "interact"
        ),
        "move_count": sum(1 for e in guests if e["applied_action"]["type"] == "move"),
        "clue_like_enrich_count": clue_like_enrich,
        "avg_coherence": round(
            sum(m["coherence_score"] for m in metrics) / len(metrics), 3
        )
        if metrics
        else None,
        "avg_entertainment": round(
            sum(m["entertainment_score"] for m in metrics) / len(metrics), 3
        )
        if metrics
        else None,
        "avg_novelty": round(sum(m["novelty_score"] for m in metrics) / len(metrics), 3)
        if metrics
        else None,
        "zero_entertainment_ticks": sum(
            1 for m in metrics if m["entertainment_score"] == 0.0
        ),
        "completed_ticks": len(metrics),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Periodically summarize a run directory")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--expected-steps", type=int, required=True)
    ap.add_argument("--interval-seconds", type=int, default=600)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    run_dir = repo_root / "runs" / args.run_id
    summary_path = repo_root / "runs" / f"{args.run_id}_monitor.jsonl"

    while not run_dir.exists():
        time.sleep(5)

    while True:
        payload = _summarize(run_dir)
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True))
            f.write("\n")

        if int(payload["completed_ticks"] or 0) >= int(args.expected_steps):
            break

        time.sleep(int(args.interval_seconds))


if __name__ == "__main__":
    main()
