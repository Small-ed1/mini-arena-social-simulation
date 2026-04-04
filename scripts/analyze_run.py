from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


CLUE_MARKERS = (
    "clue",
    "hint",
    "riddle",
    "secret",
    "hidden message",
    "cipher",
    "code",
)


def _load_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze a mini-arena run")
    ap.add_argument("run_dir")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    events = _load_jsonl(run_dir / "events.jsonl")
    metrics = _load_jsonl(run_dir / "metrics.jsonl")

    setup = [e for e in events if e["tick"] <= 0]
    proper = [e for e in events if e["tick"] > 0]
    host = [e for e in proper if e["phase"] == "host"]
    guests = [e for e in proper if e["phase"] == "guest" and e["turn_index"] >= 1]
    spawns = [e for e in proper if e["phase"] == "guest" and e["turn_index"] == -1]

    def summarize(es):
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
    collaboration = sum(
        1 for e in guests if e["applied_action"]["type"] == "collaborate"
    )
    interact = sum(1 for e in guests if e["applied_action"]["type"] == "interact")
    move = sum(1 for e in guests if e["applied_action"]["type"] == "move")

    out = {
        "run_dir": str(run_dir),
        "setup": summarize(setup),
        "host": summarize(host),
        "guests": summarize(guests),
        "spawns": len(spawns),
        "collaboration_count": collaboration,
        "interact_count": interact,
        "move_count": move,
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
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
