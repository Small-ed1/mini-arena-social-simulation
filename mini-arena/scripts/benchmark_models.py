from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
CONFIGS_DIR = ROOT / "configs"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _summarize_run(run_id: str) -> Dict[str, Any]:
    run_dir = RUNS_DIR / run_id
    events = _read_jsonl(run_dir / "events.jsonl")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    total_events = len(events)
    parse_failures = sum(1 for e in events if e.get("error"))
    invalid_world_ref = sum(
        1
        for e in events
        if "invalid_world_ref" in ((e.get("safety") or {}).get("categories") or [])
    )
    fallback_count = sum(
        1 for e in events if e.get("proposed_action") != e.get("applied_action")
    )
    raw_outputs = [((e.get("raw_model_io") or {}).get("output")) for e in events]
    json_adherence = sum(
        1 for out in raw_outputs if isinstance(out, str) and out.strip().startswith("{")
    )

    action_types = Counter(
        str((e.get("applied_action") or {}).get("type", "unknown")) for e in events
    )
    host_events = [e for e in events if e.get("phase") == "host"]
    thread_engagement = sum(
        1
        for e in host_events
        if str((e.get("applied_action") or {}).get("type")) != "spawn_event"
    )
    latencies = [
        int(mi["latency_ms"])
        for e in events
        if isinstance((mi := (e.get("model_info") or {})).get("latency_ms"), int)
    ]

    return {
        "run_id": run_id,
        "model": manifest["baseline_cfg"]["inference"]["model"],
        "seed": manifest["baseline_cfg"]["seed"],
        "guest_count": manifest["baseline_cfg"]["guest_count"],
        "episode_steps": manifest["baseline_cfg"]["episode_steps"],
        "total_events": total_events,
        "json_adherence_rate": round(json_adherence / total_events, 4)
        if total_events
        else 0.0,
        "invalid_world_reference_rate": round(invalid_world_ref / total_events, 4)
        if total_events
        else 0.0,
        "fallback_rate": round(fallback_count / total_events, 4)
        if total_events
        else 0.0,
        "parse_failure_rate": round(parse_failures / total_events, 4)
        if total_events
        else 0.0,
        "action_diversity": len(action_types),
        "action_type_counts": dict(action_types),
        "thread_engagement_rate": round(
            thread_engagement / max(1, len(host_events)), 4
        ),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1)
        if latencies
        else None,
        "max_latency_ms": max(latencies) if latencies else None,
    }


def _run_one(config_name: str, run_id: str) -> Dict[str, Any]:
    baseline_path = CONFIGS_DIR / "baseline.yaml"
    config_path = CONFIGS_DIR / config_name
    backup = baseline_path.read_text(encoding="utf-8")
    try:
        shutil.copyfile(config_path, baseline_path)
        subprocess.run(
            ["python", "-m", "scripts.run_baseline", "--run-id", run_id],
            cwd=ROOT,
            check=True,
        )
    finally:
        baseline_path.write_text(backup, encoding="utf-8")
    return _summarize_run(run_id)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Benchmark Ollama models on fixed 10-turn runs"
    )
    ap.add_argument("--output", default="runs/benchmark_seed1337_10turn_summary.json")
    args = ap.parse_args()

    jobs = [
        ("benchmark_10turn_llama31_latest.yaml", "bench_seed1337_llama31_latest"),
        ("benchmark_10turn_llama31_70b.yaml", "bench_seed1337_llama31_70b"),
        ("benchmark_10turn_qwen35_9b.yaml", "bench_seed1337_qwen35_9b"),
    ]

    results = [_run_one(config_name, run_id) for config_name, run_id in jobs]
    out_path = ROOT / args.output
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
