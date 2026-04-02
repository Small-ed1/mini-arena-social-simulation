from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "configs"
RUNS_DIR = ROOT / "runs"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_checkpoint_world(run_dir: Path, *, tick: int) -> Optional[Dict[str, Any]]:
    ck = run_dir / "checkpoints" / f"checkpoint_t{int(tick)}.json"
    if not ck.exists():
        return None
    payload = json.loads(ck.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    world = payload.get("world")
    return world if isinstance(world, dict) else None


def _pct(n: int, d: int) -> float:
    if d <= 0:
        return 0.0
    return (100.0 * float(n)) / float(d)


def _rate(n: int, d: int) -> float:
    if d <= 0:
        return 0.0
    return float(n) / float(d)


def _host_action_mix(events: List[Dict[str, Any]]) -> Counter:
    c: Counter = Counter()
    for e in events:
        if e.get("phase") != "host":
            continue
        at = (e.get("applied_action") or {}).get("type")
        c[str(at or "unknown")] += 1
    return c


def _guest_gg_rate(events: List[Dict[str, Any]]) -> Tuple[int, int]:
    total = 0
    gg = 0
    for e in events:
        if e.get("phase") != "guest":
            continue
        total += 1
        act = e.get("applied_action") or {}
        at = str(act.get("type") or "")
        if at == "collaborate":
            gg += 1
        elif at == "speak":
            if act.get("target_guest_id") is not None:
                gg += 1
    return gg, total


def _thread_progression(
    events: List[Dict[str, Any]], run_dir: Path, *, steps: int
) -> Dict[str, Any]:
    spawned = 0
    closed_msgs = 0
    for e in events:
        if e.get("phase") == "host":
            act = e.get("applied_action") or {}
            if str(act.get("type")) == "spawn_event":
                spawned += 1
        if e.get("phase") == "guest":
            env = e.get("env") or {}
            msgs = env.get("messages") or []
            if isinstance(msgs, list):
                for m in msgs:
                    if isinstance(m, str) and m.startswith("closed thread "):
                        closed_msgs += 1

    world = _load_checkpoint_world(run_dir, tick=steps)
    end_total = None
    end_closed = None
    end_open = None
    if world is not None:
        threads = world.get("open_threads") or {}
        if isinstance(threads, dict):
            end_total = len(threads)
            end_closed = sum(
                1
                for td in threads.values()
                if isinstance(td, dict) and str(td.get("status")) == "closed"
            )
            end_open = sum(
                1
                for td in threads.values()
                if isinstance(td, dict) and str(td.get("status")) == "open"
            )

    return {
        "threads_spawned": spawned,
        "closed_thread_msgs": closed_msgs,
        "threads_end_total": end_total,
        "threads_end_closed": end_closed,
        "threads_end_open": end_open,
    }


def _fallback_and_invalid_rates(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(events)
    fallback = 0
    invalid_world_ref = 0
    parse_failure = 0
    for e in events:
        if e.get("error"):
            parse_failure += 1
        if e.get("proposed_action") != e.get("applied_action"):
            fallback += 1
        cats = (e.get("safety") or {}).get("categories") or []
        if isinstance(cats, list) and "invalid_world_ref" in cats:
            invalid_world_ref += 1
    return {
        "total_events": total,
        "fallback": fallback,
        "invalid_world_ref": invalid_world_ref,
        "parse_failure": parse_failure,
    }


def _raw_cleanliness(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(events)
    raw_present = 0
    starts_with_brace = 0
    clean_object = 0
    has_fence = 0
    has_newline_prefix = 0
    for e in events:
        raw = e.get("raw_model_io") or {}
        out = raw.get("output")
        if not isinstance(out, str):
            continue
        raw_present += 1
        t = out.strip()
        if "```" in out:
            has_fence += 1
        if out[:1] in ("\n", "\r"):
            has_newline_prefix += 1
        if t.startswith("{"):
            starts_with_brace += 1
        if t.startswith("{") and t.endswith("}"):
            clean_object += 1

    return {
        "total_events": total,
        "raw_present": raw_present,
        "starts_with_brace": starts_with_brace,
        "clean_object": clean_object,
        "has_fence": has_fence,
        "has_newline_prefix": has_newline_prefix,
    }


def summarize_run(run_id: str) -> Dict[str, Any]:
    run_dir = RUNS_DIR / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    baseline = manifest.get("baseline_cfg") or {}
    steps = int(baseline.get("episode_steps") or 0)

    events = _read_jsonl(run_dir / "events.jsonl")

    host_mix = _host_action_mix(events)
    gg, gg_total = _guest_gg_rate(events)
    thread = _thread_progression(events, run_dir, steps=steps)
    fb = _fallback_and_invalid_rates(events)
    raw = _raw_cleanliness(events)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "seed": baseline.get("seed"),
        "episode_steps": steps,
        "inference": (baseline.get("inference") or {}),
        "host_action_mix": dict(host_mix),
        "guest_gg": {"count": gg, "total": gg_total, "rate": _rate(gg, gg_total)},
        "thread": thread,
        "fallback": {
            **fb,
            "fallback_rate": _rate(int(fb["fallback"]), int(fb["total_events"])),
            "invalid_world_ref_rate": _rate(
                int(fb["invalid_world_ref"]), int(fb["total_events"])
            ),
        },
        "raw": {
            **raw,
            "raw_present_rate": _rate(
                int(raw["raw_present"]), int(raw["total_events"])
            ),
            "starts_with_brace_rate": _rate(
                int(raw["starts_with_brace"]), int(raw["raw_present"])
            ),
            "clean_object_rate": _rate(
                int(raw["clean_object"]), int(raw["raw_present"])
            ),
            "fence_rate": _rate(int(raw["has_fence"]), int(raw["raw_present"])),
        },
    }


def _run_one(config_name: str, run_id: str) -> None:
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


def _fmt_mix(mix: Dict[str, Any]) -> str:
    items = [(k, int(v)) for k, v in mix.items()]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{k}={v}" for k, v in items)


def _compare(a: Dict[str, Any], b: Dict[str, Any]) -> str:
    # b - a deltas.
    lines: List[str] = []

    a_fb = a["fallback"]
    b_fb = b["fallback"]
    a_raw = a["raw"]
    b_raw = b["raw"]

    lines.append(
        f"fallback_rate: {a_fb['fallback_rate']:.3f} -> {b_fb['fallback_rate']:.3f} (d={(b_fb['fallback_rate'] - a_fb['fallback_rate']):+.3f})"
    )
    lines.append(
        f"invalid_world_ref_rate: {a_fb['invalid_world_ref_rate']:.3f} -> {b_fb['invalid_world_ref_rate']:.3f} (d={(b_fb['invalid_world_ref_rate'] - a_fb['invalid_world_ref_rate']):+.3f})"
    )
    lines.append(
        f"raw_clean_object_rate: {a_raw['clean_object_rate']:.3f} -> {b_raw['clean_object_rate']:.3f} (d={(b_raw['clean_object_rate'] - a_raw['clean_object_rate']):+.3f})"
    )
    lines.append(
        f"raw_fence_rate: {a_raw['fence_rate']:.3f} -> {b_raw['fence_rate']:.3f} (d={(b_raw['fence_rate'] - a_raw['fence_rate']):+.3f})"
    )

    a_gg = a["guest_gg"]
    b_gg = b["guest_gg"]
    lines.append(
        f"guest_gg_rate: {a_gg['rate']:.3f} -> {b_gg['rate']:.3f} (d={(b_gg['rate'] - a_gg['rate']):+.3f})"
    )

    a_t = a["thread"]
    b_t = b["thread"]

    def _closure_rate(t: Dict[str, Any]) -> Optional[float]:
        spawned = int(t.get("threads_spawned") or 0)
        closed = t.get("threads_end_closed")
        if closed is None:
            return None
        if spawned <= 0:
            return 0.0
        return float(closed) / float(spawned)

    ar = _closure_rate(a_t)
    br = _closure_rate(b_t)
    if ar is not None and br is not None:
        lines.append(
            f"thread_closure_rate(end/ spawned): {ar:.3f} -> {br:.3f} (d={(br - ar):+.3f})"
        )

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="A/B test prompt framing (current vs neutral) on fixed models"
    )
    ap.add_argument(
        "--current-config",
        default="run_25turn_mixed_host70b_guestslatest.yaml",
        help="Config to use for current prompts (copied into configs/baseline.yaml)",
    )
    ap.add_argument(
        "--neutral-config",
        default="run_25turn_mixed_host70b_guestslatest_neutralprompts.yaml",
        help="Config to use for neutral prompts (copied into configs/baseline.yaml)",
    )
    ap.add_argument(
        "--run-id-current",
        default="ab_current_seed1337_host70b_guestslatest",
        help="Run id for current prompts",
    )
    ap.add_argument(
        "--run-id-neutral",
        default="ab_neutral_seed1337_host70b_guestslatest",
        help="Run id for neutral prompts",
    )
    ap.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip running episodes; only summarize existing runs",
    )
    args = ap.parse_args()

    if not args.skip_run:
        _run_one(args.current_config, args.run_id_current)
        _run_one(args.neutral_config, args.run_id_neutral)

    cur = summarize_run(args.run_id_current)
    neu = summarize_run(args.run_id_neutral)

    print("CURRENT")
    print(f"run_id={cur['run_id']}")
    print(f"inference.host_model={(cur['inference'] or {}).get('host_model')}")
    print(f"inference.guest_model={(cur['inference'] or {}).get('guest_model')}")
    print(
        f"inference.prompt_variant={(cur['inference'] or {}).get('prompt_variant', 'current')}"
    )
    print(f"host_action_mix: {_fmt_mix(cur['host_action_mix'])}")
    print(
        f"guest_to_guest_speech_or_collab: {cur['guest_gg']['count']}/{cur['guest_gg']['total']} ({cur['guest_gg']['rate']:.3f})"
    )
    print(
        "threads: spawned={threads_spawned} end_total={threads_end_total} end_closed={threads_end_closed} end_open={threads_end_open} closed_msgs={closed_thread_msgs}".format(
            **cur["thread"]
        )
    )
    print(
        f"fallback_rate={cur['fallback']['fallback_rate']:.3f} invalid_world_ref_rate={cur['fallback']['invalid_world_ref_rate']:.3f} parse_failure={cur['fallback']['parse_failure']}"
    )
    print(
        f"raw: present={cur['raw']['raw_present']}/{cur['raw']['total_events']} clean_object_rate={cur['raw']['clean_object_rate']:.3f} fence_rate={cur['raw']['fence_rate']:.3f}"
    )

    print("\nNEUTRAL")
    print(f"run_id={neu['run_id']}")
    print(f"inference.host_model={(neu['inference'] or {}).get('host_model')}")
    print(f"inference.guest_model={(neu['inference'] or {}).get('guest_model')}")
    print(
        f"inference.prompt_variant={(neu['inference'] or {}).get('prompt_variant', 'current')}"
    )
    print(f"host_action_mix: {_fmt_mix(neu['host_action_mix'])}")
    print(
        f"guest_to_guest_speech_or_collab: {neu['guest_gg']['count']}/{neu['guest_gg']['total']} ({neu['guest_gg']['rate']:.3f})"
    )
    print(
        "threads: spawned={threads_spawned} end_total={threads_end_total} end_closed={threads_end_closed} end_open={threads_end_open} closed_msgs={closed_thread_msgs}".format(
            **neu["thread"]
        )
    )
    print(
        f"fallback_rate={neu['fallback']['fallback_rate']:.3f} invalid_world_ref_rate={neu['fallback']['invalid_world_ref_rate']:.3f} parse_failure={neu['fallback']['parse_failure']}"
    )
    print(
        f"raw: present={neu['raw']['raw_present']}/{neu['raw']['total_events']} clean_object_rate={neu['raw']['clean_object_rate']:.3f} fence_rate={neu['raw']['fence_rate']:.3f}"
    )

    print("\nDELTA (neutral - current)")
    print(_compare(cur, neu))


if __name__ == "__main__":
    main()
