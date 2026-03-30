from __future__ import annotations

from typing import Any, Dict, List, Tuple

from pydantic import TypeAdapter

from sim.checkpoint import world_from_dict
from sim.env import Environment
from sim.logging_utils import hash_json, read_json, read_jsonl
from sim.schemas import EventRecord
from sim.world_state import Rulebook


def replay_run(run_dir: str) -> Tuple[bool, List[str]]:
    """Replay a run from events.jsonl and verify world hashes."""
    manifest_path = f"{run_dir}/manifest.json"
    events_path = f"{run_dir}/events.jsonl"

    manifest = read_json(manifest_path)
    rulebook = None
    if isinstance(manifest.get("rulebook"), dict):
        rb = manifest["rulebook"]
        rulebook = Rulebook(
            hard_deny=list(rb.get("hard_deny") or []),
            soft_flag=list(rb.get("soft_flag") or []),
            fallbacks=dict(rb.get("fallbacks") or {}),
            scoring_weights=dict(rb.get("scoring_weights") or {}),
        )

    world = world_from_dict(manifest["initial_world"], rulebook=rulebook)
    env = Environment()

    adapter = TypeAdapter(EventRecord)
    errors: List[str] = []
    ok = True

    for raw in read_jsonl(events_path):
        ev = adapter.validate_python(raw)
        # The canonical world state includes tick.
        world.tick = int(ev.tick)
        before = hash_json(world.to_dict())
        if before != ev.env.world_hash_before:
            ok = False
            errors.append(
                f"tick {ev.tick} {ev.phase}#{ev.turn_index}: world_hash_before mismatch ({before} != {ev.env.world_hash_before})"
            )
            # Continue to collect more errors.

        if ev.phase == "host":
            env.apply_host_action(world, ev.applied_action)  # type: ignore[arg-type]
        else:
            env.apply_guest_action(world, ev.actor_id, ev.applied_action)

        env.tick_postprocess(world)
        after = hash_json(world.to_dict())
        if after != ev.env.world_hash_after:
            ok = False
            errors.append(
                f"tick {ev.tick} {ev.phase}#{ev.turn_index}: world_hash_after mismatch ({after} != {ev.env.world_hash_after})"
            )

    return ok, errors
