from __future__ import annotations

from typing import Any, Dict, List, Tuple

from pydantic import TypeAdapter

from sim.checkpoint import world_from_dict
from sim.env import Environment
from sim.logging_utils import hash_json, read_json, read_jsonl
from sim.orchestrator import _update_guest_turn_fairness
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

    raw_events = list(read_jsonl(events_path))
    acted_guest_order: List[str] = []

    for idx, raw in enumerate(raw_events):
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

        processed = False
        if ev.phase == "guest" and ev.turn_index == -1:
            gid = str(ev.actor_id)
            if gid in world.unspawned_guest_ids:
                world.unspawned_guest_ids = [
                    x for x in world.unspawned_guest_ids if x != gid
                ]
            if gid not in world.spawned_guest_ids:
                world.spawned_guest_ids.append(gid)
            world.guests[gid].spawn_tick = int(ev.tick)
            env.tick_postprocess(world)
            processed = True
        elif ev.phase == "host":
            env.apply_host_action(world, ev.applied_action)  # type: ignore[arg-type]
        else:
            env.apply_guest_action(world, ev.actor_id, ev.applied_action)
            if ev.turn_index >= 1:
                acted_guest_order.append(str(ev.actor_id))

        if not processed:
            env.tick_postprocess(world)
        after = hash_json(world.to_dict())
        if after != ev.env.world_hash_after:
            ok = False
            errors.append(
                f"tick {ev.tick} {ev.phase}#{ev.turn_index}: world_hash_after mismatch ({after} != {ev.env.world_hash_after})"
            )

        next_tick = None
        if idx + 1 < len(raw_events):
            next_tick = int(raw_events[idx + 1]["tick"])
        if next_tick is None or next_tick != int(ev.tick):
            _update_guest_turn_fairness(world, acted_guest_order)
            acted_guest_order = []

    return ok, errors
