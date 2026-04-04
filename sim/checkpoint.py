from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

from sim.logging_utils import ensure_dir, hash_json, json_dumps_canonical
from sim.memory import MemoryStore
from sim.world_state import (
    GuestState,
    OpenThread,
    Prop,
    Rulebook,
    WorldState,
    default_conceptual_levels,
)


def save_checkpoint(
    *, path: str, world: WorldState, memory: MemoryStore
) -> Tuple[str, str]:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    payload = {
        "world": world.to_dict(),
        "memory": memory.to_dict(),
    }
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json_dumps_canonical(payload))
        f.write("\n")
    os.replace(tmp, path)
    return (hash_json(payload["world"]), hash_json(payload["memory"]))


def load_checkpoint(
    *,
    path: str,
    personas_cfg: Dict[str, Any],
    memory_cfg: Dict[str, Any],
    rulebook: Optional[Rulebook] = None,
) -> Tuple[WorldState, MemoryStore]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    world = world_from_dict(payload["world"], rulebook=rulebook)
    mem = MemoryStore.from_dict(
        payload["memory"],
        personas_cfg=personas_cfg,
        last_n_events_host=int(memory_cfg["last_n_events_host"]),
        last_n_events_guest=int(memory_cfg["last_n_events_guest"]),
        top_k_semantic=int(memory_cfg["top_k_semantic"]),
        reflection_chars=int(memory_cfg["reflection_chars"]),
        world_summary_chars=int(memory_cfg["world_summary_chars"]),
    )
    return world, mem


def world_from_dict(
    data: Dict[str, Any], *, rulebook: Optional[Rulebook] = None
) -> WorldState:
    props: Dict[str, Prop] = {}
    for pid, pd in (data.get("props") or {}).items():
        props[pid] = Prop(
            prop_id=str(pd["prop_id"]),
            prop_type=str(pd["prop_type"]),
            location=pd.get("location"),
            portable=bool(pd["portable"]),
            held_by=pd.get("held_by"),
            state=dict(pd.get("state") or {}),
        )

    guests: Dict[str, GuestState] = {}
    for gid, gd in (data.get("guests") or {}).items():
        guests[gid] = GuestState(
            guest_id=str(gd["guest_id"]),
            persona_id=str(gd.get("persona_id") or gid),
            name=str(gd.get("name") or gid),
            location=str(gd["location"]),
            inventory=list(gd.get("inventory") or []),
            mood=dict(gd.get("mood") or {}),
            trust=dict(gd.get("trust") or {}),
            tension=dict(gd.get("tension") or {}),
            familiarity=dict(gd.get("familiarity") or {}),
            current_goal=str(gd.get("current_goal") or "Explore the arena"),
            last_action=gd.get("last_action"),
            spotlight_weight=float(gd.get("spotlight_weight", 0.0)),
            reflection_requested=bool(gd.get("reflection_requested", False)),
            spawn_tick=gd.get("spawn_tick"),
        )

    open_threads: Dict[str, OpenThread] = {}
    for tid, td in (data.get("open_threads") or {}).items():
        open_threads[tid] = OpenThread(
            thread_id=str(td["thread_id"]),
            thread_type=str(td["thread_type"]),
            status=str(td["status"]),
            description=str(td["description"]),
            location=(str(td["location"]) if td.get("location") is not None else None),
            involved_guest_ids=list(td.get("involved_guest_ids") or []),
        )

    spawned_guest_ids = data.get("spawned_guest_ids")
    if spawned_guest_ids is None:
        spawned_guest_ids = sorted(guests)
    unspawned_guest_ids = data.get("unspawned_guest_ids")
    if unspawned_guest_ids is None:
        unspawned_guest_ids = []

    return WorldState(
        arena_id=str(data["arena_id"]),
        tick=int(data["tick"]),
        locations=dict(data.get("locations") or {}),
        props=props,
        guests=guests,
        spawned_guest_ids=list(spawned_guest_ids),
        unspawned_guest_ids=list(unspawned_guest_ids),
        open_threads=open_threads,
        location_details={
            str(k): list(v or [])
            for k, v in (data.get("location_details") or {}).items()
        },
        conceptual_global={
            **default_conceptual_levels(),
            **{
                str(k): float(v)
                for k, v in (data.get("conceptual_global") or {}).items()
            },
        },
        conceptual_by_location={
            str(loc): {
                **default_conceptual_levels(),
                **{str(k): float(v) for k, v in (vals or {}).items()},
            }
            for loc, vals in (data.get("conceptual_by_location") or {}).items()
        },
        conceptual_by_guest={
            str(gid): {
                **default_conceptual_levels(),
                **{str(k): float(v) for k, v in (vals or {}).items()},
            }
            for gid, vals in (data.get("conceptual_by_guest") or {}).items()
        },
        guest_turn_fairness={
            str(gid): float(v)
            for gid, v in (data.get("guest_turn_fairness") or {}).items()
        },
        host_style=str(data.get("host_style") or "neutral"),
        host_last_actions=list(data.get("host_last_actions") or []),
        rulebook=rulebook,
    )
