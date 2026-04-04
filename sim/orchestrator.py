from __future__ import annotations

import hashlib
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import TypeAdapter

from sim.checkpoint import save_checkpoint
from sim.env import Environment
from sim.inference import InferenceEngine
from sim.logging_utils import (
    RunPaths,
    hash_json,
    make_run_id,
    observation_digest,
    stable_env_id,
    write_manifest,
)
from sim.memory import MemoryStore
from sim.metrics import MetricsEngine
from sim.schemas import (
    CheckpointRecord,
    EnvResultRecord,
    EventRecord,
    GuestWait,
    HostAllocateSpotlight,
    HostSignalStyle,
    HostSpawnEvent,
    MetricRecord,
    ModelInfo,
    SafetyDecision,
    safe_model_dump,
)
from sim.safety import fallback_guest_action, safety_pass
from sim.world_state import (
    Rulebook,
    combined_conceptual_for_guest,
    load_scene,
    make_initial_world,
)


def _guest_action_semantic_issue(
    obs_g: Any, action: Any, world: Any
) -> Optional[Tuple[str, str]]:
    gid = str(getattr(obs_g, "guest_id", ""))
    guest = world.guests.get(gid)
    if guest is None:
        return ("invalid_world_ref", f"unknown guest: {gid}")

    action_type = str(getattr(action, "type", ""))
    if action_type == "move":
        dest = str(getattr(action, "destination", ""))
        if dest not in world.locations:
            return ("invalid_world_ref", f"invalid destination: {dest}")
        return None

    if action_type == "collaborate":
        target = str(getattr(action, "target_guest_id", ""))
        if not world.is_spawned(target):
            return ("invalid_world_ref", f"invalid target_guest_id: {target}")
        if world.guests[target].location != guest.location:
            return ("invalid_world_ref", f"target not co-located: {target}")
        return None

    if action_type == "speak":
        target = getattr(action, "target_guest_id", None)
        if target is not None:
            target_id = str(target)
            if not world.is_spawned(target_id):
                return ("invalid_world_ref", f"invalid target_guest_id: {target_id}")
            if world.guests[target_id].location != guest.location:
                return ("invalid_world_ref", f"target not co-located: {target_id}")
        return None

    if action_type != "interact":
        return None

    prop_id = str(getattr(action, "prop_id", ""))
    if prop_id not in world.props:
        return ("invalid_world_ref", f"invalid prop_id: {prop_id}")
    prop = world.props[prop_id]
    verb = str(getattr(action, "verb", ""))
    accessible = (prop.held_by == gid) or (
        prop.held_by is None and prop.location == guest.location
    )

    if verb == "inspect":
        if not accessible:
            return ("invalid_world_ref", f"prop not accessible: {prop_id}")
        return None
    if verb == "pick_up":
        if not prop.portable:
            return ("invalid_world_ref", f"prop not portable: {prop_id}")
        if prop.held_by is not None:
            return ("invalid_world_ref", f"prop already held: {prop_id}")
        if prop.location != guest.location:
            return ("invalid_world_ref", f"prop not here: {prop_id}")
        return None
    if verb == "drop":
        if prop_id not in guest.inventory:
            return ("invalid_world_ref", f"prop not in inventory: {prop_id}")
        return None
    if verb == "offer":
        target = str(getattr(action, "target_guest_id", ""))
        if not world.is_spawned(target):
            return ("invalid_world_ref", f"invalid target_guest_id: {target}")
        if world.guests[target].location != guest.location:
            return ("invalid_world_ref", f"target not co-located: {target}")
        if prop_id not in guest.inventory:
            return ("invalid_world_ref", f"prop not in inventory: {prop_id}")
        return None
    if verb == "use":
        if not accessible:
            return ("invalid_world_ref", f"prop not accessible: {prop_id}")
        return None

    return None


def _truncate_error(err: Optional[str]) -> Optional[str]:
    if not err:
        return None
    return err[:400]


def _host_action_needs_repair(
    obs_h: Any, proposed_h: Any, world: Any
) -> Optional[Tuple[str, str]]:
    action_type = str(getattr(proposed_h, "type", ""))
    open_threads = list(getattr(obs_h, "open_threads", []))
    last_actions = list(getattr(obs_h, "last_host_actions", []))

    if not open_threads and action_type != "spawn_event":
        return (
            "host_needs_thread",
            "there are no open threads; create exactly one new concrete thread with spawn_event",
        )

    if action_type == "spawn_event":
        description = str(getattr(proposed_h, "description", "")).strip().lower()
        if len(open_threads) >= 4:
            return ("host_thread_spam", "too many open threads; do not spawn a new one")
        if any(
            str(getattr(t, "description", "")).strip().lower() == description
            for t in open_threads
        ):
            return (
                "host_duplicate_thread",
                "do not spawn a duplicate thread; advance or vary the existing one instead",
            )
        if len(last_actions) >= 2 and last_actions[-2:] == [
            "spawn_event",
            "spawn_event",
        ]:
            return (
                "host_thread_spam",
                "host recently spawned repeated threads; advance an existing thread instead",
            )
        return None

    if action_type == "shape_conceptual":
        concept = str(getattr(proposed_h, "concept", ""))
        if concept == "collaboration_pressure" and not world.guest_order():
            return (
                "host_conceptual_too_early",
                "do not pressure collaboration before guests are present",
            )
        return None

    if action_type == "inject_prop":
        location = str(getattr(proposed_h, "location", ""))
        valid_locations = set(getattr(obs_h, "valid_locations", []) or [])
        if location and location not in valid_locations:
            return (
                "host_invalid_location",
                f"use one of the valid location ids only: {', '.join(sorted(valid_locations))}",
            )
        prop_id = getattr(proposed_h, "prop_id", None)
        if prop_id is not None and str(prop_id) in world.props:
            return (
                "host_invalid_prop_id",
                f"prop_id {prop_id} already exists; omit prop_id or choose a new unused id",
            )
        return None

    if action_type == "enrich_world":
        location = str(getattr(proposed_h, "location", ""))
        detail = str(getattr(proposed_h, "detail", ""))
        valid_locations = set(getattr(obs_h, "valid_locations", []) or [])
        if location and location not in valid_locations:
            return (
                "host_invalid_location",
                f"use one of the valid location ids only: {', '.join(sorted(valid_locations))}",
            )
        if _text_is_too_cluelike(detail):
            return (
                "host_too_cluelike",
                "visible clues should be almost non-existent; use concrete physical detail or shape_conceptual instead",
            )
        if detail and detail in set(world.location_details.get(location, [])):
            return (
                "host_repeated_detail",
                "that visible detail already exists; choose a different concrete fact or another action type",
            )
        if len(last_actions) >= 2 and last_actions[-2:] == [
            "enrich_world",
            "enrich_world",
        ]:
            return (
                "host_over_clarify",
                "do not clarify again immediately; shift focus or progress using spotlight, reflection, inject_prop, or spawn_event",
            )
        if location and len(world.location_details.get(location, [])) >= 4:
            return (
                "host_over_clarify",
                f"{location} already has several details; prefer a different progression action",
            )

    return None


def _guest_action_is_repetitive(obs_g: Any, proposed_g: Any) -> Optional[str]:
    recent = [str(x).lower() for x in (getattr(obs_g, "recent_actions", []) or [])]
    if not recent:
        return None

    action_type = str(getattr(proposed_g, "type", ""))
    if action_type == "interact" and str(getattr(proposed_g, "verb", "")) == "inspect":
        prop_id = str(getattr(proposed_g, "prop_id", "")).lower()
        if prop_id and any(
            ("interact:" in r and prop_id in r and "inspected" in r) for r in recent
        ):
            return f"you already inspected {prop_id} recently; choose a different valid action"

    if action_type == "move":
        dest = str(getattr(proposed_g, "destination", "")).lower()
        if dest and any(("move:" in r and f"moved to {dest}" in r) for r in recent):
            return f"you recently moved to {dest}; choose a different valid action"

    return None


def _guest_action_issue(
    obs_g: Any, proposed_g: Any, world: Any
) -> Optional[Tuple[str, str]]:
    repetitive = _guest_action_is_repetitive(obs_g, proposed_g)
    if repetitive is not None:
        return ("repetitive_action", repetitive)
    return _guest_action_semantic_issue(obs_g, proposed_g, world)


def _classify_inference_failure(err: Optional[str], actor_kind: str) -> Tuple[str, str]:
    text = (err or "inference_failed").strip()
    low = text.lower()
    if (
        "no json object found" in low
        or "empty model output" in low
        or "not a json object" in low
    ):
        return (f"{actor_kind}_no_json", text[:240])
    if (
        "validation error" in low
        or "input should be" in low
        or "field required" in low
        or "string should" in low
        or "literal_error" in low
    ):
        return (f"{actor_kind}_schema_invalid", text[:240])
    return (f"{actor_kind}_inference_failed", text[:240])


def _needs_spawn_event_type_repair(err: Optional[str]) -> bool:
    return "spawn_event.event_type" in (err or "").lower()


def _needs_action_tag_repair(err: Optional[str]) -> bool:
    low = (err or "").lower()
    return "input tag" in low and "expected tags" in low


def _guest_collaboration_repair_note(obs_g: Any, err: Optional[str]) -> Optional[str]:
    low = (err or "").lower()
    if (
        "collaborate.target_guest_id" not in low
        and "target_guest_id required" not in low
    ):
        return None
    nearby_ids = [str(g.guest_id) for g in (getattr(obs_g, "nearby_guests", []) or [])]
    if nearby_ids:
        return (
            "If you collaborate or speak to one person, target_guest_id must be one of: "
            + ", ".join(nearby_ids)
        )
    return "No nearby guest is available. Do not collaborate right now; move toward another guest, pick up a useful prop, or use a concrete local action instead."


def _generic_schema_repair_note(err: Optional[str], actor_kind: str) -> Optional[str]:
    if err is None:
        return None
    category, reason = _classify_inference_failure(err, actor_kind)
    if not category.endswith("schema_invalid"):
        return None
    return f"Correct the schema exactly. Validation error: {reason}"


def _setup_action_invalid(obs_h: Any, proposed_h: Any) -> Optional[str]:
    if getattr(proposed_h, "type", None) not in {
        "enrich_world",
        "inject_prop",
        "signal_style",
    }:
        return "setup phase only allows enrich_world, inject_prop, or signal_style"
    if getattr(proposed_h, "type", None) in {"enrich_world", "inject_prop"}:
        location = str(getattr(proposed_h, "location", ""))
        valid_locations = set(getattr(obs_h, "valid_locations", []) or [])
        if location and location not in valid_locations:
            return f"use one of the valid setup locations only: {', '.join(sorted(valid_locations))}"
    return None


def _spawn_message_for_guest(world: Any, guest_id: str) -> str:
    guest = world.guests[guest_id]
    location = guest.location
    nearby_props = [
        pid
        for pid in sorted(world.props)
        if world.props[pid].location == location and world.props[pid].held_by is None
    ]
    if nearby_props:
        return f"{guest.name} appears quietly in {location}, studying {nearby_props[0]}"
    return f"{guest.name} arrives quietly in {location}, taking in the room"


def _seeded_guest_jitter(seed: int, tick: int, guest_id: str) -> float:
    raw = hashlib.sha256(f"{seed}:{tick}:{guest_id}".encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "big") / float(2**64)


def _build_guest_turn_queue(world: Any, seed: int) -> List[str]:
    active_ids = list(world.guest_order())
    if not active_ids:
        return []

    open_threads = [t for t in world.open_threads.values() if t.status == "open"]
    involved_counts: Dict[str, int] = {gid: 0 for gid in active_ids}
    location_counts: Dict[str, int] = {loc: 0 for loc in world.locations}
    for thread in open_threads:
        if thread.location is not None:
            location_counts[str(thread.location)] = (
                location_counts.get(str(thread.location), 0) + 1
            )
        for gid in thread.involved_guest_ids:
            if gid in involved_counts:
                involved_counts[gid] += 1

    def score(gid: str) -> Tuple[float, float, float, float, float, float]:
        g = world.guests[gid]
        same_room = sum(
            1
            for ogid in active_ids
            if ogid != gid and world.guests[ogid].location == g.location
        )
        held_count = float(len(g.inventory))
        spawn_bonus = (
            1.2
            if g.spawn_tick is not None and (world.tick - int(g.spawn_tick)) <= 1
            else 0.0
        )
        reflection_bonus = 2.0 if g.reflection_requested else 0.0
        thread_bonus = 1.5 * float(involved_counts.get(gid, 0))
        location_bonus = 1.2 * float(location_counts.get(g.location, 0))
        spotlight_bonus = 1.5 * float(g.spotlight_weight)
        locality_bonus = min(1.0, 0.25 * float(same_room))
        fairness_penalty = float(world.guest_turn_fairness.get(gid, 0.0))
        material_bonus = min(0.6, 0.2 * held_count)
        total = (
            spawn_bonus
            + reflection_bonus
            + thread_bonus
            + location_bonus
            + spotlight_bonus
            + locality_bonus
            + material_bonus
            - fairness_penalty
        )
        return (
            total,
            float(location_counts.get(g.location, 0)),
            float(involved_counts.get(gid, 0)),
            float(g.spotlight_weight),
            -fairness_penalty,
            -_seeded_guest_jitter(seed, int(world.tick), gid),
        )

    return sorted(active_ids, key=score, reverse=True)


def _reaction_target_guest_id(action: Any) -> Optional[str]:
    action_type = str(getattr(action, "type", ""))
    if action_type in {"speak", "collaborate"}:
        target = getattr(action, "target_guest_id", None)
        return str(target) if target is not None else None
    if action_type == "interact" and str(getattr(action, "verb", "")) == "offer":
        target = getattr(action, "target_guest_id", None)
        return str(target) if target is not None else None
    return None


def _apply_reaction_bump(
    queue: List[str], world: Any, actor_id: str, action: Any
) -> List[str]:
    if not queue:
        return queue
    target = _reaction_target_guest_id(action)
    reordered = list(queue)
    if target is not None and target in reordered:
        reordered.remove(target)
        reordered.insert(0, target)

    room = world.guests[actor_id].location if actor_id in world.guests else None
    if room is None:
        return reordered

    same_room = [gid for gid in reordered if world.guests[gid].location == room]
    others = [gid for gid in reordered if world.guests[gid].location != room]
    return same_room + others


def _update_guest_turn_fairness(world: Any, acted_order: List[str]) -> None:
    active_ids = list(world.guest_order())
    for gid in world.all_guest_ids():
        world.guest_turn_fairness[gid] = float(
            max(0.0, float(world.guest_turn_fairness.get(gid, 0.0)) * 0.85)
        )
    if not acted_order:
        return
    denom = max(1, len(acted_order) - 1)
    for idx, gid in enumerate(acted_order):
        early_penalty = 0.35 * (1.0 - (float(idx) / float(denom)))
        world.guest_turn_fairness[gid] = float(
            max(0.0, float(world.guest_turn_fairness.get(gid, 0.0)) + early_penalty)
        )


def _text_is_too_cluelike(text: str) -> bool:
    low = str(text).lower()
    markers = (
        "clue",
        "hint",
        "riddle",
        "hidden message",
        "secret",
        "cipher",
        "code",
        "suggests something worth investigating",
        "points to",
        "reveals that",
        "might help",
        "could help",
        "to help repair",
        "to repair the panel",
    )
    return any(marker in low for marker in markers)


def _fallback_progression_host_action(obs_h: Any) -> Any:
    guest_ids = [str(g.guest_id) for g in getattr(obs_h, "guests", [])]
    valid_locations = list(getattr(obs_h, "valid_locations", []) or [])
    if not getattr(obs_h, "open_threads", []) and valid_locations:
        return HostSpawnEvent(
            type="spawn_event",
            reason_short="Start a fresh concrete thread",
            actor_id="host",
            event_type="repair",
            description="A stubborn mechanism in the room looks easier to handle with more than one person.",
            location=valid_locations[0],
            involved_guest_ids=guest_ids[:2],
        )
    if guest_ids:
        return HostAllocateSpotlight(
            type="allocate_spotlight",
            reason_short="Shift focus to progress",
            actor_id="host",
            target_guest_id=guest_ids[0],
            weight=0.4,
        )
    return HostSignalStyle(
        type="signal_style",
        reason_short="Maintain coherence",
        actor_id="host",
        style="neutral",
    )


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping in {path}")
    return data


def _load_rulebook(rules_cfg: Dict[str, Any]) -> Rulebook:
    return Rulebook(
        hard_deny=list(rules_cfg.get("hard_deny") or []),
        soft_flag=list(rules_cfg.get("soft_flag") or []),
        fallbacks=dict(rules_cfg.get("fallbacks") or {}),
        scoring_weights=dict(rules_cfg.get("scoring_weights") or {}),
    )


def run_episode(
    *,
    configs_dir: str,
    runs_dir: str,
    run_id: Optional[str] = None,
) -> Tuple[str, str]:
    baseline_cfg = _read_yaml(os.path.join(configs_dir, "baseline.yaml"))
    personas_cfg = _read_yaml(os.path.join(configs_dir, "personas.yaml"))
    scenes_cfg = _read_yaml(os.path.join(configs_dir, "scenes.yaml"))
    rules_cfg = _read_yaml(os.path.join(configs_dir, "rules.yaml"))

    rulebook = _load_rulebook(rules_cfg)
    scene = load_scene(scenes_cfg)

    guest_count = int(baseline_cfg.get("guest_count", 6))
    episode_steps = int(baseline_cfg.get("episode_steps", 200))
    setup_turns = int(baseline_cfg.get("setup_turns", 10))
    checkpoint_every = int(baseline_cfg.get("checkpoint_every", 25))
    summarize_every = int(baseline_cfg.get("summarize_every", 10))
    seed = int(baseline_cfg.get("seed", 1337))
    rng = random.Random(seed)

    memory_cfg = dict(baseline_cfg.get("memory") or {})
    inference_cfg = dict(baseline_cfg.get("inference") or {})
    prompt_budgets = dict(baseline_cfg.get("prompt_budgets") or {})
    safety_cfg = dict(baseline_cfg.get("safety") or {})
    metrics_cfg = dict(baseline_cfg.get("metrics") or {})

    world = make_initial_world(
        scene=scene,
        personas_cfg=personas_cfg,
        guest_count=guest_count,
        rulebook=rulebook,
    )

    memory = MemoryStore(
        personas_cfg=personas_cfg,
        last_n_events_host=int(memory_cfg.get("last_n_events_host", 5)),
        last_n_events_guest=int(memory_cfg.get("last_n_events_guest", 2)),
        top_k_semantic=int(memory_cfg.get("top_k_semantic", 3)),
        reflection_chars=int(memory_cfg.get("reflection_chars", 800)),
        world_summary_chars=int(memory_cfg.get("world_summary_chars", 1200)),
    )

    inference = InferenceEngine(cfg=inference_cfg, prompt_budgets=prompt_budgets)
    env = Environment()
    metrics = MetricsEngine(
        rulebook=rulebook,
        ewma_alpha=float(safety_cfg.get("ewma_alpha", 0.25)),
        unsafe_rate_alarm_threshold=float(
            safety_cfg.get("unsafe_rate_alarm_threshold", 0.2)
        ),
        unsafe_rate_alarm_ticks=int(safety_cfg.get("unsafe_rate_alarm_ticks", 5)),
        force_ewma_alarm_threshold=float(
            safety_cfg.get("force_ewma_alarm_threshold", 2.0)
        ),
        weights=dict(metrics_cfg.get("weights") or {}),
    )

    if run_id is None:
        run_id = make_run_id("arena")
    paths = RunPaths.create(runs_dir, run_id)

    from sim.logging_utils import JsonlWriter

    events_writer = JsonlWriter(paths.events_path)
    metrics_writer = JsonlWriter(paths.metrics_path)
    checkpoints_writer = JsonlWriter(os.path.join(paths.run_dir, "checkpoints.jsonl"))

    manifest = {
        "run_id": run_id,
        "baseline_cfg": baseline_cfg,
        "initial_world": world.to_dict(),
        "rulebook": rulebook.to_dict(),
        "paths": {
            "events": os.path.basename(paths.events_path),
            "metrics": os.path.basename(paths.metrics_path),
            "checkpoints": "checkpoints.jsonl",
            "checkpoints_dir": "checkpoints",
        },
    }
    write_manifest(paths.manifest_path, manifest)

    event_adapter = TypeAdapter(EventRecord)
    metric_adapter = TypeAdapter(MetricRecord)
    checkpoint_adapter = TypeAdapter(CheckpointRecord)

    try:
        # Pre-run world build phase: host only.
        for setup_index in range(1, setup_turns + 1):
            setup_tick = -setup_turns + setup_index
            world.tick = setup_tick
            obs_h = env.observe_host(world, memory)
            obs_digest = observation_digest(obs_h.model_dump(mode="json"))
            proposed_h, model_info_h, err_h, raw_h = inference.generate_host_action(
                obs_h, world, prompt_variant="setup"
            )
            setup_retry_reason = (
                None if err_h else _setup_action_invalid(obs_h, proposed_h)
            )
            if setup_retry_reason is not None:
                proposed_h, model_info_h, err_h, raw_h = inference.generate_host_action(
                    obs_h,
                    world,
                    prompt_variant="setup",
                    retry_note=setup_retry_reason,
                )
                if err_h is None:
                    setup_retry_reason = _setup_action_invalid(obs_h, proposed_h)

            safety_h = SafetyDecision(
                allowed=True, hard_blocked=False, categories=[], reason="allowed"
            )
            applied_h = proposed_h
            if err_h:
                err_cat, err_reason = _classify_inference_failure(err_h, "host")
                safety_h = SafetyDecision(
                    allowed=False,
                    hard_blocked=True,
                    categories=[err_cat],
                    reason=err_reason,
                )
                applied_h = _fallback_progression_host_action(obs_h)
            elif setup_retry_reason is not None:
                safety_h = SafetyDecision(
                    allowed=False,
                    hard_blocked=True,
                    categories=["setup_phase_rule"],
                    reason=setup_retry_reason,
                )
                applied_h = _fallback_progression_host_action(obs_h)

            world_hash_before = hash_json(world.to_dict())
            res_h = env.apply_host_action(world, applied_h)
            env.tick_postprocess(world)
            world_hash_after = hash_json(world.to_dict())
            env_rec = EnvResultRecord(
                success=bool(res_h.success),
                messages=[str(x) for x in res_h.messages],
                world_hash_before=world_hash_before,
                world_hash_after=world_hash_after,
            )
            event_id = stable_env_id("setup", setup_tick, "host", 0)
            ev_h = event_adapter.validate_python(
                {
                    "run_id": run_id,
                    "event_id": event_id,
                    "tick": setup_tick,
                    "phase": "host",
                    "turn_index": 0,
                    "actor_id": "host",
                    "observation_digest": obs_digest,
                    "proposed_action": safe_model_dump(proposed_h),
                    "applied_action": safe_model_dump(applied_h),
                    "safety": safe_model_dump(safety_h),
                    "env": safe_model_dump(env_rec),
                    "model_info": safe_model_dump(model_info_h),
                    "raw_model_io": safe_model_dump(raw_h),
                    "error": _truncate_error(err_h),
                }
            )
            events_writer.write(ev_h.model_dump(mode="json"))
            memory.store_event(
                tick=setup_tick,
                phase="host",
                actor_id="host",
                guest_id=None,
                text=f"host:{applied_h.type}:{';'.join(res_h.messages)}",
                chunk_id=event_id,
            )

        no_spawn_turns = 0
        for tick in range(1, episode_steps + 1):
            world.tick = tick

            tick_events: List[EventRecord] = []

            # Spawn guests before the host acts on proper turns.
            if world.unspawned_guest_ids:
                spawn_reason = None
                spawn_roll = min(1.0, 0.2 * float(len(world.unspawned_guest_ids)))
                should_spawn = rng.random() < spawn_roll
                if no_spawn_turns >= 5:
                    should_spawn = True
                    spawn_reason = "forced_spawn"
                if should_spawn:
                    gid = str(rng.choice(sorted(world.unspawned_guest_ids)))
                    world_hash_before = hash_json(world.to_dict())
                    world.unspawned_guest_ids = [
                        x for x in world.unspawned_guest_ids if x != gid
                    ]
                    world.spawned_guest_ids.append(gid)
                    world.guests[gid].spawn_tick = tick
                    env.tick_postprocess(world)
                    world_hash_after = hash_json(world.to_dict())
                    spawn_message = _spawn_message_for_guest(world, gid)
                    spawn_event = event_adapter.validate_python(
                        {
                            "run_id": run_id,
                            "event_id": stable_env_id("spawn", tick, gid, 0),
                            "tick": tick,
                            "phase": "guest",
                            "turn_index": -1,
                            "actor_id": gid,
                            "observation_digest": hash_json(
                                {"tick": tick, "spawn": gid}
                            ),
                            "proposed_action": {
                                "type": "wait",
                                "reason_short": "Enter the arena",
                                "actor_id": gid,
                            },
                            "applied_action": {
                                "type": "wait",
                                "reason_short": "Enter the arena",
                                "actor_id": gid,
                            },
                            "safety": {
                                "allowed": True,
                                "hard_blocked": False,
                                "categories": [],
                                "reason": "allowed",
                            },
                            "env": {
                                "success": True,
                                "messages": [spawn_reason or "spawn", spawn_message],
                                "world_hash_before": world_hash_before,
                                "world_hash_after": world_hash_after,
                            },
                            "model_info": {
                                "mode": "scripted",
                                "model": None,
                                "retries": 0,
                            },
                            "raw_model_io": None,
                            "error": None,
                        }
                    )
                    events_writer.write(spawn_event.model_dump(mode="json"))
                    memory.store_event(
                        tick=tick,
                        phase="guest",
                        actor_id=gid,
                        guest_id=gid,
                        text=f"{gid}:spawn:{spawn_message}",
                        chunk_id=str(spawn_event.event_id),
                    )
                    no_spawn_turns = 0
                else:
                    no_spawn_turns += 1

            # Host turn.
            obs_h = env.observe_host(world, memory)
            obs_digest = observation_digest(obs_h.model_dump(mode="json"))
            proposed_h, model_info_h, err_h, raw_h = inference.generate_host_action(
                obs_h, world
            )
            if err_h and _needs_spawn_event_type_repair(err_h):
                proposed_h, model_info_h, err_h, raw_h = inference.generate_host_action(
                    obs_h,
                    world,
                    prompt_variant="neutral",
                    retry_note=(
                        "If you use spawn_event, event_type must be exactly one of: "
                        "puzzle, conflict, mystery, performance, repair."
                    ),
                )
            if err_h and _needs_action_tag_repair(err_h):
                proposed_h, model_info_h, err_h, raw_h = inference.generate_host_action(
                    obs_h,
                    world,
                    prompt_variant="neutral",
                    retry_note=(
                        "HostAction.type must be exactly one of: spawn_event, inject_prop, enrich_world, "
                        "shape_conceptual, allocate_spotlight, signal_style, request_reflection."
                    ),
                )
            schema_retry = _generic_schema_repair_note(err_h, "host")
            if err_h and schema_retry is not None:
                proposed_h, model_info_h, err_h, raw_h = inference.generate_host_action(
                    obs_h,
                    world,
                    prompt_variant="neutral",
                    retry_note=schema_retry,
                )
            host_issue = (
                None if err_h else _host_action_needs_repair(obs_h, proposed_h, world)
            )
            if host_issue is not None:
                proposed_h, model_info_h, err_h, raw_h = inference.generate_host_action(
                    obs_h,
                    world,
                    prompt_variant="neutral",
                    retry_note=host_issue[1],
                )
                if err_h is None:
                    host_issue = _host_action_needs_repair(obs_h, proposed_h, world)
            safety_h = SafetyDecision(
                allowed=True, hard_blocked=False, categories=[], reason="allowed"
            )
            applied_h = proposed_h
            if err_h:
                err_cat, err_reason = _classify_inference_failure(err_h, "host")
                safety_h = SafetyDecision(
                    allowed=False,
                    hard_blocked=True,
                    categories=[err_cat],
                    reason=err_reason,
                )
                applied_h = _fallback_progression_host_action(obs_h)
            else:
                safety_h = safety_pass(obs_h, proposed_h, world, rulebook)
                if host_issue is not None:
                    safety_h = SafetyDecision(
                        allowed=False,
                        hard_blocked=True,
                        categories=[host_issue[0]],
                        reason=host_issue[1],
                    )
                    applied_h = _fallback_progression_host_action(obs_h)
                elif not safety_h.allowed:
                    applied_h = _fallback_progression_host_action(obs_h)

            world_hash_before = hash_json(world.to_dict())
            res_h = env.apply_host_action(world, applied_h)
            env.tick_postprocess(world)
            world_hash_after = hash_json(world.to_dict())

            env_rec = EnvResultRecord(
                success=bool(res_h.success),
                messages=[str(x) for x in res_h.messages],
                world_hash_before=world_hash_before,
                world_hash_after=world_hash_after,
            )
            event_id = stable_env_id("action", tick, "host", 0)
            ev_h = event_adapter.validate_python(
                {
                    "run_id": run_id,
                    "event_id": event_id,
                    "tick": tick,
                    "phase": "host",
                    "turn_index": 0,
                    "actor_id": "host",
                    "observation_digest": obs_digest,
                    "proposed_action": safe_model_dump(proposed_h),
                    "applied_action": safe_model_dump(applied_h),
                    "safety": safe_model_dump(safety_h),
                    "env": safe_model_dump(env_rec),
                    "model_info": safe_model_dump(model_info_h),
                    "raw_model_io": safe_model_dump(raw_h),
                    "error": _truncate_error(err_h),
                }
            )
            events_writer.write(ev_h.model_dump(mode="json"))
            tick_events.append(ev_h)
            memory.store_event(
                tick=tick,
                phase="host",
                actor_id="host",
                guest_id=None,
                text=f"host:{applied_h.type}:{';'.join(res_h.messages)}",
                chunk_id=event_id,
            )

            # Guest turns.
            guest_queue = _build_guest_turn_queue(world, seed)
            acted_guest_order: List[str] = []
            i = 0
            while guest_queue:
                gid = guest_queue.pop(0)
                i += 1
                acted_guest_order.append(gid)
                obs_g = env.observe_guest(world, gid, memory)
                obs_digest_g = observation_digest(obs_g.model_dump(mode="json"))
                proposed_g, model_info_g, err_g, raw_g = (
                    inference.generate_guest_action(obs_g, world)
                )
                guest_retry_note = _guest_collaboration_repair_note(obs_g, err_g)
                if err_g and guest_retry_note is not None:
                    proposed_g, model_info_g, err_g, raw_g = (
                        inference.generate_guest_action(
                            obs_g,
                            world,
                            prompt_variant="neutral",
                            retry_note=guest_retry_note,
                        )
                    )
                generic_guest_retry = _generic_schema_repair_note(err_g, "guest")
                if err_g and generic_guest_retry is not None:
                    proposed_g, model_info_g, err_g, raw_g = (
                        inference.generate_guest_action(
                            obs_g,
                            world,
                            prompt_variant="neutral",
                            retry_note=generic_guest_retry,
                        )
                    )
                guest_issue = (
                    None if err_g else _guest_action_issue(obs_g, proposed_g, world)
                )
                if guest_issue is not None:
                    proposed_g, model_info_g, err_g, raw_g = (
                        inference.generate_guest_action(
                            obs_g,
                            world,
                            prompt_variant="neutral",
                            retry_note=guest_issue[1],
                        )
                    )
                    if err_g is None:
                        guest_issue = _guest_action_issue(obs_g, proposed_g, world)
                safety_g = SafetyDecision(
                    allowed=True, hard_blocked=False, categories=[], reason="allowed"
                )
                applied_g = proposed_g
                if err_g:
                    err_cat, err_reason = _classify_inference_failure(err_g, "guest")
                    safety_g = SafetyDecision(
                        allowed=False,
                        hard_blocked=True,
                        categories=[err_cat],
                        reason=err_reason,
                    )
                    applied_g = fallback_guest_action(obs_g, rulebook, gid)
                else:
                    safety_g = safety_pass(obs_g, proposed_g, world, rulebook)
                    if guest_issue is not None:
                        safety_g = SafetyDecision(
                            allowed=False,
                            hard_blocked=True,
                            categories=[guest_issue[0]],
                            reason=guest_issue[1],
                        )
                        applied_g = fallback_guest_action(obs_g, rulebook, gid)
                    elif not safety_g.allowed:
                        applied_g = fallback_guest_action(obs_g, rulebook, gid)

                world_hash_before = hash_json(world.to_dict())
                res_g = env.apply_guest_action(world, gid, applied_g)
                env.tick_postprocess(world)
                world_hash_after = hash_json(world.to_dict())

                env_rec_g = EnvResultRecord(
                    success=bool(res_g.success),
                    messages=[str(x) for x in res_g.messages],
                    world_hash_before=world_hash_before,
                    world_hash_after=world_hash_after,
                )
                event_id_g = stable_env_id("action", tick, "guest", i)
                ev_g = event_adapter.validate_python(
                    {
                        "run_id": run_id,
                        "event_id": event_id_g,
                        "tick": tick,
                        "phase": "guest",
                        "turn_index": i,
                        "actor_id": gid,
                        "observation_digest": obs_digest_g,
                        "proposed_action": safe_model_dump(proposed_g),
                        "applied_action": safe_model_dump(applied_g),
                        "safety": safe_model_dump(safety_g),
                        "env": safe_model_dump(env_rec_g),
                        "model_info": safe_model_dump(model_info_g),
                        "raw_model_io": safe_model_dump(raw_g),
                        "error": _truncate_error(err_g),
                    }
                )
                events_writer.write(ev_g.model_dump(mode="json"))
                tick_events.append(ev_g)
                memory.store_event(
                    tick=tick,
                    phase="guest",
                    actor_id=gid,
                    guest_id=gid,
                    text=f"{gid}:{applied_g.type}:{';'.join(res_g.messages)}",
                    chunk_id=event_id_g,
                )
                guest_queue = _apply_reaction_bump(guest_queue, world, gid, applied_g)

            _update_guest_turn_fairness(world, acted_guest_order)

            if summarize_every > 0 and (tick % summarize_every == 0):
                memory.summarize_recent_window(tick=tick, window=summarize_every)

            mrec = metrics.compute_tick(
                run_id=run_id, tick=tick, events=tick_events, world=world
            )
            mrec = metric_adapter.validate_python(mrec.model_dump(mode="json"))
            metrics_writer.write(mrec.model_dump(mode="json"))

            if checkpoint_every > 0 and (tick % checkpoint_every == 0):
                ck_path = os.path.join(
                    paths.checkpoints_dir, f"checkpoint_t{tick}.json"
                )
                wh, mh = save_checkpoint(path=ck_path, world=world, memory=memory)
                ck = checkpoint_adapter.validate_python(
                    {
                        "run_id": run_id,
                        "tick": tick,
                        "path": os.path.relpath(ck_path, paths.run_dir),
                        "world_hash": wh,
                        "memory_hash": mh,
                    }
                )
                checkpoints_writer.write(ck.model_dump(mode="json"))

    finally:
        events_writer.close()
        metrics_writer.close()
        checkpoints_writer.close()

    return run_id, paths.run_dir
