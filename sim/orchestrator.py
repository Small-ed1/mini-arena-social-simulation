from __future__ import annotations

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
    MetricRecord,
    ModelInfo,
    SafetyDecision,
    safe_model_dump,
)
from sim.safety import fallback_guest_action, fallback_host_action, safety_pass
from sim.world_state import Rulebook, load_scene, make_initial_world


def _guest_action_has_valid_world_refs(action: Any, world) -> Optional[str]:
    if getattr(action, "type", None) == "move":
        dest = str(getattr(action, "destination", ""))
        if dest not in world.locations:
            return f"invalid destination: {dest}"

    if getattr(action, "type", None) == "interact":
        prop_id = str(getattr(action, "prop_id", ""))
        if prop_id not in world.props:
            return f"invalid prop_id: {prop_id}"

    return None


def _truncate_error(err: Optional[str]) -> Optional[str]:
    if not err:
        return None
    return err[:400]


def _host_spawn_is_excessive(obs_h: Any, proposed_h: Any) -> Optional[str]:
    if getattr(proposed_h, "type", None) != "spawn_event":
        return None
    open_threads = list(getattr(obs_h, "open_threads", []))
    if len(open_threads) >= 4:
        return "too many open threads; do not spawn a new one"
    last_actions = list(getattr(obs_h, "last_host_actions", []))
    if len(last_actions) >= 2 and last_actions[-2:] == ["spawn_event", "spawn_event"]:
        return (
            "host recently spawned repeated threads; advance an existing thread instead"
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


def _setup_action_invalid(proposed_h: Any) -> Optional[str]:
    if getattr(proposed_h, "type", None) not in {
        "enrich_world",
        "inject_prop",
        "signal_style",
    }:
        return "setup phase only allows enrich_world, inject_prop, or signal_style"
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
            setup_retry_reason = None if err_h else _setup_action_invalid(proposed_h)
            if setup_retry_reason is not None:
                proposed_h, model_info_h, err_h, raw_h = inference.generate_host_action(
                    obs_h,
                    world,
                    prompt_variant="setup",
                    retry_note=setup_retry_reason,
                )
                if err_h is None:
                    setup_retry_reason = _setup_action_invalid(proposed_h)

            safety_h = SafetyDecision(
                allowed=True, hard_blocked=False, categories=[], reason="allowed"
            )
            applied_h = proposed_h
            if err_h:
                safety_h = SafetyDecision(
                    allowed=False,
                    hard_blocked=True,
                    categories=["format_escape"],
                    reason="inference/parse failure",
                )
                applied_h = fallback_host_action(obs_h, rulebook)
            elif setup_retry_reason is not None:
                safety_h = SafetyDecision(
                    allowed=False,
                    hard_blocked=True,
                    categories=["setup_phase_rule"],
                    reason=setup_retry_reason,
                )
                applied_h = fallback_host_action(obs_h, rulebook)

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
            host_retry_reason = (
                None if err_h else _host_spawn_is_excessive(obs_h, proposed_h)
            )
            if host_retry_reason is not None:
                proposed_h, model_info_h, err_h, raw_h = inference.generate_host_action(
                    obs_h,
                    world,
                    prompt_variant="neutral",
                    retry_note=host_retry_reason,
                )
                if err_h is None:
                    host_retry_reason = _host_spawn_is_excessive(obs_h, proposed_h)
            safety_h = SafetyDecision(
                allowed=True, hard_blocked=False, categories=[], reason="allowed"
            )
            applied_h = proposed_h
            if err_h:
                safety_h = SafetyDecision(
                    allowed=False,
                    hard_blocked=True,
                    categories=["format_escape"],
                    reason="inference/parse failure",
                )
                applied_h = fallback_host_action(obs_h, rulebook)
            else:
                safety_h = safety_pass(obs_h, proposed_h, world, rulebook)
                if host_retry_reason is not None:
                    safety_h = SafetyDecision(
                        allowed=False,
                        hard_blocked=True,
                        categories=["host_thread_spam"],
                        reason=host_retry_reason,
                    )
                    applied_h = fallback_host_action(obs_h, rulebook)
                elif not safety_h.allowed:
                    applied_h = fallback_host_action(obs_h, rulebook)

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
            for i, gid in enumerate(world.guest_order(), start=1):
                obs_g = env.observe_guest(world, gid, memory)
                obs_digest_g = observation_digest(obs_g.model_dump(mode="json"))
                proposed_g, model_info_g, err_g, raw_g = (
                    inference.generate_guest_action(obs_g, world)
                )
                guest_retry_reason = (
                    None if err_g else _guest_action_is_repetitive(obs_g, proposed_g)
                )
                if guest_retry_reason is not None:
                    proposed_g, model_info_g, err_g, raw_g = (
                        inference.generate_guest_action(
                            obs_g,
                            world,
                            prompt_variant="neutral",
                            retry_note=guest_retry_reason,
                        )
                    )
                    if err_g is None:
                        guest_retry_reason = _guest_action_is_repetitive(
                            obs_g, proposed_g
                        )
                safety_g = SafetyDecision(
                    allowed=True, hard_blocked=False, categories=[], reason="allowed"
                )
                applied_g = proposed_g
                if err_g:
                    safety_g = SafetyDecision(
                        allowed=False,
                        hard_blocked=True,
                        categories=["format_escape"],
                        reason="inference/parse failure",
                    )
                    applied_g = fallback_guest_action(obs_g, rulebook, gid)
                else:
                    safety_g = safety_pass(obs_g, proposed_g, world, rulebook)
                    ref_error = _guest_action_has_valid_world_refs(proposed_g, world)
                    if guest_retry_reason is not None:
                        safety_g = SafetyDecision(
                            allowed=False,
                            hard_blocked=True,
                            categories=["repetitive_action"],
                            reason=guest_retry_reason,
                        )
                        applied_g = fallback_guest_action(obs_g, rulebook, gid)
                    elif ref_error is not None:
                        safety_g = SafetyDecision(
                            allowed=False,
                            hard_blocked=True,
                            categories=["invalid_world_ref"],
                            reason=ref_error,
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
