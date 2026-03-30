from __future__ import annotations

import os
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
    MetricRecord,
    ModelInfo,
    SafetyDecision,
    safe_model_dump,
)
from sim.safety import fallback_guest_action, fallback_host_action, safety_pass
from sim.world_state import Rulebook, load_scene, make_initial_world


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
    checkpoint_every = int(baseline_cfg.get("checkpoint_every", 25))
    summarize_every = int(baseline_cfg.get("summarize_every", 10))

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
        for tick in range(1, episode_steps + 1):
            world.tick = tick

            tick_events: List[EventRecord] = []

            # Host turn.
            obs_h = env.observe_host(world, memory)
            obs_digest = observation_digest(obs_h.model_dump(mode="json"))
            proposed_h, model_info_h, err_h = inference.generate_host_action(
                obs_h, world
            )
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
                if not safety_h.allowed:
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
                    "error": err_h,
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
                proposed_g, model_info_g, err_g = inference.generate_guest_action(
                    obs_g, world
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
                    if not safety_g.allowed:
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
                        "error": err_g,
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
