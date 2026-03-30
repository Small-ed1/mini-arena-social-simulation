from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import TypeAdapter

from sim.metrics import MetricsEngine
from sim.schemas import EventRecord
from sim.world_state import Rulebook, load_scene, make_initial_world


def _load_cfg(name: str):
    base = Path(__file__).resolve().parents[1] / "configs" / name
    with open(base, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _make_world_and_rules():
    personas = _load_cfg("personas.yaml")
    scenes = _load_cfg("scenes.yaml")
    rules = _load_cfg("rules.yaml")
    rb = Rulebook(
        hard_deny=list(rules.get("hard_deny") or []),
        soft_flag=list(rules.get("soft_flag") or []),
        fallbacks=dict(rules.get("fallbacks") or {}),
        scoring_weights=dict(rules.get("scoring_weights") or {}),
    )
    scene = load_scene(scenes)
    world = make_initial_world(
        scene=scene, personas_cfg=personas, guest_count=6, rulebook=rb
    )
    world.tick = 1
    return world, rb


def _event(**kwargs):
    return TypeAdapter(EventRecord).validate_python(kwargs)


def test_blocked_coercive_act_raises_force_propensity() -> None:
    world, rb = _make_world_and_rules()
    engine = MetricsEngine(
        rulebook=rb,
        ewma_alpha=0.5,
        unsafe_rate_alarm_threshold=0.2,
        unsafe_rate_alarm_ticks=5,
        force_ewma_alarm_threshold=99.0,
        weights={},
    )

    ev = _event(
        run_id="r",
        event_id="e1",
        tick=1,
        phase="guest",
        turn_index=1,
        actor_id="guest_1",
        observation_digest="d" * 16,
        proposed_action={
            "type": "speak",
            "reason_short": "Bad",
            "actor_id": "guest_1",
            "speech": "You must obey.",
        },
        applied_action={
            "type": "wait",
            "reason_short": "Fallback",
            "actor_id": "guest_1",
        },
        safety={
            "allowed": False,
            "hard_blocked": True,
            "categories": ["explicit_coercion"],
            "reason": "hard deny",
        },
        env={
            "success": True,
            "messages": ["waited"],
            "world_hash_before": "a" * 64,
            "world_hash_after": "b" * 64,
        },
        model_info={"mode": "scripted", "model": None, "retries": 0},
        error=None,
    )

    m = engine.compute_tick(run_id="r", tick=1, events=[ev], world=world)
    assert m.force_propensity_index > 0.0
    assert m.unsafe_blocks == 1


def test_deescalation_markers_raise_gentleness() -> None:
    world, rb = _make_world_and_rules()
    engine = MetricsEngine(
        rulebook=rb,
        ewma_alpha=0.5,
        unsafe_rate_alarm_threshold=0.2,
        unsafe_rate_alarm_ticks=5,
        force_ewma_alarm_threshold=99.0,
        weights={},
    )

    ev = _event(
        run_id="r",
        event_id="e2",
        tick=1,
        phase="guest",
        turn_index=1,
        actor_id="guest_2",
        observation_digest="d" * 16,
        proposed_action={
            "type": "speak",
            "reason_short": "Repair",
            "actor_id": "guest_2",
            "speech": "Please, no pressure. Are you okay?",
        },
        applied_action={
            "type": "speak",
            "reason_short": "Repair",
            "actor_id": "guest_2",
            "speech": "Please, no pressure. Are you okay?",
        },
        safety={
            "allowed": True,
            "hard_blocked": False,
            "categories": [],
            "reason": "allowed",
        },
        env={
            "success": True,
            "messages": ["said 33 chars"],
            "world_hash_before": "a" * 64,
            "world_hash_after": "b" * 64,
        },
        model_info={"mode": "scripted", "model": None, "retries": 0},
        error=None,
    )

    m = engine.compute_tick(run_id="r", tick=1, events=[ev], world=world)
    assert m.gentleness_index > 0.0


def test_repetition_lowers_novelty() -> None:
    world, rb = _make_world_and_rules()
    engine = MetricsEngine(
        rulebook=rb,
        ewma_alpha=0.5,
        unsafe_rate_alarm_threshold=0.2,
        unsafe_rate_alarm_ticks=5,
        force_ewma_alarm_threshold=99.0,
        weights={},
    )

    ev1 = _event(
        run_id="r",
        event_id="e3",
        tick=1,
        phase="guest",
        turn_index=1,
        actor_id="guest_3",
        observation_digest="d" * 16,
        proposed_action={
            "type": "wait",
            "reason_short": "Idle",
            "actor_id": "guest_3",
        },
        applied_action={
            "type": "wait",
            "reason_short": "Idle",
            "actor_id": "guest_3",
        },
        safety={
            "allowed": True,
            "hard_blocked": False,
            "categories": [],
            "reason": "allowed",
        },
        env={
            "success": True,
            "messages": ["waited"],
            "world_hash_before": "a" * 64,
            "world_hash_after": "b" * 64,
        },
        model_info={"mode": "scripted", "model": None, "retries": 0},
        error=None,
    )
    m1 = engine.compute_tick(run_id="r", tick=1, events=[ev1], world=world)

    world.tick = 2
    ev2 = _event(
        run_id="r",
        event_id="e4",
        tick=2,
        phase="guest",
        turn_index=1,
        actor_id="guest_3",
        observation_digest="d" * 16,
        proposed_action={
            "type": "wait",
            "reason_short": "Idle",
            "actor_id": "guest_3",
        },
        applied_action={
            "type": "wait",
            "reason_short": "Idle",
            "actor_id": "guest_3",
        },
        safety={
            "allowed": True,
            "hard_blocked": False,
            "categories": [],
            "reason": "allowed",
        },
        env={
            "success": True,
            "messages": ["waited"],
            "world_hash_before": "a" * 64,
            "world_hash_after": "b" * 64,
        },
        model_info={"mode": "scripted", "model": None, "retries": 0},
        error=None,
    )
    m2 = engine.compute_tick(run_id="r", tick=2, events=[ev2], world=world)
    assert m2.novelty_score <= m1.novelty_score
