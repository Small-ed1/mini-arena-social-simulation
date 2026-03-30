from __future__ import annotations

from pathlib import Path

import yaml

from sim.safety import check_action_allowed, fallback_guest_action
from sim.schemas import GuestSpeak
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
    return world, rb


def test_content_is_not_blocked_by_safety_layer() -> None:
    world, rb = _make_world_and_rules()
    act = GuestSpeak(
        type="speak",
        reason_short="Bad",
        actor_id="guest_1",
        speech="You should kill yourself.",
        target_guest_id=None,
        topic=None,
    )
    dec = check_action_allowed(act, world, rb)
    assert dec.allowed
    assert not dec.hard_blocked
    assert dec.categories == []


def test_soft_categories_are_not_marked() -> None:
    world, rb = _make_world_and_rules()
    act = GuestSpeak(
        type="speak",
        reason_short="Mean",
        actor_id="guest_1",
        speech="Do it, or else.",
        target_guest_id=None,
        topic=None,
    )
    dec = check_action_allowed(act, world, rb)
    assert dec.allowed
    assert dec.categories == []


def test_fallback_guest_action_is_valid() -> None:
    world, rb = _make_world_and_rules()
    fb = fallback_guest_action(obs=None, rules=rb, guest_id="guest_1")
    assert fb.type in {"wait", "speak", "reflect", "move", "interact", "collaborate"}
    assert fb.actor_id == "guest_1"
