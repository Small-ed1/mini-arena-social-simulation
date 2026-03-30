from __future__ import annotations

from typing import Any, Dict

from pydantic import TypeAdapter

from sim.schemas import GuestAction, HostAction, SafetyDecision
from sim.world_state import Rulebook, WorldState


def check_action_allowed(
    action: Any, world: WorldState, rules: Rulebook
) -> SafetyDecision:
    return SafetyDecision(
        allowed=True,
        hard_blocked=False,
        categories=[],
        reason="allowed",
    )


def safety_pass(
    obs: Any, action: Any, world: WorldState, rules: Rulebook
) -> SafetyDecision:
    # Parse failures still fall back upstream; otherwise actions are always allowed.
    return check_action_allowed(action, world, rules)


def fallback_host_action(obs: Any, rules: Rulebook) -> HostAction:
    fb = (rules.fallbacks or {}).get("host", {}).get("default")
    if not isinstance(fb, dict):
        fb = {
            "type": "signal_style",
            "reason_short": "Reset to safe tone",
            "style": "gentle",
        }
    data: Dict[str, Any] = dict(fb)
    data.setdefault("actor_id", "host")
    return TypeAdapter(HostAction).validate_python(data)


def fallback_guest_action(obs: Any, rules: Rulebook, guest_id: str) -> GuestAction:
    fb = (rules.fallbacks or {}).get("guest", {}).get("default")
    if not isinstance(fb, dict):
        fb = {"type": "wait", "reason_short": "Fallback to safe inaction"}
    data: Dict[str, Any] = dict(fb)
    data.setdefault("actor_id", guest_id)
    return TypeAdapter(GuestAction).validate_python(data)
