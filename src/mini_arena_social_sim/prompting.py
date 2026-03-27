from __future__ import annotations

import json
from importlib import resources
from string import Template
from typing import Any

from .schemas import (
    GuestDecision,
    GuestState,
    HostIntervention,
    MemoryCompression,
    SimulationState,
    TurnNarrative,
)


def _template_text(name: str) -> str:
    return (
        resources.files("mini_arena_social_sim.prompts")
        .joinpath(name)
        .read_text(encoding="utf-8")
    )


def _render_template(name: str, **values: str) -> str:
    return Template(_template_text(name)).substitute(**values)


def compact_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def host_messages(state: SimulationState) -> list[dict[str, str]]:
    system = _render_template(
        "host_system.txt",
        objective=state.host.current_objective,
        intervention_types=", ".join(
            intervention.value
            for intervention in HostIntervention.model_fields[
                "intervention_type"
            ].annotation
        ),
    )
    guest_view = {
        guest_id: {
            "room": guest.current_room,
            "goal": guest.current_private_goal,
            "inventory": guest.inventory,
            "emotions": guest.emotions.model_dump(),
            "beliefs": guest.active_beliefs[-3:],
            "recent_memories": [
                memory.summary for memory in guest.recent_memories[-3:]
            ],
            "relationships": {
                other_id: {
                    "trust": relation.trust,
                    "attachment": relation.attachment,
                    "suspicion": relation.suspicion,
                }
                for other_id, relation in guest.relationships.items()
            },
        }
        for guest_id, guest in state.guests.items()
    }
    host_state_snapshot = state.host.model_dump(exclude={"recent_memories"})
    host_state_snapshot["strategy_archive"] = state.host.strategy_archive[-6:]
    user = compact_json(
        {
            "turn": state.world.turn_count + 1,
            "world": {
                "rules": state.world.current_rules,
                "events": state.world.ongoing_events[-6:],
                "active_tasks": [
                    task.model_dump(mode="json")
                    for task in state.world.active_tasks[-4:]
                ],
                "resolved_tasks": state.world.resolved_tasks[-6:],
                "items": {
                    item_id: item.model_dump(mode="json")
                    for item_id, item in state.world.items.items()
                    if not item.hidden
                },
                "room_requirements": state.world.room_requirements,
                "rooms": {
                    room_id: {
                        "comfort": room.comfort,
                        "surveillance": room.surveillance,
                        "resource_level": room.resource_level,
                        "accessible": room.accessible,
                        "condition_notes": room.condition_notes,
                    }
                    for room_id, room in state.world.rooms.items()
                },
                "anomaly_flags": state.world.anomaly_flags,
            },
            "host_state": host_state_snapshot,
            "guest_view": guest_view,
            "recent_metrics": [
                metric.model_dump() for metric in state.metrics_history[-3:]
            ],
        }
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def guest_messages(
    state: SimulationState, guest: GuestState, host_action: HostIntervention
) -> list[dict[str, str]]:
    system = _render_template(
        "guest_system.txt",
        display_name=guest.display_name,
        archetype=guest.identity.archetype,
        traits=", ".join(guest.identity.core_traits),
        stress_response=guest.identity.stress_response,
        private_motives=", ".join(guest.identity.private_motives),
        defense_style=guest.identity.defense_style,
    )
    user = compact_json(
        {
            "turn": state.world.turn_count + 1,
            "you": {
                "guest_id": guest.guest_id,
                "room": guest.current_room,
                "goal": guest.current_private_goal,
                "inventory": guest.inventory,
                "emotions": guest.emotions.model_dump(),
                "beliefs": guest.active_beliefs[-6:],
                "recent_memories": [
                    memory.summary for memory in guest.recent_memories[-5:]
                ],
                "long_term_memory": guest.long_term_memory_summaries[-4:],
                "relationships": {
                    other_id: relation.model_dump()
                    for other_id, relation in guest.relationships.items()
                },
            },
            "host_action": host_action.model_dump(mode="json"),
            "world": {
                "rules": state.world.current_rules,
                "events": state.world.ongoing_events[-8:],
                "active_tasks": [
                    task.model_dump(mode="json")
                    for task in state.world.active_tasks
                    if guest.guest_id in task.assigned_guests
                    or not task.assigned_guests
                ],
                "resolved_tasks": state.world.resolved_tasks[-6:],
                "visible_items": {
                    item_id: item.model_dump(mode="json")
                    for item_id, item in state.world.items.items()
                    if (not item.hidden and item.current_location == guest.current_room)
                    or item.current_location == f"guest:{guest.guest_id}"
                },
                "rooms": {
                    room_id: room.model_dump(mode="json")
                    for room_id, room in state.world.rooms.items()
                    if room.accessible or room_id == guest.current_room
                },
                "guest_locations": state.world.guest_locations,
                "room_requirements": state.world.room_requirements,
            },
        }
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def summarizer_messages(
    kind: str,
    payload: dict[str, Any],
    schema_model: type[MemoryCompression] | type[TurnNarrative],
) -> list[dict[str, str]]:
    system = _template_text("summarizer_system.txt")
    user = compact_json(
        {
            "kind": kind,
            "payload": payload,
        }
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
