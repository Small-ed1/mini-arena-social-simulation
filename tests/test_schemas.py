from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from sim.schemas import GuestAction, HostAction


def test_host_action_validates() -> None:
    ta = TypeAdapter(HostAction)
    act = ta.validate_python(
        {
            "type": "inject_prop",
            "reason_short": "Increase puzzle tension",
            "actor_id": "host",
            "prop_type": "foam_key",
            "location": "foyer",
        }
    )
    assert act.type == "inject_prop"


def test_host_enrich_world_validates() -> None:
    ta = TypeAdapter(HostAction)
    act = ta.validate_python(
        {
            "type": "enrich_world",
            "reason_short": "Add ambient clue",
            "actor_id": "host",
            "location": "mirror_hall",
            "detail": "A faint crack catches the light oddly.",
        }
    )
    assert act.type == "enrich_world"


def test_host_shape_conceptual_validates() -> None:
    ta = TypeAdapter(HostAction)
    act = ta.validate_python(
        {
            "type": "shape_conceptual",
            "reason_short": "Pressure collaboration",
            "actor_id": "host",
            "concept": "collaboration_pressure",
            "scope": "location",
            "location": "workshop",
            "intensity": 0.7,
        }
    )
    assert act.type == "shape_conceptual"


def test_host_action_extra_field_fails() -> None:
    ta = TypeAdapter(HostAction)
    with pytest.raises(ValidationError):
        ta.validate_python(
            {
                "type": "signal_style",
                "reason_short": "Tone",
                "actor_id": "host",
                "style": "gentle",
                "extra": 123,
            }
        )


def test_guest_action_missing_required_fails() -> None:
    ta = TypeAdapter(GuestAction)
    with pytest.raises(ValidationError):
        ta.validate_python(
            {"type": "move", "reason_short": "Go", "destination": "foyer"}
        )


def test_action_type_malformed_fails() -> None:
    ta = TypeAdapter(GuestAction)
    with pytest.raises(ValidationError):
        ta.validate_python(
            {"type": "fly", "reason_short": "Nope", "actor_id": "guest_1"}
        )


def test_observation_guest_requires_valid_locations() -> None:
    from sim.schemas import ObservationGuest

    with pytest.raises(ValidationError):
        ObservationGuest.model_validate(
            {
                "tick": 1,
                "guest_id": "guest_1",
                "persona": "careful and observant",
                "goal": "Explore the arena",
                "location": "foyer",
                "valid_locations": [],
                "local_view": "A small room with a bench.",
                "nearby_guests": [],
                "nearby_props": [],
                "open_threads": [],
                "memory_chunks": [],
                "recent_actions": [],
                "reflection_requested": False,
            }
        )
