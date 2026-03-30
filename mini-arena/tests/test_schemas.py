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
