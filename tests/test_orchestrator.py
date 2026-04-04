from __future__ import annotations

from types import SimpleNamespace

from sim.orchestrator import (
    _apply_reaction_bump,
    _build_guest_turn_queue,
    _classify_inference_failure,
    _host_action_needs_repair,
    _needs_spawn_event_type_repair,
    _update_guest_turn_fairness,
)
from sim.schemas import GuestCollaborate, HostInjectProp, HostSpawnEvent


def test_classify_inference_failure_schema_invalid() -> None:
    category, reason = _classify_inference_failure(
        "1 validation error for tagged-union\nspawn_event.event_type\n  Input should be 'puzzle'",
        "host",
    )
    assert category == "host_schema_invalid"
    assert "validation error" in reason.lower()


def test_detect_spawn_event_type_repair_needed() -> None:
    assert _needs_spawn_event_type_repair(
        "spawn_event.event_type Input should be 'puzzle'"
    )


def test_host_repair_rejects_duplicate_prop_id() -> None:
    obs_h = SimpleNamespace(
        open_threads=[SimpleNamespace(thread_id="t1")],
        last_host_actions=[],
        valid_locations=["foyer", "mirror_hall", "stage_room", "workshop"],
    )
    world = SimpleNamespace(props={"prop_foam_key": object()}, location_details={})
    proposed = HostInjectProp(
        type="inject_prop",
        reason_short="Introduce foam key",
        actor_id="host",
        prop_type="foam_key",
        location="mirror_hall",
        prop_id="prop_foam_key",
    )

    issue = _host_action_needs_repair(obs_h, proposed, world)
    assert issue is not None
    assert issue[0] == "host_invalid_prop_id"


def test_host_repair_rejects_duplicate_thread_description() -> None:
    obs_h = SimpleNamespace(
        open_threads=[SimpleNamespace(thread_id="t1", description="same thread")],
        last_host_actions=[],
        valid_locations=["foyer", "mirror_hall", "stage_room", "workshop"],
    )
    world = SimpleNamespace(props={}, location_details={})
    proposed = HostSpawnEvent(
        type="spawn_event",
        reason_short="Repeat thread",
        actor_id="host",
        event_type="repair",
        description="same thread",
        location="foyer",
        involved_guest_ids=[],
    )

    issue = _host_action_needs_repair(obs_h, proposed, world)
    assert issue is not None
    assert issue[0] == "host_duplicate_thread"


def test_build_guest_turn_queue_prioritizes_thread_involvement() -> None:
    world = SimpleNamespace(
        tick=5,
        locations={"foyer": "", "workshop": ""},
        props={},
        spawned_guest_ids=["guest_1", "guest_2"],
        guest_turn_fairness={"guest_1": 0.0, "guest_2": 0.0},
        open_threads={
            "t1": SimpleNamespace(
                status="open",
                location="foyer",
                involved_guest_ids=["guest_2"],
            )
        },
        guests={
            "guest_1": SimpleNamespace(
                location="workshop",
                reflection_requested=False,
                spotlight_weight=0.0,
                inventory=[],
                spawn_tick=1,
            ),
            "guest_2": SimpleNamespace(
                location="foyer",
                reflection_requested=False,
                spotlight_weight=0.0,
                inventory=[],
                spawn_tick=1,
            ),
        },
        guest_order=lambda: ["guest_1", "guest_2"],
        all_guest_ids=lambda: ["guest_1", "guest_2"],
    )
    queue = _build_guest_turn_queue(world, 1337)
    assert queue[0] == "guest_2"


def test_reaction_bump_promotes_target_next() -> None:
    world = SimpleNamespace(
        guests={
            "guest_1": SimpleNamespace(location="foyer"),
            "guest_2": SimpleNamespace(location="foyer"),
            "guest_3": SimpleNamespace(location="workshop"),
        }
    )
    queue = ["guest_3", "guest_2"]
    action = GuestCollaborate(
        type="collaborate",
        reason_short="Coordinate",
        actor_id="guest_1",
        target_guest_id="guest_2",
        proposal="Let's solve this together.",
        speech=None,
    )
    bumped = _apply_reaction_bump(queue, world, "guest_1", action)
    assert bumped[0] == "guest_2"


def test_update_guest_turn_fairness_penalizes_early_actor() -> None:
    world = SimpleNamespace(
        guest_turn_fairness={"guest_1": 0.0, "guest_2": 0.0},
        all_guest_ids=lambda: ["guest_1", "guest_2"],
        guest_order=lambda: ["guest_1", "guest_2"],
    )
    _update_guest_turn_fairness(world, ["guest_1", "guest_2"])
    assert world.guest_turn_fairness["guest_1"] > world.guest_turn_fairness["guest_2"]
