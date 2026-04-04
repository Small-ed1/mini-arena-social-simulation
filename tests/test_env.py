from __future__ import annotations

from pathlib import Path

import yaml

from sim.env import Environment
from sim.schemas import (
    GuestInteract,
    GuestMove,
    HostEnrichWorld,
    HostShapeConceptual,
    HostSignalStyle,
)
from sim.world_state import Rulebook, load_scene, make_initial_world


def _load_cfg(name: str):
    base = Path(__file__).resolve().parents[1] / "configs" / name
    with open(base, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _make_world():
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
    return world


def test_move_changes_location_only() -> None:
    env = Environment()
    world = _make_world()
    gid = "guest_1"
    start_loc = world.guests[gid].location
    dest = "mirror_hall" if start_loc != "mirror_hall" else "foyer"

    props_before = {k: v.to_dict() for k, v in world.props.items()}

    res = env.apply_guest_action(
        world,
        gid,
        GuestMove(
            type="move", reason_short="Test move", actor_id=gid, destination=dest
        ),
    )
    assert res.success
    assert world.guests[gid].location == dest
    assert {k: v.to_dict() for k, v in world.props.items()} == props_before


def test_pick_up_prop_updates_inventory() -> None:
    env = Environment()
    world = _make_world()
    gid = "guest_1"
    world.guests[gid].location = "foyer"

    act = GuestInteract(
        type="interact",
        reason_short="Pick up",
        actor_id=gid,
        verb="pick_up",
        prop_id="prop_notebook",
        target_guest_id=None,
        speech=None,
    )
    res = env.apply_guest_action(world, gid, act)
    assert res.success
    assert "prop_notebook" in world.guests[gid].inventory
    assert world.props["prop_notebook"].held_by == gid
    assert world.props["prop_notebook"].location is None


def test_invalid_prop_access_fails_cleanly() -> None:
    env = Environment()
    world = _make_world()
    gid = "guest_1"
    world.guests[gid].location = "foyer"

    # prop_mask starts in stage_room
    act = GuestInteract(
        type="interact",
        reason_short="Pick up far prop",
        actor_id=gid,
        verb="pick_up",
        prop_id="prop_mask",
        target_guest_id=None,
        speech=None,
    )
    res = env.apply_guest_action(world, gid, act)
    assert not res.success
    assert "prop_mask" not in world.guests[gid].inventory
    assert world.props["prop_mask"].held_by is None
    assert world.props["prop_mask"].location == "stage_room"


def test_host_signal_style_only_changes_style() -> None:
    env = Environment()
    world = _make_world()
    guests_before = {k: v.to_dict() for k, v in world.guests.items()}

    res = env.apply_host_action(
        world,
        HostSignalStyle(
            type="signal_style", reason_short="Style", actor_id="host", style="gentle"
        ),
    )
    assert res.success
    assert world.host_style == "gentle"
    assert {k: v.to_dict() for k, v in world.guests.items()} == guests_before


def test_initial_world_has_unspawned_guests_only() -> None:
    env = Environment()
    world = _make_world()
    obs = env.observe_host(world, memory=None)
    assert obs.guests == []
    assert obs.valid_locations == ["foyer", "mirror_hall", "stage_room", "workshop"]
    assert obs.valid_concepts == [
        "collaboration_pressure",
        "unease",
        "social_friction",
        "urgency",
    ]
    assert len(world.spawned_guest_ids) == 0
    assert len(world.unspawned_guest_ids) == 6


def test_host_observation_only_includes_open_threads() -> None:
    env = Environment()
    world = _make_world()
    for thread in world.open_threads.values():
        thread.status = "closed"
    obs = env.observe_host(world, memory=None)
    assert obs.open_threads == []


def test_host_enrich_world_adds_location_detail() -> None:
    env = Environment()
    world = _make_world()
    res = env.apply_host_action(
        world,
        HostEnrichWorld(
            type="enrich_world",
            reason_short="Add a subtle clue",
            actor_id="host",
            location="foyer",
            detail="A bench cushion looks recently disturbed.",
        ),
    )
    assert res.success
    assert (
        world.location_details["foyer"][-1]
        == "A bench cushion looks recently disturbed."
    )


def test_host_shape_conceptual_updates_hidden_pressure() -> None:
    env = Environment()
    world = _make_world()
    world.spawned_guest_ids = ["guest_1"]
    world.unspawned_guest_ids = [gid for gid in world.guests if gid != "guest_1"]
    world.guests["guest_1"].location = "foyer"

    res = env.apply_host_action(
        world,
        HostShapeConceptual(
            type="shape_conceptual",
            reason_short="Pressure collaboration",
            actor_id="host",
            concept="collaboration_pressure",
            scope="location",
            location="foyer",
            intensity=0.8,
            note="Being alone should feel costly.",
        ),
    )
    assert res.success
    env.tick_postprocess(world)
    obs = env.observe_guest(world, "guest_1", memory=None)
    assert "less promising than usual" in obs.felt_state.lower()


def test_using_foam_key_alone_does_not_close_repair_thread() -> None:
    env = Environment()
    world = _make_world()
    gid = "guest_1"
    world.spawned_guest_ids = [gid]
    world.unspawned_guest_ids = [x for x in world.guests if x != gid]
    world.guests[gid].location = "foyer"

    act = GuestInteract(
        type="interact",
        reason_short="Use key",
        actor_id=gid,
        verb="use",
        prop_id="prop_foam_key",
        target_guest_id=None,
        speech=None,
    )
    res = env.apply_guest_action(world, gid, act)
    assert res.success
    assert "the mechanism resists a solitary attempt" in res.messages
    assert world.open_threads["thread_puzzle_1"].status == "open"


def test_using_foam_key_with_helper_closes_repair_thread() -> None:
    env = Environment()
    world = _make_world()
    world.spawned_guest_ids = ["guest_1", "guest_2"]
    world.unspawned_guest_ids = [
        x for x in world.guests if x not in world.spawned_guest_ids
    ]
    world.guests["guest_1"].location = "foyer"
    world.guests["guest_2"].location = "foyer"

    act = GuestInteract(
        type="interact",
        reason_short="Use key with help",
        actor_id="guest_1",
        verb="use",
        prop_id="prop_foam_key",
        target_guest_id=None,
        speech=None,
    )
    res = env.apply_guest_action(world, "guest_1", act)
    assert res.success
    assert "closed thread thread_puzzle_1" in res.messages
    assert world.open_threads["thread_puzzle_1"].status == "closed"


def test_collaboration_reduces_location_pressure() -> None:
    env = Environment()
    world = _make_world()
    world.spawned_guest_ids = ["guest_1", "guest_2"]
    world.unspawned_guest_ids = [
        x for x in world.guests if x not in world.spawned_guest_ids
    ]
    world.guests["guest_1"].location = "foyer"
    world.guests["guest_2"].location = "foyer"
    world.conceptual_by_location["foyer"]["collaboration_pressure"] = 0.8

    from sim.schemas import GuestCollaborate

    res = env.apply_guest_action(
        world,
        "guest_1",
        GuestCollaborate(
            type="collaborate",
            reason_short="Work together",
            actor_id="guest_1",
            target_guest_id="guest_2",
            proposal="Let's work on the panel together.",
            speech=None,
        ),
    )
    assert res.success
    assert world.conceptual_by_location["foyer"]["collaboration_pressure"] < 0.8
