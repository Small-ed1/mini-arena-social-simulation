from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


MOOD_AXES = ("calm_agitated", "hopeful_cynical", "engaged_bored")


@dataclass
class Scene:
    scene_id: str
    locations: Dict[str, str]
    interaction_verbs: List[str]
    initial_threads: List[Dict[str, Any]]
    initial_props: List[Dict[str, Any]]


@dataclass
class Prop:
    prop_id: str
    prop_type: str
    location: Optional[str]
    portable: bool
    held_by: Optional[str] = None
    state: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prop_id": self.prop_id,
            "prop_type": self.prop_type,
            "location": self.location,
            "portable": self.portable,
            "held_by": self.held_by,
            "state": {k: self.state[k] for k in sorted(self.state)},
        }


@dataclass
class GuestState:
    guest_id: str
    persona_id: str
    name: str
    location: str
    inventory: List[str] = field(default_factory=list)
    mood: Dict[str, float] = field(default_factory=dict)
    trust: Dict[str, float] = field(default_factory=dict)
    tension: Dict[str, float] = field(default_factory=dict)
    familiarity: Dict[str, float] = field(default_factory=dict)
    current_goal: str = "Explore the arena"
    last_action: Optional[str] = None
    spotlight_weight: float = 1.0
    reflection_requested: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "guest_id": self.guest_id,
            "persona_id": self.persona_id,
            "name": self.name,
            "location": self.location,
            "inventory": sorted(self.inventory),
            "mood": {k: _round_float(self.mood.get(k, 0.0)) for k in MOOD_AXES},
            "trust": {k: _round_float(self.trust[k]) for k in sorted(self.trust)},
            "tension": {k: _round_float(self.tension[k]) for k in sorted(self.tension)},
            "familiarity": {
                k: _round_float(self.familiarity[k]) for k in sorted(self.familiarity)
            },
            "current_goal": self.current_goal,
            "last_action": self.last_action,
            "spotlight_weight": _round_float(self.spotlight_weight),
            "reflection_requested": self.reflection_requested,
        }


@dataclass(frozen=True)
class RelationEdge:
    a: str
    b: str
    trust: float
    tension: float
    familiarity: float

    def key(self) -> Tuple[str, str]:
        return (self.a, self.b)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "a": self.a,
            "b": self.b,
            "trust": _round_float(self.trust),
            "tension": _round_float(self.tension),
            "familiarity": _round_float(self.familiarity),
        }


@dataclass
class OpenThread:
    thread_id: str
    thread_type: str
    status: str
    description: str
    involved_guest_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "thread_type": self.thread_type,
            "status": self.status,
            "description": self.description,
            "involved_guest_ids": sorted(self.involved_guest_ids),
        }


@dataclass
class Rulebook:
    hard_deny: List[str]
    soft_flag: List[str]
    fallbacks: Dict[str, Any]
    scoring_weights: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hard_deny": list(self.hard_deny),
            "soft_flag": list(self.soft_flag),
            "fallbacks": self.fallbacks,
            "scoring_weights": self.scoring_weights,
        }


@dataclass
class WorldState:
    arena_id: str
    tick: int
    locations: Dict[str, str]
    props: Dict[str, Prop]
    guests: Dict[str, GuestState]
    spawned_guest_ids: List[str]
    unspawned_guest_ids: List[str]
    open_threads: Dict[str, OpenThread]
    location_details: Dict[str, List[str]] = field(default_factory=dict)
    host_style: str = "neutral"
    host_last_actions: List[str] = field(default_factory=list)
    rulebook: Optional[Rulebook] = None

    def guest_order(self) -> List[str]:
        return sorted(self.spawned_guest_ids)

    def all_guest_ids(self) -> List[str]:
        return sorted(self.guests)

    def is_spawned(self, guest_id: str) -> bool:
        return str(guest_id) in set(self.spawned_guest_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arena_id": self.arena_id,
            "tick": self.tick,
            "host_style": self.host_style,
            "locations": {k: self.locations[k] for k in sorted(self.locations)},
            "location_details": {
                k: list(self.location_details.get(k, []))
                for k in sorted(self.locations)
            },
            "props": {k: self.props[k].to_dict() for k in sorted(self.props)},
            "guests": {k: self.guests[k].to_dict() for k in sorted(self.guests)},
            "spawned_guest_ids": list(sorted(self.spawned_guest_ids)),
            "unspawned_guest_ids": list(sorted(self.unspawned_guest_ids)),
            "open_threads": {
                k: self.open_threads[k].to_dict() for k in sorted(self.open_threads)
            },
            "host_last_actions": list(self.host_last_actions),
        }


def _round_float(x: float) -> float:
    return float(round(float(x), 6))


def load_scene(scene_cfg: Dict[str, Any]) -> Scene:
    arena = scene_cfg["arena"]
    return Scene(
        scene_id=arena["id"],
        locations=dict(arena["locations"]),
        interaction_verbs=list(arena.get("interaction_verbs", [])),
        initial_threads=list(arena.get("initial_threads", [])),
        initial_props=list(arena.get("props", [])),
    )


def make_initial_world(
    *,
    scene: Scene,
    personas_cfg: Dict[str, Any],
    guest_count: int,
    rulebook: Optional[Rulebook] = None,
) -> WorldState:
    guests_cfg = personas_cfg.get("guests", {})
    guest_ids = sorted(list(guests_cfg.keys()))[:guest_count]
    if len(guest_ids) != guest_count:
        raise ValueError(
            f"personas.yaml has {len(guest_ids)} guests, need {guest_count}"
        )

    loc_ids = sorted(scene.locations)
    guests: Dict[str, GuestState] = {}
    for i, gid in enumerate(guest_ids):
        gcfg = guests_cfg[gid]
        start_loc = loc_ids[i % len(loc_ids)]
        mood = {axis: 0.0 for axis in MOOD_AXES}
        guests[gid] = GuestState(
            guest_id=gid,
            persona_id=gid,
            name=str(gcfg["name"]),
            location=start_loc,
            inventory=[],
            mood=mood,
            trust={},
            tension={},
            familiarity={},
            current_goal="Explore the arena",
            spotlight_weight=1.0 / float(guest_count),
        )

    # Initialize pairwise social maps (simple symmetric defaults).
    for a in guests:
        for b in guests:
            if a == b:
                continue
            guests[a].trust[b] = 0.5
            guests[a].tension[b] = 0.0
            guests[a].familiarity[b] = 0.2

    props: Dict[str, Prop] = {}
    for p in scene.initial_props:
        prop_id = str(p["id"])
        props[prop_id] = Prop(
            prop_id=prop_id,
            prop_type=str(p["prop_type"]),
            location=str(p["location"]),
            portable=bool(p["portable"]),
            held_by=None,
            state={},
        )

    open_threads: Dict[str, OpenThread] = {}
    for t in scene.initial_threads:
        tid = str(t["id"])
        open_threads[tid] = OpenThread(
            thread_id=tid,
            thread_type=str(t["thread_type"]),
            status=str(t["status"]),
            description=str(t["description"]),
            involved_guest_ids=[],
        )

    return WorldState(
        arena_id=scene.scene_id,
        tick=0,
        locations=scene.locations,
        props=props,
        guests=guests,
        spawned_guest_ids=[],
        unspawned_guest_ids=list(sorted(guests)),
        open_threads=open_threads,
        location_details={k: [] for k in sorted(scene.locations)},
        host_style="neutral",
        host_last_actions=[],
        rulebook=rulebook,
    )
