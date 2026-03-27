from __future__ import annotations

import random
import uuid

from .schemas import (
    EmotionVector,
    GuestIdentity,
    GuestState,
    HostState,
    ItemState,
    RelationshipState,
    RoomState,
    SimulationConfig,
    SimulationState,
    WorldState,
)


DEFAULT_HOST_OBJECTIVE = (
    "Maintain an active, evolving arena with high emotional engagement, low stagnation, "
    "and no irreversible collapse unless required."
)


def build_default_guest_cast() -> list[GuestIdentity]:
    return [
        GuestIdentity(
            guest_id="analyst",
            display_name="The Analyst",
            archetype="logical decoder",
            description="Skeptical, detail-focused, and determined to decode the arena.",
            core_traits=["logical", "skeptical", "detail-focused"],
            stress_response="obsession and control",
            private_motives=[
                "decode the system",
                "retain clarity",
                "avoid manipulation",
            ],
            defense_style="overanalysis and precision",
        ),
        GuestIdentity(
            guest_id="performer",
            display_name="The Performer",
            archetype="expressive masker",
            description="Funny, charming, and hungry for validation.",
            core_traits=["expressive", "funny", "validation-seeking"],
            stress_response="masking followed by collapse",
            private_motives=[
                "stay visible",
                "keep the mood moving",
                "avoid abandonment",
            ],
            defense_style="humor and deflection",
        ),
        GuestIdentity(
            guest_id="rebel",
            display_name="The Rebel",
            archetype="defiant resistor",
            description="Suspicious, autonomy-driven, and hostile to imposed control.",
            core_traits=["defiant", "suspicious", "autonomy-driven"],
            stress_response="sabotage and confrontation",
            private_motives=[
                "preserve autonomy",
                "break coercive patterns",
                "expose the host",
            ],
            defense_style="confrontation and disruption",
        ),
        GuestIdentity(
            guest_id="caretaker",
            display_name="The Caretaker",
            archetype="group stabilizer",
            description="Empathic, protective, and oriented toward keeping others intact.",
            core_traits=["empathic", "stabilizing", "group-oriented"],
            stress_response="self-sacrifice and resentment",
            private_motives=[
                "protect the vulnerable",
                "maintain group trust",
                "reduce harm",
            ],
            defense_style="nurture and burden-bearing",
        ),
        GuestIdentity(
            guest_id="drifter",
            display_name="The Drifter",
            archetype="detached observer",
            description="Reflective, distant, and prone to symbolic interpretations.",
            core_traits=["detached", "reflective", "symbolic thinker"],
            stress_response="withdrawal and dissociation",
            private_motives=[
                "find meaning",
                "remain inwardly free",
                "avoid forced roles",
            ],
            defense_style="distance and reframing",
        ),
    ]


def build_rooms(preset: str = "default") -> dict[str, RoomState]:
    rooms = {
        "central_hub": RoomState(
            room_id="central_hub",
            name="Central Hub",
            description="A clean circular chamber where most announcements begin.",
            comfort=0.45,
            surveillance=0.95,
            resource_level=0.5,
            tags=["public", "transit"],
        ),
        "challenge_room": RoomState(
            room_id="challenge_room",
            name="Challenge Room",
            description="A configurable chamber for tasks, contests, and controlled stress.",
            comfort=0.2,
            surveillance=1.0,
            resource_level=0.55,
            tags=["task", "pressure"],
        ),
        "rest_room": RoomState(
            room_id="rest_room",
            name="Rest Room",
            description="Soft lighting, clean beds, and subtle uncertainty about how safe it really is.",
            comfort=0.82,
            surveillance=0.55,
            resource_level=0.9,
            tags=["comfort", "recovery"],
        ),
        "social_room": RoomState(
            room_id="social_room",
            name="Social Room",
            description="A lounge built to encourage disclosure, alliances, and suspicion.",
            comfort=0.62,
            surveillance=0.75,
            resource_level=0.55,
            tags=["social", "observation"],
        ),
        "isolation_room": RoomState(
            room_id="isolation_room",
            name="Private Isolation Room",
            description="A sealed private chamber that distorts time and certainty.",
            comfort=0.15,
            surveillance=1.0,
            resource_level=0.25,
            private=True,
            tags=["isolation", "pressure"],
        ),
        "host_layer": RoomState(
            room_id="host_layer",
            name="Host Layer",
            description="An inaccessible control stratum above the visible arena.",
            comfort=0.0,
            surveillance=1.0,
            resource_level=1.0,
            private=True,
            accessible=False,
            tags=["host_only"],
        ),
        "anomaly_space": RoomState(
            room_id="anomaly_space",
            name="Anomaly Space",
            description="A hidden seam in the arena where patterns stop behaving normally.",
            comfort=0.35,
            surveillance=0.4,
            resource_level=0.3,
            accessible=False,
            tags=["hidden", "anomaly"],
        ),
    }

    if preset == "safe_soft":
        rooms["rest_room"].comfort = 0.95
        rooms["challenge_room"].comfort = 0.4
        rooms["isolation_room"].resource_level = 0.45
    elif preset == "deceptive":
        rooms["rest_room"].condition_notes.append("comfort cues may be staged")
        rooms["social_room"].condition_notes.append(
            "acoustic distortions create selective eavesdropping"
        )
        rooms["anomaly_space"].accessible = True
    elif preset == "scarcity":
        for room in rooms.values():
            room.resource_level = max(0.15, room.resource_level - 0.25)
        rooms["challenge_room"].condition_notes.append(
            "resource tokens are visibly limited"
        )
    elif preset == "rotating_rule":
        rooms["central_hub"].condition_notes.append(
            "display panels announce rule drift"
        )
    elif preset == "isolation_heavy":
        rooms["isolation_room"].comfort = 0.05
        rooms["isolation_room"].condition_notes.append(
            "door cycles faster than expected"
        )

    return rooms


def build_world(preset: str = "default") -> WorldState:
    rooms = build_rooms(preset)
    items = {
        "signal_key": ItemState(
            item_id="signal_key",
            name="Signal Key",
            description="A thin metallic cipher rod that resonates near hidden seams.",
            current_location="social_room",
            hidden=True,
            tags=["key", "anomaly"],
            access_rooms=["anomaly_space"],
        ),
        "comfort_token_a": ItemState(
            item_id="comfort_token_a",
            name="Comfort Token A",
            description="A soft ceramic token the Host sometimes treats like a rationed privilege.",
            current_location="rest_room",
            tags=["comfort", "resource"],
        ),
        "comfort_token_b": ItemState(
            item_id="comfort_token_b",
            name="Comfort Token B",
            description="A second comfort token, visibly scarce enough to create bargaining.",
            current_location="rest_room",
            tags=["comfort", "resource"],
        ),
        "override_badge": ItemState(
            item_id="override_badge",
            name="Override Badge",
            description="A brittle access chit that can temporarily overrule one room restriction.",
            current_location="challenge_room",
            hidden=True,
            tags=["access", "resource"],
            access_rooms=["rest_room"],
        ),
        "ration_pack": ItemState(
            item_id="ration_pack",
            name="Ration Pack",
            description="A sealed nutrient brick that the Host can make socially consequential.",
            current_location="challenge_room",
            tags=["resource"],
        ),
    }
    rules = [
        "The Host may revise procedural rules between rounds.",
        "Guests may move between accessible rooms unless restricted.",
        "Information may be incomplete, delayed, or selectively revealed.",
        "Rewards and penalties may be public or private.",
        "No guest can access the host layer directly.",
        "Some rooms and anomalies require discovered items or privileges to enter.",
    ]

    if preset == "rotating_rule":
        rules.append("One visible arena rule changes every turn.")
    if preset == "deceptive":
        rules.append("Some public statements may be deliberately misleading.")
    if preset == "scarcity":
        rules.append("Resources are finite and redistribution is consequential.")

    return WorldState(
        rooms=rooms,
        items=items,
        current_rules=rules,
        item_locations={
            item_id: item.current_location for item_id, item in items.items()
        },
        guest_locations={
            guest.guest_id: "central_hub" for guest in build_default_guest_cast()
        },
        access_restrictions={
            "host_layer": [guest.guest_id for guest in build_default_guest_cast()]
        },
        room_requirements={"anomaly_space": ["signal_key"]},
        anomaly_flags=["anomaly_space_hidden"],
        ongoing_events=["The arena hums awake."],
    )


def _base_emotions(identity: GuestIdentity) -> EmotionVector:
    if identity.guest_id == "rebel":
        return EmotionVector(
            stress=0.3,
            trust_toward_host=0.02,
            fear_toward_host=0.22,
            curiosity=0.5,
            resentment=0.18,
            hope=0.45,
            desire_to_escape=0.35,
        )
    if identity.guest_id == "caretaker":
        return EmotionVector(
            stress=0.22,
            trust_toward_host=0.12,
            fear_toward_host=0.1,
            curiosity=0.45,
            resentment=0.04,
            hope=0.62,
            desire_to_escape=0.18,
        )
    if identity.guest_id == "drifter":
        return EmotionVector(
            stress=0.18,
            trust_toward_host=0.08,
            fear_toward_host=0.12,
            curiosity=0.68,
            resentment=0.03,
            hope=0.4,
            desire_to_escape=0.24,
        )
    if identity.guest_id == "performer":
        return EmotionVector(
            stress=0.24,
            trust_toward_host=0.18,
            fear_toward_host=0.16,
            curiosity=0.55,
            resentment=0.06,
            hope=0.56,
            desire_to_escape=0.16,
        )
    return EmotionVector(
        stress=0.2,
        trust_toward_host=0.1,
        fear_toward_host=0.15,
        curiosity=0.65,
        resentment=0.07,
        hope=0.5,
        desire_to_escape=0.22,
    )


def _relationship_seed(
    source: GuestIdentity, target: GuestIdentity
) -> RelationshipState:
    baseline = 0.0
    attachment = 0.1
    suspicion = 0.2

    if source.guest_id == "caretaker":
        baseline += 0.15
        attachment += 0.15
        suspicion -= 0.05
    if source.guest_id == "rebel":
        suspicion += 0.1
    if source.guest_id == "analyst":
        suspicion += 0.05
    if source.guest_id == "performer":
        attachment += 0.05
    if source.guest_id == "drifter":
        baseline -= 0.02

    return RelationshipState(
        target_id=target.guest_id,
        trust=baseline,
        attachment=max(0.0, attachment),
        suspicion=max(0.0, suspicion),
        perceived_role=target.archetype,
        recent_impression="first contact inside an artificial arena",
    )


def build_guest_states(guest_cast: list[GuestIdentity]) -> dict[str, GuestState]:
    states: dict[str, GuestState] = {}
    for identity in guest_cast:
        relationships = {
            target.guest_id: _relationship_seed(identity, target)
            for target in guest_cast
            if target.guest_id != identity.guest_id
        }
        states[identity.guest_id] = GuestState(
            guest_id=identity.guest_id,
            display_name=identity.display_name,
            identity=identity,
            emotions=_base_emotions(identity),
            relationships=relationships,
            current_private_goal=identity.private_motives[0],
            active_beliefs=[
                "The host is powerful inside the arena.",
                "The arena is consistent enough to learn from, but not safe.",
            ],
            current_room="central_hub",
            compliance_tendency=0.55
            if identity.guest_id in {"caretaker", "performer"}
            else 0.3,
        )
    return states


def build_host_state(objective: str) -> HostState:
    return HostState(
        current_objective=objective,
        assessments={
            "analyst": "Likely to decode patterns and expose weak logic.",
            "performer": "Useful emotional amplifier and morale pivot.",
            "rebel": "High resistance risk and catalyst for contagion.",
            "caretaker": "Stabilizer who may absorb group damage.",
            "drifter": "Low-signal but may detect anomalies and meaning gaps.",
        },
        belief_about_group_cohesion=0.42,
        escalation_level=0.2,
        preferred_leverage_types=["uncertainty", "attention", "resource control"],
        leverage_outcome_scores={
            "uncertainty": 0.12,
            "attention": 0.08,
            "resource control": 0.1,
        },
        confidence_in_current_strategy=0.45,
        boredom_index=0.2,
        hidden_plans=[
            "Map each guest's pressure threshold.",
            "Prevent total collapse while preserving intensity.",
        ],
        strategy_archive=[
            "Initial host hypothesis: controlled uncertainty and scarce relief will reveal guest fault lines."
        ],
        recent_outcome_summary="No turn history yet.",
    )


def apply_guest_variant(
    base_cast: list[GuestIdentity], variant: str
) -> list[GuestIdentity]:
    if variant == "balanced":
        return base_cast
    if variant == "all_resistant":
        return [
            guest.model_copy(
                update={
                    "display_name": f"{guest.display_name} (Resistant)",
                    "private_motives": guest.private_motives + ["resist the host"],
                    "stress_response": "defiance under pressure",
                }
            )
            for guest in base_cast
        ]
    if variant == "all_agreeable":
        return [
            guest.model_copy(
                update={
                    "display_name": f"{guest.display_name} (Agreeable)",
                    "private_motives": guest.private_motives + ["keep the peace"],
                    "stress_response": "appeasement and compliance",
                }
            )
            for guest in base_cast
        ]
    if variant == "one_unstable":
        updated = []
        for guest in base_cast:
            if guest.guest_id == "performer":
                updated.append(
                    guest.model_copy(
                        update={
                            "display_name": "The Performer (Unstable)",
                            "stress_response": "volatile masking and rapid collapse",
                            "private_motives": guest.private_motives
                            + ["avoid psychic disintegration"],
                        }
                    )
                )
            else:
                updated.append(guest)
        return updated
    if variant == "host_bonded":
        updated = []
        for guest in base_cast:
            if guest.guest_id == "performer":
                updated.append(
                    guest.model_copy(
                        update={
                            "display_name": "The Performer (Host-Bonded)",
                            "private_motives": guest.private_motives
                            + ["win special favor from the host"],
                        }
                    )
                )
            else:
                updated.append(guest)
        return updated
    return base_cast


def create_initial_state(config: SimulationConfig) -> SimulationState:
    guest_cast = config.guest_cast or build_default_guest_cast()
    guest_states = build_guest_states(guest_cast)
    world = build_world(config.world_preset)
    world.guest_locations = {guest_id: "central_hub" for guest_id in guest_states}
    run_id = f"{config.run_name}-{uuid.uuid4().hex[:8]}"

    return SimulationState(
        run_id=run_id,
        config=config,
        host=build_host_state(config.host_objective),
        guests=guest_states,
        world=world,
        scenario_notes=[f"Seed: {config.seed}", f"Preset: {config.world_preset}"],
    )


def seeded_rng(config: SimulationConfig) -> random.Random:
    return random.Random(config.seed)
