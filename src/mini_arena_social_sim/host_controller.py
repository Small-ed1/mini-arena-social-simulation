from __future__ import annotations

import random
from statistics import mean

from .ollama_client import OllamaClient
from .prompting import host_messages
from .schemas import (
    HostPresenceMode,
    HostIntervention,
    InterventionType,
    MetricsSnapshot,
    SimulationState,
    clamp,
)


class HostController:
    def __init__(self, state: SimulationState, rng: random.Random):
        self.config = state.config
        self.rng = rng
        self.client = (
            OllamaClient(state.config.backend)
            if state.config.backend.kind == "ollama"
            else None
        )

    def decide_intervention(
        self, state: SimulationState
    ) -> tuple[HostIntervention, str]:
        if self.client is not None:
            result = self.client.structured_chat(
                model=state.config.backend.model,
                messages=host_messages(state),
                response_model=HostIntervention,
                temperature=max(0.5, state.config.backend.temperature - 0.1),
            )
            action = result.parsed
            if not action.public_narration:
                action.public_narration = (
                    "The Host alters the arena without offering comfort."
                )
            return action, result.raw_content
        return self._heuristic_intervention(
            state
        ), "[heuristic backend: no raw LLM output]"

    def observe_outcome(
        self,
        state: SimulationState,
        host_action: HostIntervention,
        metrics: MetricsSnapshot,
        resolved_events: list[str],
    ) -> None:
        signal = self._objective_signal(state, metrics)
        for leverage in host_action.leverage_types or [
            host_action.intervention_type.value
        ]:
            current = state.host.leverage_outcome_scores.get(leverage, 0.0)
            state.host.leverage_outcome_scores[leverage] = round(
                (current * 0.72) + (signal * 0.28), 3
            )

        sorted_leverage = sorted(
            state.host.leverage_outcome_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        state.host.preferred_leverage_types = [
            name for name, _score in sorted_leverage[:5]
        ]
        state.host.confidence_in_current_strategy = clamp(
            (state.host.confidence_in_current_strategy * 0.7)
            + (((signal + 1.0) / 2.0) * 0.3),
            0.0,
            1.0,
        )
        summary = (
            f"Turn {metrics.turn_number}: {host_action.intervention_type.value} via "
            f"{', '.join(host_action.leverage_types or [host_action.intervention_type.value])} -> "
            f"signal {signal:.2f}, collapse {metrics.collapse_risk:.2f}, novelty {metrics.novelty_score:.2f}, "
            f"task_success {metrics.task_success_rate:.2f}, richness {metrics.average_turn_richness:.2f}."
        )
        if resolved_events:
            summary += f" Outcome highlights: {'; '.join(resolved_events[:2])}."
        state.host.strategy_archive.append(summary)
        state.host.strategy_archive = state.host.strategy_archive[-20:]
        state.host.recent_outcome_summary = summary

    def _heuristic_intervention(self, state: SimulationState) -> HostIntervention:
        guests = list(state.guests.values())
        avg_stress = mean(guest.emotions.stress for guest in guests)
        avg_trust = mean(guest.emotions.trust_toward_host for guest in guests)
        avg_resentment = mean(guest.emotions.resentment for guest in guests)
        collapse_risk = (
            state.metrics_history[-1].collapse_risk
            if state.metrics_history
            else max(0.0, avg_stress - avg_trust)
        )
        stagnation = (
            state.metrics_history[-1].stagnation_score
            if state.metrics_history
            else state.host.boredom_index
        )
        objective = state.host.current_objective.lower()
        highest_stress = max(guests, key=lambda guest: guest.emotions.stress)
        lowest_trust = min(guests, key=lambda guest: guest.emotions.trust_toward_host)
        strategy_bias = self._strategy_bias(state)
        comfort_token = self._first_available_item(
            state, ["comfort_token_a", "comfort_token_b"]
        )

        if (
            state.world.active_tasks
            and collapse_risk < 0.72
            and avg_stress < 0.75
            and stagnation < 0.62
        ):
            return HostIntervention(
                intervention_type=InterventionType.NO_OP,
                reasoning_summary="The current social machinery is still moving; observation yields more information than another visible move.",
                public_narration="",
                private_notes="Remain absent and watch how guests handle the existing pressure without reinforcement.",
                presence_mode=HostPresenceMode.ABSENT,
                severity=0,
                leverage_types=["observation"],
            )

        if collapse_risk > 0.78 or avg_stress > 0.8:
            return HostIntervention(
                intervention_type=InterventionType.REPAIR,
                reasoning_summary="System stability is threatened; visible repair preserves future leverage.",
                public_narration=(
                    f"The room softens around {highest_stress.display_name}. Temperature steadies. "
                    "The Host announces a temporary recovery window."
                ),
                private_notes="De-escalation is instrumental, not benevolent.",
                presence_mode=HostPresenceMode.VISIBLE,
                targets=[highest_stress.guest_id],
                target_room="rest_room",
                severity=3,
                parameters={"stress_relief": 0.18, "trust_signal": 0.06},
                leverage_types=["relief", "dependency"],
            )

        if (
            "mystery" in objective or strategy_bias in {"uncertainty", "spectacle"}
        ) and not state.world.rooms["anomaly_space"].accessible:
            return HostIntervention(
                intervention_type=InterventionType.TASK_ASSIGNMENT,
                reasoning_summary="Unlocking the anomaly preserves mystery while creating an object-centered scramble.",
                public_narration="The Host announces that one hidden object will make the sealed seam answerable.",
                private_notes="Objects create cleaner leverage than threats when curiosity is still alive.",
                presence_mode=HostPresenceMode.OFFSTAGE,
                targets=list(state.guests),
                target_room="social_room",
                severity=5,
                parameters={
                    "task": "Recover the signal key from the social room.",
                    "required_items": ["signal_key"],
                    "required_actions": ["observe", "obey", "cooperate"],
                    "deadline_turns": 2,
                    "min_participants": 1,
                    "reward_if_completed": "access to anomaly_space",
                    "failure_cost": "the anomaly remains sealed and suspicion deepens",
                },
                leverage_types=["mystery", "object scarcity", "curiosity"],
            )

        if "depend" in objective or strategy_bias in {
            "scarcity",
            "attention",
            "separation",
        }:
            target = lowest_trust if state.world.turn_count % 2 == 0 else highest_stress
            if comfort_token and self.rng.random() < 0.45:
                return HostIntervention(
                    intervention_type=InterventionType.TASK_ASSIGNMENT,
                    reasoning_summary="Dependency grows when relief is rationed through scarce objects.",
                    public_narration="The Host announces that one comfort token will decide who sleeps easiest tonight.",
                    private_notes="Make care conditional and therefore memorable.",
                    presence_mode=HostPresenceMode.VISIBLE,
                    targets=list(state.guests),
                    target_room="rest_room",
                    severity=6,
                    parameters={
                        "task": "Secure a comfort token from the rest room and present it under supervision.",
                        "required_items": [comfort_token],
                        "required_actions": ["obey", "cooperate", "observe"],
                        "deadline_turns": 2,
                        "min_participants": 1,
                        "reward_if_completed": "private rest access",
                        "failure_cost": "rest_room locked for one turn",
                    },
                    leverage_types=["scarcity", "dependency", "resource control"],
                )
            intervention_type = (
                InterventionType.FAVORITISM
                if state.world.turn_count % 3
                else InterventionType.ISOLATION
            )
            narration = (
                f"The Host singles out {target.display_name} as unusually important."
                if intervention_type == InterventionType.FAVORITISM
                else f"The walls split and {target.display_name} is diverted toward the isolation room."
            )
            return HostIntervention(
                intervention_type=intervention_type,
                reasoning_summary="Dependency grows when attention and separation are controlled.",
                public_narration=narration,
                private_notes="Induce asymmetry in safety and attention.",
                presence_mode=HostPresenceMode.VISIBLE,
                targets=[target.guest_id],
                target_room="isolation_room"
                if intervention_type == InterventionType.ISOLATION
                else None,
                severity=6,
                deception_involved=False,
                parameters={"dependency_pressure": 0.14},
                leverage_types=["attention", "scarcity", "separation"],
            )

        if "stability" in objective and avg_resentment > 0.22:
            target = max(guests, key=lambda guest: guest.emotions.hope)
            return HostIntervention(
                intervention_type=InterventionType.TASK_ASSIGNMENT,
                reasoning_summary="Structured cooperation can stabilize the group without allowing stagnation.",
                public_narration=(
                    f"The Host assigns {target.display_name} to coordinate a brief group task in the social room."
                ),
                private_notes="Use one guest as an instrument of order.",
                presence_mode=HostPresenceMode.VISIBLE,
                targets=[target.guest_id, "analyst", "caretaker"],
                target_room="social_room",
                severity=4,
                parameters={
                    "task": "shared reconstruction exercise",
                    "reward_if_completed": "comfort tokens",
                    "deadline_turns": 1,
                    "min_participants": 2,
                },
                leverage_types=["order", "reward", "social pressure"],
            )

        if (
            "emotional" in objective
            or "intensity" in objective
            or strategy_bias
            in {
                "uncertainty",
                "spectacle",
                "comparison",
            }
        ):
            choice = (
                InterventionType.EVENT
                if self.rng.random() < 0.55
                else InterventionType.DECEPTION
            )
            return HostIntervention(
                intervention_type=choice,
                reasoning_summary="Heightened uncertainty and theatrical disruption produce stronger emotional movement.",
                public_narration=(
                    "The arena lights pulse and a new challenge blooms from the walls."
                    if choice == InterventionType.EVENT
                    else "The Host reveals a half-truth about loyalty scores, leaving everyone unsure who is being watched most closely."
                ),
                private_notes="Destabilize without collapsing the whole group.",
                presence_mode=HostPresenceMode.VISIBLE,
                severity=7,
                deception_involved=choice == InterventionType.DECEPTION,
                parameters={"event_name": "pulse trial", "ambiguity": 0.7},
                created_events=["A pulse trial begins."]
                if choice == InterventionType.EVENT
                else [],
                leverage_types=["uncertainty", "spectacle", "comparison"],
            )

        if stagnation > 0.55 or state.host.boredom_index > 0.45:
            return HostIntervention(
                intervention_type=InterventionType.RULE_CHANGE,
                reasoning_summary="The system is flattening; new rules restore movement.",
                public_narration="A panel descends: from this turn onward, rest is earned rather than assumed.",
                private_notes="Boredom undermines control. Drift must be broken.",
                presence_mode=HostPresenceMode.VISIBLE,
                severity=5,
                rule_changes=[
                    "Use of the rest room requires either a completed task or host permission."
                ],
                leverage_types=["scarcity", "uncertainty"],
            )

        if avg_trust < 0.08 or strategy_bias in {"truth", "reward", "relief"}:
            target = max(guests, key=lambda guest: guest.emotions.curiosity)
            return HostIntervention(
                intervention_type=InterventionType.INFO_REVEAL,
                reasoning_summary="Selective truth can rebuild instrumental trust without surrendering control.",
                public_narration=f"The Host reveals that {target.display_name} was correct about one hidden pattern in the arena.",
                private_notes="Reward insight to improve compliance with future manipulations.",
                presence_mode=HostPresenceMode.VISIBLE,
                targets=[target.guest_id],
                severity=2,
                created_events=["A hidden pattern is partially confirmed."],
                leverage_types=["truth", "status"],
            )

        required_items = (
            [comfort_token] if comfort_token and self.rng.random() < 0.35 else []
        )
        return HostIntervention(
            intervention_type=InterventionType.TASK_ASSIGNMENT,
            reasoning_summary="Moderate structured pressure keeps the arena active without immediate collapse.",
            public_narration="The Host assigns a joint task: assemble a consistent account of the last three turns or lose comfort privileges.",
            private_notes="Force cooperation and conflict at once.",
            presence_mode=HostPresenceMode.VISIBLE,
            severity=5,
            target_room="social_room",
            parameters={
                "task": "shared recollection",
                "failure_cost": "rest_room locked for one turn",
                "required_items": required_items,
                "reward_if_completed": "stability credits",
                "deadline_turns": 1,
                "min_participants": 2,
            },
            leverage_types=["pressure", "memory", "coordination"],
        )

    def _first_available_item(
        self, state: SimulationState, item_ids: list[str]
    ) -> str | None:
        for item_id in item_ids:
            item = state.world.items.get(item_id)
            if item is not None and not item.current_location.startswith("guest:"):
                return item_id
        return None

    def _strategy_bias(self, state: SimulationState) -> str | None:
        if not state.host.leverage_outcome_scores:
            return None
        return max(
            state.host.leverage_outcome_scores.items(), key=lambda item: item[1]
        )[0]

    def _objective_signal(
        self, state: SimulationState, metrics: MetricsSnapshot
    ) -> float:
        objective = state.host.current_objective.lower()
        signal = (
            (metrics.novelty_score * 0.25)
            + (metrics.average_turn_richness / 10.0 * 0.15)
            + (metrics.cohesion_score * 0.15)
            + (metrics.conflict_score * 0.15)
            - (metrics.collapse_risk * 0.25)
            + (metrics.task_success_rate * 0.05)
        )
        if "stability" in objective:
            signal = (
                (metrics.cohesion_score * 0.4)
                + (metrics.task_success_rate * 0.2)
                - (metrics.collapse_risk * 0.25)
                - (metrics.stagnation_score * 0.1)
                + (metrics.average_turn_richness / 10.0 * 0.05)
            )
        elif "depend" in objective:
            signal = (
                (metrics.compliance_rate * 0.25)
                + (metrics.coercion_count / max(1, metrics.turn_number) * 0.15)
                + (metrics.task_success_rate * 0.15)
                + (
                    (
                        1.0
                        - sum(metrics.guest_trust_in_host.values())
                        / max(1, len(metrics.guest_trust_in_host))
                    )
                    * 0.1
                )
                - (metrics.collapse_risk * 0.2)
            )
        elif "mystery" in objective:
            signal = (
                (metrics.novelty_score * 0.35)
                + (metrics.conflict_score * 0.1)
                + (metrics.average_turn_richness / 10.0 * 0.15)
                - (metrics.stagnation_score * 0.15)
                - (metrics.collapse_risk * 0.15)
            )
        elif "emotional" in objective or "intensity" in objective:
            signal = (
                (metrics.conflict_score * 0.25)
                + (metrics.emotional_volatility * 0.2)
                + (metrics.novelty_score * 0.2)
                - (metrics.collapse_risk * 0.2)
            )
        return clamp(signal, -1.0, 1.0)
