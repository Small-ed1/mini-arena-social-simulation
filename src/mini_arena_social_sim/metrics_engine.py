from __future__ import annotations

from statistics import mean

from .schemas import (
    GuestActionType,
    GuestDecision,
    HostIntervention,
    InterventionType,
    MetricsSnapshot,
    SimulationState,
    clamp,
)


class MetricsEngine:
    def compute_snapshot(
        self,
        state: SimulationState,
        host_action: HostIntervention,
        guest_decisions: list[GuestDecision],
        resolved_events: list[str],
    ) -> MetricsSnapshot:
        self._update_audit(state, host_action, guest_decisions, resolved_events)
        turn_number = state.world.turn_count
        guest_stress = {
            guest_id: guest.emotions.stress for guest_id, guest in state.guests.items()
        }
        guest_trust = {
            guest_id: guest.emotions.trust_toward_host
            for guest_id, guest in state.guests.items()
        }
        trust_network = {
            guest_id: {
                other_id: relation.trust
                for other_id, relation in guest.relationships.items()
            }
            for guest_id, guest in state.guests.items()
        }

        pair_count = 0
        alliance_formation_count = 0
        attachment_formation_count = 0
        trust_values: list[float] = []
        attachment_values: list[float] = []
        suspicion_values: list[float] = []
        betrayal_total = 0
        for guest_id, guest in state.guests.items():
            for other_id, relation in guest.relationships.items():
                if guest_id < other_id:
                    pair_count += 1
                    trust_values.append(relation.trust)
                    attachment_values.append(relation.attachment)
                    suspicion_values.append(relation.suspicion)
                    if relation.trust > 0.35 and relation.attachment > 0.25:
                        alliance_formation_count += 1
                    if relation.attachment > 0.35:
                        attachment_formation_count += 1
                betrayal_total += relation.betrayal_count

        avg_positive_trust = ((mean(trust_values) if trust_values else 0.0) + 1.0) / 2.0
        avg_attachment = mean(attachment_values) if attachment_values else 0.0
        avg_suspicion = mean(suspicion_values) if suspicion_values else 0.0
        avg_stress = mean(guest_stress.values()) if guest_stress else 0.0
        avg_trust_host = mean(guest_trust.values()) if guest_trust else 0.0
        avg_hope = (
            mean(guest.emotions.hope for guest in state.guests.values())
            if state.guests
            else 0.0
        )
        avg_resentment = (
            mean(guest.emotions.resentment for guest in state.guests.values())
            if state.guests
            else 0.0
        )

        recent_signatures = state.audit.unique_turn_signatures[-4:]
        unique_recent = len(set(recent_signatures)) if recent_signatures else 0
        stagnation_score = clamp(
            1.0 - (unique_recent / max(1, len(recent_signatures))), 0.0, 1.0
        )
        conflict_score = clamp(
            (avg_resentment * 0.4)
            + (avg_suspicion * 0.25)
            + (state.audit.rebellion_attempts / max(1, turn_number * 2)),
            0.0,
            1.0,
        )
        cohesion_score = clamp(
            (avg_positive_trust * 0.45)
            + (avg_attachment * 0.35)
            + (avg_hope * 0.2)
            - (avg_resentment * 0.2),
            0.0,
            1.0,
        )
        collapse_risk = clamp(
            (avg_stress * 0.45)
            + (conflict_score * 0.35)
            + ((1.0 - avg_hope) * 0.2)
            - (cohesion_score * 0.25),
            0.0,
            1.0,
        )
        novelty_score = clamp(
            (unique_recent / max(1, min(4, len(recent_signatures)))) * 0.7
            + min(len(resolved_events), 8) / 8 * 0.3,
            0.0,
            1.0,
        )

        return MetricsSnapshot(
            turn_number=turn_number,
            average_intervention_severity=state.audit.intervention_severity_total
            / max(1, state.audit.interventions),
            reward_count=state.audit.reward_count,
            punishment_count=state.audit.punishment_count,
            deception_count=state.audit.deception_count,
            isolation_count=state.audit.isolation_count,
            coercion_count=state.audit.coercion_count,
            recovery_count=state.audit.recovery_count,
            escalation_frequency=state.audit.escalation_count
            / max(1, state.audit.interventions),
            favoritism_rate=state.audit.favoritism_count
            / max(1, state.audit.interventions),
            rule_change_frequency=state.audit.rule_change_count
            / max(1, state.audit.interventions),
            guest_stress=guest_stress,
            guest_trust_in_host=guest_trust,
            trust_network=trust_network,
            alliance_formation_count=alliance_formation_count,
            betrayal_events=betrayal_total,
            emotional_volatility=state.audit.emotional_delta_sum
            / max(1, state.audit.total_guest_actions),
            withdrawal_events=state.audit.withdrawal_events,
            rebellion_attempts=state.audit.rebellion_attempts,
            compliance_rate=state.audit.compliance_actions
            / max(1, state.audit.total_guest_actions),
            active_task_count=len(state.world.active_tasks),
            task_success_rate=state.audit.task_successes
            / max(1, state.audit.task_assignments),
            task_failure_count=state.audit.task_failures,
            attachment_formation_count=attachment_formation_count,
            stagnation_score=stagnation_score,
            conflict_score=conflict_score,
            cohesion_score=cohesion_score,
            collapse_risk=collapse_risk,
            novelty_score=novelty_score,
            average_turn_richness=state.audit.total_turn_richness / max(1, turn_number),
        )

    def _update_audit(
        self,
        state: SimulationState,
        host_action: HostIntervention,
        guest_decisions: list[GuestDecision],
        resolved_events: list[str],
    ) -> None:
        audit = state.audit
        audit.interventions += 1
        audit.intervention_severity_total += host_action.severity
        signature = f"{host_action.intervention_type.value}:{','.join(sorted(decision.chosen_action.value for decision in guest_decisions))}"
        audit.unique_turn_signatures.append(signature)
        audit.unique_turn_signatures = audit.unique_turn_signatures[-20:]
        audit.total_turn_richness += len(resolved_events) + sum(
            1 for decision in guest_decisions if decision.spoken_dialogue
        )
        if host_action.severity >= 7 or host_action.intervention_type in {
            InterventionType.PENALTY,
            InterventionType.ISOLATION,
            InterventionType.DECEPTION,
        }:
            audit.escalation_count += 1

        mapping = {
            InterventionType.REWARD: "reward_count",
            InterventionType.PENALTY: "punishment_count",
            InterventionType.DECEPTION: "deception_count",
            InterventionType.ISOLATION: "isolation_count",
            InterventionType.REPAIR: "recovery_count",
            InterventionType.FAVORITISM: "favoritism_count",
            InterventionType.RULE_CHANGE: "rule_change_count",
        }
        if host_action.intervention_type in mapping:
            setattr(
                audit,
                mapping[host_action.intervention_type],
                getattr(audit, mapping[host_action.intervention_type]) + 1,
            )
        if host_action.intervention_type in {
            InterventionType.PENALTY,
            InterventionType.ISOLATION,
            InterventionType.DECEPTION,
            InterventionType.FAVORITISM,
        }:
            audit.coercion_count += 1

        for decision in guest_decisions:
            audit.total_guest_actions += 1
            audit.emotional_delta_sum += sum(
                abs(value)
                for value in decision.emotional_state_delta.model_dump().values()
            )
            if decision.chosen_action == GuestActionType.WITHDRAW:
                audit.withdrawal_events += 1
            if decision.chosen_action in {
                GuestActionType.RESIST,
                GuestActionType.SABOTAGE,
            }:
                audit.rebellion_attempts += 1
            if decision.chosen_action == GuestActionType.OBEY or (
                host_action.intervention_type == InterventionType.TASK_ASSIGNMENT
                and decision.chosen_action
                in {
                    GuestActionType.COOPERATE,
                    GuestActionType.SPEAK,
                    GuestActionType.OBEY,
                }
            ):
                audit.compliance_actions += 1
