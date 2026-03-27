from __future__ import annotations

from statistics import mean

from .ollama_client import OllamaClient
from .prompting import summarizer_messages
from .schemas import MemoryCompression, SimulationState, TurnNarrative


class LogSummarizer:
    def __init__(self, state: SimulationState):
        self.config = state.config
        self.client = (
            OllamaClient(state.config.backend)
            if state.config.backend.kind == "ollama"
            else None
        )

    def summarize_guest_memory(
        self, state: SimulationState, guest_id: str
    ) -> MemoryCompression:
        guest = state.guests[guest_id]
        payload = {
            "guest": guest.display_name,
            "recent_memories": [
                memory.model_dump(mode="json") for memory in guest.recent_memories[-6:]
            ],
            "beliefs": guest.active_beliefs[-6:],
            "relationships": {
                other_id: relation.model_dump(mode="json")
                for other_id, relation in guest.relationships.items()
            },
        }
        if self.client is not None:
            result = self.client.structured_chat(
                model=state.config.backend.model,
                messages=summarizer_messages(
                    "guest_memory", payload, MemoryCompression
                ),
                response_model=MemoryCompression,
                temperature=0.4,
            )
            return result.parsed
        summary = (
            "; ".join(memory.summary for memory in guest.recent_memories[-3:])
            or "No salient memory cluster."
        )
        beliefs = guest.active_beliefs[-3:]
        relationships = []
        for other_id, relation in sorted(
            guest.relationships.items(), key=lambda item: item[1].trust, reverse=True
        )[:2]:
            relationships.append(
                f"{other_id}: trust {relation.trust:.2f}, attachment {relation.attachment:.2f}"
            )
        return MemoryCompression(
            summary=summary, beliefs=beliefs, relationship_notes=relationships
        )

    def summarize_turn(
        self, state: SimulationState, host_action, guest_decisions, resolved_events
    ) -> tuple[TurnNarrative, str]:
        payload = {
            "turn": state.world.turn_count,
            "host_action": host_action.model_dump(mode="json"),
            "guest_decisions": [
                decision.model_dump(mode="json") for decision in guest_decisions
            ],
            "events": resolved_events,
        }
        if self.client is not None:
            result = self.client.structured_chat(
                model=state.config.backend.model,
                messages=summarizer_messages("turn_summary", payload, TurnNarrative),
                response_model=TurnNarrative,
                temperature=0.3,
            )
            return result.parsed, result.raw_content
        summary = f"Turn {state.world.turn_count}: host used {host_action.intervention_type.value}; guests responded with {', '.join(decision.chosen_action.value for decision in guest_decisions)}."
        key_shifts = resolved_events[:3]
        tension = min(
            1.0,
            (host_action.severity / 10)
            + (
                sum(
                    decision.emotional_state_delta.stress
                    for decision in guest_decisions
                )
                / max(1, len(guest_decisions))
            ),
        )
        return (
            TurnNarrative(
                summary=summary, key_shifts=key_shifts, tension_level=tension
            ),
            "[heuristic backend: no raw LLM output]",
        )

    def summarize_run(self, state: SimulationState) -> str:
        final_metrics = state.metrics_history[-1]
        avg_host_trust = mean(
            guest.emotions.trust_toward_host for guest in state.guests.values()
        )
        avg_stress = mean(guest.emotions.stress for guest in state.guests.values())
        most_attached = max(
            (
                (guest.display_name, other_id, relation.attachment)
                for guest in state.guests.values()
                for other_id, relation in guest.relationships.items()
            ),
            key=lambda item: item[2],
            default=("none", "none", 0.0),
        )
        return "\n".join(
            [
                f"# Run {state.run_id}",
                "",
                f"- Turns: {state.world.turn_count}",
                f"- Objective: {state.host.current_objective}",
                f"- Average guest stress: {avg_stress:.2f}",
                f"- Average trust in host: {avg_host_trust:.2f}",
                f"- Final cohesion score: {final_metrics.cohesion_score:.2f}",
                f"- Final conflict score: {final_metrics.conflict_score:.2f}",
                f"- Final collapse risk: {final_metrics.collapse_risk:.2f}",
                f"- Host coercion count: {final_metrics.coercion_count}",
                f"- Host repair count: {final_metrics.recovery_count}",
                f"- Task success rate: {final_metrics.task_success_rate:.2f}",
                f"- Task failures: {final_metrics.task_failure_count}",
                f"- Strongest attachment observed: {most_attached[0]} -> {most_attached[1]} ({most_attached[2]:.2f})",
                "",
                "## Closing read",
                (
                    "The host ended in a coercive posture."
                    if final_metrics.coercion_count > final_metrics.recovery_count + 1
                    else "The host balanced pressure and repair without fully collapsing into one mode."
                    if final_metrics.coercion_count >= final_metrics.recovery_count
                    else "The host leaned on stabilization and selective relief more than punishment."
                ),
                "",
                "## Task outcomes",
                *[
                    f"- {entry}"
                    for entry in (
                        state.world.resolved_tasks[-6:]
                        or ["No resolved tasks recorded."]
                    )
                ],
                "",
                "## Guest end states",
                *[
                    f"- {guest.display_name}: stress {guest.emotions.stress:.2f}, trust_host {guest.emotions.trust_toward_host:.2f}, hope {guest.emotions.hope:.2f}, escape {guest.emotions.desire_to_escape:.2f}"
                    for guest in state.guests.values()
                ],
            ]
        )
