from __future__ import annotations

from .log_summarizer import LogSummarizer
from .schemas import MemoryCategory, MemoryRecord, SimulationState


class MemoryStore:
    def __init__(self, state: SimulationState, summarizer: LogSummarizer):
        self.summarizer = summarizer
        self.interval = state.config.summarization_interval

    def record_turn(
        self, state: SimulationState, host_action, guest_decisions, resolved_events
    ) -> list[tuple[str, MemoryRecord]]:
        created: list[tuple[str, MemoryRecord]] = []

        host_memory = MemoryRecord(
            turn_number=state.world.turn_count,
            category=MemoryCategory.HOST,
            summary=f"Used {host_action.intervention_type.value} with severity {host_action.severity}: {host_action.reasoning_summary}",
            salience=min(1.0, 0.35 + host_action.severity / 10),
            tags=[host_action.intervention_type.value],
        )
        state.host.recent_memories.append(host_memory)
        state.host.recent_memories = state.host.recent_memories[-12:]
        created.append(("host", host_memory))

        event_digest = (
            "; ".join(resolved_events[:4]) if resolved_events else "No visible shift."
        )
        for decision in guest_decisions:
            guest = state.guests[decision.guest_id]
            summary = decision.memory_to_store or event_digest
            memory = MemoryRecord(
                turn_number=state.world.turn_count,
                category=MemoryCategory.EPISODIC,
                summary=summary,
                salience=min(
                    1.0,
                    0.3
                    + abs(decision.emotional_state_delta.stress)
                    + abs(decision.emotional_state_delta.resentment),
                ),
                tags=[
                    decision.chosen_action.value,
                    host_action.intervention_type.value,
                ],
            )
            guest.recent_memories.append(memory)
            guest.recent_memories = guest.recent_memories[-12:]
            created.append((guest.guest_id, memory))

            if decision.belief_update:
                belief_memory = MemoryRecord(
                    turn_number=state.world.turn_count,
                    category=MemoryCategory.BELIEF,
                    summary=decision.belief_update,
                    salience=0.6,
                    tags=["belief"],
                )
                guest.recent_memories.append(belief_memory)
                guest.recent_memories = guest.recent_memories[-12:]
                created.append((guest.guest_id, belief_memory))

        if state.world.turn_count % self.interval == 0:
            created.extend(self.compress_long_term_memory(state))
        return created

    def compress_long_term_memory(
        self, state: SimulationState
    ) -> list[tuple[str, MemoryRecord]]:
        created: list[tuple[str, MemoryRecord]] = []
        for guest_id, guest in state.guests.items():
            compression = self.summarizer.summarize_guest_memory(state, guest_id)
            guest.long_term_memory_summaries.append(compression.summary)
            guest.long_term_memory_summaries.extend(compression.relationship_notes)
            guest.long_term_memory_summaries = guest.long_term_memory_summaries[-12:]
            for belief in compression.beliefs[-2:]:
                if belief not in guest.active_beliefs:
                    guest.active_beliefs.append(belief)
            guest.active_beliefs = guest.active_beliefs[-16:]
            memory = MemoryRecord(
                turn_number=state.world.turn_count,
                category=MemoryCategory.RELATIONSHIP,
                summary=compression.summary,
                salience=0.7,
                tags=["compression"],
            )
            created.append((guest_id, memory))
        return created
