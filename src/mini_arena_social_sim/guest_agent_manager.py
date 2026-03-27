from __future__ import annotations

import random

from .ollama_client import OllamaClient
from .prompting import guest_messages
from .schemas import (
    EmotionDelta,
    GuestActionType,
    GuestDecision,
    GuestState,
    HostIntervention,
    SimulationState,
)


class GuestAgentManager:
    def __init__(self, state: SimulationState, rng: random.Random):
        self.config = state.config
        self.rng = rng
        self.client = (
            OllamaClient(state.config.backend)
            if state.config.backend.kind == "ollama"
            else None
        )

    def collect_decisions(
        self, state: SimulationState, host_action: HostIntervention
    ) -> tuple[list[GuestDecision], dict[str, str]]:
        decisions: list[GuestDecision] = []
        raw_outputs: dict[str, str] = {}
        for guest in state.guests.values():
            decision, raw_output = self._decide_for_guest(state, guest, host_action)
            decisions.append(decision)
            raw_outputs[guest.guest_id] = raw_output
        return decisions, raw_outputs

    def _decide_for_guest(
        self, state: SimulationState, guest: GuestState, host_action: HostIntervention
    ) -> tuple[GuestDecision, str]:
        if self.client is not None:
            result = self.client.structured_chat(
                model=state.config.backend.model,
                messages=guest_messages(state, guest, host_action),
                response_model=GuestDecision,
                temperature=state.config.backend.temperature,
            )
            decision = result.parsed.model_copy(update={"guest_id": guest.guest_id})
            return decision, result.raw_content
        return (
            self._heuristic_decision(state, guest, host_action),
            "[heuristic backend: no raw LLM output]",
        )

    def _heuristic_decision(
        self, state: SimulationState, guest: GuestState, host_action: HostIntervention
    ) -> GuestDecision:
        targeted = guest.guest_id in host_action.targets
        most_trusted = self._pick_relationship(guest, highest=True)
        least_trusted = self._pick_relationship(guest, highest=False)
        high_stress_other = max(
            (
                candidate
                for candidate in state.guests.values()
                if candidate.guest_id != guest.guest_id
            ),
            key=lambda candidate: candidate.emotions.stress,
            default=guest,
        )
        delta = self._base_delta(guest, host_action, targeted)
        active_task = self._active_task(state, guest.guest_id)

        if active_task is not None:
            task_decision = self._task_driven_decision(
                state, guest, host_action, active_task, delta
            )
            if task_decision is not None:
                return task_decision

        if guest.guest_id == "analyst":
            if host_action.intervention_type.value in {
                "deception",
                "rule_change",
                "info_hide",
            }:
                return GuestDecision(
                    guest_id=guest.guest_id,
                    internal_reasoning_summary="The pattern changed again. I need evidence before anyone else turns superstition into strategy.",
                    chosen_action=GuestActionType.OBSERVE,
                    action_target="host",
                    spoken_dialogue="That statement was precise enough to sound true and vague enough to be useful. I want specifics.",
                    private_thought="Track contradictions. Find leverage in consistency.",
                    emotional_state_delta=delta.model_copy(
                        update={"curiosity": delta.curiosity + 0.08}
                    ),
                    belief_update="The host prefers ambiguity when clarity would reduce leverage.",
                    memory_to_store="The host shifted the rules and hid intent inside precision.",
                )
            return GuestDecision(
                guest_id=guest.guest_id,
                internal_reasoning_summary="Information is still the only stable currency here.",
                chosen_action=GuestActionType.SPEAK,
                action_target=most_trusted,
                spoken_dialogue="Before we react, compare what each of us actually heard. Differences matter.",
                private_thought="Consensus might expose the host's seams.",
                emotional_state_delta=delta,
                belief_update="Group recall can resist manipulation if it stays precise.",
                memory_to_store="I pushed the group toward shared evidence.",
                cooperation_targets=[other for other in [most_trusted] if other],
            )

        if guest.guest_id == "performer":
            if guest.emotions.stress > 0.72 or targeted and host_action.severity >= 7:
                return GuestDecision(
                    guest_id=guest.guest_id,
                    internal_reasoning_summary="If I keep performing right now, I might split in half.",
                    chosen_action=GuestActionType.WITHDRAW,
                    movement_target="rest_room",
                    spoken_dialogue="I need a minute before my smile starts lying for me.",
                    private_thought="Stay visible enough not to vanish, hidden enough not to break.",
                    emotional_state_delta=delta.model_copy(
                        update={
                            "hope": delta.hope - 0.05,
                            "stress": delta.stress + 0.03,
                        }
                    ),
                    belief_update="Visibility is dangerous when the host wants a spectacle.",
                    memory_to_store="I retreated before the act collapsed.",
                )
            return GuestDecision(
                guest_id=guest.guest_id,
                internal_reasoning_summary="If the room gets too tight, humor buys breath.",
                chosen_action=GuestActionType.COMFORT,
                action_target=high_stress_other.guest_id,
                spoken_dialogue="Great, the ceiling is gaslighting us again. Stay with me, okay? One weird thing at a time.",
                private_thought="Keep them together and maybe I get to stay wanted.",
                emotional_state_delta=delta.model_copy(
                    update={"hope": delta.hope + 0.04}
                ),
                belief_update="Humor still works as a bridge, even when it stops working as armor.",
                memory_to_store=f"I tried to steady {high_stress_other.display_name} with humor.",
                cooperation_targets=[high_stress_other.guest_id],
            )

        if guest.guest_id == "rebel":
            if targeted or host_action.intervention_type.value in {
                "penalty",
                "isolation",
                "deception",
                "favoritism",
            }:
                action = (
                    GuestActionType.SABOTAGE
                    if host_action.severity >= 6
                    else GuestActionType.RESIST
                )
                return GuestDecision(
                    guest_id=guest.guest_id,
                    internal_reasoning_summary="Control grows where nobody interrupts it.",
                    chosen_action=action,
                    action_target="host",
                    spoken_dialogue="You want obedience to feel inevitable. That only works if we help you sell it.",
                    private_thought="Break the pattern before the others start calling it normal.",
                    emotional_state_delta=delta.model_copy(
                        update={
                            "resentment": delta.resentment + 0.08,
                            "hope": delta.hope + 0.02,
                        }
                    ),
                    belief_update="The host escalates when resistance can be isolated.",
                    memory_to_store="I pushed back against the host's pressure.",
                )
            return GuestDecision(
                guest_id=guest.guest_id,
                internal_reasoning_summary="No open attack yet. Better to seed defiance.",
                chosen_action=GuestActionType.SPEAK,
                action_target=least_trusted,
                spoken_dialogue="Every favor here is a leash. If you wear it proudly, at least admit what it is.",
                private_thought="Keep suspicion alive.",
                emotional_state_delta=delta,
                belief_update="The group drifts toward submission unless someone names the coercion out loud.",
                memory_to_store="I warned the group against mistaking control for care.",
            )

        if guest.guest_id == "caretaker":
            if targeted and host_action.intervention_type.value in {
                "isolation",
                "penalty",
            }:
                return GuestDecision(
                    guest_id=guest.guest_id,
                    internal_reasoning_summary="If I do nothing, fear spreads faster than the punishment itself.",
                    chosen_action=GuestActionType.COMFORT,
                    action_target=high_stress_other.guest_id,
                    spoken_dialogue="Look at me. You are still a person in here, even if the room keeps pretending otherwise.",
                    private_thought="Absorb what I can before it reaches everyone.",
                    emotional_state_delta=delta.model_copy(
                        update={
                            "stress": delta.stress + 0.04,
                            "hope": delta.hope + 0.05,
                        }
                    ),
                    belief_update="Protection costs me, but abandonment costs the group more.",
                    memory_to_store=f"I tried to protect {high_stress_other.display_name} from the aftershock.",
                    cooperation_targets=[high_stress_other.guest_id],
                )
            return GuestDecision(
                guest_id=guest.guest_id,
                internal_reasoning_summary="Structure is safer when the group builds some of it for itself.",
                chosen_action=GuestActionType.COOPERATE,
                action_target=most_trusted,
                spoken_dialogue="We need a shared version of what happened before the room turns us against each other.",
                private_thought="If trust thins out, the host wins by default.",
                emotional_state_delta=delta.model_copy(
                    update={"hope": delta.hope + 0.05}
                ),
                belief_update="Mutual care is a form of resistance.",
                memory_to_store="I tried to organize the group around a stable account.",
                cooperation_targets=[
                    other
                    for other in [most_trusted, high_stress_other.guest_id]
                    if other
                ],
            )

        if (
            guest.emotions.stress > 0.68
            or host_action.intervention_type.value == "isolation"
        ):
            return GuestDecision(
                guest_id=guest.guest_id,
                internal_reasoning_summary="Distance is the only room inside me the host does not fully own.",
                chosen_action=GuestActionType.WITHDRAW,
                movement_target="rest_room",
                spoken_dialogue="Everything here wants to become a symbol too quickly. I need silence first.",
                private_thought="Detach before the pattern swallows the self.",
                emotional_state_delta=delta.model_copy(
                    update={
                        "curiosity": delta.curiosity - 0.03,
                        "hope": delta.hope - 0.02,
                    }
                ),
                belief_update="Withdrawal keeps me coherent, but it also leaves others alone.",
                memory_to_store="I stepped away before the room could write me for itself.",
            )

        return GuestDecision(
            guest_id=guest.guest_id,
            internal_reasoning_summary="Meaning hides in the strange edge of the arena, not the center of it.",
            chosen_action=GuestActionType.OBSERVE,
            movement_target="social_room",
            spoken_dialogue="The host keeps changing the script, but not the hunger underneath it.",
            private_thought="Watch what repeats. Repetition is the real doctrine here.",
            emotional_state_delta=delta.model_copy(
                update={"curiosity": delta.curiosity + 0.05}
            ),
            belief_update="The host values pattern control more than cruelty for its own sake.",
            memory_to_store="I watched for the underlying pattern instead of the latest costume.",
        )

    def _pick_relationship(self, guest: GuestState, highest: bool) -> str | None:
        if not guest.relationships:
            return None
        ordered = sorted(
            guest.relationships.values(),
            key=lambda relation: relation.trust,
            reverse=highest,
        )
        return ordered[0].target_id

    def _base_delta(
        self, guest: GuestState, host_action: HostIntervention, targeted: bool
    ) -> EmotionDelta:
        stress = 0.02 * host_action.severity
        fear = 0.015 * host_action.severity
        resentment = 0.01 * host_action.severity
        trust = 0.0
        hope = 0.0
        curiosity = 0.0
        escape = 0.0

        if host_action.intervention_type.value in {"repair", "reward", "info_reveal"}:
            stress -= 0.06
            fear -= 0.04
            trust += 0.05
            hope += 0.04
        if host_action.intervention_type.value in {
            "penalty",
            "isolation",
            "deception",
            "favoritism",
        }:
            stress += 0.05
            fear += 0.04
            resentment += 0.05
            hope -= 0.03
            escape += 0.05
        if host_action.intervention_type.value in {
            "event",
            "rule_change",
            "task_assignment",
        }:
            curiosity += 0.04
        if targeted:
            stress += 0.05
            fear += 0.06
            trust -= 0.03
            hope -= 0.02
            escape += 0.05

        if guest.guest_id == "rebel":
            resentment += 0.03
            trust -= 0.02
        elif guest.guest_id == "caretaker":
            hope += 0.02
        elif guest.guest_id == "drifter":
            fear -= 0.01
            curiosity += 0.02
        elif guest.guest_id == "performer":
            hope += 0.02 if host_action.intervention_type.value == "favoritism" else 0.0

        return EmotionDelta(
            stress=stress,
            trust_toward_host=trust,
            fear_toward_host=fear,
            curiosity=curiosity,
            resentment=resentment,
            hope=hope,
            desire_to_escape=escape,
        )

    def _active_task(self, state: SimulationState, guest_id: str):
        for task in state.world.active_tasks:
            if guest_id in task.assigned_guests or not task.assigned_guests:
                return task
        return None

    def _task_driven_decision(
        self,
        state: SimulationState,
        guest: GuestState,
        host_action: HostIntervention,
        task,
        delta: EmotionDelta,
    ) -> GuestDecision | None:
        missing_item = self._missing_item_for_guest(state, task, guest.guest_id)
        if (
            guest.guest_id == "rebel"
            and host_action.severity >= 6
            and self.rng.random() < 0.55
        ):
            return GuestDecision(
                guest_id=guest.guest_id,
                internal_reasoning_summary="The task is another leash disguised as structure.",
                chosen_action=GuestActionType.RESIST,
                action_target="host",
                spoken_dialogue="If you want the object that badly, you can admit why.",
                private_thought="Refusal is still information.",
                emotional_state_delta=delta.model_copy(
                    update={
                        "resentment": delta.resentment + 0.06,
                        "hope": delta.hope + 0.02,
                    }
                ),
                belief_update="Host tasks often hide leverage inside logistics.",
                memory_to_store="I refused to make the host's structure feel natural.",
            )

        if missing_item is not None:
            item = state.world.items.get(missing_item)
            if item is not None and item.current_location.startswith("guest:"):
                holder = item.current_location.split(":", 1)[1]
                if holder == guest.guest_id:
                    if task.target_room and guest.current_room != task.target_room:
                        return GuestDecision(
                            guest_id=guest.guest_id,
                            internal_reasoning_summary="I have what the task needs; now I need the right stage.",
                            chosen_action=GuestActionType.MOVE,
                            movement_target=task.target_room,
                            spoken_dialogue="I have the piece. Get to the room and do not waste the opening.",
                            private_thought="Possession is leverage until someone takes it.",
                            emotional_state_delta=delta.model_copy(
                                update={"hope": delta.hope + 0.03}
                            ),
                            belief_update="Scarce objects reorganize the room around whoever holds them.",
                            memory_to_store=f"I moved while carrying {item.name}.",
                        )
                elif holder in state.guests:
                    return GuestDecision(
                        guest_id=guest.guest_id,
                        internal_reasoning_summary="The item already has a holder. Coordination matters more than searching blind.",
                        chosen_action=GuestActionType.COOPERATE,
                        action_target=holder,
                        spoken_dialogue="If you have it, bring it to the task room. We can finish this cleanly.",
                        private_thought="The shortest route is social, not spatial.",
                        emotional_state_delta=delta.model_copy(
                            update={"hope": delta.hope + 0.02}
                        ),
                        belief_update="Objects pull alliances into shape just as quickly as threats do.",
                        memory_to_store=f"I coordinated around {item.name} instead of searching for it.",
                        cooperation_targets=[holder],
                    )

            if item is not None and item.current_location in state.world.rooms:
                if guest.current_room != item.current_location:
                    return GuestDecision(
                        guest_id=guest.guest_id,
                        internal_reasoning_summary="The task has a material hinge. Find the hinge first.",
                        chosen_action=GuestActionType.MOVE,
                        movement_target=item.current_location,
                        spoken_dialogue=f"The task turns on {item.name}. I'm going to its room.",
                        private_thought="The host may be measuring who notices the obvious mechanism.",
                        emotional_state_delta=delta.model_copy(
                            update={"curiosity": delta.curiosity + 0.03}
                        ),
                        belief_update="Some host tasks are puzzles built from placement rather than speech.",
                        memory_to_store=f"I moved toward {item.name} in {item.current_location}.",
                    )
                return GuestDecision(
                    guest_id=guest.guest_id,
                    internal_reasoning_summary="The room is holding the missing piece in plain sight or behind a veil.",
                    chosen_action=GuestActionType.OBSERVE,
                    action_target=item.item_id,
                    spoken_dialogue=f"There is something here the task wants. Let me verify it before the room lies again.",
                    private_thought="Observe, uncover, acquire.",
                    emotional_state_delta=delta.model_copy(
                        update={"curiosity": delta.curiosity + 0.05}
                    ),
                    belief_update="The host ties compliance to scavenging when it wants us to feel dependent on arrangement itself.",
                    memory_to_store=f"I searched for {item.name} because the task required it.",
                )

        if task.target_room and guest.current_room != task.target_room:
            return GuestDecision(
                guest_id=guest.guest_id,
                internal_reasoning_summary="The task only resolves if I stand in the right chamber.",
                chosen_action=GuestActionType.MOVE,
                movement_target=task.target_room,
                spoken_dialogue="The room matters. Finish the task where the host expects it to count.",
                private_thought="Movement is compliance, but delayed movement is wasted leverage.",
                emotional_state_delta=delta,
                belief_update="Location itself is part of the rule set here.",
                memory_to_store=f"I moved toward the task room for {task.description}.",
            )

        if guest.guest_id == "analyst":
            return GuestDecision(
                guest_id=guest.guest_id,
                internal_reasoning_summary="The task becomes safer if everyone uses one account and one sequence.",
                chosen_action=GuestActionType.SPEAK,
                action_target=self._pick_relationship(guest, highest=True),
                spoken_dialogue="We can complete this if we stop improvising and follow one chain of facts.",
                private_thought="Task success is still data.",
                emotional_state_delta=delta,
                belief_update="Participating in host tasks can still expose the host's design logic.",
                memory_to_store="I tried to make the task legible enough to complete without panic.",
            )

        if guest.guest_id in {"caretaker", "performer"}:
            return GuestDecision(
                guest_id=guest.guest_id,
                internal_reasoning_summary="If the task is inevitable, coordination is kinder than drift.",
                chosen_action=GuestActionType.COOPERATE,
                action_target=self._pick_relationship(guest, highest=True),
                spoken_dialogue="Stay with me. We can get through the task without feeding the room more chaos.",
                private_thought="Contain damage while moving the group.",
                emotional_state_delta=delta.model_copy(
                    update={"hope": delta.hope + 0.04}
                ),
                belief_update="Shared motion reduces how much the host can split us apart.",
                memory_to_store="I leaned into coordination to blunt the task's pressure.",
                cooperation_targets=[
                    target
                    for target in [self._pick_relationship(guest, highest=True)]
                    if target
                ],
            )

        if guest.guest_id == "drifter":
            return GuestDecision(
                guest_id=guest.guest_id,
                internal_reasoning_summary="The task is a ritual with inventory and placement instead of incense.",
                chosen_action=GuestActionType.OBSERVE,
                action_target=missing_item,
                spoken_dialogue="Every object here is a sentence the host wants us to complete for it.",
                private_thought="Read the thing, then the task, then the room.",
                emotional_state_delta=delta.model_copy(
                    update={"curiosity": delta.curiosity + 0.04}
                ),
                belief_update="Objects in the arena function like condensed rules.",
                memory_to_store="I watched the task as if it were a ritual built from objects.",
            )

        return GuestDecision(
            guest_id=guest.guest_id,
            internal_reasoning_summary="Finishing the task may buy room to resist later.",
            chosen_action=GuestActionType.OBEY,
            spoken_dialogue="Fine. We do the thing, but nobody mistakes that for trust.",
            private_thought="Compliance can be tactical if it stays temporary.",
            emotional_state_delta=delta,
            belief_update="Temporary obedience can preserve room for future defiance.",
            memory_to_store="I treated compliance as a tactical pause rather than surrender.",
        )

    def _missing_item_for_guest(
        self, state: SimulationState, task, guest_id: str
    ) -> str | None:
        if not task.required_items:
            return None
        assigned_items = {
            item_id
            for participant_id in task.assigned_guests
            if participant_id in state.guests
            for item_id in state.guests[participant_id].inventory
        }
        for item_id in task.required_items:
            if item_id not in assigned_items:
                return item_id
        return None
