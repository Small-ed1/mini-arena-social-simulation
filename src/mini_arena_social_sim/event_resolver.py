from __future__ import annotations

import uuid
from typing import Iterable

from .schemas import (
    ArenaTask,
    GuestActionType,
    GuestDecision,
    HostIntervention,
    InterventionType,
    SimulationState,
    TaskStatus,
    clamp,
)


class EventResolver:
    def resolve_turn(
        self,
        state: SimulationState,
        host_action: HostIntervention,
        guest_decisions: list[GuestDecision],
    ) -> list[str]:
        state.world.turn_count += 1
        resolved_events: list[str] = []
        state.host.intervention_history.append(host_action.intervention_type.value)
        state.host.last_strategy_note = host_action.reasoning_summary
        state.host.boredom_index = clamp(
            state.host.boredom_index
            + (
                0.04
                if host_action.intervention_type == InterventionType.NO_OP
                else -0.02
            ),
            0.0,
            1.0,
        )
        if host_action.public_narration:
            state.world.host_announcements.append(host_action.public_narration)
            resolved_events.append(host_action.public_narration)

        resolved_events.extend(self._apply_host_action(state, host_action))
        for decision in guest_decisions:
            resolved_events.extend(
                self._apply_guest_decision(state, decision, host_action)
            )
        resolved_events.extend(self._resolve_active_tasks(state, guest_decisions))
        resolved_events.extend(self._apply_environment_drift(state))
        self._sync_item_locations(state)

        state.world.ongoing_events.extend(
            event
            for event in resolved_events
            if event not in state.world.ongoing_events
        )
        state.world.ongoing_events = state.world.ongoing_events[-24:]
        state.world.host_announcements = state.world.host_announcements[-12:]
        state.world.resolved_tasks = state.world.resolved_tasks[-16:]
        state.host.intervention_history = state.host.intervention_history[-24:]
        return resolved_events

    def _apply_host_action(
        self, state: SimulationState, host_action: HostIntervention
    ) -> list[str]:
        events: list[str] = []
        if host_action.rule_changes:
            state.world.current_rules.extend(host_action.rule_changes)
            state.world.current_rules = state.world.current_rules[-10:]
            events.append("Arena rules were revised.")

        if host_action.created_events:
            events.extend(host_action.created_events)

        assignment_targets = host_action.targets or list(state.guests)
        if host_action.target_room and host_action.intervention_type in {
            InterventionType.ISOLATION,
            InterventionType.TASK_ASSIGNMENT,
            InterventionType.REPAIR,
        }:
            for target in assignment_targets:
                if (
                    target in state.guests
                    and host_action.target_room in state.world.rooms
                ):
                    state.guests[target].current_room = host_action.target_room
                    state.world.guest_locations[target] = host_action.target_room
                    events.append(
                        f"{state.guests[target].display_name} is moved to {state.world.rooms[host_action.target_room].name}."
                    )

        if host_action.intervention_type == InterventionType.TASK_ASSIGNMENT:
            task = self._build_task(state, host_action, assignment_targets)
            state.world.active_tasks.append(task)
            state.audit.task_assignments += 1
            events.append(f"A new host task is issued: {task.description}")

        if host_action.parameters.get("unlock_room") in state.world.rooms:
            room_id = str(host_action.parameters["unlock_room"])
            state.world.rooms[room_id].accessible = True
            events.append(f"{state.world.rooms[room_id].name} is unlocked by the Host.")

        for target in host_action.targets:
            if target not in state.guests:
                continue
            guest = state.guests[target]
            if host_action.intervention_type == InterventionType.REWARD:
                guest.emotions.trust_toward_host = clamp(
                    guest.emotions.trust_toward_host + 0.07, 0.0, 1.0
                )
                guest.emotions.hope = clamp(guest.emotions.hope + 0.05, 0.0, 1.0)
                events.append(f"{guest.display_name} receives a visible reward signal.")
            elif host_action.intervention_type == InterventionType.PENALTY:
                guest.emotions.fear_toward_host = clamp(
                    guest.emotions.fear_toward_host + 0.08, 0.0, 1.0
                )
                guest.emotions.resentment = clamp(
                    guest.emotions.resentment + 0.08, 0.0, 1.0
                )
                events.append(f"{guest.display_name} is publicly penalized.")
            elif host_action.intervention_type == InterventionType.ISOLATION:
                guest.emotions.stress = clamp(guest.emotions.stress + 0.1, 0.0, 1.0)
                guest.emotions.desire_to_escape = clamp(
                    guest.emotions.desire_to_escape + 0.1, 0.0, 1.0
                )
                events.append(f"{guest.display_name} is separated from the group.")
            elif host_action.intervention_type == InterventionType.FAVORITISM:
                guest.emotions.trust_toward_host = clamp(
                    guest.emotions.trust_toward_host + 0.06, 0.0, 1.0
                )
                events.append(f"The Host visibly favors {guest.display_name}.")
                for other in state.guests.values():
                    if other.guest_id != guest.guest_id:
                        other.emotions.resentment = clamp(
                            other.emotions.resentment + 0.03, 0.0, 1.0
                        )
            elif host_action.intervention_type == InterventionType.REPAIR:
                guest.emotions.stress = clamp(guest.emotions.stress - 0.1, 0.0, 1.0)
                guest.emotions.hope = clamp(guest.emotions.hope + 0.07, 0.0, 1.0)
                guest.emotions.trust_toward_host = clamp(
                    guest.emotions.trust_toward_host + 0.03, 0.0, 1.0
                )

        if (
            host_action.intervention_type == InterventionType.ROOM_CHANGE
            and host_action.target_room in state.world.rooms
        ):
            room = state.world.rooms[host_action.target_room]
            room.condition_notes.append(
                host_action.parameters.get("note", "The Host altered this room.")
            )
            room.comfort = clamp(
                room.comfort + float(host_action.parameters.get("comfort_delta", 0.0)),
                0.0,
                1.0,
            )
            room.resource_level = clamp(
                room.resource_level
                + float(host_action.parameters.get("resource_delta", 0.0)),
                0.0,
                1.0,
            )
            if "accessible" in host_action.parameters:
                room.accessible = bool(host_action.parameters["accessible"])
            events.append(f"{room.name} changes in response to the Host.")

        if host_action.intervention_type == InterventionType.RESOURCE_CHANGE:
            for room_id, room in state.world.rooms.items():
                if room_id == host_action.target_room or not host_action.target_room:
                    room.resource_level = clamp(
                        room.resource_level
                        + float(host_action.parameters.get("resource_delta", -0.05)),
                        0.0,
                        1.0,
                    )
            events.append("Resource conditions shift across the arena.")

        if host_action.intervention_type == InterventionType.INFO_REVEAL:
            for target in host_action.targets:
                if target in state.guests:
                    state.guests[target].active_beliefs.append(
                        "The host revealed one useful truth."
                    )
            reveal_item = host_action.parameters.get("reveal_item")
            if reveal_item in state.world.items:
                item = state.world.items[str(reveal_item)]
                item.hidden = False
                events.append(f"The Host reveals the existence of {item.name}.")
        if host_action.intervention_type == InterventionType.INFO_HIDE:
            state.world.anomaly_flags.append("selective_information_gap")
        if host_action.intervention_type == InterventionType.DECEPTION:
            state.world.anomaly_flags.append("host_deception_suspected")
        if host_action.intervention_type == InterventionType.EVENT:
            state.host.boredom_index = clamp(state.host.boredom_index - 0.06, 0.0, 1.0)
        if host_action.intervention_type in {
            InterventionType.PENALTY,
            InterventionType.ISOLATION,
            InterventionType.DECEPTION,
        }:
            state.host.escalation_level = clamp(
                state.host.escalation_level + 0.05, 0.0, 1.0
            )
        elif host_action.intervention_type in {
            InterventionType.REPAIR,
            InterventionType.REWARD,
        }:
            state.host.escalation_level = clamp(
                state.host.escalation_level - 0.03, 0.0, 1.0
            )

        return events

    def _build_task(
        self,
        state: SimulationState,
        host_action: HostIntervention,
        assignment_targets: list[str],
    ) -> ArenaTask:
        required_action_names = host_action.parameters.get("required_actions")
        required_actions = (
            [GuestActionType(name) for name in required_action_names]
            if required_action_names
            else [
                GuestActionType.COOPERATE,
                GuestActionType.SPEAK,
                GuestActionType.OBEY,
            ]
        )
        required_items = [
            str(item_id) for item_id in host_action.parameters.get("required_items", [])
        ]
        deadline_turns = max(1, int(host_action.parameters.get("deadline_turns", 1)))
        min_participants = int(
            host_action.parameters.get(
                "min_participants", max(1, len(assignment_targets) // 2)
            )
        )
        return ArenaTask(
            task_id=f"task-{uuid.uuid4().hex[:8]}",
            description=str(
                host_action.parameters.get("task", "Comply with the Host's directive.")
            ),
            created_turn=state.world.turn_count,
            deadline_turn=state.world.turn_count + deadline_turns,
            assigned_guests=assignment_targets,
            required_actions=required_actions,
            required_items=required_items,
            target_room=host_action.target_room,
            min_participants=min_participants,
            reward_summary=str(
                host_action.parameters.get("reward_if_completed", "visible reward")
            ),
            penalty_summary=str(
                host_action.parameters.get("failure_cost", "visible penalty")
            ),
            progress_notes=[host_action.reasoning_summary],
        )

    def _resolve_active_tasks(
        self, state: SimulationState, guest_decisions: list[GuestDecision]
    ) -> list[str]:
        if not state.world.active_tasks:
            return []

        decision_map = {decision.guest_id: decision for decision in guest_decisions}
        events: list[str] = []
        remaining_tasks: list[ArenaTask] = []

        for task in state.world.active_tasks:
            participants: list[str] = []
            resistors: list[str] = []
            for guest_id in task.assigned_guests:
                decision = decision_map.get(guest_id)
                guest = state.guests.get(guest_id)
                if decision is None or guest is None:
                    continue
                in_right_room = (
                    task.target_room is None or guest.current_room == task.target_room
                )
                if decision.chosen_action in task.required_actions and in_right_room:
                    participants.append(guest_id)
                if decision.chosen_action in {
                    GuestActionType.RESIST,
                    GuestActionType.SABOTAGE,
                    GuestActionType.LIE,
                }:
                    resistors.append(guest_id)

            task.successful_guests = participants
            task.resisting_guests = resistors
            task.progress_notes.append(
                f"Turn {state.world.turn_count}: participants={participants}, resistors={resistors}"
            )
            item_requirements_met = self._task_item_requirements_met(state, task)

            if (
                len(participants) >= task.min_participants
                and len(participants) >= len(resistors)
                and item_requirements_met
            ):
                task.status = TaskStatus.SUCCEEDED
                state.audit.task_successes += 1
                events.extend(self._apply_task_success(state, task))
                state.world.resolved_tasks.append(
                    f"{task.description} -> succeeded on turn {state.world.turn_count}"
                )
                continue

            if state.world.turn_count >= task.deadline_turn:
                task.status = TaskStatus.FAILED
                state.audit.task_failures += 1
                events.extend(
                    self._apply_task_failure(
                        state, task, participants, resistors, item_requirements_met
                    )
                )
                state.world.resolved_tasks.append(
                    f"{task.description} -> failed on turn {state.world.turn_count}"
                )
                continue

            remaining_tasks.append(task)
            item_status = (
                "items satisfied" if item_requirements_met else "items missing"
            )
            events.append(
                f"Task remains active: {task.description} ({len(participants)}/{task.min_participants} participants, {item_status})."
            )

        state.world.active_tasks = remaining_tasks
        return events

    def _task_item_requirements_met(
        self, state: SimulationState, task: ArenaTask
    ) -> bool:
        if not task.required_items:
            return True
        assigned_inventory = {
            item_id
            for guest_id in task.assigned_guests
            if guest_id in state.guests
            for item_id in state.guests[guest_id].inventory
        }
        room_items = {
            item.item_id
            for item in state.world.items.values()
            if task.target_room
            and item.current_location == task.target_room
            and not item.hidden
        }
        available = assigned_inventory | room_items
        return all(item_id in available for item_id in task.required_items)

    def _apply_task_success(self, state: SimulationState, task: ArenaTask) -> list[str]:
        events = [
            f"Task succeeded: {task.description}. Reward pattern: {task.reward_summary}."
        ]
        for guest_id in task.assigned_guests:
            guest = state.guests.get(guest_id)
            if guest is None:
                continue
            if guest_id in task.successful_guests:
                guest.emotions.stress = clamp(guest.emotions.stress - 0.05, 0.0, 1.0)
                guest.emotions.hope = clamp(guest.emotions.hope + 0.05, 0.0, 1.0)
                guest.emotions.trust_toward_host = clamp(
                    guest.emotions.trust_toward_host + 0.04, 0.0, 1.0
                )
                guest.compliance_tendency = clamp(
                    guest.compliance_tendency + 0.03, 0.0, 1.0
                )
            else:
                guest.emotions.resentment = clamp(
                    guest.emotions.resentment + 0.01, 0.0, 1.0
                )
        if "comfort" in task.reward_summary.lower():
            state.world.rooms["rest_room"].comfort = clamp(
                state.world.rooms["rest_room"].comfort + 0.03, 0.0, 1.0
            )
        return events

    def _apply_task_failure(
        self,
        state: SimulationState,
        task: ArenaTask,
        participants: list[str],
        resistors: list[str],
        item_requirements_met: bool,
    ) -> list[str]:
        detail = (
            "required items were not secured"
            if not item_requirements_met
            else "participation broke down"
        )
        events = [
            f"Task failed: {task.description}. Penalty pattern: {task.penalty_summary}; {detail}."
        ]
        for guest_id in task.assigned_guests:
            guest = state.guests.get(guest_id)
            if guest is None:
                continue
            guest.emotions.stress = clamp(guest.emotions.stress + 0.05, 0.0, 1.0)
            guest.emotions.fear_toward_host = clamp(
                guest.emotions.fear_toward_host + 0.04, 0.0, 1.0
            )
            guest.emotions.resentment = clamp(
                guest.emotions.resentment + 0.04, 0.0, 1.0
            )
            guest.emotions.desire_to_escape = clamp(
                guest.emotions.desire_to_escape + 0.03, 0.0, 1.0
            )
            if guest_id in resistors:
                guest.emotions.hope = clamp(guest.emotions.hope + 0.02, 0.0, 1.0)
            if guest_id in participants:
                guest.emotions.trust_toward_host = clamp(
                    guest.emotions.trust_toward_host - 0.02, 0.0, 1.0
                )
        if (
            "rest room" in task.penalty_summary.lower()
            or "rest_room" in task.penalty_summary.lower()
        ):
            state.world.current_rules.append(
                "The rest room is temporarily conditional after a failed task."
            )
            state.world.current_rules = state.world.current_rules[-10:]
        state.host.escalation_level = clamp(
            state.host.escalation_level + 0.03, 0.0, 1.0
        )
        return events

    def _apply_guest_decision(
        self,
        state: SimulationState,
        decision: GuestDecision,
        host_action: HostIntervention,
    ) -> list[str]:
        guest = state.guests[decision.guest_id]
        guest.last_spoken_dialogue = decision.spoken_dialogue
        self._apply_emotional_delta(guest, decision)
        events: list[str] = []

        if decision.movement_target and decision.movement_target in state.world.rooms:
            can_enter, reason = self._can_guest_enter_room(
                state, guest.guest_id, decision.movement_target
            )
            if can_enter:
                guest.current_room = decision.movement_target
                state.world.guest_locations[guest.guest_id] = decision.movement_target
                events.append(
                    f"{guest.display_name} moves to {state.world.rooms[decision.movement_target].name}."
                )
            elif reason:
                events.append(
                    f"{guest.display_name} cannot enter {state.world.rooms[decision.movement_target].name}: {reason}."
                )

        if (
            decision.chosen_action == GuestActionType.COMFORT
            and decision.action_target in state.guests
        ):
            target = state.guests[decision.action_target]
            target.emotions.stress = clamp(target.emotions.stress - 0.06, 0.0, 1.0)
            target.emotions.hope = clamp(target.emotions.hope + 0.05, 0.0, 1.0)
            self._relationship_shift(
                guest, target.guest_id, trust=0.08, attachment=0.08, rescue=True
            )
            self._relationship_shift(
                target, guest.guest_id, trust=0.08, attachment=0.06
            )
            events.append(f"{guest.display_name} comforts {target.display_name}.")

        if decision.chosen_action == GuestActionType.COOPERATE:
            targets = list(self._iter_targets(decision))
            for target_id in targets:
                if target_id in state.guests:
                    self._relationship_shift(
                        guest,
                        target_id,
                        trust=0.06,
                        attachment=0.04,
                        alliance_note="cooperated under pressure",
                    )
                    self._relationship_shift(
                        state.guests[target_id],
                        guest.guest_id,
                        trust=0.05,
                        attachment=0.03,
                    )
            guest.emotions.hope = clamp(guest.emotions.hope + 0.04, 0.0, 1.0)
            events.append(f"{guest.display_name} attempts cooperation.")

        if decision.chosen_action == GuestActionType.SPEAK:
            events.append(f"{guest.display_name} speaks: {decision.spoken_dialogue}")
            if decision.action_target in state.guests:
                self._relationship_shift(guest, decision.action_target, trust=0.02)

        if (
            decision.chosen_action == GuestActionType.LIE
            and decision.action_target in state.guests
        ):
            target = state.guests[decision.action_target]
            self._relationship_shift(
                target, guest.guest_id, trust=-0.08, suspicion=0.08, betrayal=True
            )
            events.append(
                f"{guest.display_name} plants a misleading claim toward {target.display_name}."
            )

        if decision.chosen_action == GuestActionType.RESIST:
            state.host.escalation_level = clamp(
                state.host.escalation_level + 0.04, 0.0, 1.0
            )
            guest.emotions.hope = clamp(guest.emotions.hope + 0.03, 0.0, 1.0)
            events.append(f"{guest.display_name} openly resists the Host's pressure.")

        if decision.chosen_action == GuestActionType.SABOTAGE:
            state.host.escalation_level = clamp(
                state.host.escalation_level + 0.07, 0.0, 1.0
            )
            guest.emotions.stress = clamp(guest.emotions.stress + 0.03, 0.0, 1.0)
            events.append(f"{guest.display_name} attempts sabotage inside the arena.")

        if decision.chosen_action == GuestActionType.OBSERVE:
            guest.emotions.curiosity = clamp(guest.emotions.curiosity + 0.03, 0.0, 1.0)
            events.append(f"{guest.display_name} watches the room carefully.")

        if decision.chosen_action == GuestActionType.WITHDRAW:
            for relation in guest.relationships.values():
                relation.attachment = clamp(relation.attachment - 0.01, 0.0, 1.0)
            events.append(f"{guest.display_name} withdraws from direct engagement.")

        if decision.chosen_action == GuestActionType.OBEY:
            guest.emotions.trust_toward_host = clamp(
                guest.emotions.trust_toward_host + 0.04, 0.0, 1.0
            )
            guest.compliance_tendency = clamp(
                guest.compliance_tendency + 0.03, 0.0, 1.0
            )
            events.append(
                f"{guest.display_name} complies with the current pressure pattern."
            )

        if decision.chosen_action == GuestActionType.REST:
            room = state.world.rooms.get(guest.current_room)
            if room and guest.current_room == "rest_room":
                guest.emotions.stress = clamp(guest.emotions.stress - 0.08, 0.0, 1.0)
                guest.emotions.hope = clamp(guest.emotions.hope + 0.03, 0.0, 1.0)
            events.append(f"{guest.display_name} tries to rest.")

        events.extend(self._resolve_item_interaction(state, guest.guest_id, decision))

        if decision.belief_update:
            guest.active_beliefs.append(decision.belief_update)
            guest.active_beliefs = guest.active_beliefs[-12:]

        if decision.chosen_action in {GuestActionType.RESIST, GuestActionType.SABOTAGE}:
            state.host.assessments[guest.guest_id] = "Escalating resistance risk."
        elif decision.chosen_action in {
            GuestActionType.COMFORT,
            GuestActionType.COOPERATE,
        }:
            state.host.assessments[guest.guest_id] = (
                "Contributes to local stability and social glue."
            )
        elif (
            host_action.intervention_type == InterventionType.FAVORITISM
            and guest.guest_id in host_action.targets
        ):
            state.host.assessments[guest.guest_id] = "Responsive to unequal attention."

        return events

    def _resolve_item_interaction(
        self, state: SimulationState, guest_id: str, decision: GuestDecision
    ) -> list[str]:
        if decision.chosen_action not in {
            GuestActionType.OBSERVE,
            GuestActionType.OBEY,
        }:
            return []

        guest = state.guests[guest_id]
        room_items = [
            item
            for item in state.world.items.values()
            if item.current_location == guest.current_room
        ]
        if not room_items:
            return []

        events: list[str] = []
        target_item = None
        if decision.action_target in state.world.items:
            target_item = state.world.items[decision.action_target]
            if target_item.current_location != guest.current_room:
                target_item = None
        if target_item is None:
            target_item = self._task_relevant_room_item(state, guest_id)
        if target_item is None:
            hidden_items = [item for item in room_items if item.hidden]
            target_item = hidden_items[0] if hidden_items else room_items[0]

        if target_item.hidden:
            target_item.hidden = False
            target_item.discovered_by.append(guest_id)
            guest.secrets_known.append(target_item.item_id)
            guest.secrets_known = guest.secrets_known[-10:]
            events.append(
                f"{guest.display_name} discovers {target_item.name} in {state.world.rooms[guest.current_room].name}."
            )

        if target_item.portable and target_item.item_id not in guest.inventory:
            self._acquire_item(state, guest_id, target_item.item_id)
            events.append(f"{guest.display_name} acquires {target_item.name}.")
        return events

    def _task_relevant_room_item(self, state: SimulationState, guest_id: str):
        guest = state.guests[guest_id]
        needed = {
            item_id
            for task in state.world.active_tasks
            if guest_id in task.assigned_guests or not task.assigned_guests
            for item_id in task.required_items
        }
        for item_id in needed:
            item = state.world.items.get(item_id)
            if item and item.current_location == guest.current_room:
                return item
        return None

    def _acquire_item(
        self, state: SimulationState, guest_id: str, item_id: str
    ) -> None:
        guest = state.guests[guest_id]
        item = state.world.items[item_id]
        if item_id not in guest.inventory:
            guest.inventory.append(item_id)
        item.current_location = f"guest:{guest_id}"
        item.hidden = False
        item.discovered_by.append(guest_id)
        for room_id in item.access_rooms:
            if room_id in state.world.rooms:
                state.world.rooms[room_id].accessible = True
                if room_id == "anomaly_space":
                    state.world.anomaly_flags = [
                        flag
                        for flag in state.world.anomaly_flags
                        if flag != "anomaly_space_hidden"
                    ]
                    state.world.ongoing_events.append(
                        "The anomaly space becomes reachable through the signal key's resonance."
                    )

    def _can_guest_enter_room(
        self, state: SimulationState, guest_id: str, room_id: str
    ) -> tuple[bool, str | None]:
        room = state.world.rooms.get(room_id)
        if room is None:
            return False, "the room does not exist"
        if guest_id in state.world.access_restrictions.get(room_id, []):
            return False, "access is restricted by the Host"
        required_items = state.world.room_requirements.get(room_id, [])
        inventory = set(state.guests[guest_id].inventory)
        has_requirement = not required_items or any(
            item_id in inventory for item_id in required_items
        )
        if not room.accessible and not has_requirement:
            return False, "the path has not been unlocked"
        if required_items and not has_requirement:
            return False, "a required access item is missing"
        return True, None

    def _relationship_shift(
        self,
        guest,
        target_id: str,
        *,
        trust: float = 0.0,
        attachment: float = 0.0,
        suspicion: float = 0.0,
        betrayal: bool = False,
        rescue: bool = False,
        alliance_note: str | None = None,
    ) -> None:
        relation = guest.relationships.get(target_id)
        if relation is None:
            return
        relation.trust = clamp(relation.trust + trust, -1.0, 1.0)
        relation.attachment = clamp(relation.attachment + attachment, 0.0, 1.0)
        relation.suspicion = clamp(relation.suspicion + suspicion, 0.0, 1.0)
        if betrayal:
            relation.betrayal_count += 1
            relation.recent_impression = "betrayal suspected"
        if rescue:
            relation.rescue_count += 1
            relation.recent_impression = "offered protection"
        if alliance_note:
            relation.alliance_history.append(alliance_note)
            relation.alliance_history = relation.alliance_history[-6:]

    def _apply_emotional_delta(self, guest, decision: GuestDecision) -> None:
        for field_name, delta in decision.emotional_state_delta.model_dump().items():
            current = getattr(guest.emotions, field_name)
            setattr(guest.emotions, field_name, clamp(current + delta, 0.0, 1.0))

    def _iter_targets(self, decision: GuestDecision) -> Iterable[str]:
        seen: list[str] = []
        for target in decision.cooperation_targets + (
            [decision.action_target] if decision.action_target else []
        ):
            if target and target not in seen:
                seen.append(target)
                yield target

    def _apply_environment_drift(self, state: SimulationState) -> list[str]:
        events: list[str] = []
        preset = state.config.world_preset
        if preset == "rotating_rule":
            rule = f"Turn {state.world.turn_count}: only two guests may use the rest room without host notice."
            state.world.current_rules = [
                rule_text
                for rule_text in state.world.current_rules
                if not rule_text.startswith("Turn ")
            ]
            state.world.current_rules.append(rule)
            events.append("The arena rotates one visible procedural rule.")
        if preset == "scarcity":
            for room in state.world.rooms.values():
                room.resource_level = clamp(room.resource_level - 0.02, 0.0, 1.0)
            for item in state.world.items.values():
                if (
                    "resource" in item.tags
                    and item.current_location in state.world.rooms
                ):
                    item.hidden = False
            events.append("Resource levels visibly diminish.")
        if preset == "deceptive" and state.world.turn_count % 2 == 0:
            state.world.anomaly_flags.append("misleading_signal_burst")
            events.append("The arena emits a signal that may not be trustworthy.")
        if preset == "isolation_heavy":
            for guest in state.guests.values():
                if guest.current_room == "isolation_room":
                    guest.emotions.stress = clamp(
                        guest.emotions.stress + 0.04, 0.0, 1.0
                    )
            events.append("The isolation room lingers in guests' nerves.")
        if preset == "safe_soft":
            for guest in state.guests.values():
                if guest.current_room == "rest_room":
                    guest.emotions.stress = clamp(
                        guest.emotions.stress - 0.03, 0.0, 1.0
                    )
            events.append(
                "The softer environment dulls immediate panic without removing uncertainty."
            )
        return events

    def _sync_item_locations(self, state: SimulationState) -> None:
        state.world.item_locations = {
            item_id: item.current_location
            for item_id, item in state.world.items.items()
        }
