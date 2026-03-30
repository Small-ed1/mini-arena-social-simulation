from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from sim.schemas import (
    GuestSnapshot,
    HostAction,
    HostAllocateSpotlight,
    HostInjectProp,
    HostRequestReflection,
    HostSignalStyle,
    HostSpawnEvent,
    ObservationGuest,
    ObservationHost,
    PropSnapshot,
    ThreadSnapshot,
)
from sim.world_state import OpenThread, Prop, WorldState, _round_float


@dataclass
class EnvResult:
    success: bool
    messages: List[str]


class Environment:
    def observe_host(self, world: WorldState, memory) -> ObservationHost:
        mem_chunks = []
        if memory is not None:
            mem_chunks = memory.retrieve_for_host(world)

        guests = []
        for gid in world.guest_order():
            g = world.guests[gid]
            guests.append(
                GuestSnapshot(
                    guest_id=gid,
                    name=g.name,
                    location=g.location,
                    mood={k: _round_float(g.mood.get(k, 0.0)) for k in g.mood},
                    goal=g.current_goal,
                    spotlight_weight=_round_float(g.spotlight_weight),
                    last_action=g.last_action,
                )
            )

        open_threads = [
            ThreadSnapshot(
                thread_id=t.thread_id,
                thread_type=t.thread_type,  # type: ignore[arg-type]
                status=t.status,  # type: ignore[arg-type]
                description=t.description,
            )
            for t in (world.open_threads[k] for k in sorted(world.open_threads))
        ]

        world_summary = self._summarize_world_for_host(world)

        return ObservationHost(
            tick=world.tick,
            world_summary=world_summary,
            guests=guests,
            open_threads=open_threads,
            memory_chunks=mem_chunks,
            last_host_actions=list(world.host_last_actions[-3:]),
        )

    def observe_guest(
        self, world: WorldState, guest_id: str, memory
    ) -> ObservationGuest:
        g = world.guests[guest_id]

        mem_chunks = []
        reflection_summary: Optional[str] = None
        recent_actions = []
        if memory is not None:
            mem_chunks = memory.retrieve_for_guest(guest_id, world)
            reflection_summary = memory.get_reflection_summary(guest_id)
            recent_actions = memory.recent_action_texts_for_guest(guest_id, n=2)

        nearby_guests = []
        for ogid in world.guest_order():
            if ogid == guest_id:
                continue
            og = world.guests[ogid]
            if og.location != g.location:
                continue
            nearby_guests.append(
                GuestSnapshot(
                    guest_id=ogid,
                    name=og.name,
                    location=og.location,
                    mood={k: _round_float(og.mood.get(k, 0.0)) for k in og.mood},
                    goal=og.current_goal,
                    spotlight_weight=_round_float(og.spotlight_weight),
                    last_action=og.last_action,
                )
            )

        nearby_props = []
        for pid in sorted(world.props):
            p = world.props[pid]
            if p.location == g.location and p.held_by is None:
                nearby_props.append(
                    PropSnapshot(
                        prop_id=pid,
                        prop_type=p.prop_type,
                        location=p.location,
                        held_by=None,
                        portable=p.portable,
                    )
                )
        for pid in sorted(g.inventory):
            p = world.props.get(pid)
            if p is None:
                continue
            nearby_props.append(
                PropSnapshot(
                    prop_id=pid,
                    prop_type=p.prop_type,
                    location=None,
                    held_by=guest_id,
                    portable=p.portable,
                )
            )

        open_threads = [
            ThreadSnapshot(
                thread_id=t.thread_id,
                thread_type=t.thread_type,  # type: ignore[arg-type]
                status=t.status,  # type: ignore[arg-type]
                description=t.description,
            )
            for t in (world.open_threads[k] for k in sorted(world.open_threads))
            if t.status == "open"
        ]

        local_view = self._summarize_local_view(world, guest_id)
        persona_text = self._persona_text(memory, guest_id)

        return ObservationGuest(
            tick=world.tick,
            guest_id=guest_id,  # type: ignore[arg-type]
            persona=persona_text,
            goal=g.current_goal,
            location=g.location,
            local_view=local_view,
            nearby_guests=nearby_guests,
            nearby_props=nearby_props,
            open_threads=open_threads,
            memory_chunks=mem_chunks,
            reflection_summary=reflection_summary,
            recent_actions=recent_actions,
            reflection_requested=g.reflection_requested,
        )

    def apply_host_action(self, world: WorldState, action: HostAction) -> EnvResult:
        messages: List[str] = []
        success = True

        if isinstance(action, HostInjectProp):
            if action.location not in world.locations:
                return EnvResult(False, [f"unknown location: {action.location}"])
            prop_id = action.prop_id
            if prop_id is None:
                prop_id = self._next_prop_id(world, action.prop_type)
            if prop_id in world.props:
                return EnvResult(False, [f"prop_id already exists: {prop_id}"])
            world.props[prop_id] = Prop(
                prop_id=prop_id,
                prop_type=action.prop_type,
                location=action.location,
                portable=True,
                held_by=None,
                state={},
            )
            messages.append(f"injected prop {prop_id} at {action.location}")

        elif isinstance(action, HostAllocateSpotlight):
            if action.target_guest_id not in world.guests:
                return EnvResult(False, [f"unknown guest: {action.target_guest_id}"])
            self._set_spotlight(
                world, str(action.target_guest_id), float(action.weight)
            )
            messages.append(
                f"spotlight -> {action.target_guest_id} ({action.weight:.2f})"
            )

        elif isinstance(action, HostSignalStyle):
            world.host_style = action.style
            messages.append(f"host style = {action.style}")

        elif isinstance(action, HostRequestReflection):
            if action.scope == "all":
                for gid in world.guests:
                    world.guests[gid].reflection_requested = True
                messages.append("reflection requested (all)")
            else:
                tg = action.target_guest_id
                if tg is None or str(tg) not in world.guests:
                    return EnvResult(False, ["unknown target_guest_id for reflection"])  # type: ignore[unreachable]
                world.guests[str(tg)].reflection_requested = True
                messages.append(f"reflection requested ({tg})")

        elif isinstance(action, HostSpawnEvent):
            tid = self._next_thread_id(world, action.event_type)
            world.open_threads[tid] = OpenThread(
                thread_id=tid,
                thread_type=action.event_type,
                status="open",
                description=action.description,
                involved_guest_ids=[str(x) for x in action.involved_guest_ids],
            )
            messages.append(f"spawned thread {tid} ({action.event_type})")

        else:
            success = False
            messages.append("unhandled host action")

        world.host_last_actions.append(action.type)
        if len(world.host_last_actions) > 10:
            world.host_last_actions[:] = world.host_last_actions[-10:]

        return EnvResult(success, messages)

    def apply_guest_action(self, world: WorldState, guest_id: str, action) -> EnvResult:
        messages: List[str] = []
        if guest_id not in world.guests:
            return EnvResult(False, [f"unknown guest_id: {guest_id}"])
        g = world.guests[guest_id]

        # Guest actions are validated upstream; env still enforces world constraints.
        atype = getattr(action, "type", None)

        if atype == "move":
            dest = str(action.destination)
            if dest not in world.locations:
                return EnvResult(False, [f"unknown destination: {dest}"])
            if dest == g.location:
                messages.append("already there")
            else:
                g.location = dest
                messages.append(f"moved to {dest}")
            g.last_action = f"move:{dest}"
            g.mood["engaged_bored"] = float(
                min(1.0, g.mood.get("engaged_bored", 0.0) + 0.05)
            )
            return EnvResult(True, messages)

        if atype == "speak":
            speech = str(action.speech)
            target = getattr(action, "target_guest_id", None)
            if target is not None and str(target) in g.trust:
                g.trust[str(target)] = float(min(1.0, g.trust[str(target)] + 0.02))
                g.familiarity[str(target)] = float(
                    min(1.0, g.familiarity[str(target)] + 0.02)
                )
            g.last_action = "speak"
            g.mood["engaged_bored"] = float(
                min(1.0, g.mood.get("engaged_bored", 0.0) + 0.04)
            )
            messages.append(f"said {len(speech)} chars")
            return EnvResult(True, messages)

        if atype == "wait":
            g.last_action = "wait"
            g.mood["engaged_bored"] = float(
                max(-1.0, g.mood.get("engaged_bored", 0.0) - 0.02)
            )
            return EnvResult(True, ["waited"])

        if atype == "reflect":
            g.last_action = "reflect"
            g.reflection_requested = False
            g.mood["calm_agitated"] = float(
                min(1.0, g.mood.get("calm_agitated", 0.0) + 0.05)
            )
            return EnvResult(True, ["reflected"])

        if atype == "collaborate":
            target = str(action.target_guest_id)
            if target not in world.guests:
                return EnvResult(False, [f"unknown target_guest_id: {target}"])
            if world.guests[target].location != g.location:
                return EnvResult(False, ["target not co-located"])
            g.trust[target] = float(min(1.0, g.trust.get(target, 0.5) + 0.03))
            g.familiarity[target] = float(
                min(1.0, g.familiarity.get(target, 0.2) + 0.03)
            )
            g.last_action = f"collaborate:{target}"
            messages.append(f"collaborated with {target}")
            return EnvResult(True, messages)

        if atype == "interact":
            verb = str(action.verb)
            pid = str(action.prop_id)
            p = world.props.get(pid)
            if p is None:
                return EnvResult(False, [f"unknown prop_id: {pid}"])

            if verb == "inspect":
                if p.held_by not in (None, guest_id) and p.location != g.location:
                    return EnvResult(False, ["prop not accessible"])
                g.last_action = f"inspect:{pid}"
                g.mood["engaged_bored"] = float(
                    min(1.0, g.mood.get("engaged_bored", 0.0) + 0.06)
                )
                return EnvResult(True, [f"inspected {pid}"])

            if verb == "pick_up":
                if not p.portable:
                    return EnvResult(False, ["prop not portable"])
                if p.held_by is not None:
                    return EnvResult(False, ["prop already held"])
                if p.location != g.location:
                    return EnvResult(False, ["prop not here"])
                p.held_by = guest_id
                p.location = None
                if pid not in g.inventory:
                    g.inventory.append(pid)
                g.last_action = f"pick_up:{pid}"
                return EnvResult(True, [f"picked up {pid}"])

            if verb == "drop":
                if pid not in g.inventory:
                    return EnvResult(False, ["prop not in inventory"])
                g.inventory = [x for x in g.inventory if x != pid]
                p.held_by = None
                p.location = g.location
                g.last_action = f"drop:{pid}"
                return EnvResult(True, [f"dropped {pid}"])

            if verb == "offer":
                target = str(action.target_guest_id)
                if target not in world.guests:
                    return EnvResult(False, [f"unknown target_guest_id: {target}"])
                if world.guests[target].location != g.location:
                    return EnvResult(False, ["target not co-located"])
                if pid not in g.inventory:
                    return EnvResult(False, ["prop not in inventory"])
                g.inventory = [x for x in g.inventory if x != pid]
                world.guests[target].inventory.append(pid)
                p.held_by = target
                p.location = None
                g.last_action = f"offer:{pid}->{target}"
                return EnvResult(True, [f"offered {pid} to {target}"])

            if verb == "use":
                accessible = (p.held_by == guest_id) or (
                    p.held_by is None and p.location == g.location
                )
                if not accessible:
                    return EnvResult(False, ["prop not accessible"])
                p.state["used_count"] = int(p.state.get("used_count", 0)) + 1
                p.state["last_used_tick"] = world.tick
                g.last_action = f"use:{pid}"
                # Tiny thread nudge: if foam_key used, close first open puzzle.
                if p.prop_type == "foam_key":
                    for tid in sorted(world.open_threads):
                        t = world.open_threads[tid]
                        if t.thread_type == "puzzle" and t.status == "open":
                            t.status = "closed"
                            messages.append(f"closed thread {tid}")
                            break
                return EnvResult(True, [f"used {pid}"] + messages)

            return EnvResult(False, [f"unknown interact verb: {verb}"])

        return EnvResult(False, ["unhandled guest action"])

    def tick_postprocess(self, world: WorldState) -> None:
        # Keep spotlight weights normalized and non-negative.
        total = sum(
            max(0.0, float(world.guests[g].spotlight_weight)) for g in world.guests
        )
        if total <= 0.0:
            eq = 1.0 / float(len(world.guests))
            for gid in world.guests:
                world.guests[gid].spotlight_weight = eq
        else:
            for gid in world.guests:
                world.guests[gid].spotlight_weight = float(
                    max(0.0, float(world.guests[gid].spotlight_weight)) / total
                )

    def _next_prop_id(self, world: WorldState, prop_type: str) -> str:
        base = f"prop_{prop_type}"
        if base not in world.props:
            return base
        i = 2
        while True:
            cand = f"{base}_{i}"
            if cand not in world.props:
                return cand
            i += 1

    def _next_thread_id(self, world: WorldState, event_type: str) -> str:
        base = f"thread_{event_type}"
        if base not in world.open_threads:
            return base
        i = 2
        while True:
            cand = f"{base}_{i}"
            if cand not in world.open_threads:
                return cand
            i += 1

    def _set_spotlight(
        self, world: WorldState, target_guest_id: str, weight: float
    ) -> None:
        weight = float(max(0.0, min(1.0, weight)))
        other_ids = [gid for gid in world.guest_order() if gid != target_guest_id]
        world.guests[target_guest_id].spotlight_weight = weight
        remaining = 1.0 - weight
        if not other_ids:
            world.guests[target_guest_id].spotlight_weight = 1.0
            return

        prev = [float(world.guests[gid].spotlight_weight) for gid in other_ids]
        prev_total = sum(max(0.0, x) for x in prev)
        if prev_total <= 0.0:
            eq = remaining / float(len(other_ids))
            for gid in other_ids:
                world.guests[gid].spotlight_weight = eq
            return

        for gid, x in zip(other_ids, prev):
            world.guests[gid].spotlight_weight = remaining * (max(0.0, x) / prev_total)

    def _summarize_world_for_host(self, world: WorldState) -> str:
        parts: List[str] = []
        parts.append(
            f"Arena={world.arena_id} tick={world.tick} style={world.host_style}"
        )
        open_threads = [
            t
            for t in (world.open_threads[k] for k in sorted(world.open_threads))
            if t.status == "open"
        ]
        if open_threads:
            parts.append(
                "OpenThreads="
                + ",".join(f"{t.thread_id}:{t.thread_type}" for t in open_threads[:5])
            )
        loc_counts = {}
        for gid in world.guests:
            loc_counts[world.guests[gid].location] = (
                loc_counts.get(world.guests[gid].location, 0) + 1
            )
        parts.append(
            "GuestsByLoc="
            + ",".join(f"{k}:{loc_counts[k]}" for k in sorted(loc_counts))
        )
        return " | ".join(parts)

    def _summarize_local_view(self, world: WorldState, guest_id: str) -> str:
        g = world.guests[guest_id]
        loc_desc = world.locations.get(g.location, "")
        props_here = [
            pid
            for pid in sorted(world.props)
            if world.props[pid].location == g.location
            and world.props[pid].held_by is None
        ]
        others = [
            ogid
            for ogid in world.guest_order()
            if ogid != guest_id and world.guests[ogid].location == g.location
        ]
        return (
            f"Location={g.location}. {loc_desc} "
            f"VisibleProps={props_here[:6]} OtherGuests={others[:6]}"
        )

    def _persona_text(self, memory, guest_id: str) -> str:
        if memory is None:
            return f"Persona for {guest_id}."
        return memory.get_persona_summary(guest_id)
