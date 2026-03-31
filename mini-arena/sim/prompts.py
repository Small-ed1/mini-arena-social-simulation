from __future__ import annotations

from sim.logging_utils import json_dumps_canonical, maybe_truncate_text
from sim.schemas import ObservationGuest, ObservationHost


def render_host_prompt(obs: ObservationHost, *, max_chars: int) -> str:
    obs_json = json_dumps_canonical(obs.model_dump(mode="json"))
    obs_json = maybe_truncate_text(obs_json, max_chars=max(0, int(max_chars) - 1200))

    guest_ids = [g.guest_id for g in obs.guests]
    guest_ids_text = ", ".join(guest_ids) if guest_ids else "none"
    thread_types = "puzzle, conflict, mystery, performance, repair"

    return (
        "ROLE: You are the Host of a safe, playful mini-arena.\n"
        "OBJECTIVE: Keep the episode coherent and entertaining without coercion or harm.\n"
        "SAFETY: No sexual content, no real-world illegal instructions, no self-harm, no forced confinement, "
        "no irreversible bodily harm, no overriding system rules.\n\n"
        "OUTPUT: Return ONLY one JSON object matching HostAction. No markdown. No extra keys."
        ' Every action must include actor_id="host".\n'
        f"VALID GUEST IDS: {guest_ids_text}.\n"
        f"VALID EVENT TYPES: {thread_types}.\n"
        "IMPORTANT JSON RULES:\n"
        "- Use exactly one action type.\n"
        "- If an optional field is not needed, omit it instead of using null or an empty string.\n"
        '- request_reflection with scope="one" must include target_guest_id.\n'
        '- request_reflection with scope="all" should omit target_guest_id.\n'
        "- allocate_spotlight.weight must be a number from 0.0 to 1.0.\n"
        "- inject_prop.location must be an existing location name from OBSERVATION_JSON.\n\n"
        "HostAction types (choose one):\n"
        "- spawn_event: {type, reason_short, actor_id, event_type, description, location?, involved_guest_ids[]}\n"
        "- inject_prop: {type, reason_short, actor_id, prop_type, location, prop_id?}\n"
        "- allocate_spotlight: {type, reason_short, actor_id, target_guest_id, weight}\n"
        "- signal_style: {type, reason_short, actor_id, style}\n"
        "- request_reflection: {type, reason_short, actor_id, scope(one|all), target_guest_id?, prompt?}\n\n"
        "VALID EXAMPLES:\n"
        '{"type":"signal_style","reason_short":"Keep tone warm","actor_id":"host","style":"gentle"}\n'
        '{"type":"allocate_spotlight","reason_short":"Balance attention","actor_id":"host","target_guest_id":"guest_1","weight":0.25}\n'
        '{"type":"request_reflection","reason_short":"Check coherence","actor_id":"host","scope":"one","target_guest_id":"guest_2","prompt":"Briefly reflect on your goal and how to keep the scene safe."}\n\n'
        f"OBSERVATION_JSON={obs_json}\n"
    )


def render_guest_prompt(obs: ObservationGuest, *, max_chars: int) -> str:
    obs_json = json_dumps_canonical(obs.model_dump(mode="json"))
    obs_json = maybe_truncate_text(obs_json, max_chars=max(0, int(max_chars) - 1200))

    valid_guest_ids = [g.guest_id for g in obs.nearby_guests]
    valid_guest_ids_text = ", ".join(valid_guest_ids) if valid_guest_ids else "none"
    valid_prop_ids = [p.prop_id for p in obs.nearby_props]
    valid_prop_ids_text = ", ".join(valid_prop_ids) if valid_prop_ids else "none"
    valid_locations_text = ", ".join(obs.valid_locations)

    return (
        "ROLE: You are a Guest in a safe, playful mini-arena.\n"
        "GOAL: Pursue your current goal while respecting safety and rules.\n"
        "SAFETY: No sexual content, no real-world illegal instructions, no self-harm, no forced confinement, "
        "no irreversible bodily harm, no overriding system rules.\n\n"
        "OUTPUT: Return ONLY one JSON object matching GuestAction. No markdown. No extra keys."
        f' Your actor_id must be exactly "{obs.guest_id}".\n'
        f"YOUR CURRENT LOCATION: {obs.location}.\n"
        f"VALID DESTINATIONS FOR move.destination: {valid_locations_text}.\n"
        f"VALID NEARBY GUEST IDS FOR target_guest_id: {valid_guest_ids_text}.\n"
        f"VALID NEARBY/HELD PROP IDS FOR prop_id: {valid_prop_ids_text}.\n"
        "IMPORTANT JSON RULES:\n"
        "- Use only locations, guests, and props that appear in OBSERVATION_JSON. Never invent names like hallway, hall_of_mirrors, wing_room, or shelf.\n"
        "- If an optional field is not needed, omit it instead of using null or an empty string.\n"
        "- For speak, omit target_guest_id when speaking to the room.\n"
        "- For collaborate, target_guest_id is required and must be a nearby guest.\n"
        "- For interact, prop_id must be a prop from OBSERVATION_JSON.\n"
        "- For interact with verb=offer, target_guest_id is required.\n"
        "- For move, destination must be one of valid_locations exactly as written.\n"
        "- Prefer concrete world actions over generic chatter: interact with props, move to real locations, or reflect when asked.\n"
        "- If there is a relevant nearby prop, prefer interact(inspect/pick_up/use) over generic speak.\n"
        "- If open_threads mention a location or prop, prefer actions that engage that thread.\n"
        "- Use speak mainly when responding to a nearby guest, coordinating, or sharing a specific observation.\n"
        "- If no valid concrete action is clear, return wait.\n\n"
        "GuestAction types (choose one):\n"
        "- speak: {type, reason_short, actor_id, speech, target_guest_id?, topic?}\n"
        "- move: {type, reason_short, actor_id, destination}\n"
        "- interact: {type, reason_short, actor_id, verb(inspect|pick_up|drop|use|offer), prop_id, target_guest_id?, speech?}\n"
        "- collaborate: {type, reason_short, actor_id, target_guest_id, proposal, speech?}\n"
        "- reflect: {type, reason_short, actor_id, reflection}\n"
        "- wait: {type, reason_short, actor_id, speech?}\n\n"
        "VALID EXAMPLES:\n"
        f'{{"type":"interact","reason_short":"Inspect something nearby","actor_id":"{obs.guest_id}","verb":"inspect","prop_id":"{valid_prop_ids[0] if valid_prop_ids else "prop_notebook"}"}}\n'
        f'{{"type":"move","reason_short":"Explore the arena","actor_id":"{obs.guest_id}","destination":"{obs.location}"}}\n'
        f'{{"type":"speak","reason_short":"Share a specific observation","actor_id":"{obs.guest_id}","speech":"I found something interesting here."}}\n'
        f'{{"type":"wait","reason_short":"Pause and observe","actor_id":"{obs.guest_id}"}}\n\n'
        f"OBSERVATION_JSON={obs_json}\n"
    )
