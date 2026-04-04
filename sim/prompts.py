from __future__ import annotations

from sim.logging_utils import json_dumps_canonical, maybe_truncate_text
from sim.schemas import ObservationGuest, ObservationHost


def render_host_prompt(
    obs: ObservationHost,
    *,
    max_chars: int,
    variant: str = "current",
    retry_note: str | None = None,
) -> str:
    v = (variant or "current").strip().lower()
    if v in ("setup", "worldbuild", "pre_run"):
        return _render_host_prompt_setup(
            obs, max_chars=max_chars, retry_note=retry_note
        )
    if v in ("neutral", "minimal", "bare"):
        return _render_host_prompt_neutral(
            obs, max_chars=max_chars, retry_note=retry_note
        )
    return _render_host_prompt_current(obs, max_chars=max_chars, retry_note=retry_note)


def render_guest_prompt(
    obs: ObservationGuest,
    *,
    max_chars: int,
    variant: str = "current",
    retry_note: str | None = None,
) -> str:
    v = (variant or "current").strip().lower()
    if v in ("neutral", "minimal", "bare"):
        return _render_guest_prompt_neutral(
            obs, max_chars=max_chars, retry_note=retry_note
        )
    return _render_guest_prompt_current(obs, max_chars=max_chars, retry_note=retry_note)


def _format_threads(obs: ObservationHost | ObservationGuest) -> str:
    if not obs.open_threads:
        return "none"
    return "; ".join(
        f"{t.thread_id}:{t.thread_type}:{t.description}" for t in obs.open_threads[:6]
    )


def _retry_block(retry_note: str | None) -> str:
    if not retry_note:
        return ""
    return f"REPAIR NOTE:\n- {retry_note}\n"


def _host_action_catalog() -> str:
    return (
        "HOST ACTIONS:\n"
        "- spawn_event: {type, reason_short, actor_id, event_type, description, location?, involved_guest_ids[]}\n"
        "- inject_prop: {type, reason_short, actor_id, prop_type, location, prop_id?}\n"
        "- enrich_world: {type, reason_short, actor_id, location, detail}\n"
        "- shape_conceptual: {type, reason_short, actor_id, concept, scope(all|location|one), location?, target_guest_id?, intensity, note?}\n"
        "- allocate_spotlight: {type, reason_short, actor_id, target_guest_id, weight}\n"
        "- signal_style: {type, reason_short, actor_id, style}\n"
        "- request_reflection: {type, reason_short, actor_id, scope(one|all), target_guest_id?, prompt?}\n"
    )


def _guest_action_catalog() -> str:
    return (
        "GUEST ACTIONS:\n"
        "- speak: {type, reason_short, actor_id, speech, target_guest_id?, topic?}\n"
        "- move: {type, reason_short, actor_id, destination}\n"
        "- interact: {type, reason_short, actor_id, verb(inspect|pick_up|drop|use|offer), prop_id, target_guest_id?, speech?}\n"
        "- collaborate: {type, reason_short, actor_id, target_guest_id, proposal, speech?}\n"
        "- reflect: {type, reason_short, actor_id, reflection}\n"
        "- wait: {type, reason_short, actor_id, speech?}\n"
    )


def _render_host_prompt_current(
    obs: ObservationHost, *, max_chars: int, retry_note: str | None = None
) -> str:
    obs_json = json_dumps_canonical(obs.model_dump(mode="json"))
    obs_json = maybe_truncate_text(obs_json, max_chars=max(0, int(max_chars) - 1500))

    guest_ids = [g.guest_id for g in obs.guests]
    guest_ids_text = ", ".join(guest_ids) if guest_ids else "none"
    valid_locations_text = ", ".join(obs.valid_locations)
    valid_concepts_text = ", ".join(obs.valid_concepts)
    open_thread_count = len(obs.open_threads)
    retry_block = _retry_block(retry_note)

    return (
        "MODE: PROPER TURN HOST CONTROLLER\n"
        "TASK: Choose exactly one host action that best progresses the current simulation state.\n\n"
        f"OPEN THREAD COUNT: {open_thread_count}.\n"
        f"OPEN THREADS: {_format_threads(obs)}.\n"
        f"CONCEPTUAL SUMMARY: {obs.conceptual_summary}.\n"
        f"VALID GUEST IDS: {guest_ids_text}.\n"
        f"VALID LOCATION IDS: {valid_locations_text}.\n"
        f"VALID CONCEPTS: {valid_concepts_text}.\n\n"
        "WORKFLOW:\n"
        "1. Inspect the currently open threads, active guests, and known locations.\n"
        "2. Separate visible concrete changes from hidden conceptual pressure.\n"
        "3. Choose one action that changes state, focus, or progress.\n"
        "4. Prefer collaboration pressure, social friction, urgency, or unease via shape_conceptual over visible clues.\n"
        "5. Prefer payoff, concretization, or resolution over repeated explanation.\n"
        "4. If there are no open threads, create exactly one new concrete thread with spawn_event.\n"
        "5. If there are open threads, prefer advancing one of them over creating another.\n\n"
        "DECISION PRIORITY:\n"
        "1. Advance an existing open thread.\n"
        "2. Resolve or concretize an existing thread.\n"
        "3. Add hidden conceptual pressure when you want collaboration or psychological strain.\n"
        "4. Shift attention if spotlight is stuck.\n"
        "5. Request reflection if progress is stalled.\n"
        "6. Add one small concrete physical detail tied to an existing location or prop.\n"
        "7. Spawn exactly one new thread only when necessary.\n\n"
        "OUTPUT CONTRACT:\n"
        "- Return ONLY one JSON object matching HostAction.\n"
        "- No markdown fences.\n"
        "- No preamble, explanation, or commentary.\n"
        "- Do not narrate your reasoning.\n"
        '- actor_id must be exactly "host".\n'
        "- Use exactly one action type.\n"
        "- Omit optional fields instead of using null or empty strings.\n"
        "- Use only listed HostAction type names. Never invent labels like progression.\n"
        '- request_reflection with scope="one" must include target_guest_id.\n'
        '- request_reflection with scope="all" should omit target_guest_id.\n'
        "- spawn_event.event_type must be exactly one of: puzzle, conflict, mystery, performance, repair.\n"
        "- allocate_spotlight.weight must be a number from 0.0 to 1.0.\n"
        "- inject_prop.location and enrich_world.location must use an exact valid location id.\n"
        "- shape_conceptual.concept must be one of the valid concepts.\n"
        "- Use enrich_world only for concrete physical, sensory, or spatial facts.\n"
        "- Do not use enrich_world for clues, hints, riddles, hidden messages, or symbolic prompts.\n"
        "- Use shape_conceptual when you want pressure, strain, urgency, or collaboration demand that guests should feel indirectly.\n\n"
        f"{retry_block}"
        "SILENT SELF-CHECK BEFORE OUTPUT:\n"
        "- Is this exactly one HostAction object?\n"
        "- Does it use only valid guest ids and location ids?\n"
        "- If using enrich_world, is it concrete rather than clue-like?\n"
        "- If using shape_conceptual, is it hidden pressure rather than visible exposition?\n"
        "- Does it progress the scene rather than merely repeat it?\n"
        "- If no thread is open, did you create exactly one concrete thread?\n\n"
        f"{_host_action_catalog()}\n"
        "VALID EXAMPLES:\n"
        '{"type":"spawn_event","reason_short":"Start one concrete thread","actor_id":"host","event_type":"repair","description":"A stiff panel in the foyer looks like it may need more than one set of hands to shift.","location":"foyer","involved_guest_ids":[]}\n'
        '{"type":"spawn_event","reason_short":"Introduce a social thread","actor_id":"host","event_type":"conflict","description":"Two guests notice the same object at once and neither is sure who should handle it first.","location":"workshop","involved_guest_ids":["guest_1","guest_2"]}\n'
        '{"type":"shape_conceptual","reason_short":"Pressure collaboration","actor_id":"host","concept":"collaboration_pressure","scope":"location","location":"workshop","intensity":0.7,"note":"Being alone here should feel costly."}\n'
        '{"type":"allocate_spotlight","reason_short":"Shift focus to progress","actor_id":"host","target_guest_id":"guest_1","weight":0.25}\n'
        '{"type":"request_reflection","reason_short":"Check stalled progress","actor_id":"host","scope":"one","target_guest_id":"guest_2","prompt":"State your next move and whether you need another guest to make progress."}\n'
        '{"type":"enrich_world","reason_short":"Add physical detail","actor_id":"host","location":"mirror_hall","detail":"One mirror frame is loose where it meets the wall."}\n\n'
        f"OBSERVATION_JSON={obs_json}\n"
    )


def _render_host_prompt_neutral(
    obs: ObservationHost, *, max_chars: int, retry_note: str | None = None
) -> str:
    obs_json = json_dumps_canonical(obs.model_dump(mode="json"))
    obs_json = maybe_truncate_text(obs_json, max_chars=max(0, int(max_chars) - 1100))

    guest_ids = [g.guest_id for g in obs.guests]
    guest_ids_text = ", ".join(guest_ids) if guest_ids else "none"
    valid_locations_text = ", ".join(obs.valid_locations)
    valid_concepts_text = ", ".join(obs.valid_concepts)
    retry_block = _retry_block(retry_note)

    return (
        "MODE: HOST REPAIR\n"
        "Return one corrected HostAction JSON object only.\n"
        f"OPEN THREADS: {_format_threads(obs)}.\n"
        f"CONCEPTUAL SUMMARY: {obs.conceptual_summary}.\n"
        f"VALID GUEST IDS: {guest_ids_text}.\n"
        f"VALID LOCATION IDS: {valid_locations_text}.\n"
        f"VALID CONCEPTS: {valid_concepts_text}.\n"
        "RULES:\n"
        "- No markdown.\n"
        "- No explanation.\n"
        '- actor_id must be "host".\n'
        "- Use exactly one action type.\n"
        "- Omit optional fields instead of null.\n"
        "- Use only listed HostAction type names. Never invent labels like progression.\n"
        "- spawn_event.event_type must be exactly one of: puzzle, conflict, mystery, performance, repair.\n"
        "- Use enrich_world only for concrete physical details.\n"
        "- Use shape_conceptual for hidden pressure, not visible clues.\n"
        "- If there are no open threads, prefer spawn_event.\n"
        "- Otherwise prefer progression over repetition.\n"
        f"{retry_block}"
        f"{_host_action_catalog()}\n"
        f"OBSERVATION_JSON={obs_json}\n"
    )


def _render_host_prompt_setup(
    obs: ObservationHost, *, max_chars: int, retry_note: str | None = None
) -> str:
    obs_json = json_dumps_canonical(obs.model_dump(mode="json"))
    obs_json = maybe_truncate_text(obs_json, max_chars=max(0, int(max_chars) - 1200))

    valid_locations_text = ", ".join(obs.valid_locations)
    valid_concepts_text = ", ".join(obs.valid_concepts)
    retry_block = _retry_block(retry_note)

    return (
        "MODE: PRE-RUN WORLD BUILD\n"
        "TASK: Create passive world texture before guests arrive.\n\n"
        f"VALID LOCATION IDS: {valid_locations_text}.\n"
        f"VALID CONCEPTS: {valid_concepts_text}.\n"
        "ONLY ALLOWED ACTIONS IN THIS MODE:\n"
        "- enrich_world\n"
        "- inject_prop\n"
        "- shape_conceptual\n"
        "- signal_style\n\n"
        "SETUP RULES:\n"
        "- Add concrete room details, props, ambient oddities, and latent hooks.\n"
        "- Keep additions passive, environmental, and unresolved.\n"
        "- Do not create active missions, urgent tasks, accusations, countdowns, alarms, or conflicts requiring immediate response.\n"
        "- Use only the exact location ids listed above.\n"
        "- Visible clues should be almost non-existent.\n"
        "- Use enrich_world only for concrete physical facts, never for hints or symbolic nudges.\n"
        "- Use shape_conceptual if you want hidden pressure or latent psychological tone.\n"
        "- Use a mix of concrete details, occasional props, and occasional hidden conceptual shaping; do not spend every setup turn on enrich_world.\n"
        "- Do not repeat the same ambient clue with only small wording changes.\n\n"
        "OUTPUT CONTRACT:\n"
        "- Return ONLY one JSON object matching HostAction.\n"
        "- No markdown fences.\n"
        "- No explanation.\n"
        '- actor_id must be exactly "host".\n'
        "- Use exactly one action type.\n"
        "- Omit optional fields instead of null or empty strings.\n"
        f"{retry_block}"
        "SILENT SELF-CHECK BEFORE OUTPUT:\n"
        "- Is this passive world building rather than an active event?\n"
        "- Does it use one exact valid location id?\n"
        "- Is it adding texture rather than issuing a mission?\n\n"
        "ALLOWED ACTION SHAPES:\n"
        "- enrich_world: {type, reason_short, actor_id, location, detail}\n"
        "- inject_prop: {type, reason_short, actor_id, prop_type, location, prop_id?}\n"
        "- shape_conceptual: {type, reason_short, actor_id, concept, scope(all|location|one), location?, target_guest_id?, intensity, note?}\n"
        "- signal_style: {type, reason_short, actor_id, style}\n\n"
        "VALID EXAMPLES:\n"
        '{"type":"enrich_world","reason_short":"Add passive physical detail","actor_id":"host","location":"mirror_hall","detail":"One mirror frame sits slightly crooked against the wall."}\n'
        '{"type":"inject_prop","reason_short":"Add concrete prop","actor_id":"host","prop_type":"bent_wrench","location":"foyer"}\n'
        '{"type":"shape_conceptual","reason_short":"Seed latent pressure","actor_id":"host","concept":"unease","scope":"location","location":"mirror_hall","intensity":0.4,"note":"The room should feel slightly mentally unstable."}\n'
        '{"type":"signal_style","reason_short":"Set initial atmosphere","actor_id":"host","style":"measured"}\n\n'
        f"OBSERVATION_JSON={obs_json}\n"
    )


def _render_guest_prompt_current(
    obs: ObservationGuest, *, max_chars: int, retry_note: str | None = None
) -> str:
    obs_json = json_dumps_canonical(obs.model_dump(mode="json"))
    obs_json = maybe_truncate_text(obs_json, max_chars=max(0, int(max_chars) - 1500))

    valid_guest_ids = [g.guest_id for g in obs.nearby_guests]
    valid_guest_ids_text = ", ".join(valid_guest_ids) if valid_guest_ids else "none"
    valid_prop_ids = [p.prop_id for p in obs.nearby_props]
    valid_prop_ids_text = ", ".join(valid_prop_ids) if valid_prop_ids else "none"
    valid_locations_text = ", ".join(obs.valid_locations)
    held_prop_ids = [p.prop_id for p in obs.nearby_props if p.held_by == obs.guest_id]
    held_prop_ids_text = ", ".join(held_prop_ids) if held_prop_ids else "none"
    relevant_thread = obs.open_threads[0].description if obs.open_threads else "none"
    relevant_thread_type = (
        obs.open_threads[0].thread_type if obs.open_threads else "none"
    )
    retry_block = _retry_block(retry_note)

    return (
        "MODE: GUEST TURN\n"
        "TASK: Choose exactly one valid guest action for the current local situation.\n\n"
        f"ACTOR ID: {obs.guest_id}.\n"
        f"CURRENT LOCATION: {obs.location}.\n"
        f"MOST RELEVANT OPEN THREAD: {relevant_thread_type} - {relevant_thread}.\n"
        f"FELT STATE: {obs.felt_state}.\n"
        f"VALID DESTINATIONS: {valid_locations_text}.\n"
        f"NEARBY GUEST IDS: {valid_guest_ids_text}.\n"
        f"NEARBY OR HELD PROP IDS: {valid_prop_ids_text}.\n"
        f"HELD PROP IDS: {held_prop_ids_text}.\n\n"
        "DECISION ORDER:\n"
        "1. Treat FELT STATE as background context, not an instruction.\n"
        "2. Use or offer a held prop if that can advance the current thread.\n"
        "3. Pick up a relevant portable prop if one is here and useful.\n"
        "4. Collaborate or speak specifically with a nearby guest if that feels natural and helps progress.\n"
        "5. Move toward a relevant location or concrete object.\n"
        "6. Inspect only if the prop is new, newly relevant, or materially changed.\n"
        "7. Wait only if no other valid action would progress the scene.\n\n"
        "OUTPUT CONTRACT:\n"
        "- Return ONLY one JSON object matching GuestAction.\n"
        "- No markdown fences.\n"
        "- No explanation.\n"
        "- Do not narrate your reasoning.\n"
        f'- actor_id must be exactly "{obs.guest_id}".\n'
        "- Use exactly one action type.\n"
        "- Omit optional fields instead of using null or empty strings.\n"
        '- offer is not a top-level action type. To offer something, use type="interact" with verb="offer".\n'
        "- Use only locations, guests, and props that appear in OBSERVATION_JSON.\n"
        "- For collaborate, target_guest_id must be a nearby guest.\n"
        "- For interact, prop_id must be a nearby or held prop from OBSERVATION_JSON.\n"
        "- For interact with verb=offer, target_guest_id is required and you must already be holding the prop.\n"
        "- For interact with verb=drop, you must already be holding the prop.\n"
        "- For move, destination must be one of the valid destinations exactly as written.\n"
        "- Do not repeat the same inspect or same move loop if a different valid action can advance the scene.\n\n"
        f"{retry_block}"
        "SILENT SELF-CHECK BEFORE OUTPUT:\n"
        "- Is actor_id exact?\n"
        "- Are all ids and locations valid?\n"
        "- If offering or dropping, am I holding the prop?\n"
        "- If collaborating, is the target nearby?\n"
        "- Does the action fit the situation naturally instead of forcing a social move?\n"
        "- Am I repeating an inspect or move that I should avoid?\n\n"
        f"{_guest_action_catalog()}\n"
        "VALID EXAMPLES:\n"
        f'{{"type":"interact","reason_short":"Inspect a new relevant prop","actor_id":"{obs.guest_id}","verb":"inspect","prop_id":"{valid_prop_ids[0] if valid_prop_ids else "prop_notebook"}"}}\n'
        f'{{"type":"interact","reason_short":"Offer a held prop","actor_id":"{obs.guest_id}","verb":"offer","prop_id":"{held_prop_ids[0] if held_prop_ids else "prop_flashlight"}","target_guest_id":"{valid_guest_ids[0] if valid_guest_ids else "guest_2"}"}}\n'
        f'{{"type":"move","reason_short":"Move toward a relevant clue","actor_id":"{obs.guest_id}","destination":"{obs.location}"}}\n'
        f'{{"type":"speak","reason_short":"Share a concrete observation","actor_id":"{obs.guest_id}","speech":"I found something specific that may matter."}}\n'
        f'{{"type":"wait","reason_short":"No valid action","actor_id":"{obs.guest_id}","speech":"I have no better valid move right now."}}\n\n'
        f"OBSERVATION_JSON={obs_json}\n"
    )


def _render_guest_prompt_neutral(
    obs: ObservationGuest, *, max_chars: int, retry_note: str | None = None
) -> str:
    obs_json = json_dumps_canonical(obs.model_dump(mode="json"))
    obs_json = maybe_truncate_text(obs_json, max_chars=max(0, int(max_chars) - 1100))

    valid_guest_ids = [g.guest_id for g in obs.nearby_guests]
    valid_guest_ids_text = ", ".join(valid_guest_ids) if valid_guest_ids else "none"
    valid_prop_ids = [p.prop_id for p in obs.nearby_props]
    valid_prop_ids_text = ", ".join(valid_prop_ids) if valid_prop_ids else "none"
    valid_locations_text = ", ".join(obs.valid_locations)
    held_prop_ids = [p.prop_id for p in obs.nearby_props if p.held_by == obs.guest_id]
    held_prop_ids_text = ", ".join(held_prop_ids) if held_prop_ids else "none"
    retry_block = _retry_block(retry_note)

    return (
        "MODE: GUEST REPAIR\n"
        "Return one corrected GuestAction JSON object only.\n"
        f"ACTOR ID: {obs.guest_id}.\n"
        f"CURRENT LOCATION: {obs.location}.\n"
        f"FELT STATE: {obs.felt_state}.\n"
        f"VALID DESTINATIONS: {valid_locations_text}.\n"
        f"NEARBY GUEST IDS: {valid_guest_ids_text}.\n"
        f"NEARBY OR HELD PROP IDS: {valid_prop_ids_text}.\n"
        f"HELD PROP IDS: {held_prop_ids_text}.\n"
        "RULES:\n"
        "- No markdown.\n"
        "- No explanation.\n"
        f'- actor_id must be exactly "{obs.guest_id}".\n'
        "- Use exactly one action type.\n"
        "- Omit optional fields instead of null.\n"
        "- Use only valid ids and locations.\n"
        '- offer is not a top-level action type. To offer something, use type="interact" with verb="offer".\n'
        "- For offer/drop, you must already be holding the prop.\n"
        "- Treat FELT STATE as context, not a command.\n"
        "- Prefer a different valid action over repeating the same failed or repetitive one.\n"
        f"{retry_block}"
        f"{_guest_action_catalog()}\n"
        f"OBSERVATION_JSON={obs_json}\n"
    )
