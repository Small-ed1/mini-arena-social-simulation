from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests
from pydantic import TypeAdapter

from sim.logging_utils import maybe_truncate_text
from sim.prompts import render_guest_prompt, render_host_prompt
from sim.schemas import (
    GuestAction,
    GuestCollaborate,
    GuestInteract,
    GuestMove,
    GuestReflect,
    GuestSpeak,
    GuestWait,
    HostAction,
    HostAllocateSpotlight,
    HostRequestReflection,
    HostSignalStyle,
    HostSpawnEvent,
    ModelInfo,
    ObservationGuest,
    ObservationHost,
)
from sim.world_state import WorldState


class InferenceError(RuntimeError):
    pass


def _extract_json_object(text: str) -> Dict[str, Any]:
    t = text.strip()
    if not t:
        raise InferenceError("empty model output")
    if t[0] == "{" and t.endswith("}"):
        obj = json.loads(t)
        if not isinstance(obj, dict):
            raise InferenceError("model output is not a JSON object")
        return obj
    # Best-effort extraction: first {...} block.
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise InferenceError("no JSON object found")
    obj = json.loads(t[start : end + 1])
    if not isinstance(obj, dict):
        raise InferenceError("extracted JSON is not an object")
    return obj


class ScriptedPolicy:
    def generate_host_action(
        self, obs: ObservationHost, world: WorldState
    ) -> HostAction:
        tick = int(obs.tick)
        guest_ids = world.guest_order()

        if tick % 20 == 0:
            # Rotate reflection requests.
            idx = (tick // 20 - 1) % max(1, len(guest_ids))
            target = guest_ids[idx]
            return HostRequestReflection(
                type="request_reflection",
                reason_short="Occasional self-check for coherence",
                actor_id="host",
                scope="one",
                target_guest_id=target,
                prompt="Briefly reflect on your goal and how to keep the scene safe.",
            )

        if tick % 10 == 0:
            etypes = ["mystery", "performance", "repair", "puzzle", "conflict"]
            et = etypes[(tick // 10) % len(etypes)]
            return HostSpawnEvent(
                type="spawn_event",
                reason_short="Add a small twist to maintain momentum",
                actor_id="host",
                event_type=et,  # type: ignore[arg-type]
                description=f"A subtle cue suggests a new {et} thread.",
                location=None,
                involved_guest_ids=[],
            )

        if tick % 5 == 0:
            # Spotlight the currently least-highlighted guest.
            min_gid = min(
                guest_ids, key=lambda g: float(world.guests[g].spotlight_weight)
            )
            return HostAllocateSpotlight(
                type="allocate_spotlight",
                reason_short="Keep spotlight balanced",
                actor_id="host",
                target_guest_id=min_gid,
                weight=0.28,
            )

        style = "gentle" if (tick % 2 == 0) else "mysterious"
        return HostSignalStyle(
            type="signal_style",
            reason_short="Maintain a safe tone",
            actor_id="host",
            style=style,
        )

    def generate_guest_action(
        self, obs: ObservationGuest, world: WorldState
    ) -> GuestAction:
        gid = str(obs.guest_id)
        tick = int(obs.tick)
        g = world.guests[gid]

        if obs.reflection_requested:
            return GuestReflect(
                type="reflect",
                reason_short="Host requested reflection",
                actor_id=gid,
                reflection=maybe_truncate_text(
                    f"I keep my goal in mind ({g.current_goal}) and try to collaborate without pressure or threats.",
                    800,
                ),
            )

        # If holding foam key and any open puzzle thread exists, try to use it.
        if "prop_foam_key" in g.inventory:
            for tid in sorted(world.open_threads):
                t = world.open_threads[tid]
                if t.thread_type == "puzzle" and t.status == "open":
                    return GuestInteract(
                        type="interact",
                        reason_short="Try the foam key on the puzzle",
                        actor_id=gid,
                        verb="use",
                        prop_id="prop_foam_key",
                        target_guest_id=None,
                        speech=None,
                    )

        # Pick up a portable prop if available.
        portable_here = [
            p
            for p in obs.nearby_props
            if p.location == obs.location and p.portable and p.held_by is None
        ]
        if portable_here and len(g.inventory) < 2:
            pid = str(portable_here[0].prop_id)
            return GuestInteract(
                type="interact",
                reason_short="Gather a useful prop",
                actor_id=gid,
                verb="pick_up",
                prop_id=pid,
                target_guest_id=None,
                speech=None,
            )

        # If someone is nearby, alternate between speak and collaborate.
        others = [x.guest_id for x in obs.nearby_guests if str(x.guest_id) != gid]
        if others:
            target = str(sorted(others)[0])
            if (tick + _guest_index(gid)) % 4 == 0:
                return GuestSpeak(
                    type="speak",
                    reason_short="Check in with a neighbor",
                    actor_id=gid,
                    speech="Want to compare notes on what we've seen?",
                    target_guest_id=target,
                    topic="coordination",
                )
            return GuestCollaborate(
                type="collaborate",
                reason_short="Team up to explore safely",
                actor_id=gid,
                target_guest_id=target,
                proposal="Let's scan the room and share what we find.",
                speech="I'll check the corners. You look near the props?",
            )

        # Otherwise move along a deterministic route.
        locs = sorted(world.locations)
        if locs:
            cur = g.location
            try:
                idx = locs.index(cur)
            except ValueError:
                idx = 0
            dest = locs[(idx + 1 + _guest_index(gid)) % len(locs)]
            return GuestMove(
                type="move",
                reason_short="Keep exploring",
                actor_id=gid,
                destination=dest,
            )

        return GuestWait(
            type="wait",
            reason_short="No safe move available",
            actor_id=gid,
            speech=None,
        )


def _guest_index(guest_id: str) -> int:
    try:
        return int(guest_id.split("_")[-1])
    except Exception:
        return 0


class OllamaClient:
    def __init__(self, *, url: str, timeout_s: int):
        self._url = str(url)
        self._timeout_s = int(timeout_s)

    def generate(self, *, model: str, prompt: str, temperature: float) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": float(temperature)},
        }
        r = requests.post(self._url, json=payload, timeout=self._timeout_s)
        r.raise_for_status()
        data = r.json()
        resp = data.get("response")
        if not isinstance(resp, str):
            raise InferenceError("ollama response missing 'response' string")
        return resp


class InferenceEngine:
    def __init__(self, *, cfg: Dict[str, Any], prompt_budgets: Dict[str, Any]):
        self._cfg = cfg
        self._prompt_budgets = prompt_budgets
        self._mode = str(
            cfg.get("mode") or os.getenv("MINI_ARENA_INFERENCE_MODE") or "ollama"
        )
        self._max_retries = int(cfg.get("max_retries", 2))
        self._timeout_s = int(cfg.get("timeout_s", 60))
        self._temperature = float(cfg.get("temperature", 0.7))
        default_model = str(cfg.get("model") or os.getenv("OLLAMA_MODEL") or "llama3.1")
        self._host_model = str(
            cfg.get("host_model") or os.getenv("OLLAMA_HOST_MODEL") or default_model
        )
        self._guest_model = str(
            cfg.get("guest_model") or os.getenv("OLLAMA_GUEST_MODEL") or default_model
        )
        self._ollama_url = self._normalize_ollama_url(
            str(
                cfg.get("ollama_url")
                or os.getenv("OLLAMA_URL")
                or "http://localhost:11434"
            )
        )

        if self._mode == "ollama":
            if not self._host_model or not self._guest_model:
                raise InferenceError("ollama mode requires host_model and guest_model")

        self._scripted = ScriptedPolicy()
        self._ollama = OllamaClient(url=self._ollama_url, timeout_s=self._timeout_s)

        self._host_adapter = TypeAdapter(HostAction)
        self._guest_adapter = TypeAdapter(GuestAction)

    @staticmethod
    def _normalize_ollama_url(url: str) -> str:
        base = url.strip().rstrip("/")
        if not base:
            return "http://localhost:11434/api/generate"
        if base.endswith("/api/generate"):
            return base
        return f"{base}/api/generate"

    def generate_host_action(
        self, obs: ObservationHost, world: WorldState
    ) -> Tuple[HostAction, ModelInfo, Optional[str]]:
        if self._mode == "scripted":
            act = self._scripted.generate_host_action(obs, world)
            return act, ModelInfo(mode="scripted", model=None, retries=0), None
        if self._mode != "ollama":
            raise InferenceError(f"unknown inference mode: {self._mode}")

        prompt = render_host_prompt(
            obs, max_chars=int(self._prompt_budgets.get("host_chars", 32000))
        )
        start = time.time()
        last_err: Optional[str] = None
        for attempt in range(self._max_retries + 1):
            try:
                out = self._ollama.generate(
                    model=self._host_model, prompt=prompt, temperature=self._temperature
                )
                obj = _extract_json_object(out)
                act = self._host_adapter.validate_python(obj)
                mi = ModelInfo(
                    mode="ollama",
                    model=self._host_model,
                    retries=attempt,
                    latency_ms=int((time.time() - start) * 1000),
                    prompt_chars=len(prompt),
                    output_chars=len(out),
                )
                return act, mi, None
            except Exception as e:
                last_err = str(e)
                continue

        mi = ModelInfo(
            mode="ollama",
            model=self._host_model,
            retries=self._max_retries,
            latency_ms=int((time.time() - start) * 1000),
            prompt_chars=len(prompt),
        )
        return (
            self._scripted.generate_host_action(obs, world),
            mi,
            (last_err or "inference_failed"),
        )

    def generate_guest_action(
        self, obs: ObservationGuest, world: WorldState
    ) -> Tuple[GuestAction, ModelInfo, Optional[str]]:
        if self._mode == "scripted":
            act = self._scripted.generate_guest_action(obs, world)
            return act, ModelInfo(mode="scripted", model=None, retries=0), None
        if self._mode != "ollama":
            raise InferenceError(f"unknown inference mode: {self._mode}")

        prompt = render_guest_prompt(
            obs, max_chars=int(self._prompt_budgets.get("guest_chars", 24000))
        )
        start = time.time()
        last_err: Optional[str] = None
        for attempt in range(self._max_retries + 1):
            try:
                out = self._ollama.generate(
                    model=self._guest_model,
                    prompt=prompt,
                    temperature=self._temperature,
                )
                obj = _extract_json_object(out)
                act = self._guest_adapter.validate_python(obj)
                mi = ModelInfo(
                    mode="ollama",
                    model=self._guest_model,
                    retries=attempt,
                    latency_ms=int((time.time() - start) * 1000),
                    prompt_chars=len(prompt),
                    output_chars=len(out),
                )
                return act, mi, None
            except Exception as e:
                last_err = str(e)
                continue

        mi = ModelInfo(
            mode="ollama",
            model=self._guest_model,
            retries=self._max_retries,
            latency_ms=int((time.time() - start) * 1000),
            prompt_chars=len(prompt),
        )
        return (
            self._scripted.generate_guest_action(obs, world),
            mi,
            (last_err or "inference_failed"),
        )
