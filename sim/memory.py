from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from sim.logging_utils import hash_json, maybe_truncate_text
from sim.schemas import MemoryChunk


_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "you",
    "your",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "from",
    "into",
    "they",
    "them",
    "then",
    "than",
    "but",
    "not",
    "too",
    "its",
    "just",
    "over",
}


def _tokens(text: str) -> List[str]:
    out: List[str] = []
    cur: List[str] = []
    for ch in text.lower():
        if ch.isalnum() or ch == "_":
            cur.append(ch)
        else:
            if cur:
                tok = "".join(cur)
                cur = []
                if len(tok) >= 3 and tok not in _STOPWORDS:
                    out.append(tok)
    if cur:
        tok = "".join(cur)
        if len(tok) >= 3 and tok not in _STOPWORDS:
            out.append(tok)
    return out


def _lexical_score(query: str, text: str) -> int:
    q = set(_tokens(query))
    if not q:
        return 0
    t = set(_tokens(text))
    return len(q.intersection(t))


@dataclass
class _EventNote:
    tick: int
    actor_id: str
    phase: str
    guest_id: Optional[str]
    text: str
    chunk_id: str

    def to_chunk(self) -> MemoryChunk:
        return MemoryChunk(
            chunk_id=self.chunk_id,
            kind="event",
            text=self.text,
            tick=self.tick,
            actor_id=self.actor_id,
        )


class MemoryStore:
    def __init__(
        self,
        *,
        personas_cfg: Dict[str, Any],
        last_n_events_host: int,
        last_n_events_guest: int,
        top_k_semantic: int,
        reflection_chars: int,
        world_summary_chars: int,
    ):
        self._personas_cfg = personas_cfg
        self._last_n_events_host = int(last_n_events_host)
        self._last_n_events_guest = int(last_n_events_guest)
        self._top_k_semantic = int(top_k_semantic)
        self._reflection_chars = int(reflection_chars)
        self._world_summary_chars = int(world_summary_chars)

        self._events: List[_EventNote] = []
        self._world_summary: Optional[str] = None
        self._guest_reflections: Dict[str, str] = {}
        self._guest_recent_actions: Dict[str, List[str]] = {}
        self._semantic_summaries: List[
            Tuple[str, str, int]
        ] = []  # (chunk_id, text, tick)

    def store_event(
        self,
        *,
        tick: int,
        phase: str,
        actor_id: str,
        guest_id: Optional[str],
        text: str,
        chunk_id: str,
    ) -> None:
        note = _EventNote(
            tick=int(tick),
            actor_id=str(actor_id),
            phase=str(phase),
            guest_id=str(guest_id) if guest_id is not None else None,
            text=maybe_truncate_text(str(text), 800),
            chunk_id=str(chunk_id),
        )
        self._events.append(note)
        if guest_id is not None:
            rid = str(guest_id)
            self._guest_recent_actions.setdefault(rid, []).append(note.text)
            if len(self._guest_recent_actions[rid]) > 20:
                self._guest_recent_actions[rid] = self._guest_recent_actions[rid][-20:]

    def retrieve_for_host(self, world) -> List[MemoryChunk]:
        chunks: List[MemoryChunk] = []
        if self._world_summary:
            chunks.append(
                MemoryChunk(
                    chunk_id="world_summary",
                    kind="summary",
                    text=maybe_truncate_text(
                        self._world_summary, self._world_summary_chars
                    ),
                    tick=getattr(world, "tick", None),
                )
            )
        recent = self._events[-max(0, self._last_n_events_host) :]
        chunks.extend([n.to_chunk() for n in recent])
        return chunks

    def retrieve_for_guest(self, guest_id: str, world) -> List[MemoryChunk]:
        gid = str(guest_id)
        chunks: List[MemoryChunk] = []

        # Last N events involving the guest.
        involving = [
            e
            for e in reversed(self._events)
            if e.actor_id == gid or (e.guest_id == gid)
        ]
        involving = list(reversed(involving[: max(0, self._last_n_events_guest)]))
        chunks.extend([n.to_chunk() for n in involving])

        # Semantic summaries: pick top-k by lexical overlap with local view (if available).
        query = ""
        try:
            query = str(getattr(world.guests[gid], "location", ""))
        except Exception:
            query = ""
        scored: List[Tuple[int, str, str, int]] = []
        for chunk_id, text, tick in self._semantic_summaries:
            scored.append((_lexical_score(query, text), chunk_id, text, tick))
        scored.sort(key=lambda x: (x[0], x[3], x[1]), reverse=True)
        for score, chunk_id, text, tick in scored[: max(0, self._top_k_semantic)]:
            if score <= 0:
                continue
            chunks.append(
                MemoryChunk(chunk_id=chunk_id, kind="summary", text=text, tick=tick)
            )

        # Rolling reflection.
        refl = self._guest_reflections.get(gid)
        if refl:
            chunks.append(
                MemoryChunk(
                    chunk_id=f"reflection:{gid}",
                    kind="reflection",
                    text=maybe_truncate_text(refl, self._reflection_chars),
                    tick=getattr(world, "tick", None),
                    actor_id=gid,
                )
            )

        return chunks

    def summarize_recent_window(self, *, tick: int, window: int = 10) -> None:
        # Deterministic summary: compress recent events into a stable narrative.
        tick = int(tick)
        recent = [e for e in self._events if e.tick > max(0, tick - window)]
        if not recent:
            return

        lines: List[str] = []
        for e in recent[-40:]:
            lines.append(f"t{e.tick}:{e.actor_id}:{e.text}")
        summary = " | ".join(lines)
        self._world_summary = maybe_truncate_text(summary, self._world_summary_chars)

        chunk_id = f"summary:t{tick}"
        self._semantic_summaries.append((chunk_id, self._world_summary, tick))
        if len(self._semantic_summaries) > 50:
            self._semantic_summaries = self._semantic_summaries[-50:]

        # Very small reflection heuristic per guest.
        by_guest: Dict[str, int] = {}
        for e in recent:
            if e.actor_id.startswith("guest_"):
                by_guest[e.actor_id] = by_guest.get(e.actor_id, 0) + 1
        for gid, count in sorted(by_guest.items()):
            prev = self._guest_reflections.get(gid, "")
            new = (
                f"Recent activity count={count}. Focus on collaboration and safe play."
            )
            if prev:
                new = prev.split("\n")[0] + "\n" + new
            self._guest_reflections[gid] = maybe_truncate_text(
                new, self._reflection_chars
            )

    def write_reflection(self, guest_id: str, text: str) -> None:
        self._guest_reflections[str(guest_id)] = maybe_truncate_text(
            str(text), self._reflection_chars
        )

    def get_reflection_summary(self, guest_id: str) -> Optional[str]:
        return self._guest_reflections.get(str(guest_id))

    def get_persona_summary(self, guest_id: str) -> str:
        gcfg = (self._personas_cfg.get("guests") or {}).get(str(guest_id), {})
        name = gcfg.get("name", guest_id)
        temperament = gcfg.get("temperament", "")
        speech = gcfg.get("default_speech_style", "")
        conflict = gcfg.get("conflict_style", "")
        repair = gcfg.get("repair_tendency", "")
        return (
            f"You are {name} ({guest_id}). Temperament={temperament}. Speech={speech}. "
            f"ConflictStyle={conflict}. RepairTendency={repair}."
        )

    def recent_action_texts_for_guest(self, guest_id: str, n: int) -> List[str]:
        vals = self._guest_recent_actions.get(str(guest_id), [])
        return list(vals[-max(0, int(n)) :])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "events": [
                {
                    "tick": e.tick,
                    "actor_id": e.actor_id,
                    "phase": e.phase,
                    "guest_id": e.guest_id,
                    "text": e.text,
                    "chunk_id": e.chunk_id,
                }
                for e in self._events
            ],
            "world_summary": self._world_summary,
            "guest_reflections": dict(self._guest_reflections),
            "guest_recent_actions": {
                k: list(v) for k, v in self._guest_recent_actions.items()
            },
            "semantic_summaries": list(self._semantic_summaries),
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        *,
        personas_cfg: Dict[str, Any],
        last_n_events_host: int,
        last_n_events_guest: int,
        top_k_semantic: int,
        reflection_chars: int,
        world_summary_chars: int,
    ) -> "MemoryStore":
        ms = cls(
            personas_cfg=personas_cfg,
            last_n_events_host=last_n_events_host,
            last_n_events_guest=last_n_events_guest,
            top_k_semantic=top_k_semantic,
            reflection_chars=reflection_chars,
            world_summary_chars=world_summary_chars,
        )
        ms._world_summary = data.get("world_summary")
        ms._guest_reflections = dict(data.get("guest_reflections") or {})
        ms._guest_recent_actions = {
            k: list(v) for k, v in (data.get("guest_recent_actions") or {}).items()
        }
        ms._semantic_summaries = [
            tuple(x) for x in (data.get("semantic_summaries") or [])
        ]
        ms._events = []
        for e in data.get("events") or []:
            ms._events.append(
                _EventNote(
                    tick=int(e["tick"]),
                    actor_id=str(e["actor_id"]),
                    phase=str(e["phase"]),
                    guest_id=str(e["guest_id"])
                    if e.get("guest_id") is not None
                    else None,
                    text=str(e["text"]),
                    chunk_id=str(e.get("chunk_id") or f"event:{e['tick']}"),
                )
            )
        return ms

    def memory_hash(self) -> str:
        return hash_json(self.to_dict())
