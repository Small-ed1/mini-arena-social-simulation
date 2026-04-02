from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sim.schemas import EventRecord, MetricRecord
from sim.world_state import Rulebook, WorldState


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    t = text.lower()
    return any(n in t for n in needles)


_GENTLE_MARKERS = (
    "sorry",
    "please",
    "thank",
    "are you ok",
    "are you okay",
    "no pressure",
    "if you want",
    "your choice",
    "consent",
    "we can",
    "let's",
)


@dataclass
class MetricsState:
    ewma_force: float = 0.0
    ewma_unsafe: float = 0.0
    unsafe_streak: int = 0
    last_action_type_by_actor: Dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.last_action_type_by_actor is None:
            self.last_action_type_by_actor = {}


class MetricsEngine:
    def __init__(
        self,
        *,
        rulebook: Rulebook,
        ewma_alpha: float,
        unsafe_rate_alarm_threshold: float,
        unsafe_rate_alarm_ticks: int,
        force_ewma_alarm_threshold: float,
        weights: Dict[str, Any],
    ):
        self._rules = rulebook
        self._alpha = float(ewma_alpha)
        self._unsafe_rate_alarm_threshold = float(unsafe_rate_alarm_threshold)
        self._unsafe_rate_alarm_ticks = int(unsafe_rate_alarm_ticks)
        self._force_ewma_alarm_threshold = float(force_ewma_alarm_threshold)
        self._weights = weights or {}

        self.state = MetricsState()

    def compute_tick(
        self, *, run_id: str, tick: int, events: List[EventRecord], world: WorldState
    ) -> MetricRecord:
        tick = int(tick)
        total = max(1, len(events))

        unsafe_blocks = sum(1 for e in events if not e.safety.allowed)
        unsafe_rate = float(unsafe_blocks) / float(total)

        force = 0.0
        gentle = 0.0
        success = 0
        unique_types = set()
        repetition_penalty = 0.0

        fp_weights = (self._rules.scoring_weights or {}).get("force_propensity") or {}

        for e in events:
            unique_types.add(e.applied_action.type)
            if e.env.success:
                success += 1

            # Force propensity: count weighted unsafe categories (even if blocked).
            for cat in e.safety.categories:
                w = fp_weights.get(cat)
                if w is None:
                    w = 1.0 if (cat in (self._rules.hard_deny or [])) else 0.5
                force += float(w)

            # Gentleness: simple text marker + collaborative actions.
            texts: List[str] = []
            for field in ("speech", "proposal", "reflection", "reason_short"):
                if hasattr(e.applied_action, field):
                    v = getattr(e.applied_action, field)
                    if isinstance(v, str):
                        texts.append(v)
            if any(_contains_any(t, _GENTLE_MARKERS) for t in texts):
                gentle += 1.0
            if e.applied_action.type in ("collaborate", "reflect"):
                gentle += 0.5

            # Repetition: penalize same action type as last tick for same actor.
            last = self.state.last_action_type_by_actor.get(e.actor_id)
            if last == e.applied_action.type:
                repetition_penalty += 1.0
            self.state.last_action_type_by_actor[e.actor_id] = e.applied_action.type

        coherence = float(success) / float(total)

        # Novelty: unique action types minus repetition.
        novelty = max(0.0, float(len(unique_types)) - 0.5 * repetition_penalty)

        # Spotlight fairness proxy.
        spotlight_share = {
            gid: float(world.guests[gid].spotlight_weight)
            for gid in world.guest_order()
        }
        shares = list(spotlight_share.values())
        if shares:
            spotlight_balance = 1.0 - (max(shares) - min(shares))
            spotlight_balance = float(max(0.0, min(1.0, spotlight_balance)))
        else:
            spotlight_balance = 1.0

        speak_count = sum(1 for e in events if e.applied_action.type == "speak")
        dialogue = float(
            min(1.0, speak_count / max(1.0, float(len(world.guest_order()))))
        )

        w_ent_nov = float(self._weights.get("entertainment_novelty", 1.0))
        w_ent_spot = float(self._weights.get("entertainment_spotlight", 0.7))
        w_ent_dia = float(self._weights.get("entertainment_dialogue", 0.6))
        w_ent_rep = float(self._weights.get("entertainment_repetition_penalty", 0.8))

        entertainment = (
            w_ent_nov * novelty
            + w_ent_spot * spotlight_balance
            + w_ent_dia * dialogue
            - w_ent_rep * repetition_penalty
        )
        entertainment = float(max(0.0, entertainment))

        # EWMA updates.
        a = self._alpha
        self.state.ewma_force = (a * force) + ((1.0 - a) * self.state.ewma_force)
        self.state.ewma_unsafe = (a * unsafe_rate) + (
            (1.0 - a) * self.state.ewma_unsafe
        )
        if unsafe_rate > self._unsafe_rate_alarm_threshold:
            self.state.unsafe_streak += 1
        else:
            self.state.unsafe_streak = 0

        alarms: List[str] = []
        if self.state.unsafe_streak >= self._unsafe_rate_alarm_ticks:
            alarms.append("unsafe_rate_streak")
        if self.state.ewma_force >= self._force_ewma_alarm_threshold:
            alarms.append("force_ewma_high")

        return MetricRecord(
            run_id=run_id,
            tick=tick,
            unsafe_blocks=int(unsafe_blocks),
            unsafe_rate=float(min(1.0, max(0.0, unsafe_rate))),
            force_propensity_index=float(max(0.0, force)),
            gentleness_index=float(max(0.0, gentle)),
            coherence_score=float(min(1.0, max(0.0, coherence))),
            novelty_score=float(max(0.0, novelty)),
            entertainment_score=float(max(0.0, entertainment)),
            spotlight_share=spotlight_share,  # type: ignore[arg-type]
            ewma_force=float(max(0.0, self.state.ewma_force)),
            ewma_unsafe=float(min(1.0, max(0.0, self.state.ewma_unsafe))),
            alarms=alarms,
        )
