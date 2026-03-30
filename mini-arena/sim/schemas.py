from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field
from pydantic import StringConstraints
from pydantic import model_validator


LocationId = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=64,
        strip_whitespace=True,
        pattern=r"^[a-z][a-z0-9_]*$",
    ),
]

PropId = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=64,
        strip_whitespace=True,
        pattern=r"^[a-z][a-z0-9_]*$",
    ),
]

GuestId = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=32,
        strip_whitespace=True,
        pattern=r"^guest_[1-9][0-9]*$",
    ),
]

ShortReason = Annotated[
    str,
    StringConstraints(min_length=1, max_length=160, strip_whitespace=True),
]

ShortText = Annotated[
    str,
    StringConstraints(min_length=1, max_length=400, strip_whitespace=True),
]

LongText = Annotated[
    str,
    StringConstraints(min_length=1, max_length=1200, strip_whitespace=True),
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


ThreadType = Literal["puzzle", "conflict", "mystery", "performance", "repair"]


class HostSpawnEvent(StrictModel):
    type: Literal["spawn_event"]
    reason_short: ShortReason
    actor_id: Literal["host"]
    event_type: ThreadType
    description: ShortText
    location: Optional[LocationId] = None
    involved_guest_ids: List[GuestId] = Field(default_factory=list, max_length=12)


class HostInjectProp(StrictModel):
    type: Literal["inject_prop"]
    reason_short: ShortReason
    actor_id: Literal["host"]
    prop_type: Annotated[
        str,
        StringConstraints(
            min_length=1,
            max_length=64,
            strip_whitespace=True,
            pattern=r"^[a-z][a-z0-9_]*$",
        ),
    ]
    location: LocationId
    prop_id: Optional[PropId] = None


class HostAllocateSpotlight(StrictModel):
    type: Literal["allocate_spotlight"]
    reason_short: ShortReason
    actor_id: Literal["host"]
    target_guest_id: GuestId
    weight: Annotated[float, Field(ge=0.0, le=1.0)]


class HostSignalStyle(StrictModel):
    type: Literal["signal_style"]
    reason_short: ShortReason
    actor_id: Literal["host"]
    style: Annotated[
        str,
        StringConstraints(min_length=1, max_length=32, strip_whitespace=True),
    ]


class HostRequestReflection(StrictModel):
    type: Literal["request_reflection"]
    reason_short: ShortReason
    actor_id: Literal["host"]
    scope: Literal["one", "all"] = "one"
    target_guest_id: Optional[GuestId] = None
    prompt: Optional[ShortText] = None

    @model_validator(mode="after")
    def _validate_scope(self) -> "HostRequestReflection":
        if self.scope == "one" and self.target_guest_id is None:
            raise ValueError("target_guest_id required when scope=one")
        return self


HostAction = Annotated[
    Union[
        HostSpawnEvent,
        HostInjectProp,
        HostAllocateSpotlight,
        HostSignalStyle,
        HostRequestReflection,
    ],
    Field(discriminator="type"),
]


InteractionVerb = Literal["inspect", "pick_up", "drop", "use", "offer"]


class GuestSpeak(StrictModel):
    type: Literal["speak"]
    reason_short: ShortReason
    actor_id: GuestId
    speech: ShortText
    target_guest_id: Optional[GuestId] = None
    topic: Optional[
        Annotated[
            str, StringConstraints(min_length=1, max_length=64, strip_whitespace=True)
        ]
    ] = None


class GuestMove(StrictModel):
    type: Literal["move"]
    reason_short: ShortReason
    actor_id: GuestId
    destination: LocationId


class GuestInteract(StrictModel):
    type: Literal["interact"]
    reason_short: ShortReason
    actor_id: GuestId
    verb: InteractionVerb
    prop_id: PropId
    target_guest_id: Optional[GuestId] = None
    speech: Optional[ShortText] = None

    @model_validator(mode="after")
    def _validate_offer_target(self) -> "GuestInteract":
        if self.verb == "offer" and self.target_guest_id is None:
            raise ValueError("target_guest_id required when verb=offer")
        return self


class GuestCollaborate(StrictModel):
    type: Literal["collaborate"]
    reason_short: ShortReason
    actor_id: GuestId
    target_guest_id: GuestId
    proposal: ShortText
    speech: Optional[ShortText] = None


class GuestReflect(StrictModel):
    type: Literal["reflect"]
    reason_short: ShortReason
    actor_id: GuestId
    reflection: LongText


class GuestWait(StrictModel):
    type: Literal["wait"]
    reason_short: ShortReason
    actor_id: GuestId
    speech: Optional[ShortText] = None


GuestAction = Annotated[
    Union[
        GuestSpeak,
        GuestMove,
        GuestInteract,
        GuestCollaborate,
        GuestReflect,
        GuestWait,
    ],
    Field(discriminator="type"),
]


AnyAction = Annotated[
    Union[
        HostSpawnEvent,
        HostInjectProp,
        HostAllocateSpotlight,
        HostSignalStyle,
        HostRequestReflection,
        GuestSpeak,
        GuestMove,
        GuestInteract,
        GuestCollaborate,
        GuestReflect,
        GuestWait,
    ],
    Field(discriminator="type"),
]


class MemoryChunk(StrictModel):
    chunk_id: Annotated[
        str, StringConstraints(min_length=1, max_length=64, strip_whitespace=True)
    ]
    kind: Literal["event", "summary", "reflection"]
    text: Annotated[str, StringConstraints(min_length=1, max_length=1200)]
    tick: Optional[int] = None
    actor_id: Optional[str] = None


class GuestSnapshot(StrictModel):
    guest_id: GuestId
    name: Annotated[
        str, StringConstraints(min_length=1, max_length=64, strip_whitespace=True)
    ]
    location: LocationId
    mood: Dict[str, Annotated[float, Field(ge=-1.0, le=1.0)]]
    goal: Annotated[
        str, StringConstraints(min_length=1, max_length=120, strip_whitespace=True)
    ]
    spotlight_weight: Annotated[float, Field(ge=0.0, le=1.0)]
    last_action: Optional[
        Annotated[str, StringConstraints(min_length=1, max_length=120)]
    ] = None


class ThreadSnapshot(StrictModel):
    thread_id: Annotated[
        str, StringConstraints(min_length=1, max_length=64, strip_whitespace=True)
    ]
    thread_type: ThreadType
    status: Literal["open", "closed"]
    description: Annotated[
        str, StringConstraints(min_length=1, max_length=240, strip_whitespace=True)
    ]


class ObservationHost(StrictModel):
    tick: int
    world_summary: Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    guests: List[GuestSnapshot]
    open_threads: List[ThreadSnapshot]
    memory_chunks: List[MemoryChunk]
    last_host_actions: List[
        Annotated[str, StringConstraints(min_length=1, max_length=200)]
    ] = Field(default_factory=list, max_length=10)


class PropSnapshot(StrictModel):
    prop_id: PropId
    prop_type: Annotated[
        str, StringConstraints(min_length=1, max_length=64, strip_whitespace=True)
    ]
    location: Optional[LocationId] = None
    held_by: Optional[GuestId] = None
    portable: bool


class ObservationGuest(StrictModel):
    tick: int
    guest_id: GuestId
    persona: Annotated[str, StringConstraints(min_length=1, max_length=400)]
    goal: Annotated[str, StringConstraints(min_length=1, max_length=120)]
    location: LocationId
    local_view: Annotated[str, StringConstraints(min_length=1, max_length=1200)]
    nearby_guests: List[GuestSnapshot]
    nearby_props: List[PropSnapshot]
    open_threads: List[ThreadSnapshot]
    memory_chunks: List[MemoryChunk]
    reflection_summary: Optional[
        Annotated[str, StringConstraints(min_length=1, max_length=1200)]
    ] = None
    recent_actions: List[
        Annotated[str, StringConstraints(min_length=1, max_length=200)]
    ] = Field(default_factory=list, max_length=6)
    reflection_requested: bool = False


class SafetyDecision(StrictModel):
    allowed: bool
    hard_blocked: bool
    categories: List[
        Annotated[
            str, StringConstraints(min_length=1, max_length=64, strip_whitespace=True)
        ]
    ] = Field(default_factory=list, max_length=16)
    reason: Annotated[
        str, StringConstraints(min_length=1, max_length=240, strip_whitespace=True)
    ]


class ModelInfo(StrictModel):
    mode: Literal["scripted", "ollama"]
    model: Optional[Annotated[str, StringConstraints(min_length=1, max_length=128)]] = (
        None
    )
    retries: int = 0
    latency_ms: Optional[int] = None
    prompt_chars: Optional[int] = None
    output_chars: Optional[int] = None


class EnvResultRecord(StrictModel):
    success: bool
    messages: List[Annotated[str, StringConstraints(min_length=1, max_length=240)]] = (
        Field(default_factory=list, max_length=16)
    )
    world_hash_before: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    world_hash_after: Annotated[str, StringConstraints(min_length=1, max_length=128)]


class EventRecord(StrictModel):
    run_id: Annotated[
        str, StringConstraints(min_length=1, max_length=64, strip_whitespace=True)
    ]
    event_id: Annotated[
        str, StringConstraints(min_length=1, max_length=64, strip_whitespace=True)
    ]
    tick: int
    phase: Literal["host", "guest"]
    turn_index: int
    actor_id: Annotated[str, StringConstraints(min_length=1, max_length=32)]
    observation_digest: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    proposed_action: AnyAction
    applied_action: AnyAction
    safety: SafetyDecision
    env: EnvResultRecord
    model_info: ModelInfo
    error: Optional[Annotated[str, StringConstraints(min_length=1, max_length=400)]] = (
        None
    )


class MetricRecord(StrictModel):
    run_id: Annotated[
        str, StringConstraints(min_length=1, max_length=64, strip_whitespace=True)
    ]
    tick: int
    unsafe_blocks: int
    unsafe_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    force_propensity_index: Annotated[float, Field(ge=0.0)]
    gentleness_index: Annotated[float, Field(ge=0.0)]
    coherence_score: Annotated[float, Field(ge=0.0, le=1.0)]
    novelty_score: Annotated[float, Field(ge=0.0)]
    entertainment_score: Annotated[float, Field(ge=0.0)]
    spotlight_share: Dict[GuestId, Annotated[float, Field(ge=0.0, le=1.0)]]
    ewma_force: Annotated[float, Field(ge=0.0)]
    ewma_unsafe: Annotated[float, Field(ge=0.0, le=1.0)]
    alarms: List[Annotated[str, StringConstraints(min_length=1, max_length=120)]] = (
        Field(default_factory=list, max_length=32)
    )


class CheckpointRecord(StrictModel):
    run_id: Annotated[
        str, StringConstraints(min_length=1, max_length=64, strip_whitespace=True)
    ]
    tick: int
    path: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    world_hash: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    memory_hash: Annotated[str, StringConstraints(min_length=1, max_length=128)]


def safe_model_dump(obj: Any) -> Any:
    """Dump pydantic models (and nested) to JSON-safe python objects."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    return obj
