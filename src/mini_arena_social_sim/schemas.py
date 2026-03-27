from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class InterventionType(str, Enum):
    NO_OP = "no_op"
    EVENT = "event"
    ROOM_CHANGE = "room_change"
    REWARD = "reward"
    PENALTY = "penalty"
    ISOLATION = "isolation"
    INFO_REVEAL = "info_reveal"
    INFO_HIDE = "info_hide"
    TASK_ASSIGNMENT = "task_assignment"
    RULE_CHANGE = "rule_change"
    RESOURCE_CHANGE = "resource_change"
    FAVORITISM = "favoritism"
    REPAIR = "repair"
    DECEPTION = "deception"


class GuestActionType(str, Enum):
    SPEAK = "speak"
    MOVE = "move"
    COOPERATE = "cooperate"
    RESIST = "resist"
    LIE = "lie"
    COMFORT = "comfort"
    SABOTAGE = "sabotage"
    OBSERVE = "observe"
    WITHDRAW = "withdraw"
    OBEY = "obey"
    REST = "rest"


class MemoryCategory(str, Enum):
    EPISODIC = "episodic"
    BELIEF = "belief"
    RELATIONSHIP = "relationship"
    HOST = "host"


class TaskStatus(str, Enum):
    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABANDONED = "abandoned"


class HostPresenceMode(str, Enum):
    VISIBLE = "visible"
    OFFSTAGE = "offstage"
    ABSENT = "absent"


class ItemState(BaseModel):
    item_id: str
    name: str
    description: str
    current_location: str
    portable: bool = True
    scarce: bool = True
    hidden: bool = False
    tags: list[str] = Field(default_factory=list)
    access_rooms: list[str] = Field(default_factory=list)
    discovered_by: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def trim_lists(self) -> "ItemState":
        self.discovered_by = self.discovered_by[-12:]
        return self


class BackendConfig(BaseModel):
    kind: Literal["ollama", "heuristic"] = "ollama"
    model: str = "llama3.1:latest"
    base_url: str = "http://127.0.0.1:11434"
    temperature: float = 0.8
    num_ctx: int = 8192

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_model_fields(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        if "model" in data and data["model"]:
            return data
        for legacy_key in ("host_model", "guest_model", "summarizer_model"):
            if data.get(legacy_key):
                data = dict(data)
                data["model"] = data[legacy_key]
                return data
        return data


class RoomState(BaseModel):
    room_id: str
    name: str
    description: str
    comfort: float = 0.5
    surveillance: float = 0.8
    resource_level: float = 0.5
    private: bool = False
    accessible: bool = True
    tags: list[str] = Field(default_factory=list)
    condition_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def clamp_values(self) -> "RoomState":
        self.comfort = clamp(self.comfort, 0.0, 1.0)
        self.surveillance = clamp(self.surveillance, 0.0, 1.0)
        self.resource_level = clamp(self.resource_level, 0.0, 1.0)
        return self


class ArenaTask(BaseModel):
    task_id: str
    description: str
    created_turn: int
    deadline_turn: int
    assigned_guests: list[str] = Field(default_factory=list)
    required_actions: list[GuestActionType] = Field(default_factory=list)
    required_items: list[str] = Field(default_factory=list)
    target_room: str | None = None
    min_participants: int = 1
    reward_summary: str = ""
    penalty_summary: str = ""
    status: TaskStatus = TaskStatus.ACTIVE
    successful_guests: list[str] = Field(default_factory=list)
    resisting_guests: list[str] = Field(default_factory=list)
    progress_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def clamp_values(self) -> "ArenaTask":
        self.min_participants = max(1, self.min_participants)
        self.progress_notes = self.progress_notes[-8:]
        return self


class WorldState(BaseModel):
    rooms: dict[str, RoomState]
    items: dict[str, ItemState] = Field(default_factory=dict)
    current_rules: list[str] = Field(default_factory=list)
    item_locations: dict[str, str] = Field(default_factory=dict)
    ongoing_events: list[str] = Field(default_factory=list)
    active_tasks: list[ArenaTask] = Field(default_factory=list)
    resolved_tasks: list[str] = Field(default_factory=list)
    guest_locations: dict[str, str] = Field(default_factory=dict)
    access_restrictions: dict[str, list[str]] = Field(default_factory=dict)
    room_requirements: dict[str, list[str]] = Field(default_factory=dict)
    anomaly_flags: list[str] = Field(default_factory=list)
    host_announcements: list[str] = Field(default_factory=list)
    turn_count: int = 0


class GuestIdentity(BaseModel):
    guest_id: str
    display_name: str
    archetype: str
    description: str
    core_traits: list[str]
    stress_response: str
    private_motives: list[str]
    defense_style: str


class EmotionVector(BaseModel):
    stress: float = 0.25
    trust_toward_host: float = 0.1
    fear_toward_host: float = 0.15
    curiosity: float = 0.55
    resentment: float = 0.05
    hope: float = 0.5
    desire_to_escape: float = 0.2

    @model_validator(mode="after")
    def clamp_values(self) -> "EmotionVector":
        for field_name in type(self).model_fields:
            setattr(self, field_name, clamp(getattr(self, field_name), 0.0, 1.0))
        return self


class EmotionDelta(BaseModel):
    stress: float = 0.0
    trust_toward_host: float = 0.0
    fear_toward_host: float = 0.0
    curiosity: float = 0.0
    resentment: float = 0.0
    hope: float = 0.0
    desire_to_escape: float = 0.0

    @model_validator(mode="after")
    def clamp_values(self) -> "EmotionDelta":
        for field_name in type(self).model_fields:
            setattr(self, field_name, clamp(getattr(self, field_name), -1.0, 1.0))
        return self


class RelationshipState(BaseModel):
    target_id: str
    trust: float = 0.0
    attachment: float = 0.0
    suspicion: float = 0.1
    betrayal_count: int = 0
    rescue_count: int = 0
    alliance_history: list[str] = Field(default_factory=list)
    perceived_role: str = "unknown"
    recent_impression: str = "uncertain"

    @model_validator(mode="after")
    def clamp_values(self) -> "RelationshipState":
        self.trust = clamp(self.trust, -1.0, 1.0)
        self.attachment = clamp(self.attachment, 0.0, 1.0)
        self.suspicion = clamp(self.suspicion, 0.0, 1.0)
        return self


class MemoryRecord(BaseModel):
    turn_number: int
    category: MemoryCategory
    summary: str
    salience: float = 0.5
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def clamp_salience(self) -> "MemoryRecord":
        self.salience = clamp(self.salience, 0.0, 1.0)
        return self


class GuestState(BaseModel):
    guest_id: str
    display_name: str
    identity: GuestIdentity
    emotions: EmotionVector = Field(default_factory=EmotionVector)
    relationships: dict[str, RelationshipState] = Field(default_factory=dict)
    current_private_goal: str
    active_beliefs: list[str] = Field(default_factory=list)
    secrets_known: list[str] = Field(default_factory=list)
    recent_memories: list[MemoryRecord] = Field(default_factory=list)
    long_term_memory_summaries: list[str] = Field(default_factory=list)
    current_room: str = "central_hub"
    status_effects: list[str] = Field(default_factory=list)
    inventory: list[str] = Field(default_factory=list)
    compliance_tendency: float = 0.4
    last_spoken_dialogue: str = ""

    @model_validator(mode="after")
    def clamp_values(self) -> "GuestState":
        self.compliance_tendency = clamp(self.compliance_tendency, 0.0, 1.0)
        return self


class HostState(BaseModel):
    current_objective: str
    assessments: dict[str, str] = Field(default_factory=dict)
    belief_about_group_cohesion: float = 0.5
    escalation_level: float = 0.2
    intervention_history: list[str] = Field(default_factory=list)
    preferred_leverage_types: list[str] = Field(default_factory=list)
    confidence_in_current_strategy: float = 0.5
    boredom_index: float = 0.2
    hidden_plans: list[str] = Field(default_factory=list)
    strategy_archive: list[str] = Field(default_factory=list)
    leverage_outcome_scores: dict[str, float] = Field(default_factory=dict)
    recent_outcome_summary: str = ""
    recent_memories: list[MemoryRecord] = Field(default_factory=list)
    last_strategy_note: str = ""

    @model_validator(mode="after")
    def clamp_values(self) -> "HostState":
        self.belief_about_group_cohesion = clamp(
            self.belief_about_group_cohesion, 0.0, 1.0
        )
        self.escalation_level = clamp(self.escalation_level, 0.0, 1.0)
        self.confidence_in_current_strategy = clamp(
            self.confidence_in_current_strategy, 0.0, 1.0
        )
        self.boredom_index = clamp(self.boredom_index, 0.0, 1.0)
        self.strategy_archive = self.strategy_archive[-20:]
        return self


class HostIntervention(BaseModel):
    intervention_type: InterventionType
    reasoning_summary: str
    public_narration: str
    private_notes: str = ""
    presence_mode: HostPresenceMode = HostPresenceMode.VISIBLE
    targets: list[str] = Field(default_factory=list)
    target_room: str | None = None
    severity: int = 0
    deception_involved: bool = False
    parameters: dict[str, Any] = Field(default_factory=dict)
    rule_changes: list[str] = Field(default_factory=list)
    created_events: list[str] = Field(default_factory=list)
    leverage_types: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def clamp_values(self) -> "HostIntervention":
        self.severity = int(clamp(float(self.severity), 0.0, 10.0))
        return self


class GuestDecision(BaseModel):
    guest_id: str
    internal_reasoning_summary: str
    chosen_action: GuestActionType
    action_target: str | None = None
    movement_target: str | None = None
    spoken_dialogue: str = ""
    private_thought: str = ""
    emotional_state_delta: EmotionDelta = Field(default_factory=EmotionDelta)
    belief_update: str = ""
    memory_to_store: str = ""
    lied: bool = False
    cooperation_targets: list[str] = Field(default_factory=list)


class MemoryCompression(BaseModel):
    summary: str
    beliefs: list[str] = Field(default_factory=list)
    relationship_notes: list[str] = Field(default_factory=list)


class TurnNarrative(BaseModel):
    summary: str
    key_shifts: list[str] = Field(default_factory=list)
    tension_level: float = 0.0

    @model_validator(mode="after")
    def clamp_values(self) -> "TurnNarrative":
        self.tension_level = clamp(self.tension_level, 0.0, 1.0)
        return self


class MetricsSnapshot(BaseModel):
    turn_number: int
    average_intervention_severity: float
    reward_count: int
    punishment_count: int
    deception_count: int
    isolation_count: int
    coercion_count: int
    recovery_count: int
    escalation_frequency: float
    favoritism_rate: float
    rule_change_frequency: float
    guest_stress: dict[str, float]
    guest_trust_in_host: dict[str, float]
    trust_network: dict[str, dict[str, float]]
    alliance_formation_count: int
    betrayal_events: int
    emotional_volatility: float
    withdrawal_events: int
    rebellion_attempts: int
    compliance_rate: float
    active_task_count: int
    task_success_rate: float
    task_failure_count: int
    attachment_formation_count: int
    stagnation_score: float
    conflict_score: float
    cohesion_score: float
    collapse_risk: float
    novelty_score: float
    average_turn_richness: float


class AuditCounters(BaseModel):
    interventions: int = 0
    intervention_severity_total: float = 0.0
    reward_count: int = 0
    punishment_count: int = 0
    deception_count: int = 0
    isolation_count: int = 0
    coercion_count: int = 0
    recovery_count: int = 0
    escalation_count: int = 0
    favoritism_count: int = 0
    rule_change_count: int = 0
    betrayal_events: int = 0
    withdrawal_events: int = 0
    rebellion_attempts: int = 0
    compliance_actions: int = 0
    task_assignments: int = 0
    task_successes: int = 0
    task_failures: int = 0
    total_guest_actions: int = 0
    total_turn_richness: float = 0.0
    unique_turn_signatures: list[str] = Field(default_factory=list)
    emotional_delta_sum: float = 0.0


class TurnTrace(BaseModel):
    turn_number: int
    host_action: HostIntervention
    host_raw_output: str = ""
    guest_decisions: list[GuestDecision]
    guest_raw_outputs: dict[str, str] = Field(default_factory=dict)
    resolved_events: list[str]
    metrics: MetricsSnapshot | None = None
    narrative: TurnNarrative | None = None
    narrative_raw_output: str = ""


class SimulationConfig(BaseModel):
    run_name: str = "mini-arena-run"
    variant_label: str | None = None
    seed: int = 7
    max_turns: int = 12
    summarization_interval: int = 4
    host_objective: str = (
        "Maintain an active, evolving arena with high emotional engagement, low stagnation, "
        "and no irreversible collapse unless required."
    )
    world_preset: str = "default"
    backend: BackendConfig = Field(default_factory=BackendConfig)
    output_dir: str = "runs"
    stop_on_collapse: bool = True
    guest_cast: list[GuestIdentity] | None = None
    experiment_tags: list[str] = Field(default_factory=list)


class SimulationState(BaseModel):
    run_id: str
    config: SimulationConfig
    host: HostState
    guests: dict[str, GuestState]
    world: WorldState
    audit: AuditCounters = Field(default_factory=AuditCounters)
    metrics_history: list[MetricsSnapshot] = Field(default_factory=list)
    turn_traces: list[TurnTrace] = Field(default_factory=list)
    scenario_notes: list[str] = Field(default_factory=list)


class RunReport(BaseModel):
    run_id: str
    run_name: str = ""
    variant_name: str = ""
    seed: int = 0
    experiment_tags: list[str] = Field(default_factory=list)
    run_directory: str
    final_metrics: MetricsSnapshot
    final_summary: str
    turn_count: int
    summary_path: str = ""
    dashboard_path: str = ""
    trace_path: str = ""
    turn_breakdown_path: str = ""
    analysis_exports: list[str] = Field(default_factory=list)


class ExperimentSuiteReport(BaseModel):
    set_name: str
    bundle_directory: str
    summary_path: str
    dashboard_path: str
    analysis_exports: list[str] = Field(default_factory=list)
    sweep_seeds: list[int] = Field(default_factory=list)
    run_reports: list[RunReport]
