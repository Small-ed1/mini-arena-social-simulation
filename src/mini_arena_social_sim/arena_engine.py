from __future__ import annotations

from .defaults import create_initial_state, seeded_rng
from .event_resolver import EventResolver
from .guest_agent_manager import GuestAgentManager
from .host_controller import HostController
from .log_summarizer import LogSummarizer
from .memory_store import MemoryStore
from .metrics_engine import MetricsEngine
from .schemas import RunReport, SimulationConfig, TurnTrace
from .state_store import StateStore


class ArenaEngine:
    def __init__(self, config: SimulationConfig):
        self.config = config

    def run(self) -> RunReport:
        state = create_initial_state(self.config)
        rng = seeded_rng(self.config)
        host_controller = HostController(state, rng)
        guest_manager = GuestAgentManager(state, rng)
        resolver = EventResolver()
        metrics_engine = MetricsEngine()
        summarizer = LogSummarizer(state)
        memory_store = MemoryStore(state, summarizer)
        state_store = StateStore(self.config)
        run_dir = state_store.start_run(state)

        for _ in range(self.config.max_turns):
            host_action, host_raw_output = host_controller.decide_intervention(state)
            guest_decisions, guest_raw_outputs = guest_manager.collect_decisions(
                state, host_action
            )
            resolved_events = resolver.resolve_turn(state, host_action, guest_decisions)
            metrics = metrics_engine.compute_snapshot(
                state, host_action, guest_decisions, resolved_events
            )
            state.metrics_history.append(metrics)
            host_controller.observe_outcome(
                state, host_action, metrics, resolved_events
            )
            narrative, narrative_raw_output = summarizer.summarize_turn(
                state, host_action, guest_decisions, resolved_events
            )
            trace = TurnTrace(
                turn_number=state.world.turn_count,
                host_action=host_action,
                host_raw_output=host_raw_output,
                guest_decisions=guest_decisions,
                guest_raw_outputs=guest_raw_outputs,
                resolved_events=resolved_events,
                metrics=metrics,
                narrative=narrative,
                narrative_raw_output=narrative_raw_output,
            )
            state.turn_traces.append(trace)
            memory_records = memory_store.record_turn(
                state, host_action, guest_decisions, resolved_events
            )
            state_store.record_turn(state, trace, memory_records)

            if self.config.stop_on_collapse and metrics.collapse_risk >= 0.97:
                state.scenario_notes.append(
                    "Run stopped early because collapse risk crossed the hard threshold."
                )
                break

        final_summary = summarizer.summarize_run(state)
        (
            summary_path,
            dashboard_path,
            analysis_exports,
            turn_breakdown_path,
        ) = state_store.finalize_run(state, final_summary)
        return RunReport(
            run_id=state.run_id,
            run_name=self.config.run_name,
            variant_name=self.config.variant_label or self.config.run_name,
            seed=self.config.seed,
            experiment_tags=self.config.experiment_tags,
            run_directory=run_dir,
            final_metrics=state.metrics_history[-1],
            final_summary=final_summary,
            turn_count=state.world.turn_count,
            summary_path=summary_path,
            dashboard_path=dashboard_path,
            trace_path=f"{run_dir}/trace.jsonl",
            turn_breakdown_path=turn_breakdown_path,
            analysis_exports=analysis_exports,
        )
