from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
import json
import csv
import random
from argparse import Namespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mini_arena_social_sim.arena_engine import ArenaEngine
from mini_arena_social_sim.analysis import load_run_state
from mini_arena_social_sim.defaults import create_initial_state
from mini_arena_social_sim.event_resolver import EventResolver
from mini_arena_social_sim.experiment_runner import ExperimentRunner
from mini_arena_social_sim.host_controller import HostController
from mini_arena_social_sim.cli import build_config
from mini_arena_social_sim.reporting import load_run_report, write_experiment_bundle
from mini_arena_social_sim.ollama_client import OllamaClient
from mini_arena_social_sim.schemas import (
    ArenaTask,
    BackendConfig,
    GuestActionType,
    GuestDecision,
    HostPresenceMode,
    HostIntervention,
    InterventionType,
    SimulationConfig,
)


class SimulationSmokeTests(unittest.TestCase):
    def test_initial_state_has_expected_cast_and_rooms(self) -> None:
        config = SimulationConfig(backend=BackendConfig(kind="heuristic"))
        state = create_initial_state(config)
        self.assertEqual(len(state.guests), 5)
        self.assertIn("central_hub", state.world.rooms)
        self.assertIn("anomaly_space", state.world.rooms)

    def test_heuristic_run_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = SimulationConfig(
                run_name="smoke",
                max_turns=3,
                backend=BackendConfig(kind="heuristic"),
                output_dir=tmp_dir,
            )
            report = ArenaEngine(config).run()
            run_dir = Path(report.run_directory)
            self.assertTrue((run_dir / "config.json").exists())
            self.assertTrue((run_dir / "summary.md").exists())
            self.assertTrue((run_dir / "dashboard.html").exists())
            self.assertTrue((run_dir / "turn_breakdown.md").exists())
            self.assertTrue((run_dir / "metrics_history.json").exists())
            self.assertTrue((run_dir / "turn_metrics.csv").exists())
            self.assertTrue((run_dir / "guest_end_state.csv").exists())
            self.assertTrue((run_dir / "item_state.csv").exists())
            self.assertTrue((run_dir / "task_history.csv").exists())
            self.assertGreaterEqual(report.turn_count, 1)

            final_state = json.loads((run_dir / "final_state.json").read_text())
            self.assertIn("resolved_tasks", final_state["world"])
            self.assertTrue(final_state["turn_traces"][0]["host_raw_output"])
            self.assertIn("guest_raw_outputs", final_state["turn_traces"][0])

    def test_experiment_runner_builds_variants(self) -> None:
        config = SimulationConfig(backend=BackendConfig(kind="heuristic"))
        runner = ExperimentRunner(config)
        variants = runner.build_set("A")
        self.assertEqual(len(variants), 5)

    def test_backend_config_uses_single_model_field(self) -> None:
        backend = BackendConfig.model_validate({"host_model": "llama3.1:latest"})
        self.assertEqual(backend.model, "llama3.1:latest")

    def test_ollama_client_extracts_balanced_json_from_extra_text(self) -> None:
        content = '{"guest_id":"caretaker","chosen_action":"observe"}\nextra text'
        parsed = OllamaClient._extract_json(content)
        self.assertEqual(parsed["guest_id"], "caretaker")

    def test_cli_build_config_uses_shared_model(self) -> None:
        args = Namespace(
            backend="ollama",
            model="llama3.1:latest",
            host_model="",
            guest_model="",
            summarizer_model="",
            base_url="http://127.0.0.1:11434",
            run_name="model-test",
            seed=7,
            turns=1,
            summary_interval=4,
            objective="test objective",
            world_preset="default",
            output_dir="runs",
            keep_going_on_collapse=False,
        )
        config = build_config(args)
        self.assertEqual(config.backend.model, "llama3.1:latest")

    def test_turn_breakdown_report_contains_reasoning_and_raw_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = SimulationConfig(
                run_name="breakdown",
                max_turns=1,
                backend=BackendConfig(kind="heuristic"),
                output_dir=tmp_dir,
            )
            report = ArenaEngine(config).run()
            breakdown = Path(report.turn_breakdown_path).read_text(encoding="utf-8")
            self.assertIn("## Turn 1", breakdown)
            self.assertIn("### Host", breakdown)
            self.assertIn("Internal reasoning", breakdown)
            self.assertIn("Raw LLM output", breakdown)

    def test_host_can_be_absent_when_existing_task_is_running(self) -> None:
        config = SimulationConfig(backend=BackendConfig(kind="heuristic"))
        state = create_initial_state(config)
        state.world.active_tasks.append(
            ArenaTask(
                task_id="task-active",
                description="Observe ongoing task dynamics.",
                created_turn=0,
                deadline_turn=2,
                assigned_guests=["analyst", "caretaker"],
            )
        )
        controller = HostController(state, random.Random(config.seed))
        action, raw_output = controller.decide_intervention(state)
        self.assertEqual(action.intervention_type, InterventionType.NO_OP)
        self.assertEqual(action.presence_mode, HostPresenceMode.ABSENT)
        self.assertIn("heuristic backend", raw_output)

    def test_experiment_suite_writes_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = SimulationConfig(
                max_turns=1,
                backend=BackendConfig(kind="heuristic"),
                output_dir=tmp_dir,
            )
            suite = ExperimentRunner(config).run_suite("A")
            self.assertEqual(len(suite.run_reports), 5)
            self.assertTrue(Path(suite.summary_path).exists())
            self.assertTrue(Path(suite.dashboard_path).exists())
            self.assertTrue(
                any(path.endswith("suite_runs.csv") for path in suite.analysis_exports)
            )
            self.assertTrue(
                any(
                    path.endswith("variant_aggregates.csv")
                    for path in suite.analysis_exports
                )
            )
            self.assertTrue(
                any(
                    path.endswith("seed_aggregates.csv")
                    for path in suite.analysis_exports
                )
            )

    def test_single_config_seed_sweep_writes_seed_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = SimulationConfig(
                run_name="seed-sweep",
                variant_label="seed-sweep",
                max_turns=1,
                backend=BackendConfig(kind="heuristic"),
                output_dir=tmp_dir,
            )
            suite = ExperimentRunner(config).run_seed_sweep([3, 4, 5])
            self.assertEqual(len(suite.run_reports), 3)
            self.assertEqual(suite.sweep_seeds, [3, 4, 5])
            seed_csv = next(
                path
                for path in suite.analysis_exports
                if path.endswith("seed_aggregates.csv")
            )
            with Path(seed_csv).open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual({int(row["seed"]) for row in rows}, {3, 4, 5})

    def test_experiment_seed_sweep_writes_variant_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = SimulationConfig(
                max_turns=1,
                backend=BackendConfig(kind="heuristic"),
                output_dir=tmp_dir,
            )
            suite = ExperimentRunner(config).run_suite("A", seeds=[1, 2])
            self.assertEqual(len(suite.run_reports), 10)
            self.assertEqual(
                len({report.variant_name for report in suite.run_reports}), 5
            )
            variant_csv = next(
                path
                for path in suite.analysis_exports
                if path.endswith("variant_aggregates.csv")
            )
            with Path(variant_csv).open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(int(row["seed_count"]) == 2 for row in rows))

    def test_task_assignment_resolves_and_updates_audit(self) -> None:
        state = create_initial_state(
            SimulationConfig(backend=BackendConfig(kind="heuristic"))
        )
        resolver = EventResolver()
        host_action = HostIntervention(
            intervention_type=InterventionType.TASK_ASSIGNMENT,
            reasoning_summary="Force coordinated recall.",
            public_narration="The Host orders a shared recollection task.",
            target_room="social_room",
            severity=5,
            parameters={
                "task": "shared recollection",
                "required_actions": ["cooperate", "speak", "obey"],
                "min_participants": 2,
                "deadline_turns": 1,
                "reward_if_completed": "comfort tokens",
            },
        )
        decisions = [
            GuestDecision(
                guest_id="analyst",
                internal_reasoning_summary="Need a clean account.",
                chosen_action=GuestActionType.SPEAK,
            ),
            GuestDecision(
                guest_id="performer",
                internal_reasoning_summary="Keep the mood moving.",
                chosen_action=GuestActionType.OBSERVE,
            ),
            GuestDecision(
                guest_id="rebel",
                internal_reasoning_summary="Refuse submission.",
                chosen_action=GuestActionType.RESIST,
            ),
            GuestDecision(
                guest_id="caretaker",
                internal_reasoning_summary="Keep everyone aligned.",
                chosen_action=GuestActionType.COOPERATE,
            ),
            GuestDecision(
                guest_id="drifter",
                internal_reasoning_summary="Watch the pattern.",
                chosen_action=GuestActionType.OBSERVE,
            ),
        ]

        resolver.resolve_turn(state, host_action, decisions)
        self.assertTrue(state.world.resolved_tasks)
        self.assertEqual(state.audit.task_assignments, 1)
        self.assertEqual(state.audit.task_successes, 1)
        self.assertFalse(state.world.active_tasks)

    def test_hidden_item_unlocks_anomaly_access(self) -> None:
        state = create_initial_state(
            SimulationConfig(backend=BackendConfig(kind="heuristic"))
        )
        resolver = EventResolver()
        state.guests["analyst"].current_room = "social_room"
        state.world.guest_locations["analyst"] = "social_room"
        host_action = HostIntervention(
            intervention_type=InterventionType.NO_OP,
            reasoning_summary="Observe natural item discovery.",
            public_narration="The Host watches without interrupting.",
        )
        decision = GuestDecision(
            guest_id="analyst",
            internal_reasoning_summary="The room probably contains a hidden mechanism.",
            chosen_action=GuestActionType.OBSERVE,
            action_target="signal_key",
        )
        filler = [
            GuestDecision(
                guest_id=guest_id,
                internal_reasoning_summary="Hold position.",
                chosen_action=GuestActionType.OBSERVE,
            )
            for guest_id in ["performer", "rebel", "caretaker", "drifter"]
        ]
        resolver.resolve_turn(state, host_action, [decision, *filler])
        self.assertIn("signal_key", state.guests["analyst"].inventory)
        self.assertTrue(state.world.rooms["anomaly_space"].accessible)

    def test_host_strategy_archive_updates_after_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = SimulationConfig(
                run_name="strategy",
                max_turns=2,
                backend=BackendConfig(kind="heuristic"),
                output_dir=tmp_dir,
            )
            report = ArenaEngine(config).run()
            state = load_run_state(report.run_directory)
            self.assertGreaterEqual(len(state.host.strategy_archive), 2)
            self.assertNotEqual(
                state.host.recent_outcome_summary, "No turn history yet."
            )

    def test_compare_bundle_from_existing_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = SimulationConfig(
                run_name="compare-a",
                max_turns=1,
                backend=BackendConfig(kind="heuristic"),
                output_dir=tmp_dir,
            )
            report_a = ArenaEngine(config).run()
            report_b = ArenaEngine(
                config.model_copy(update={"run_name": "compare-b"})
            ).run()
            loaded_reports = [
                load_run_report(report_a.run_directory),
                load_run_report(report_b.run_directory),
            ]
            suite = write_experiment_bundle("compare-test", loaded_reports, tmp_dir)
            self.assertTrue(Path(suite.summary_path).exists())
            self.assertTrue(Path(suite.dashboard_path).exists())


if __name__ == "__main__":
    unittest.main()
