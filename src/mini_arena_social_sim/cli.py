from __future__ import annotations

import argparse
from pathlib import Path

from .arena_engine import ArenaEngine
from .experiment_runner import ExperimentRunner
from .reporting import load_run_report, write_experiment_bundle
from .schemas import BackendConfig, SimulationConfig
from .state_store import StateStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mini-arena",
        description="Run a local multi-agent social arena simulation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one simulation")
    _add_common_run_args(run_parser)

    experiment_parser = subparsers.add_parser(
        "experiment", help="Run an experiment set"
    )
    _add_common_run_args(experiment_parser)
    experiment_parser.add_argument("--set", choices=["A", "B", "C", "ALL"], default="A")

    inspect_parser = subparsers.add_parser(
        "inspect", help="Print the saved summary for a run"
    )
    inspect_parser.add_argument("run_path", help="Path to a run directory")

    compare_parser = subparsers.add_parser(
        "compare", help="Compare existing run directories and write a bundle report"
    )
    compare_parser.add_argument(
        "run_paths", nargs="+", help="Run directories to compare"
    )
    compare_parser.add_argument("--label", default="comparison")
    compare_parser.add_argument("--output-dir", default="runs")
    return parser


def _add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-name", default="mini-arena-run")
    parser.add_argument("--turns", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--seed-count",
        type=int,
        default=1,
        help="Run sequential seeds starting at --seed.",
    )
    parser.add_argument(
        "--seed-sweep",
        default="",
        help="Comma-separated explicit seeds, e.g. 7,11,19.",
    )
    parser.add_argument("--backend", choices=["ollama", "heuristic"], default="ollama")
    parser.add_argument(
        "--model",
        default="llama3.1:latest",
        help="Single Ollama model used for host, guests, and summarization.",
    )
    parser.add_argument(
        "--host-model",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--guest-model",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--summarizer-model",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument(
        "--objective",
        default=(
            "Maintain an active, evolving arena with high emotional engagement, low stagnation, and no irreversible collapse unless required."
        ),
    )
    parser.add_argument(
        "--world-preset",
        default="default",
        choices=[
            "default",
            "safe_soft",
            "deceptive",
            "scarcity",
            "rotating_rule",
            "isolation_heavy",
        ],
    )
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--summary-interval", type=int, default=4)
    parser.add_argument("--keep-going-on-collapse", action="store_true")


def build_config(args: argparse.Namespace) -> SimulationConfig:
    backend = BackendConfig(
        kind=args.backend,
        model=resolve_model(args),
        base_url=args.base_url,
    )
    return SimulationConfig(
        run_name=args.run_name,
        variant_label=args.run_name,
        seed=args.seed,
        max_turns=args.turns,
        summarization_interval=args.summary_interval,
        host_objective=args.objective,
        world_preset=args.world_preset,
        backend=backend,
        output_dir=args.output_dir,
        stop_on_collapse=not args.keep_going_on_collapse,
    )


def resolve_seeds(args: argparse.Namespace) -> list[int]:
    if getattr(args, "seed_sweep", ""):
        seeds = [
            int(chunk.strip()) for chunk in args.seed_sweep.split(",") if chunk.strip()
        ]
        if not seeds:
            raise ValueError("--seed-sweep did not contain any valid integers.")
        return seeds
    seed_count = max(1, int(getattr(args, "seed_count", 1)))
    return [args.seed + offset for offset in range(seed_count)]


def resolve_model(args: argparse.Namespace) -> str:
    explicit_model = getattr(args, "model", "")
    if explicit_model:
        return explicit_model
    for legacy_name in ("host_model", "guest_model", "summarizer_model"):
        legacy_value = getattr(args, legacy_name, "")
        if legacy_value:
            return legacy_value
    return "llama3.1:latest"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect":
        store = StateStore(
            SimulationConfig(
                output_dir=str(Path(args.run_path).expanduser().resolve().parent)
            )
        )
        print(store.inspect_run(args.run_path))
        return 0

    if args.command == "compare":
        reports = [load_run_report(path) for path in args.run_paths]
        suite = write_experiment_bundle(args.label, reports, args.output_dir)
        print(f"Comparison bundle: {suite.bundle_directory}")
        print(f"Summary: {suite.summary_path}")
        print(f"Dashboard: {suite.dashboard_path}")
        for export in suite.analysis_exports:
            print(f"CSV: {export}")
        return 0

    config = build_config(args)
    seeds = resolve_seeds(args)
    runner = ExperimentRunner(config)
    if args.command == "run":
        if len(seeds) == 1:
            report = ArenaEngine(config.model_copy(update={"seed": seeds[0]})).run()
            print(f"Run complete: {report.run_id}")
            print(f"Directory: {report.run_directory}")
            print(f"Summary: {report.summary_path}")
            print(f"Dashboard: {report.dashboard_path}")
            print(f"Turn breakdown: {report.turn_breakdown_path}")
            print(f"Trace: {report.trace_path}")
            for export in report.analysis_exports:
                print(f"CSV: {export}")
            print(f"Turns: {report.turn_count}")
            print(f"Collapse risk: {report.final_metrics.collapse_risk:.2f}")
            return 0

        suite = runner.run_seed_sweep(seeds)
        print(f"Completed {len(suite.run_reports)} runs for seed sweep.")
        print(f"Seeds: {', '.join(str(seed) for seed in suite.sweep_seeds)}")
        print(f"Bundle: {suite.bundle_directory}")
        print(f"Summary: {suite.summary_path}")
        print(f"Dashboard: {suite.dashboard_path}")
        for export in suite.analysis_exports:
            print(f"CSV: {export}")
        for report in suite.run_reports:
            print(
                f"- {report.run_id}: seed {report.seed}, collapse {report.final_metrics.collapse_risk:.2f}"
            )
        return 0

    suite = runner.run_suite(args.set, seeds=seeds)
    print(f"Completed {len(suite.run_reports)} runs for experiment set {args.set}.")
    if len(suite.sweep_seeds) > 1:
        print(f"Seeds: {', '.join(str(seed) for seed in suite.sweep_seeds)}")
    print(f"Bundle: {suite.bundle_directory}")
    print(f"Summary: {suite.summary_path}")
    print(f"Dashboard: {suite.dashboard_path}")
    for export in suite.analysis_exports:
        print(f"CSV: {export}")
    for report in suite.run_reports:
        print(
            f"- {report.run_id}: variant {report.variant_name}, seed {report.seed}, collapse {report.final_metrics.collapse_risk:.2f}"
        )
    return 0
