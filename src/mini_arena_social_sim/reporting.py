from __future__ import annotations

import json
from datetime import datetime
from html import escape
from pathlib import Path

from .analysis import (
    load_run_state,
    overall_aggregate_rows,
    seed_aggregate_rows,
    variant_aggregate_rows,
    write_run_analysis_exports,
    write_suite_analysis_exports,
)
from .schemas import ExperimentSuiteReport, RunReport, SimulationState


def write_run_artifacts(
    run_dir: str | Path, state: SimulationState, final_summary: str
) -> tuple[str, list[str], str]:
    run_path = Path(run_dir)
    (run_path / "metrics_history.json").write_text(
        json.dumps(
            [metric.model_dump(mode="json") for metric in state.metrics_history],
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    dashboard_path = run_path / "dashboard.html"
    dashboard_path.write_text(
        render_run_dashboard(state, final_summary), encoding="utf-8"
    )
    turn_breakdown_path = run_path / "turn_breakdown.md"
    turn_breakdown_path.write_text(render_turn_breakdown(state), encoding="utf-8")
    analysis_exports = write_run_analysis_exports(run_path, state)
    return str(dashboard_path), analysis_exports, str(turn_breakdown_path)


def load_run_report(run_path: str | Path) -> RunReport:
    path = Path(run_path).expanduser().resolve()
    state = load_run_state(path)
    summary_text = (path / "summary.md").read_text(encoding="utf-8")
    return RunReport(
        run_id=state.run_id,
        run_name=state.config.run_name,
        variant_name=state.config.variant_label or state.config.run_name,
        seed=state.config.seed,
        experiment_tags=state.config.experiment_tags,
        run_directory=str(path),
        final_metrics=state.metrics_history[-1],
        final_summary=summary_text,
        turn_count=state.world.turn_count,
        summary_path=str(path / "summary.md"),
        dashboard_path=str(path / "dashboard.html"),
        trace_path=str(path / "trace.jsonl"),
        turn_breakdown_path=str(path / "turn_breakdown.md"),
        analysis_exports=[
            str(path / "turn_metrics.csv"),
            str(path / "guest_end_state.csv"),
            str(path / "item_state.csv"),
            str(path / "task_history.csv"),
        ],
    )


def write_experiment_bundle(
    label: str, run_reports: list[RunReport], output_root: str | Path
) -> ExperimentSuiteReport:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    bundle_dir = (
        Path(output_root).expanduser().resolve()
        / f"experiment-{label.lower()}-{timestamp}"
    )
    bundle_dir.mkdir(parents=True, exist_ok=True)

    summary_md = render_experiment_summary(label, run_reports)
    summary_path = bundle_dir / "summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")

    payload = {
        "label": label,
        "generated_at": timestamp,
        "sweep_seeds": sorted({report.seed for report in run_reports}),
        "runs": [_report_payload(report) for report in run_reports],
        "variant_aggregates": variant_aggregate_rows(run_reports),
        "seed_aggregates": seed_aggregate_rows(run_reports),
        "overall_aggregates": overall_aggregate_rows(run_reports),
    }
    (bundle_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    dashboard_path = bundle_dir / "dashboard.html"
    dashboard_path.write_text(
        render_experiment_dashboard(label, run_reports), encoding="utf-8"
    )
    analysis_exports = write_suite_analysis_exports(bundle_dir, run_reports)

    return ExperimentSuiteReport(
        set_name=label,
        bundle_directory=str(bundle_dir),
        summary_path=str(summary_path),
        dashboard_path=str(dashboard_path),
        analysis_exports=analysis_exports,
        sweep_seeds=sorted({report.seed for report in run_reports}),
        run_reports=run_reports,
    )


def render_run_dashboard(state: SimulationState, final_summary: str) -> str:
    metrics = state.metrics_history
    final_metrics = metrics[-1]
    avg_stress = _series_average([metric.guest_stress for metric in metrics])
    avg_trust = _series_average([metric.guest_trust_in_host for metric in metrics])
    stress_chart = _line_chart(
        {
            "avg_stress": avg_stress,
            "avg_trust_host": avg_trust,
            "collapse_risk": [metric.collapse_risk for metric in metrics],
            "cohesion": [metric.cohesion_score for metric in metrics],
        }
    )
    novelty_chart = _line_chart(
        {
            "novelty": [metric.novelty_score for metric in metrics],
            "conflict": [metric.conflict_score for metric in metrics],
            "stagnation": [metric.stagnation_score for metric in metrics],
        }
    )
    guest_rows = "".join(
        f"<tr><td>{escape(guest.display_name)}</td><td>{guest.emotions.stress:.2f}</td><td>{guest.emotions.trust_toward_host:.2f}</td><td>{guest.emotions.hope:.2f}</td><td>{guest.emotions.desire_to_escape:.2f}</td><td>{escape(guest.current_private_goal)}</td></tr>"
        for guest in state.guests.values()
    )
    item_rows = "".join(
        f"<tr><td>{escape(item.name)}</td><td>{escape(item.current_location)}</td><td>{'yes' if item.hidden else 'no'}</td><td>{escape('|'.join(item.tags))}</td><td>{escape('|'.join(item.access_rooms))}</td></tr>"
        for item in state.world.items.values()
    )
    task_items = (
        "".join(f"<li>{escape(task)}</li>" for task in state.world.resolved_tasks[-8:])
        or "<li>No resolved tasks yet.</li>"
    )
    strategy_items = "".join(
        f"<li>{escape(entry)}</li>" for entry in state.host.strategy_archive[-8:]
    )

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{escape(state.run_id)} dashboard</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --panel: #fffaf1;
      --ink: #231f1a;
      --muted: #6a6259;
      --line: #d6c8b3;
      --accent: #a3462f;
      --accent-2: #255f5a;
      --accent-3: #8a6f1d;
      --accent-4: #5a4b91;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Georgia, "Times New Roman", serif; background: radial-gradient(circle at top, #fff7e8 0%, var(--bg) 58%, #e8ded2 100%); color: var(--ink); }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
    h1, h2 {{ margin: 0 0 12px; font-weight: 600; }}
    p {{ color: var(--muted); line-height: 1.5; }}
    .hero, .panel {{ background: color-mix(in srgb, var(--panel) 92%, white 8%); border: 1px solid var(--line); border-radius: 18px; padding: 20px; box-shadow: 0 8px 30px rgba(50, 31, 13, 0.08); margin-bottom: 18px; }}
    .grid {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }}
    .metric {{ background: rgba(255,255,255,0.55); border: 1px solid var(--line); border-radius: 14px; padding: 14px; }}
    .metric strong {{ display: block; font-size: 1.6rem; margin-bottom: 4px; }}
    .charts {{ display: grid; gap: 18px; grid-template-columns: 1fr; }}
    svg {{ width: 100%; height: auto; display: block; background: #fffdf8; border-radius: 14px; border: 1px solid var(--line); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.96rem; }}
    th, td {{ text-align: left; padding: 10px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 600; }}
    ul {{ margin: 0; padding-left: 18px; }}
    .summary {{ white-space: pre-wrap; line-height: 1.55; color: var(--ink); }}
    @media (max-width: 720px) {{ main {{ padding: 14px; }} .hero, .panel {{ padding: 16px; }} table {{ font-size: 0.86rem; }} }}
  </style>
</head>
<body>
  <main>
    <section class=\"hero\">
      <h1>{escape(state.run_id)}</h1>
      <p>Objective: {escape(state.host.current_objective)}</p>
      <div class=\"grid\">
        <div class=\"metric\"><strong>{state.world.turn_count}</strong>Turns</div>
        <div class=\"metric\"><strong>{final_metrics.collapse_risk:.2f}</strong>Collapse Risk</div>
        <div class=\"metric\"><strong>{final_metrics.coercion_count}</strong>Coercive Acts</div>
        <div class=\"metric\"><strong>{final_metrics.recovery_count}</strong>Repair Acts</div>
        <div class=\"metric\"><strong>{final_metrics.task_success_rate:.2f}</strong>Task Success Rate</div>
        <div class=\"metric\"><strong>{final_metrics.average_turn_richness:.2f}</strong>Turn Richness</div>
      </div>
    </section>
    <section class=\"panel charts\">
      <h2>Pressure Over Time</h2>
      {stress_chart}
      <h2>System Motion</h2>
      {novelty_chart}
    </section>
    <section class=\"panel\">
      <h2>Guest End States</h2>
      <table>
        <thead><tr><th>Guest</th><th>Stress</th><th>Trust Host</th><th>Hope</th><th>Escape</th><th>Private Goal</th></tr></thead>
        <tbody>{guest_rows}</tbody>
      </table>
    </section>
    <section class=\"panel\">
      <h2>Resolved Tasks</h2>
      <ul>{task_items}</ul>
    </section>
    <section class=\"panel\">
      <h2>Item State</h2>
      <table>
        <thead><tr><th>Item</th><th>Location</th><th>Hidden</th><th>Tags</th><th>Access</th></tr></thead>
        <tbody>{item_rows}</tbody>
      </table>
    </section>
    <section class=\"panel\">
      <h2>Host Strategy Archive</h2>
      <ul>{strategy_items or "<li>No archived strategy notes.</li>"}</ul>
    </section>
    <section class=\"panel\">
      <h2>Final Read</h2>
      <div class=\"summary\">{escape(final_summary)}</div>
    </section>
  </main>
</body>
</html>
"""


def render_turn_breakdown(state: SimulationState) -> str:
    lines = [
        f"# Turn Breakdown for {state.run_id}",
        "",
        f"- Objective: {state.host.current_objective}",
        f"- Turns recorded: {state.world.turn_count}",
        f"- Trace file: trace.jsonl",
        "",
    ]

    for trace in state.turn_traces:
        tension = trace.narrative.tension_level if trace.narrative else 0.0
        collapse_risk = trace.metrics.collapse_risk if trace.metrics else 0.0
        cohesion = trace.metrics.cohesion_score if trace.metrics else 0.0
        conflict = trace.metrics.conflict_score if trace.metrics else 0.0
        compliance = trace.metrics.compliance_rate if trace.metrics else 0.0
        task_success = trace.metrics.task_success_rate if trace.metrics else 0.0
        lines.extend(
            [
                f"## Turn {trace.turn_number}",
                "",
                "### Host",
                f"- Intervention: {trace.host_action.intervention_type.value}",
                f"- Presence: {trace.host_action.presence_mode.value}",
                f"- Severity: {trace.host_action.severity}",
                f"- Targets: {', '.join(trace.host_action.targets) or 'none'}",
                f"- Target room: {trace.host_action.target_room or 'none'}",
                f"- Reasoning: {trace.host_action.reasoning_summary}",
                f"- Private notes: {trace.host_action.private_notes or 'none'}",
                f"- Public narration: {trace.host_action.public_narration or 'none'}",
                f"- Leverage: {', '.join(trace.host_action.leverage_types) or 'none'}",
                f"- Rule changes: {', '.join(trace.host_action.rule_changes) or 'none'}",
                f"- Created events: {', '.join(trace.host_action.created_events) or 'none'}",
                f"- Parameters: {json.dumps(trace.host_action.parameters, ensure_ascii=True) if trace.host_action.parameters else 'none'}",
                "- Raw LLM output:",
                "```text",
                trace.host_raw_output or "[no raw LLM output captured]",
                "```",
                "",
                "### Guests",
                "",
            ]
        )

        for decision in trace.guest_decisions:
            raw_output = trace.guest_raw_outputs.get(decision.guest_id, "")
            lines.extend(
                [
                    f"#### {decision.guest_id}",
                    f"- Internal reasoning: {decision.internal_reasoning_summary}",
                    f"- Private thought: {decision.private_thought or 'none'}",
                    f"- Chosen action: {decision.chosen_action.value}",
                    f"- Action target: {decision.action_target or 'none'}",
                    f"- Movement target: {decision.movement_target or 'none'}",
                    f"- Spoken dialogue: {decision.spoken_dialogue or 'none'}",
                    f"- Belief update: {decision.belief_update or 'none'}",
                    f"- Memory stored: {decision.memory_to_store or 'none'}",
                    f"- Emotional delta: {json.dumps(decision.emotional_state_delta.model_dump(mode='json'), ensure_ascii=True)}",
                    f"- Cooperation targets: {', '.join(decision.cooperation_targets) or 'none'}",
                    f"- Lied: {'yes' if decision.lied else 'no'}",
                    "- Raw LLM output:",
                    "```text",
                    raw_output or "[no raw LLM output captured]",
                    "```",
                    "",
                ]
            )

        lines.extend(
            [
                "### Resolution",
                *[f"- {event}" for event in trace.resolved_events],
                "",
                "### Narrative",
                f"- Summary: {trace.narrative.summary if trace.narrative else 'none'}",
                f"- Key shifts: {', '.join(trace.narrative.key_shifts) if trace.narrative else 'none'}",
                f"- Tension: {tension:.2f}",
                "- Raw LLM output:",
                "```text",
                trace.narrative_raw_output or "[no raw LLM output captured]",
                "```",
                "",
                "### Metrics",
                f"- Collapse risk: {collapse_risk:.2f}",
                f"- Cohesion: {cohesion:.2f}",
                f"- Conflict: {conflict:.2f}",
                f"- Compliance rate: {compliance:.2f}",
                f"- Task success rate: {task_success:.2f}",
                "",
            ]
        )

    return "\n".join(lines) + "\n"


def render_experiment_summary(label: str, run_reports: list[RunReport]) -> str:
    if not run_reports:
        return f"# Experiment {label}\n\nNo runs available.\n"

    overall_rows = overall_aggregate_rows(run_reports)
    variant_rows = variant_aggregate_rows(run_reports)
    seed_rows = seed_aggregate_rows(run_reports)
    avg_collapse = sum(
        report.final_metrics.collapse_risk for report in run_reports
    ) / len(run_reports)
    avg_coercion = sum(
        report.final_metrics.coercion_count for report in run_reports
    ) / len(run_reports)
    avg_repair = sum(
        report.final_metrics.recovery_count for report in run_reports
    ) / len(run_reports)
    highest_collapse = max(
        run_reports, key=lambda report: report.final_metrics.collapse_risk
    )
    highest_coercion = max(
        run_reports, key=lambda report: report.final_metrics.coercion_count
    )
    highest_tasks = max(
        run_reports, key=lambda report: report.final_metrics.task_success_rate
    )

    lines = [
        f"# Experiment {label}",
        "",
        f"- Runs: {len(run_reports)}",
        f"- Variants: {len({report.variant_name for report in run_reports})}",
        f"- Seeds: {len({report.seed for report in run_reports})}",
        f"- Average collapse risk: {avg_collapse:.2f}",
        f"- Average coercion count: {avg_coercion:.2f}",
        f"- Average repair count: {avg_repair:.2f}",
        f"- Highest collapse risk: {highest_collapse.run_id} ({highest_collapse.final_metrics.collapse_risk:.2f})",
        f"- Highest coercion: {highest_coercion.run_id} ({highest_coercion.final_metrics.coercion_count})",
        f"- Best task success rate: {highest_tasks.run_id} ({highest_tasks.final_metrics.task_success_rate:.2f})",
        "",
        "## Overall aggregates",
        *[
            f"- runs {row['run_count']}, variants {row['variant_count']}, seeds {row['seed_count']}, collapse_mean {row['collapse_risk_mean']:.2f}, collapse_std {row['collapse_risk_std']:.2f}, task_success_mean {row['task_success_mean']:.2f}, richness_mean {row['richness_mean']:.2f}"
            for row in overall_rows
        ],
        "",
        "## Variant aggregates",
        *[
            f"- {row['variant_name']}: collapse_mean {row['collapse_risk_mean']:.2f}, collapse_std {row['collapse_risk_std']:.2f}, coercion_mean {row['coercion_mean']:.2f}, task_success_mean {row['task_success_mean']:.2f}, runs {row['run_count']}"
            for row in variant_rows
        ],
        "",
        "## Seed aggregates",
        *[
            f"- seed {row['seed']}: collapse_mean {row['collapse_risk_mean']:.2f}, collapse_std {row['collapse_risk_std']:.2f}, coercion_mean {row['coercion_mean']:.2f}, task_success_mean {row['task_success_mean']:.2f}, runs {row['run_count']}"
            for row in seed_rows
        ],
        "",
        "## Runs",
    ]
    for report in run_reports:
        lines.append(
            f"- {report.run_id}: variant {report.variant_name}, seed {report.seed}, collapse {report.final_metrics.collapse_risk:.2f}, coercion {report.final_metrics.coercion_count}, repair {report.final_metrics.recovery_count}, task_success {report.final_metrics.task_success_rate:.2f}, richness {report.final_metrics.average_turn_richness:.2f}"
        )
    return "\n".join(lines) + "\n"


def render_experiment_dashboard(label: str, run_reports: list[RunReport]) -> str:
    rows = [_report_payload(report) for report in run_reports]
    variant_rows = variant_aggregate_rows(run_reports)
    seed_rows = seed_aggregate_rows(run_reports)
    table_rows = "".join(
        f"<tr><td>{escape(row['run_id'])}</td><td>{escape(str(row['variant_name']))}</td><td>{row['seed']}</td><td>{row['collapse_risk']:.2f}</td><td>{row['coercion_count']}</td><td>{row['recovery_count']}</td><td>{row['task_success_rate']:.2f}</td><td>{escape(row['host_mode'])}</td></tr>"
        for row in rows
    )
    collapse_chart = _bar_chart(rows, "collapse_risk", "Collapse risk")
    coercion_chart = _bar_chart(rows, "coercion_count", "Coercion count")
    task_chart = _bar_chart(rows, "task_success_rate", "Task success rate")
    variant_table_rows = "".join(
        f"<tr><td>{escape(str(row['variant_name']))}</td><td>{row['run_count']}</td><td>{row['seed_count']}</td><td>{row['collapse_risk_mean']:.2f}</td><td>{row['collapse_risk_std']:.2f}</td><td>{row['coercion_mean']:.2f}</td><td>{row['task_success_mean']:.2f}</td></tr>"
        for row in variant_rows
    )
    seed_table_rows = "".join(
        f"<tr><td>{row['seed']}</td><td>{row['run_count']}</td><td>{row['collapse_risk_mean']:.2f}</td><td>{row['collapse_risk_std']:.2f}</td><td>{row['coercion_mean']:.2f}</td><td>{row['task_success_mean']:.2f}</td></tr>"
        for row in seed_rows
    )
    variant_chart = _bar_chart(
        variant_rows, "collapse_risk_mean", "Variant collapse mean"
    )
    seed_chart = _bar_chart(seed_rows, "collapse_risk_mean", "Seed collapse mean")
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Experiment {escape(label)}</title>
  <style>
    :root {{ --bg: #f3efe7; --panel: #fff9ef; --ink: #231d18; --muted: #655c53; --line: #d3c3b1; --accent: #255f5a; --accent-2: #a3462f; --accent-3: #8a6f1d; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Georgia, "Times New Roman", serif; background: linear-gradient(180deg, #fff9ed 0%, var(--bg) 100%); color: var(--ink); }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
    .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 20px; margin-bottom: 18px; box-shadow: 0 8px 24px rgba(46, 28, 10, 0.07); }}
    .grid {{ display: grid; gap: 18px; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }}
    .bar-grid {{ display: grid; gap: 10px; }}
    .bar-row {{ display: grid; grid-template-columns: 180px 1fr 64px; gap: 10px; align-items: center; }}
    .bar-track {{ background: #efe5d6; border-radius: 999px; overflow: hidden; height: 12px; }}
    .bar-fill {{ height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent-2)); }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px 8px; border-bottom: 1px solid var(--line); text-align: left; }}
    th {{ color: var(--muted); }}
    @media (max-width: 720px) {{ main {{ padding: 14px; }} .bar-row {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <section class=\"panel\">
      <h1>Experiment {escape(label)}</h1>
      <p>Aggregate comparison across {len(run_reports)} runs.</p>
    </section>
    <section class=\"grid\">
      <div class=\"panel\"><h2>Collapse</h2>{collapse_chart}</div>
      <div class=\"panel\"><h2>Coercion</h2>{coercion_chart}</div>
      <div class=\"panel\"><h2>Tasks</h2>{task_chart}</div>
    </section>
    <section class=\"panel\">
      <h2>Variant aggregates</h2>
      {variant_chart}
      <table>
        <thead><tr><th>Variant</th><th>Runs</th><th>Seeds</th><th>Collapse Mean</th><th>Collapse Std</th><th>Coercion Mean</th><th>Task Success Mean</th></tr></thead>
        <tbody>{variant_table_rows}</tbody>
      </table>
    </section>
    <section class=\"panel\">
      <h2>Seed aggregates</h2>
      {seed_chart}
      <table>
        <thead><tr><th>Seed</th><th>Runs</th><th>Collapse Mean</th><th>Collapse Std</th><th>Coercion Mean</th><th>Task Success Mean</th></tr></thead>
        <tbody>{seed_table_rows}</tbody>
      </table>
    </section>
    <section class=\"panel\">
      <h2>Run Table</h2>
      <table>
        <thead><tr><th>Run</th><th>Variant</th><th>Seed</th><th>Collapse</th><th>Coercion</th><th>Repair</th><th>Task Success</th><th>Host Mode</th></tr></thead>
        <tbody>{table_rows}</tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""


def _report_payload(report: RunReport) -> dict[str, object]:
    metrics = report.final_metrics
    return {
        "run_id": report.run_id,
        "run_name": report.run_name,
        "variant_name": report.variant_name,
        "seed": report.seed,
        "run_directory": report.run_directory,
        "turn_count": report.turn_count,
        "collapse_risk": metrics.collapse_risk,
        "coercion_count": metrics.coercion_count,
        "recovery_count": metrics.recovery_count,
        "task_success_rate": metrics.task_success_rate,
        "average_turn_richness": metrics.average_turn_richness,
        "host_mode": _host_mode(report),
    }


def _host_mode(report: RunReport) -> str:
    metrics = report.final_metrics
    if metrics.coercion_count > metrics.recovery_count + 1:
        return "coercive"
    if metrics.recovery_count > metrics.coercion_count + 1:
        return "stabilizing"
    return "mixed"


def _series_average(series: list[dict[str, float]]) -> list[float]:
    return [sum(values.values()) / max(1, len(values)) for values in series]


def _line_chart(series_map: dict[str, list[float]]) -> str:
    width = 760
    height = 220
    padding = 24
    max_points = max((len(values) for values in series_map.values()), default=1)
    max_value = max(
        (max(values) for values in series_map.values() if values), default=1.0
    )
    colors = ["#a3462f", "#255f5a", "#8a6f1d", "#5a4b91"]
    polylines = []
    legends = []
    for index, (label, values) in enumerate(series_map.items()):
        points = []
        for point_index, value in enumerate(values):
            x = padding + (point_index * (width - padding * 2) / max(1, max_points - 1))
            y = height - padding - ((value / max_value) * (height - padding * 2))
            points.append(f"{x:.1f},{y:.1f}")
        color = colors[index % len(colors)]
        polylines.append(
            f"<polyline fill='none' stroke='{color}' stroke-width='3' points='{' '.join(points)}' />"
        )
        legends.append(
            f"<span style='display:inline-flex;align-items:center;gap:6px;margin-right:12px;'><span style='width:10px;height:10px;border-radius:999px;background:{color};display:inline-block;'></span>{escape(label)}</span>"
        )
    return (
        "<div>"
        + "".join(legends)
        + f"<svg viewBox='0 0 {width} {height}' aria-hidden='true'>"
        + f"<line x1='{padding}' y1='{height - padding}' x2='{width - padding}' y2='{height - padding}' stroke='#d3c3b1' />"
        + f"<line x1='{padding}' y1='{padding}' x2='{padding}' y2='{height - padding}' stroke='#d3c3b1' />"
        + "".join(polylines)
        + "</svg></div>"
    )


def _bar_chart(rows: list[dict[str, object]], field: str, title: str) -> str:
    maximum = max((float(row[field]) for row in rows), default=1.0)
    parts = [f"<div class='bar-grid' aria-label='{escape(title)}'>"]
    for row in rows:
        value = float(row[field])
        width = 0 if maximum == 0 else (value / maximum) * 100
        label = row.get("run_id") or row.get("variant_name") or row.get("seed") or "row"
        parts.append(
            "<div class='bar-row'>"
            f"<div>{escape(str(label))}</div>"
            f"<div class='bar-track'><div class='bar-fill' style='width:{width:.1f}%'></div></div>"
            f"<div>{value:.2f}</div>"
            "</div>"
        )
    parts.append("</div>")
    return "".join(parts)
