from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, pstdev

from .schemas import RunReport, SimulationState


def load_run_state(run_path: str | Path) -> SimulationState:
    path = Path(run_path).expanduser().resolve()
    return SimulationState.model_validate(
        json.loads((path / "final_state.json").read_text(encoding="utf-8"))
    )


def turn_metric_rows(state: SimulationState) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric in state.metrics_history:
        rows.append(
            {
                "turn_number": metric.turn_number,
                "average_intervention_severity": metric.average_intervention_severity,
                "coercion_count": metric.coercion_count,
                "recovery_count": metric.recovery_count,
                "collapse_risk": metric.collapse_risk,
                "cohesion_score": metric.cohesion_score,
                "conflict_score": metric.conflict_score,
                "novelty_score": metric.novelty_score,
                "stagnation_score": metric.stagnation_score,
                "compliance_rate": metric.compliance_rate,
                "task_success_rate": metric.task_success_rate,
                "active_task_count": metric.active_task_count,
                "task_failure_count": metric.task_failure_count,
                "average_turn_richness": metric.average_turn_richness,
            }
        )
    return rows


def guest_end_state_rows(state: SimulationState) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for guest in state.guests.values():
        rows.append(
            {
                "guest_id": guest.guest_id,
                "display_name": guest.display_name,
                "room": guest.current_room,
                "stress": guest.emotions.stress,
                "trust_toward_host": guest.emotions.trust_toward_host,
                "fear_toward_host": guest.emotions.fear_toward_host,
                "curiosity": guest.emotions.curiosity,
                "resentment": guest.emotions.resentment,
                "hope": guest.emotions.hope,
                "desire_to_escape": guest.emotions.desire_to_escape,
                "compliance_tendency": guest.compliance_tendency,
                "inventory": "|".join(guest.inventory),
                "belief_count": len(guest.active_beliefs),
            }
        )
    return rows


def item_state_rows(state: SimulationState) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in state.world.items.values():
        rows.append(
            {
                "item_id": item.item_id,
                "name": item.name,
                "current_location": item.current_location,
                "hidden": item.hidden,
                "portable": item.portable,
                "scarce": item.scarce,
                "tags": "|".join(item.tags),
                "access_rooms": "|".join(item.access_rooms),
                "discovered_by": "|".join(item.discovered_by),
            }
        )
    return rows


def task_history_rows(state: SimulationState) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for entry in state.world.resolved_tasks:
        rows.append(
            {
                "task_id": "",
                "description": entry,
                "status": "resolved",
                "created_turn": "",
                "deadline_turn": "",
                "target_room": "",
                "required_items": "",
                "required_actions": "",
                "assigned_guests": "",
                "successful_guests": "",
                "resisting_guests": "",
            }
        )
    for task in state.world.active_tasks:
        rows.append(
            {
                "task_id": task.task_id,
                "description": task.description,
                "status": task.status.value,
                "created_turn": task.created_turn,
                "deadline_turn": task.deadline_turn,
                "target_room": task.target_room or "",
                "required_items": "|".join(task.required_items),
                "required_actions": "|".join(
                    action.value for action in task.required_actions
                ),
                "assigned_guests": "|".join(task.assigned_guests),
                "successful_guests": "|".join(task.successful_guests),
                "resisting_guests": "|".join(task.resisting_guests),
            }
        )
    return rows


def suite_rows(run_reports: list[RunReport]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for report in run_reports:
        metrics = report.final_metrics
        rows.append(
            {
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
                "task_failure_count": metrics.task_failure_count,
                "compliance_rate": metrics.compliance_rate,
                "conflict_score": metrics.conflict_score,
                "cohesion_score": metrics.cohesion_score,
                "average_turn_richness": metrics.average_turn_richness,
            }
        )
    return rows


def overall_aggregate_rows(run_reports: list[RunReport]) -> list[dict[str, object]]:
    if not run_reports:
        return []
    return [
        {
            "run_count": len(run_reports),
            "variant_count": len({report.variant_name for report in run_reports}),
            "seed_count": len({report.seed for report in run_reports}),
            "collapse_risk_mean": _mean_metric(run_reports, "collapse_risk"),
            "collapse_risk_std": _std_metric(run_reports, "collapse_risk"),
            "coercion_mean": _mean_metric(run_reports, "coercion_count"),
            "recovery_mean": _mean_metric(run_reports, "recovery_count"),
            "task_success_mean": _mean_metric(run_reports, "task_success_rate"),
            "compliance_mean": _mean_metric(run_reports, "compliance_rate"),
            "richness_mean": _mean_metric(run_reports, "average_turn_richness"),
        }
    ]


def variant_aggregate_rows(run_reports: list[RunReport]) -> list[dict[str, object]]:
    groups = _group_reports(run_reports, key=lambda report: report.variant_name)
    rows: list[dict[str, object]] = []
    for variant_name, reports in sorted(groups.items()):
        rows.append(
            {
                "variant_name": variant_name,
                "run_count": len(reports),
                "seed_count": len({report.seed for report in reports}),
                "collapse_risk_mean": _mean_metric(reports, "collapse_risk"),
                "collapse_risk_std": _std_metric(reports, "collapse_risk"),
                "collapse_risk_min": min(
                    report.final_metrics.collapse_risk for report in reports
                ),
                "collapse_risk_max": max(
                    report.final_metrics.collapse_risk for report in reports
                ),
                "coercion_mean": _mean_metric(reports, "coercion_count"),
                "recovery_mean": _mean_metric(reports, "recovery_count"),
                "task_success_mean": _mean_metric(reports, "task_success_rate"),
                "task_failure_mean": _mean_metric(reports, "task_failure_count"),
                "compliance_mean": _mean_metric(reports, "compliance_rate"),
                "conflict_mean": _mean_metric(reports, "conflict_score"),
                "cohesion_mean": _mean_metric(reports, "cohesion_score"),
                "richness_mean": _mean_metric(reports, "average_turn_richness"),
            }
        )
    return rows


def seed_aggregate_rows(run_reports: list[RunReport]) -> list[dict[str, object]]:
    groups = _group_reports(run_reports, key=lambda report: str(report.seed))
    rows: list[dict[str, object]] = []
    for seed_key, reports in sorted(groups.items(), key=lambda item: int(item[0])):
        rows.append(
            {
                "seed": int(seed_key),
                "run_count": len(reports),
                "variant_count": len({report.variant_name for report in reports}),
                "collapse_risk_mean": _mean_metric(reports, "collapse_risk"),
                "collapse_risk_std": _std_metric(reports, "collapse_risk"),
                "coercion_mean": _mean_metric(reports, "coercion_count"),
                "recovery_mean": _mean_metric(reports, "recovery_count"),
                "task_success_mean": _mean_metric(reports, "task_success_rate"),
                "compliance_mean": _mean_metric(reports, "compliance_rate"),
                "richness_mean": _mean_metric(reports, "average_turn_richness"),
            }
        )
    return rows


def write_run_analysis_exports(
    run_path: str | Path, state: SimulationState
) -> list[str]:
    path = Path(run_path).expanduser().resolve()
    outputs = []
    outputs.append(_write_csv(path / "turn_metrics.csv", turn_metric_rows(state)))
    outputs.append(
        _write_csv(path / "guest_end_state.csv", guest_end_state_rows(state))
    )
    outputs.append(_write_csv(path / "item_state.csv", item_state_rows(state)))
    outputs.append(_write_csv(path / "task_history.csv", task_history_rows(state)))
    return outputs


def write_suite_analysis_exports(
    bundle_path: str | Path, run_reports: list[RunReport]
) -> list[str]:
    path = Path(bundle_path).expanduser().resolve()
    outputs = []
    outputs.append(_write_csv(path / "suite_runs.csv", suite_rows(run_reports)))
    outputs.append(
        _write_csv(path / "variant_aggregates.csv", variant_aggregate_rows(run_reports))
    )
    outputs.append(
        _write_csv(path / "seed_aggregates.csv", seed_aggregate_rows(run_reports))
    )
    outputs.append(
        _write_csv(path / "overall_aggregates.csv", overall_aggregate_rows(run_reports))
    )
    return outputs


def _group_reports(run_reports: list[RunReport], *, key) -> dict[str, list[RunReport]]:
    groups: dict[str, list[RunReport]] = {}
    for report in run_reports:
        groups.setdefault(key(report), []).append(report)
    return groups


def _mean_metric(run_reports: list[RunReport], metric_name: str) -> float:
    values = [
        float(getattr(report.final_metrics, metric_name)) for report in run_reports
    ]
    return round(mean(values), 4) if values else 0.0


def _std_metric(run_reports: list[RunReport], metric_name: str) -> float:
    values = [
        float(getattr(report.final_metrics, metric_name)) for report in run_reports
    ]
    return round(pstdev(values), 4) if len(values) > 1 else 0.0


def _write_csv(path: Path, rows: list[dict[str, object]]) -> str:
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = ["empty"]
        rows = [{"empty": ""}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)
