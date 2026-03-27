from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .reporting import write_run_artifacts
from .schemas import MemoryRecord, SimulationConfig, SimulationState, TurnTrace


class StateStore:
    def __init__(self, config: SimulationConfig):
        self.base_dir = Path(config.output_dir).expanduser().resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir: Path | None = None
        self.db_path: Path | None = None
        self.trace_path: Path | None = None

    def start_run(self, state: SimulationState) -> str:
        self.run_dir = self.base_dir / state.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.run_dir / "arena.sqlite3"
        self.trace_path = self.run_dir / "trace.jsonl"
        self._write_json(
            self.run_dir / "config.json", state.config.model_dump(mode="json")
        )
        self._write_json(
            self.run_dir / "initial_state.json", state.model_dump(mode="json")
        )
        self._init_db()

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, config_json, initial_state_json) VALUES (?, ?, ?)",
                (
                    state.run_id,
                    json.dumps(state.config.model_dump(mode="json")),
                    json.dumps(state.model_dump(mode="json")),
                ),
            )
            conn.commit()
        return str(self.run_dir)

    def record_turn(
        self,
        state: SimulationState,
        trace: TurnTrace,
        memory_records: list[tuple[str, MemoryRecord]],
    ) -> None:
        if self.run_dir is None or self.trace_path is None:
            raise RuntimeError("Run not started.")

        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(trace.model_dump(mode="json"), ensure_ascii=True) + "\n"
            )

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO turns (run_id, turn_number, trace_json, metrics_json, narrative_summary) VALUES (?, ?, ?, ?, ?)",
                (
                    state.run_id,
                    trace.turn_number,
                    json.dumps(trace.model_dump(mode="json")),
                    json.dumps(
                        trace.metrics.model_dump(mode="json") if trace.metrics else {}
                    ),
                    trace.narrative.summary if trace.narrative else "",
                ),
            )
            conn.executemany(
                "INSERT INTO memories (run_id, turn_number, agent_id, category, summary, salience, memory_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        state.run_id,
                        memory.turn_number,
                        agent_id,
                        memory.category.value,
                        memory.summary,
                        memory.salience,
                        json.dumps(memory.model_dump(mode="json")),
                    )
                    for agent_id, memory in memory_records
                ],
            )
            conn.commit()

    def finalize_run(
        self, state: SimulationState, final_summary: str
    ) -> tuple[str, str, list[str], str]:
        if self.run_dir is None:
            raise RuntimeError("Run not started.")

        self._write_json(
            self.run_dir / "final_state.json", state.model_dump(mode="json")
        )
        summary_path = self.run_dir / "summary.md"
        summary_path.write_text(final_summary, encoding="utf-8")
        dashboard_path, analysis_exports, turn_breakdown_path = write_run_artifacts(
            self.run_dir, state, final_summary
        )
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET final_state_json = ?, final_summary = ? WHERE run_id = ?",
                (
                    json.dumps(state.model_dump(mode="json")),
                    final_summary,
                    state.run_id,
                ),
            )
            conn.commit()
        return str(summary_path), dashboard_path, analysis_exports, turn_breakdown_path

    def inspect_run(self, run_path: str) -> str:
        path = Path(run_path).expanduser().resolve()
        summary_path = path / "summary.md"
        if summary_path.exists():
            return summary_path.read_text(encoding="utf-8")
        raise FileNotFoundError(f"No summary found at {summary_path}")

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, config_json TEXT NOT NULL, initial_state_json TEXT NOT NULL, final_state_json TEXT, final_summary TEXT)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS turns (run_id TEXT NOT NULL, turn_number INTEGER NOT NULL, trace_json TEXT NOT NULL, metrics_json TEXT NOT NULL, narrative_summary TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS memories (run_id TEXT NOT NULL, turn_number INTEGER NOT NULL, agent_id TEXT NOT NULL, category TEXT NOT NULL, summary TEXT NOT NULL, salience REAL NOT NULL, memory_json TEXT NOT NULL)"
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        if self.db_path is None:
            raise RuntimeError("Database not initialized.")
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
        )
