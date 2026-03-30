from __future__ import annotations

from pathlib import Path

import yaml

from sim.orchestrator import run_episode
from sim.replay import replay_run


def _read_repo_cfg(name: str):
    base = Path(__file__).resolve().parents[1] / "configs" / name
    with open(base, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_replay_matches_hashes(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "configs"
    runs_dir = tmp_path / "runs"
    cfg_dir.mkdir(parents=True)
    runs_dir.mkdir(parents=True)

    baseline = _read_repo_cfg("baseline.yaml")
    baseline["episode_steps"] = 5
    baseline["checkpoint_every"] = 0
    baseline["summarize_every"] = 0
    baseline.setdefault("inference", {})
    baseline["inference"]["mode"] = "scripted"

    for name, obj in (
        ("baseline.yaml", baseline),
        ("personas.yaml", _read_repo_cfg("personas.yaml")),
        ("scenes.yaml", _read_repo_cfg("scenes.yaml")),
        ("rules.yaml", _read_repo_cfg("rules.yaml")),
    ):
        with open(cfg_dir / name, "w", encoding="utf-8") as f:
            yaml.safe_dump(obj, f, sort_keys=True)

    run_id, run_dir = run_episode(
        configs_dir=str(cfg_dir), runs_dir=str(runs_dir), run_id="test_run"
    )
    assert run_id == "test_run"

    ok, errors = replay_run(run_dir)
    assert ok, "\n".join(errors[:5])
