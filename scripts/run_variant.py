from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import yaml

from sim.orchestrator import run_episode


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a temporary config variant")
    ap.add_argument(
        "--base-config", required=True, help="Base config yaml file in configs/"
    )
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--episode-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--host-model", default=None)
    ap.add_argument("--guest-model", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--timeout-s", type=int, default=None)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    configs_src = repo_root / "configs"
    runs_dir = repo_root / "runs"

    with tempfile.TemporaryDirectory(prefix="mini_arena_cfg_") as tmp:
        cfg_dir = Path(tmp)
        for name in ("personas.yaml", "scenes.yaml", "rules.yaml"):
            shutil.copyfile(configs_src / name, cfg_dir / name)

        base = yaml.safe_load(
            (configs_src / args.base_config).read_text(encoding="utf-8")
        )
        if args.episode_steps is not None:
            base["episode_steps"] = int(args.episode_steps)
        if args.seed is not None:
            base["seed"] = int(args.seed)
        base.setdefault("inference", {})
        if args.timeout_s is not None:
            base["inference"]["timeout_s"] = int(args.timeout_s)
        if args.model is not None:
            base["inference"]["model"] = str(args.model)
        if args.host_model is not None:
            base["inference"]["host_model"] = str(args.host_model)
        if args.guest_model is not None:
            base["inference"]["guest_model"] = str(args.guest_model)

        with open(cfg_dir / "baseline.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(base, f, sort_keys=True)

        run_id, run_dir = run_episode(
            configs_dir=os.path.abspath(str(cfg_dir)),
            runs_dir=os.path.abspath(str(runs_dir)),
            run_id=args.run_id,
        )
        print(run_id)
        print(run_dir)


if __name__ == "__main__":
    main()
