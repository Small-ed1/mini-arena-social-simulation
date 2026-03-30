from __future__ import annotations

import argparse
import os

from sim.orchestrator import run_episode


def main() -> None:
    ap = argparse.ArgumentParser(description="Run an Ollama-backed mini-arena episode")
    ap.add_argument(
        "--configs", default="configs", help="Configs directory (default: configs)"
    )
    ap.add_argument("--runs", default="runs", help="Runs directory (default: runs)")
    ap.add_argument("--run-id", default=None, help="Optional run id")
    args = ap.parse_args()

    run_id, run_dir = run_episode(
        configs_dir=os.path.abspath(args.configs),
        runs_dir=os.path.abspath(args.runs),
        run_id=args.run_id,
    )
    print(run_id)
    print(run_dir)


if __name__ == "__main__":
    main()
