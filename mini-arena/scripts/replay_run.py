from __future__ import annotations

import argparse
import os
import sys

from sim.replay import replay_run


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay a run and verify state hashes")
    ap.add_argument("run_dir", help="Run directory (e.g., runs/<run_id>)")
    args = ap.parse_args()

    ok, errors = replay_run(os.path.abspath(args.run_dir))
    if ok:
        print("OK")
        return
    print("FAIL")
    for e in errors[:20]:
        print(e)
    if len(errors) > 20:
        print(f"... {len(errors) - 20} more")
    sys.exit(1)


if __name__ == "__main__":
    main()
