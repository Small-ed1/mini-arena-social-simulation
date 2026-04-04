# Mistral Iteration Results

This note summarizes the recent Mistral-focused iteration loop and the long 200-turn background run.

## Validation Loop

Ten iterations were run with real episodes plus the unit test suite.

Episode lengths used:

- 6 turns
- 10 turns
- 15 turns
- 15 turns after targeted fixes
- 15 turns after schema repairs
- 18 turns
- 12 turns
- 20 turns
- 25 turns
- 30 turns

The loop focused on:

- hidden conceptual pressure versus visible concrete world state
- guest action naturalness
- schema repair and alias normalization
- reduced clue-like host behavior
- improved guest-to-guest ordering

By the end of the loop:

- `pytest -q` passed with `37` tests
- host schema failures in short and medium runs were reduced to zero
- guest schema failures in short and medium runs were reduced to zero
- clue-like visible enrichments were reduced to zero in the analyzed runs
- guest ordering became situational and deterministic rather than fixed alphabetical order

## Dynamic Guest Order

The guest loop is now:

- sequential
- deterministic from seed
- re-ordered per tick by local context

The queue now prefers guests who:

- are involved in the currently open thread
- are located where the active thread is happening
- are currently spotlighted or asked to reflect
- just spawned
- are holding relevant material

It also bumps direct interaction targets forward so guest-to-guest exchanges feel less artificial.

## 200-Turn Mistral Run

Run id:

- `runs/bg_run200_mistral_len200`

Final high-level numbers:

- completed ticks: `200`
- average coherence: `1.0`
- average entertainment: `0.151`
- average novelty: `0.615`
- zero-entertainment ticks: `178`

Final host action mix:

- `allocate_spotlight`: `146`
- `inject_prop`: `22`
- `shape_conceptual`: `15`
- `request_reflection`: `11`
- `spawn_event`: `3`
- `enrich_world`: `3`

Final guest action mix:

- `interact`: `1001`
- `collaborate`: `94`
- `move`: `58`
- `wait`: `20`

Observed long-run failure mode:

- The run stayed structurally stable.
- Coherence did not collapse.
- The system still flattened behaviorally over long horizons.
- Host spotlighting and repeated progression management dominated.
- Guests drifted back toward high-volume interaction with declining novelty.

So the current state is:

- strong structural robustness
- improved guest-to-guest ordering and short-run naturalness
- remaining long-run weakness is behavioral stagnation rather than formatting or schema collapse

## Main Changes Included In This Batch

- hidden conceptual host tool: `shape_conceptual`
- softer guest felt-state handling
- concrete world/hidden conceptual separation
- stronger schema repair paths
- alias normalization for common model mistakes
- dynamic guest turn ordering with reaction bumps and fairness memory
- reusable run helpers:
  - `scripts/run_variant.py`
  - `scripts/analyze_run.py`
  - `scripts/monitor_run.py`
