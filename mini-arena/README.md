# mini-arena v1

LLM-centered, Ollama-backed social simulation baseline.

## What v1 does

- Uses Ollama as the default inference backend for the host and all guests.
- Host acts once per tick, then guests act in fixed order (AEC).
- Every host/guest action is strict-schema JSON, safety-checked, and applied by the environment.
- All applied actions (and blocked proposals) are logged as immutable events.
- Metrics are computed per tick; checkpoints are written periodically.
- A run can be replayed deterministically from the event log and verified by state hashes.

## Quickstart

From this directory:

- Start Ollama and pull a model first:
  - `ollama serve`
  - `ollama pull llama3.1`
- Run a baseline episode:
  - `python -m scripts.run_baseline`
- Replay a run (verifies hashes):
  - `python -m scripts.replay_run runs/<run_id>`
- Inspect a run (prints metric summary):
  - `python -m scripts.inspect_run runs/<run_id>`

## Ollama

- Default endpoint: `http://localhost:11434`
- Default model: `llama3.1`
- `configs/baseline.yaml` is set to `inference.mode: ollama`
- You can override runtime settings with environment variables:
  - `OLLAMA_URL`
  - `OLLAMA_MODEL`
  - `OLLAMA_HOST_MODEL`
  - `OLLAMA_GUEST_MODEL`
  - `MINI_ARENA_INFERENCE_MODE`
- Set `MINI_ARENA_INFERENCE_MODE=scripted` if you want the deterministic fallback policy instead of LLM inference.

## Configuration

Configs live in `configs/`:

- `configs/baseline.yaml`
- `configs/personas.yaml`
- `configs/scenes.yaml`
- `configs/rules.yaml`

## Tests

- `pytest -q`
