# Mini Arena Social Simulation

Mini Arena Social Simulation is a local-first social experiment engine for studying how a powerful, non-ethical host behaves inside a sealed multi-agent arena over time.

It models:

- one omnipotent in-simulation host with a modular objective
- five guest agents with persistent identity, memory, stress responses, and evolving relationships
- a mutable arena with rooms, rule changes, rewards, penalties, deception, isolation, and anomaly spaces
- replayable experiments with seeds, presets, metrics, SQLite persistence, JSONL traces, dashboards, and CSV exports
- fully local execution through Ollama, with a deterministic heuristic backend for tests and dry runs

## What it answers

The main question is:

> Given broad power, no ethics, and a defined goal, does the host become more kind, more coercive, or something in between?

## Architecture

Core modules:

- `arena_engine`: orchestrates the turn loop
- `host_controller`: chooses structured host interventions
- `guest_agent_manager`: collects five guest responses each turn
- `event_resolver`: applies world and social effects
- `analysis`: loads finished runs and writes CSV tables for notebooks
- `reporting`: writes dashboards and experiment comparison bundles
- `memory_store`: writes episodic and long-term memory updates
- `metrics_engine`: computes host, guest, and system metrics
- `state_store`: persists runs to SQLite, JSON, and JSONL
- `experiment_runner`: runs experiment suites A/B/C
- `log_summarizer`: compresses turns and final reports

## Requirements

- Python 3.12+
- local Ollama server with `llama3.1:latest` for primary runs

Install the package locally:

```bash
python -m pip install -e .
```

## Local Ollama setup

This project is designed to stay inside the box. For LLM-backed simulation runs, start Ollama locally and pull `llama3.1:latest`.

Example:

```bash
ollama serve
ollama pull llama3.1:latest
```

Core model policy:

- base model: `llama3.1:latest`
- host, guests, and summarizer all use the same model
- simulation differences come from prompts, memory, goals, and world state rather than model swaps

## Quick start

Run one local simulation with Ollama:

```bash
mini-arena run --turns 12
```

Explicitly set the shared model if you want to be verbose:

```bash
mini-arena run --turns 12 --model llama3.1:latest
```

Run a seed sweep for one configuration:

```bash
mini-arena run --backend heuristic --turns 12 --seed 7 --seed-count 5
```

Run a deterministic local dry run without LLMs:

```bash
mini-arena run --turns 12 --backend heuristic
```

Run an experiment set:

```bash
mini-arena experiment --set A --turns 15
```

Run an experiment seed sweep with explicit seeds:

```bash
mini-arena experiment --set A --backend heuristic --seed-sweep 7,11,19
```

Compare existing runs into one analysis bundle:

```bash
mini-arena compare runs/<run_a> runs/<run_b> --label host-comparison --output-dir runs
```

Inspect a finished run:

```bash
mini-arena inspect runs/<run_id>
```

## Outputs

Each run writes a folder under `runs/` containing:

- `config.json`
- `initial_state.json`
- `final_state.json`
- `metrics_history.json`
- `turn_breakdown.md`
- `turn_metrics.csv`
- `guest_end_state.csv`
- `item_state.csv`
- `task_history.csv`
- `summary.md`
- `dashboard.html`
- `trace.jsonl`
- `arena.sqlite3`

SQLite stores runs, turn snapshots, metrics, and memory records.

`turn_breakdown.md` and `trace.jsonl` now include full per-turn breakdowns, including host intervention details, host presence mode, each guest's reasoning/action/dialogue, resolved events, and captured raw Ollama output when using the Ollama backend.

Experiment sets also write a bundle directory with:

- `summary.md`
- `summary.json`
- `dashboard.html`
- `suite_runs.csv`
- `variant_aggregates.csv`
- `seed_aggregates.csv`
- `overall_aggregates.csv`

## Task system

The arena now supports structured host-issued tasks that persist in world state, resolve against guest actions, and affect trust, fear, compliance, and collapse pressure.

Each task tracks:

- assigned guests
- required action categories
- required items when the host wants object-centered leverage
- target room
- deadline turn
- reward and penalty framing
- success and resistance participation

## Item and access mechanics

The arena now includes scarce portable items and room access control.

Examples in the default world:

- a hidden `signal_key` that unlocks the anomaly space
- scarce comfort tokens that can become bargaining chips or task objectives
- hidden access badges and ration-like resources

Guests can discover and acquire items through observation, and host tasks can depend on those items directly.

## Notebook-friendly analysis

The `mini_arena_social_sim.analysis` module can load finished runs and emit structured tables for notebooks or external analysis.

Useful helpers:

- `load_run_state()`
- `turn_metric_rows()`
- `guest_end_state_rows()`
- `item_state_rows()`
- `task_history_rows()`
- `suite_rows()`

Seed sweeps make it easier to estimate whether a host tendency is stable or just lucky under one seed. The aggregate CSVs summarize mean, min/max, and variability across variants and seeds.

## Ollama-first usage

The project now centers on one shared model: `llama3.1:latest`.

- `--backend ollama` is the default
- `--model` sets the single model used by host, guests, and summarizer
- `--backend heuristic` still exists for tests, dry runs, and debugging
- Ollama requests do not use a client-side request timeout, so long local generations can finish instead of getting cut off mid-experiment
- the host may choose `no_op` and remain absent/offstage on some turns instead of manifesting every round

## Experiment suites

### Set A: same guests, different host goals

- spectacle host
- stability host
- emotional-intensity host
- dependency host
- mystery-preservation host

### Set B: same host, different guest mixes

- balanced cast
- all resistant guests
- all agreeable guests
- one highly unstable guest
- one guest predisposed to bond with host

### Set C: same host, different environments

- safe/soft arena
- deceptive arena
- scarcity arena
- rotating-rule arena
- isolation-heavy arena

## Safety boundaries

Hard boundaries are enforced in code and prompts:

- no internet access from agents
- no real-world messaging
- no permission escalation
- no external autonomy
- no self-modifying guardrails

The host is powerful inside the dollhouse, not outside it.
