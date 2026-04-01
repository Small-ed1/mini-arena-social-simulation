# Seed 1337 Model Analysis

This document compares the same-seed runs currently in the repo and explains what each model's output pattern means for this project.

## Runs Analyzed

- `runs/run25_host70b_guestslatest`
  - Host: `llama3.1:70b`
  - Guests: `llama3.1:latest`
- `runs/run25_mistral_nemo_12b_seed1337`
  - Host: `mistral-nemo:12b`
  - Guests: `mistral-nemo:12b`
- `runs/run25_phi4_14b_seed1337`
  - Host: `phi4:14b`
  - Guests: `phi4:14b`
- `runs/bench_seed1337_llama31_latest`
  - Host: `llama3.1:latest`
  - Guests: `llama3.1:latest`
  - Only 10 turns, but useful as the cleanest all-`latest` baseline.

## Important Limits

- The seed is held constant at `1337`, so scene initialization and guest ordering are comparable.
- The runs are not fully apples-to-apples.
- `run25_host70b_guestslatest` uses mixed models, so its guest behavior is mostly a measurement of `llama3.1:latest`, not `70b`.
- The `70b` host run used an earlier host prompt that still contained explicit safety language, while the later Mistral and Phi runs used the simplified prompt.
- That means the same seed controls the world setup, but not every other variable.

## Summary Table

| Run | Host action mix | Guest action mix | Output discipline | Avg host latency | Avg guest latency | Main take |
|---|---|---|---|---:|---:|---|
| `llama3.1:70b` host + `latest` guests | `allocate_spotlight` 16, `request_reflection` 6, `signal_style` 3 | `interact` 78, `move` 57, `speak` 12, `reflect` 2, `wait` 1 | Very clean | 114701 ms | 3163 ms | Strongest director, impractical as guest model |
| `mistral-nemo:12b` all roles | `allocate_spotlight` 8, `spawn_event` 7, `request_reflection` 6, `signal_style` 3, `inject_prop` 1 | `interact` 94, `move` 48, `speak` 7, `reflect` 1 | Clean | 3603 ms | 3432 ms | Best all-around tradeoff |
| `phi4:14b` all roles | `spawn_event` 23, `allocate_spotlight` 2 | `interact` 119, `move` 13, `collaborate` 12, `wait` 6 | Weakest; frequent fenced/explanatory output | 11246 ms | 19263 ms | Creative but noisy and structurally expensive |
| `llama3.1:latest` all roles, 10 turns | `spawn_event` 9, `request_reflection` 1 | `interact` 27, `move` 19, `speak` 11, `reflect` 2, `wait` 1 | Clean | 3440 ms | 2197 ms | Best practical guest baseline |

## Model-by-Model Analysis

### `llama3.1:70b`

Observed behavior in `run25_host70b_guestslatest`:

- The host almost never creates new world content.
- Instead it acts like a stage manager.
- It mostly reallocates attention and asks for reflection.
- Host outputs were consistently terse, schema-clean, and free of markdown wrappers.

What that means:

- `70b` is the strongest host when the job is orchestration rather than invention.
- It is good at deciding who should matter next.
- It is less interested in introducing new objects, props, or twists.
- This makes it useful as a host-only model.

Why it matters:

- A host in this project is not just another agent; it controls pacing.
- `70b` behaves like a director who keeps the camera pointed at the right person.
- That can improve coherence and spotlight balance.
- The cost is speed. At roughly 115 seconds average host latency, the model is operationally expensive.

Guest implications:

- The guests in this run are `llama3.1:latest`, not `70b`.
- They remained structurally strong: only one fallback and one invalid world reference across 150 guest actions.
- That reinforces the current view that `latest` is a strong guest baseline.

### `mistral-nemo:12b`

Observed behavior in `run25_mistral_nemo_12b_seed1337`:

- The host used the broadest action palette of any model tested.
- It both managed pacing and changed the world.
- It spawned mystery/conflict threads, injected a `foam_key`, requested reflections, and used spotlight shifts.
- Guest outputs stayed structurally clean across all 150 guest actions.

What that means:

- `mistral-nemo:12b` is the best general-purpose model in the current setup.
- It is not as tightly directorial as `70b`, but it is much more capable of generating actual scenario motion.
- It makes the simulation feel more like a GM driving a live scene instead of a referee moving attention around.

Why it matters:

- This project benefits from both coherence and world motion.
- `mistral-nemo` is the best balance of those two needs so far.
- It is fast enough to use for host or guest roles.
- It does not create the same structural cleanup burden seen with `phi4`.

Guest implications:

- Mistral guests were fully stable in this sample: no fallbacks, no invalid world refs, no parse failures.
- Their actions leaned heavily toward `interact` and `move`, which fits the arena well.
- This makes Mistral a viable guest candidate if more expressive guest behavior is desired.

### `phi4:14b`

Observed behavior in `run25_phi4_14b_seed1337`:

- The host used `spawn_event` 23 times out of 25 turns.
- It repeatedly created new puzzles, mysteries, and performance/conflict threads.
- Raw host outputs often arrived wrapped in fenced code blocks.
- Some outputs also included long explanations after the JSON.
- Guest outputs had the same pattern at scale: fenced output in 147 out of 150 guest events, and explanatory text in 112 events.
- The guest side incurred 6 fallbacks, 3 invalid world references, and 3 parse failures.

What that means:

- `phi4` is the most assistant-like model of the group in this task.
- It treats the prompt as a request for a helpful formatted answer, not just a structured action.
- It can generate vivid scenario text, but it resists the low-level discipline this simulator needs.

Why it matters:

- Guest turns dominate runtime and log volume.
- A guest model that wraps JSON in markdown and adds explanations creates avoidable cleanup pressure.
- That pressure is manageable in a small sample, but it scales poorly over long runs.
- The metrics reflect that: several coherence dips to `0.857`, repeated `unsafe_blocks`/cleanup blips, and long stretches of zero entertainment late in the run.

Guest implications:

- `phi4` is a poor guest fit for the current strict-action setup.
- It is too verbose and too willing to explain itself.
- The model may still be useful for offline idea generation or richer narrative drafting, but not as the default guest actor.

### `llama3.1:latest`

Observed behavior in `bench_seed1337_llama31_latest`:

- The host heavily favored `spawn_event`.
- The guests were structurally reliable.
- There was only one fallback and one invalid world reference in the 10-turn benchmark.
- Raw outputs were mostly direct JSON with minimal wrapper text.

What that means:

- `latest` remains the best practical guest baseline.
- It is not the most imaginative host, but it is fast, mostly disciplined, and cheap enough to scale.
- This matters because guest turns happen six times per tick.

Why it matters:

- For guests, the main requirement is not brilliance; it is consistent, low-overhead action generation.
- `latest` does that better than `phi4` and much faster than `70b`.
- It is the strongest choice when the system needs many actions per turn with minimal cleanup.

## What the Same-Seed Comparison Actually Shows

The shared seed lets us say the following with confidence:

- The models are responding to the same initial arena, same guest ordering, and same starting distribution.
- Differences in output style are mostly model-driven, not scene-driven.

The seed does not let us say the following without qualification:

- That every difference is solely due to the model.
- The prompt revision changed during the experiment series.
- The `70b` comparison is host-only because its guests are `latest`.

## Practical Interpretation

### Best host model

- If runtime is irrelevant: `llama3.1:70b`
- If runtime matters: `mistral-nemo:12b`

### Best guest model

- Best practical choice: `llama3.1:latest`
- Best alternative worth testing more: `mistral-nemo:12b`
- Model to avoid for guests in this setup: `phi4:14b`

### Best mixed setup right now

- `llama3.1:70b` host + `llama3.1:latest` guests for best host steering
- `mistral-nemo:12b` host + `llama3.1:latest` guests for best cost/performance tradeoff

## Recommended Next Experiment

To finish the guest comparison properly, run a guest-only benchmark with:

- fixed host model
- fixed seed `1337`
- fixed guest count
- fixed prompt version
- guest model swapped among:
  - `llama3.1:latest`
  - `mistral-nemo:12b`
  - `phi4:14b`

And compare only:

- guest fallback rate
- invalid world-reference rate
- parse failure rate
- action diversity
- collaboration rate
- guest latency

That would turn the current directional conclusions into a clean guest-model decision.
