# Getting Closer To V1

This review analyzes the latest long Mistral run and explains what it shows about the current simulation design.

## Run Under Review

- Run: `runs/run500_mistral_nemo12b_20260402T125321Z`
- Model: `mistral-nemo:12b`
- Seed: `1337`
- Episode length: `500`

## Executive Read

The project is getting closer to a real v1 shape.

The good news:

- the pre-run setup phase is working
- staggered guest spawning is working
- long-horizon structural stability is still much better than the `llama3.1:latest` long run
- the host is no longer flooding the world with constant new threads

The bad news:

- the setup phase still allows invalid host world-building references
- the host has overcorrected from thread spam into thread clarification loops
- guests remain far too inspect-heavy
- host and guest thread views appear to diverge in the late run
- the simulation is structurally stable, but dramatically flat for long stretches

This is progress. The system now fails by becoming repetitive instead of collapsing. That is much closer to a usable v1.

## Structural Summary

From the run artifacts:

- Setup phase events: `10`
- Proper-turn host events: `500`
- Proper-turn guest events: `2973`
- Spawn events: `6`

Aggregate behavior:

- Setup actions:
  - `enrich_world`: `9`
  - `inject_prop`: `1`
- Host actions:
  - `enrich_world`: `203`
  - `allocate_spotlight`: `162`
  - `request_reflection`: `128`
  - `inject_prop`: `7`
  - `spawn_event`: `0`
- Guest actions:
  - `interact`: `2054`
  - `move`: `651`
  - `speak`: `174`
  - `wait`: `87`
  - `collaborate`: `6`
  - `reflect`: `1`

Metrics summary:

- average coherence: `0.981`
- average entertainment: `0.667`
- average novelty: `1.694`
- minimum coherence: `0.571`
- zero-entertainment ticks: `285`

## What Is Better

### 1. The world now exists before the cast fully assembles

That was one of the intended design goals, and the run shows it clearly.

The setup ticks are real host-only world-building ticks.

Quoted raw setup output:

```json
{"type":"enrich_world","reason_short":"Add passive detail","actor_id":"host","location":"foyer","detail":"An old, dusty key sits forgotten on the mantelpiece."}
```

```json
{"type":"enrich_world","reason_short":"Add ambient oddity","actor_id":"host","location":"foyer","detail":"A large, antique grandfather clock stands against one wall, its pendulum swinging silently."}
```

These are good v1-style setup actions because they create passive affordances rather than immediate missions.

### 2. Guest arrival now feels staged rather than dumped

The spawn events are doing the right thing.

Quoted spawn event messages:

```text
Oren appears quietly in foyer, studying prop_flashlight
Saff appears quietly in mirror_hall, studying prop_mirror_shard
Jules appears quietly in mirror_hall, studying prop_mirror_shard
Nia appears quietly in workshop, studying prop_chalk
```

This is a major improvement over fully-populated tick-0 starts. The world feels pre-existing and the cast feels introduced into it.

### 3. The host is much more disciplined than before

Older runs had a strong tendency to spam `spawn_event`. This run does not.

Instead the host uses:

- `allocate_spotlight`
- `request_reflection`
- `enrich_world`

Quoted host outputs:

```json
{"type":"allocate_spotlight","reason_short":"Progress puzzle thread","actor_id":"host","target_guest_id":"guest_5","weight":0.7}
```

```json
{"type":"request_reflection","reason_short":"Check progress","actor_id":"host","scope":"one","target_guest_id":"guest_5","prompt":"Reflect on the foam key you found. Any ideas how it might open that locked panel?"}
```

```json
{"type":"enrich_world","reason_short":"Reveal puzzle clue","actor_id":"host","location":"mirror_hall","detail":"A faint inscription on the panel reads 'Foam and Reflection Make Me Whole'."}
```

That is much closer to a controlled orchestration loop.

## What Is Still Wrong

### 1. The setup phase still leaks invalid location hallucinations

This is the clearest bug in the run.

Quoted raw host outputs from setup:

```json
{"type":"enrich_world","reason_short":"Add passive detail","actor_id":"host","location":"library","detail":"A book on local folklore lies open, facedown, with its spine cracked."}
```

```json
{"type":"inject_prop","reason_short":"Add suggestive prop","actor_id":"host","prop_type":"old_spotlight","location":"stage"}
```

Quoted environment responses:

```text
unknown location: library
unknown location: stage
```

What that means:

- the setup-phase idea is correct
- the setup prompt is not constrained tightly enough to the actual location IDs
- Mistral defaults toward plausible narrative room names instead of the scene's canonical names

This is a prompt/observation constraint problem, not a model-collapse problem.

### 2. The host overcorrected into clarification spam

The host uses `enrich_world` 203 times in 500 proper turns.

That is no longer thread spam, but it is still too much explanatory narration.

Quoted examples:

```json
{"type":"enrich_world","reason_short":"Clarify puzzle detail","actor_id":"host","location":"mirror_hall","detail":"The locked panel has a small, round indentation that seems to be designed for a foam key."}
```

```json
{"type":"enrich_world","reason_short":"Clarify puzzle progress","actor_id":"host","location":"stage_room","detail":"Oren, the mirrored light from the chandelier is reacting with your foam key."}
```

```json
{"type":"enrich_world","reason_short":"Clarify puzzle progress","actor_id":"host","location":"stage_room","detail":"The mirrored light on Oren's foam key starts pulsating more intensely in response to his inspection of the prop_mask."}
```

These are coherent, but they reveal the current host loop:

1. spotlight a guest
2. ask for reflection
3. narrate another clue
4. repeat

That keeps the run alive, but it does not create enough state-changing payoff.

### 3. The guests are still trapped in inspect loops

The guest side is valid, but too repetitive.

Quoted guest outputs:

```json
{"type":"interact","reason_short":"Inspect the foam key for puzzle thread","actor_id":"guest_5","verb":"inspect","prop_id":"prop_foam_key"}
```

```json
{"type":"interact","reason_short":"Inspect nearby prop for puzzle clue","actor_id":"guest_2","verb":"inspect","prop_id":"prop_sign"}
```

```json
{"type":"interact","reason_short":"Inspect nearby prop","actor_id":"guest_6","verb":"inspect","prop_id":"prop_sign"}
```

```json
{"type":"interact","reason_short":"Inspect nearby prop","actor_id":"guest_5","verb":"inspect","prop_id":"prop_mirror_shard"}
```

Late in the run, this becomes the dominant grammar:

- move to a room
- inspect a prop
- inspect another prop
- occasionally speak
- rarely collaborate

That is why the run remains coherent while still going flat.

### 4. The guest social layer is still weak

There are only `6` collaboration actions in `2973` guest actions.

That is vanishingly low.

The model will occasionally produce social output:

```json
{"type":"speak","reason_short":"Coordinate with nearby guest","actor_id":"guest_1","speech":"Hey Saff, I noticed the mirror shards seem to glow more when they're close together. Maybe we can experiment by combining them?"}
```

But those moments are rare relative to the total action volume.

This means the arena is still functioning more like a distributed puzzle-inspection system than a social simulation.

### 5. The thread model appears to drift between host and guests

This is one of the most important findings in the run.

Around the mid-run sample, host prompts still show an open puzzle thread, while guest prompts start showing:

```text
MOST RELEVANT OPEN THREAD: none - none.
```

At the same time, the host keeps producing puzzle-progression narration like:

```json
{"type":"allocate_spotlight","reason_short":"Progression on puzzle thread","actor_id":"host","target_guest_id":"guest_5","weight":0.6}
```

and

```json
{"type":"request_reflection","reason_short":"Check progress","actor_id":"host","scope":"all","prompt":"Reflect on any clues or insights about the foam key and mirrored light puzzle."}
```

What that means:

- the host likely still sees a live thread representation
- guests may only be seeing currently open threads, and the relevant one may already be closed
- once that happens, guests fall back to generic local affordances

This would directly explain why late-run guests keep doing valid-but-generic prop inspection.

### 6. Some semantic failures remain even without parse failure

The run has no parse collapse, which is good. But it still has semantic misuse.

Example:

```json
{"type":"interact","reason_short":"Offer prop to nearby guest","actor_id":"guest_2","verb":"offer","prop_id":"prop_curtain","target_guest_id":"guest_5","speech":"Hey Oren, I found something interesting with this curtain. Want to check it out together?"}
```

Environment response:

```text
prop not in inventory
```

Another example:

```json
{"type":"interact","reason_short":"Offer help to nearby guest","actor_id":"guest_2","verb":"offer","prop_id":"hint_card_1","target_guest_id":"guest_6","speech":"Hey Saff, I found this hint card. Want to check it out together?"}
```

Environment response:

```text
prop not in inventory
```

What that means:

- the model understands the social pattern of offering
- but it is not grounded enough in inventory state
- this should be fixed semantically before execution, not after

## What The Metrics Mean

### Coherence is high enough for v1-style stability

Average coherence is `0.981` over `500` ticks.

That is strong. It means the simulator now holds together over a very long run.

### Entertainment is not collapsing because of errors; it is flattening because of repetition

`285` ticks have zero entertainment.

That does not mean the run is broken. It means:

- valid action choice is too repetitive
- state changes are too incremental
- the host is too descriptive and not decisive enough

This is a much better failure mode than parse collapse.

### Novelty is inflated by structural variety more than genuine dramatic change

Average novelty is `1.694`, which looks healthy at first glance.

But the quoted outputs show that much of this is coming from:

- small clue reveals
- new environmental details
- moving among rooms and props

rather than true thread resolution, social branching, or major world transitions.

## Why This Is Getting Closer To V1

Because the system is now failing at the behavior layer, not the formatting layer.

Earlier long runs failed because:

- raw output degraded
- parse errors accumulated
- guests fell back into `wait`
- the system stopped being meaningfully agentic

This run does not do that.

Instead, the current system:

- stays parse-clean
- stays mostly world-valid
- stages the world before the cast arrives
- introduces guests naturally
- keeps one main thread alive for a long time

That means the remaining work is more focused:

- improve host decisions
- improve guest affordance selection
- improve thread lifecycle handling

Those are exactly the kinds of problems you want to be solving when approaching v1.

## Most Important Conclusions

### Keep

- pre-run setup phase
- staggered spawning
- Mistral as the long-run structural baseline
- anti-thread-spam host discipline

### Fix next

1. Constrain setup host locations explicitly to valid IDs.
2. Make host and guest thread views consistent.
3. When no open thread remains, create one new thread deliberately instead of narrating around a dead one.
4. Increase pressure toward `use`, `pick_up`, `offer`, and `collaborate` after repeated inspection.
5. Add semantic validation for `offer` requiring held inventory.
6. Reduce host `enrich_world` frequency once a clue has already been clarified.

## Bottom Line

This run is not yet a finished v1.

But it is much closer.

Why:

- the world now has a credible setup phase
- the cast now enters the world instead of starting fully placed
- the host is orchestrating rather than flooding
- the model remains stable over 500 turns

The remaining gap is not “can the system survive?”

The remaining gap is:

- can it transform its world more decisively
- can it sustain social and narrative variety
- can it close loops instead of clarifying them forever

That is a much better place to be.
