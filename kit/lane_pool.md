# Concurrent scoring on one allocated GPU

`lane_pool.py` is an opt-in runner for ordered commands belonging
to **one** assignment lane. Request one node, one GPU and a per-GPU lease from the
existing scheduling wrapper. All checkpoints and panels must be staged and bound
by CPU jobs first. List every required input in `requires`; the first command exits
PENDING (75) before GPU admission if its inputs are absent. Later inputs are checked
per command; outputs from a running predecessor can satisfy them. Once no command
is running, an absent input returns PENDING immediately. A child may also exit 75 or
the existing kit NotReady code 3;
the runner stops admitting commands, lets active peers finish and returns PENDING.
Other child failures mark the lane FAILED and return 1. There are no retries.
Existing commands returning zero after printing a line starting `PENDING:` or
`NOT READY:` (or a standalone `PENDING`) also mark the lane PENDING. Complete
`requires` lists remain mandatory so unavailable work exits before model load.

| Family | nodes | gpus | cpus | May share node |
|---|---:|---:|---:|---|
| One lane, K=1 | 1 | 1 | 4 | yes, distinct allocated GPUs |
| One lane, K=2 | 1 | 1 | 4 | yes, distinct allocated GPUs |
| One lane, K=3 | 1 | 1 | 4 | yes, distinct allocated GPUs |
| Preparation, binding, publication, reduction | 1 | 0 | 4 | yes |

Example resource line: `sbatch --nodes=1 --gpus-per-node=1 --cpus-per-task=4 ...`.
Submit through the GPU wrapper's `python -I kit/util_watch.py --out <job>/util.jsonl -- ...`
invocation when integrated. The runner itself does not duplicate the watcher or
take a second lease that would conflict with the wrapper's lease.

```json
[
 {"argv":["python","-I","kit/scorer.py","--out","scores/checkpoint-a"],
  "requires":["exports/checkpoint-a/config.json","bindings/ready.json"]},
 {"argv":["python","-I","kit/scorer.py","--out","scores/checkpoint-b"],
  "requires":["exports/checkpoint-b/config.json","bindings/ready.json"]}
]
```

Replace the illustrative scorer command with the existing complete scoring
command for the lane, including its frozen GPU assignment. Commands are argument
arrays, never shell fragments. Relative paths use the submitting working directory.
Give every command its own fresh output tree. Run:

```sh
python -I kit/lane_pool.py --plan lane-plan.json --concurrency 3 --out lane-job
```

The runner requires exactly one visible GPU, queries only the allocated device
when `CUDA_VISIBLE_DEVICES` is set, records its measured UUID, and checks it again
at every launch and completion. A historical desired UUID never refuses admission.
`lane.json` records the plan hash, K, UUID, ordered commands, exit codes and wall
times. A fresh output directory is required for each allocation. Completed command receipts
survive a PENDING result. `--resume` reuses them only when the plan hash, measured
UUID, inherited `KIT_LANE_JOB` nonce and `SLURM_JOB_ID` all match. A different
allocation must re-score the complete contrast in a fresh tree.
The U3 concurrency decision permits overlapping independent checkpoint commands;
the same-GPU comparison rule remains mandatory for every checkpoint in the lane.
List dependencies requiring scientific order on the dependent command. Historical assignment maps remain unchanged. New complete jobs bind the actual
scheduler-selected UUID and re-score every input of a contrast. See
[v6 allocation lanes](util70-integration.md).

Direct CLI starts are staggered by 20 seconds by default (configurable with
`--stagger-seconds`, 0–60); production `invocation()` uses zero. This is a fixed delay between starts of ready commands,
never a wait for another job's output. An optional `--timeout-seconds` terminates
active command process groups and writes FAILED when the controller deadline expires.
Normal command failures let active peers finish. The enclosing scheduler's wall
limit remains necessary. A partially completed plan is never a COMPLETE lane.

## Memory and unchanged scoring

`KIT_GPU_MEMORY_FRACTION` is read by one helper,
`inference_lifecycle.engine_kwargs`. Absent means all original engine kwargs,
including missing kwargs and vLLM defaults, stay unchanged. Overrides must be finite
and strictly between 0 and 1. The production likelihood, generated panel, legal,
SP, broad-pool and teacher engines all pass through the shared constructor;
the older hints and repair-target paths use the same helper. HF NLL/reference scorers (`s_probe`,
`a4_nll`, `hellaswag_reference`) have no vLLM reservation and remain unchanged.
Training rollout memory settings are unchanged. Existing generation engine receipts
and teacher reload receipts record the effective reservation. Under a lane runner,
every shared constructor also writes a separate engine receipt under
`<lane-job>/<command-index>/engines/`, including likelihood engines with no standalone
engine receipt. The scorer's score rows and response files are untouched.

K is selected per lane from every command's optional `engine` description:

```json
{"argv":["python","-I","kit/scorer.py","--model","/staged/model"],
 "engine":{"backend":"vllm","model":"/staged/model","max_model_len":12288}}
```

K=3 is permitted only for Qwen3-8B vLLM at `max_model_len <= 12288`, with TP1.
The model is the exact `Qwen/Qwen3-8B` identifier or a local `config.json` with
its 8B geometry. The shared constructor rechecks actual model/context/TP kwargs
before loading, so a plan cannot silently launch an unqualified engine. K=3 uses
`.31` per process (93% total); K=2 vLLM uses `.42`. K=1 clears the override and
preserves the original engine defaults. The lane's minimum qualified capacity
caps the requested K, including an explicit CLI `--concurrency 3`.

Absent profiles, other models, larger/unknown contexts, HF references and HF FP32
PD scoring select K=1. Existing callers without engine descriptions therefore run
serially. PD concurrency up to two requires the separately reviewed per-cell
commands and qualification; this controller does not infer that approval. No
three-process HF qualification is claimed. Do not reduce model context, sequences,
doses or prompts to make startup pass. The TP2 teacher runs outside this pool.

Concurrency changes timing and reservations; deterministic equality remains an
acceptance condition to measure. It does not establish 70% utilisation in advance.
Startup, profiling collisions, model reloads, CPU grading and the final underfilled
tail can still breach rolling windows. K=3 has the greatest memory risk.

## Prepared second measurement (do not launch during preparation)

CPU-only dry run:

```sh
python -I scripts/util70_smoke_modal.py --suite concurrency --dry-run --out /tmp/util70-concurrency-dry
```

Explicit paid launch, requiring the same prepopulated model cache as suite baseline:

```sh
python -I scripts/util70_smoke_modal.py --suite concurrency --execute --out tmp/util70/smoke/a2
```

P1–P5 remain available as `--suite baseline` (the default). The concurrency suite:

* P6: all 300 frozen general members through unchanged `score_forgetting.cmd_generate` solo.
* P7: two independent processes, each scoring the same full 300 members.
* P8: three independent processes, each scoring the same full 300 members.
* P9: unchanged HF reference on the same frozen 100 HellaSwag items solo, then two concurrent copies.

The generated replicas use a 20-second stagger. HF replicas use two seconds:
the first smoke's roughly 24-second reference run would have little useful
compute overlap with a 20-second stagger, and HF has no vLLM memory profiler.

Phases repeat complete, unchanged scorer invocations with fresh model loads until
they exceed 320 seconds, subject to the shared container deadline. This supplies
a 300-second window where feasible and measures the reload-heavy checkpoint lane,
including loads, stagger, teardown and short CPU comparisons. No GPU padding or
numerical tuning is introduced. Each invocation's generated response bytes, token
record bytes and extracted score-payload bytes must match P6; HF score files must
match the P9 solo file. `scores.json` contains exactly the original `panels`,
`total` and `decoding` fields. Full `forgetting.json` receipts also contain timestamps
and engine reservations that legitimately differ and are retained unmodified.
Missing or different compared bytes produce prominent EQUALITY FAIL output and an
incomplete suite. `equality.jsonl`, `lanes.jsonl`, per-phase receipts and the summary
are collected on the host; full worker artifacts remain in the diagnostic container.

One H100 container, four fixed CPUs, 64 GiB fixed host memory, no retries and a
hard 2400-second lifetime. The script's existing configured rate formula gives a
**$5.80 bound**, including 25% contingency and $0.75 build reserve; no paid test
has been launched by this change. A null 300-second window is INSUFFICIENT,
never success. Equality failure and telemetry failure cannot produce an OK suite.

## Infrastructure identities for integration

Do not rewrite historical pins. The integration owner must generate D10's
`resources/v6/infrastructure-update.json` and bind it through the existing
runtime-update/binding authorities before running changed pinned code.
This unit intentionally does not supply final authority hashes.

Changed bytes: `inference_lifecycle.py`, `score_forgetting.py`, `eval_bed.py`,
`v4_teacher.py`, `c_targets.py`, `hints.py`, `sp_score.py`, `repair/make_targets.py`; new runner
`lane_pool.py`. Generated general scorer identity (`scorer_files` and its
`scorer_sha256`) changes because `score_forgetting.py` changes; eval-bed's scorer
file identity changes; S registration code validation and existing m2 runtime code
pins need the prospective D10 authority. SP's pinned `sp_score.py` construction
identity changes. Specifically `c_common.protocol().implementation` changes for
`c_targets.py`, `v4_teacher.py`, `score_forgetting.py` and `eval_bed.py`;
`m2_common.protocol().c` therefore changes, as do X bindings' optional
`scoring_protocol` values containing that M2 protocol. These changes flow into
newly produced runtime/send bindings and SP manifests that bind the changed bytes.
S's historical registration hash stays unchanged; its newly bound prospective
infrastructure authority must cover the changed registered code.
No scientific panel/checker/decoding/weights/schedule identities change, and the
explicit M3 trainer `CODE_FILES` digest is unchanged by these files. The integration
owner binds the new runner and effective engine helper in the new code authority.
