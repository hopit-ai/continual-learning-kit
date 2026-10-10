# Phase 0: guide for the agent debugging it on the node

This is for whoever (person or coding agent) is running phase 0 on the 8×H100 node and hits a failure. It says what
phase 0 is for, what each row does, where to look when one fails, what you may fix yourself, and what to send back.
`README-phase0.md` remains the full runbook; this is the short version for debugging.

## What phase 0 is for

A **technical check**, not an experiment. It runs every stage of the FinQA pipeline once, at toy size, on your node:

1. a 27B teacher writes worked solutions for 20 FinQA training questions;
2. the 8B student rewrites them in its own words;
3. both become a small training set;
4. four training recipes (S, F, R, D) each train the 8B for **two optimizer steps**;
5. each result is merged, reloaded and scored on 20 held-out questions.

Nothing it produces is a result. The scores are meaningless at two steps. What we need from it:
- **every stage completes on your hardware with the shipped code**;
- **the timings, memory and logs**, so the real campaign can be sized.

So the goal when something breaks is: **make the stage run, record exactly what you changed, keep going.**

## The rows, in order

All rows run serially inside one allocation (`runner.py batch`). Each row's exact command and environment are in
`$KIT/campaigns/v4-phase0.yaml`. The allocation script `$WORK/v4/report-phase0/allocation/phase0.sh` exports the
environment and then runs the runner.

| Row | What it does | GPUs | Python env |
|---|---|---|---|
| `containment-selftest` | GPU/Slurm isolation self-test. **Informational**: a failure is recorded, later rows still run | 8 | trainer |
| `verify-prepare` | Re-hashes the files prepared on the login node | 0 | trainer |
| `score-agreement1..4-finqa-2048` | Scores the untrained 8B on 50 held-out questions, four times, to check scoring is repeatable | 1 | trainer (vLLM 0.12) |
| `scoring-agreement` | Compares those four scorings | 0 | trainer |
| `teacher-finqa` | 27B teacher: **4 engines × TP=2**, writes solutions, verifies answers, merges shards | 8 | **inference venv (vLLM 0.18)** |
| `rewrite-finqa` | 8B rewrites the verified solutions: 8 engines × TP=1 | 8 | trainer (vLLM 0.12) |
| `training-set` | Keeps questions with both a solution and a rewrite | 0 | trainer |
| `schedule-q-S-finqa` | Builds the shared 64-example order for two steps | 0 | trainer |
| `resolved-configs` | Resolves all four launchers' Hydra configs and checks key constants | 0 | trainer |
| `train-q-{S,F,R,D}-finqa` | Two optimizer steps of each recipe (S and D: verl PPO-style; F and R: verl SFT) | 8 | trainer |
| `merge-q-{S,F,R,D}-finqa` | Merges the FSDP checkpoint into a Hugging Face export | 0 | trainer |
| `score-q-{S,F,R,D}-finqa-finqa-2048` | Reloads the export and scores 20 held-out questions | 1 | trainer |
| `report` | Reads all the evidence and writes the technical reading. **Fails if the reading is incomplete** | 0 | trainer |
| `PAUSE` | Ends the allocation | 0 | trainer |

Each row needs the row before it. If one fails, the rows after it fail or are skipped for lack of input. **Debug the
first failed row; the rest usually follow from it.**

## Where to look

| What | Path |
|---|---|
| Per-row command, output and verdict | `$WORK/campaign/v4-phase0/<row>/attempt-N/` (`start.json`, `output.log`, `verdict.json`) |
| Why a row failed (exception text) | `$WORK/v4/report-phase0/failures/<row>.json` |
| A hard stop (wrong model hash, inputs, disk, writes outside WORK) | `$WORK/v4/report-phase0/hard-stop.json` |
| GPU-row containment records | `$WORK/v4/containment/v4-phase0/<row>/attempt-N/` |
| Teacher and rewrite engine logs, one per shard | `$WORK/v4/report-phase0/shard-logs/{teacher,rewrite}/finqa/<shard>.log` and `launches.json` |
| Teacher and rewrite outputs | `$WORK/v4/report-phase0/{teacher,rewrite}/finqa/` (`raw_outputs.jsonl`, `pool.jsonl`, `demonstrations.jsonl` / `rewrites.jsonl`) |
| Training runs | `$WORK/runs/q-<ARM>-finqa/` (`metrics.jsonl`, `env/`, launcher logs, checkpoints, `hf-step2/`) |
| Slurm job stdout and stderr | wherever your wrapper writes them |

`kit/v4_phase0.py` exit codes: `0` = row passed, `1` = the operation failed (the cause is in `failures/<row>.json`),
`2` = hard stop.

## Debugging loop

1. Find the first failed row, and read its `output.log`, its `failures/<row>.json`, and, for teacher or rewrite,
   the shard logs. The real error is usually in the shard log or the trainer log, not the runner's summary.
2. To iterate quickly, take an interactive allocation (`salloc` with the same GPUs). Copy the `export` and
   `source` lines from `phase0.sh`, everything above the `p4_contain.py selftest` line. Then run the failing
   row's **inner command**, the part after `--` in its `v4-phase0.yaml` entry, with that row's `env` values set.
   For example, for the teacher:
   ```sh
   export V4_PHASE0_INFERENCE=1 V4_COMMAND_TIMEOUT=6000 V4_TELEMETRY=1
   python "$KIT/v4_phase0.py" teacher --work "$WORK" --phase phase0 --row teacher-finqa --task finqa
   ```
   Hand runs write into WORK. They are for diagnosis only: some stages skip work they find already complete,
   and hand runs leave no runner records.
3. Once it works, **start a fresh WORK and launch the full allocation again** as in `README-phase0.md`. The
   Hugging Face model blobs are cached, so a fresh WORK mostly re-does the environment builds. The evidence
   we read must come from one clean, complete runner pass.

## What you may change, and what you must not

**You may change infrastructure** that does not alter what is generated, trained or scored. Record every change
(see "What to send back"). For example:
- time limits, deadlines, retries of a crashed process, waiting for a slow filesystem;
- environment variables, cache directories (for example keeping Triton, inductor and vLLM caches outside WORK so a
  second run reuses them), temporary-directory locations, Slurm directives;
- vLLM engine settings that do not change outputs: eager mode, timeouts, `max_num_seqs`, `gpu_memory_utilization`,
  starting the four teacher engines one after another instead of together;
- paths, launcher plumbing, a missing package or import (record its version);
- a bug in our code that crashes or wrongly fails a row. Fix it minimally and tell us.

**You must not change** anything that would change the science. If a fix seems to need one of these, stop and report:
- the models or their revisions, or any data: question pools, held-out panels, prompts, the 20/50 question choices;
- generation settings: temperatures, seeds, attempt schedule, token caps, the answer verifier;
- the reward, the scoring code or its settings;
- training settings: learning rate, steps, batch, sequence lengths, arm definitions (S, F, R, D);
- never mark a row passed by hand, edit an output or report file, or delete evidence. A failed row with its logs
  is more useful to us than a row made to look passed.

## What to send back

After the clean run (or after you stop), run the reconcile, collect and reader commands at the end of
`README-phase0.md`. Then send:
1. `${WORK}-return.tar.gz`, `${WORK}-reading.json` and the allocation job ID;
2. **your local changes as a diff against the tag:** `git -C "$KIT/.." diff kit-v4-phase0-v7 > "${WORK}-local.diff"`
   (include untracked files if you added any);
3. a short note: for each change, what failed, the evidence (log line), what you changed and why it does not
   alter the science. We fold your fixes into the main package.

## Failures so far, for orientation

| Version | What broke on the node | Fix |
|---|---|---|
| v1 | Teacher config keeps `vocab_size` under `text_config` | read the nested value (v2) |
| v2 | Tight unmeasured time checks, including a 5-second GPU probe | every limit ×10 and the self-test made informational (v3/v4) |
| v4 | A failed operation was recorded as a pass; S launcher dataset path | a failed operation fails its row; resolved dataset path (v5) |
| v5 | SFT's final validation record was counted as a step; GPU UUID `GPU-` prefix | both fixed (v6) |
| v6 | Four TP-2 teacher engines spent 27 min capturing CUDA graphs, then the first request hit vLLM's 300 s RPC deadline | engines eager; vLLM, NCCL and Gloo deadlines ×10; an incomplete report fails its row (v7) |

The v6 teacher failure means the teacher, rewrite, training, merge and scoring rows **have not yet run on this node**.
Expect the first problems there. Before v6 the "teacher passed" result was the silent-pass bug.
