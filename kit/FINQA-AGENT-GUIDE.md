# Main FinQA: guide for the partner’s debugging agent

Read [README-finqa.md](README-finqa.md). Owner direction is plan v4 sections
15.21 and 15.22 (10 October). Phase 0 v7 passed on job 414790: 26/26 rows,
12.06 allocation GPU-hours. Its exports are technical checks, never scientific
initialisations.

## What runs

Qualification runs first: the **full 1,441-question teacher corpus and rewrites**,
first timing/coverage/scoring measurements, saved-state restoration, and two
updates for S/F/R/D/D-R. Its report is a progress report. **There is no PAUSE,
owner-proceed receipt, or second-tag transition. Main follows automatically.**

Main runs all 23 standalone slots at forty updates: S/F/R/D seeds 101–105 and
D-R seeds 101–103, with the registered scoring and eight-trajectory probes.
The D seed-101 comparison is an analysis finding; missing its bar never stops
later slots. Low intersection coverage warns; train on surviving verified
questions with the unchanged shared seeded schedule and batch/dose. Keep all
61 BF16 exports. Trainer states retire only when the original proofs verify;
failed retirement retains state and warns instead of blocking another slot.

The allocation **ceiling is 1,000 GPU-hours of actual cumulative spend**,
including phase-0 carry, idle allocation time and failed attempts. A planned
100-hour qualification line is advice. No projection or block-fit refusal applies.
One native eight-GPU job executes qualification then all main stages. If its site
window ends, the flow submits the same retained native wrapper with
`--dependency=afterany:$SLURM_JOB_ID`; the continuation skips passed rows.
This needs no manager intervention and uses the pilot’s existing host venv route.
Request the site's MaxTime of **10,080 minutes** for the allocation. Watch the
progress reports throughout the run, including actual spend and warnings.
Scoring uses **one GPU of the eight by design**; all eight allocated GPUs are
still charged during scoring.

Hard stops are wrong model/data identity, writes outside WORK, actual disk
exhaustion before the next write, and actual 1,000 GPU-hour spend. Everything
else is a warning or a failed operation to fix locally. Generous **10×** row caps
are wall-clock backstops. A command failure remains a FAIL in its original
attempt; it never becomes a fabricated PASS or blocks independent later rows.
The overall command can return nonzero when it retained failed rows.

## Find failures and warnings

| Evidence | Location below WORK |
|---|---|
| Command, frozen inputs, log, verdict | `campaign/v4-{qualification,main}/<row>/attempt-N/{start.json,output.log,verdict.json}` |
| Operation failure | `v4/report-{qualification,main}/failures/<row>.json` |
| All warning receipts | `v4/warnings/*.json` (also printed as WARNING) |
| Phase actual-spend boundaries and cap envelopes | `v4/report-flow/progress/*.json` |
| Progress report and timings | `v4/report-{qualification,main}/{reading.json,timings/}` |
| Teacher/rewrite engines | `v4/report-<phase>/shard-logs/{teacher,rewrite}/finqa/` |
| Trainer, checkpoint/export proofs | `runs/<slot>/` |
| Actual accounting and Slurm observations | `k8b4/containment/` |
| Batch logs and continuation submissions | `v4/report-qualification/allocation/`, `v4/report-flow/submissions/` |

The planner creates Slurm log directories before submission. Keep the same WORK
and native wrapper. CPU prepare both phases before submitting the flow. Recover
frozen launch exports in a fresh login shell using
`python "$KIT/v4_finqa_site.py" exports --work "$WORK"`. Wheel/compilation caches
live in `TASK_ROOT/cache` (WORK parent/cache fallback), outside WORK; TMPDIR remains
job-private. Caches are not evidence and are never archived. The **1,879 GiB** disk
figure is advice, not a prepare gate. Check free space/quota and grow storage if
needed; the next save/export checks its own write capacity.

## Fix and rerun in place

After fixing infrastructure, activate the same trainer venv and rerun the failed
row in the same WORK and allocation (or a new flow allocation if the old one ended):

```bash
python "$KIT/runner.py" run "$KIT/campaigns/v4-main.yaml" --row ROW_ID
# A qualification row uses v4-qualification.yaml instead.
python "$KIT/v4_finqa_flow.py" --work "$WORK"
```

No review file or retry-count limit is needed. Re-running the planner with the
same `--out` keeps its original files and writes `-attempt-N` siblings in the
same WORK; use the printed new payload/header for a replacement allocation. The runner allocates a new
`attempt-N`, retains the previous attempt, and skips any row already PASS, even
when a dependency was retried. Complete valid saved optimizer state resumes;
partial state is preserved in the failed-attempt archive. It never resets a
valid completed dose. Warnings about scheduler reconciliation, versions, scoring
environment, metrics, C1 timing, determinism or batch invariance do not require
owner acknowledgement. Actual execution failures still require a working fix.
Incomplete scoring output is moved into the failed runner attempt's
`scoring-output/`. An incomplete restoration arm's observations and leg runs
are preserved in `v4/report-qualification/restoration/failed-attempts/<arm>/attempt-N/`
before a fresh restoration attempt; completed arms are reused.

## What you may change

**Infrastructure changes are free:** paths, venv activation, caches, Slurm and
GPU plumbing, crashes, sockets, patience, telemetry and resume plumbing. Fix and
rerun failed rows yourself. Preserve v7’s eager engines, RPC 3,000 s, teacher/NCCL
6,000 s, Gloo 18,000 s unless an infrastructure fix requires changing them; explain
that change in the note.

We retain the registered models/revisions, data/pools, prompts, sampling, verifier,
reward/scoring, learning rates, steps, batch, arms and seeds. A change touching
science is allowed **only when nothing else works**. Flag exactly what changed
and why in your note so the owner decides whether that run counts. The bars
D >= 58/1,147 and D minus S >= five points per primary seed are analysis criteria,
never permissions to run.

Never fabricate or hand-edit model outputs, metrics, timing receipts, verdicts
or reports. Never delete evidence. Do not manually delete checkpoints to evade
identity checks; automatic proof-bound state retirement remains supported.
Write campaign outputs inside WORK. Keep the phase-0 return immutable.

## Return along the way and at the end

Return progress reports/warnings, archive, job IDs and a short note describing
every fix. Return the diff against the original immutable tag:

```bash
git -C "$KIT/.." diff "$V4_KIT_TAG" > "$WORK/main-local.diff"
```

Include untracked helpers separately. Flag any science changes prominently and
let the owner decide whether the run counts. Do not push tags, publish changes,
or contact anyone from the kit.

## Phase 3 partner package A

Package A now uses the plain jobs in [README-p3a.md](README-p3a.md).
Pass local model paths, venv Python paths, GPU IDs and a fresh OUT directly.
Its jobs do not read this FinQA WORK or use campaign/allocation accounting.
