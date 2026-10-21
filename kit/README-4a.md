# 4A: memory plus consolidation, plan-v6 rung v2

Run after return 1 of [README-message.md](README-message.md). Historical `k4a`
is unrelated. This is a plain offline kit; the partner owns sbatch and the
existing `run_stage`/`message_handoff.py` chained-job handoff. Run exactly one
4A payload at a time. One physical GPU scores every comparison; eight GPUs train.
The binding science is copied
[verbatim, including the 21 Oct deviation](resources/a4/registration.md). No Modal preview answer is used here.

| Row | Work | Entry point |
|---|---|---|
| A1 | Restore parent OUT, verify scorer UUID/host/environment | Combined job-boundary block |
| A2 | Select W from complete unreset B; bind audited slice, base, replay and measured budget | `a4_bind.py` |
| A3 | Base general-v1 references, same physical scorer | `a4_run.py`, first rows |
| A4 | Render and archive arrival contexts and recurrence tags from earlier ledgers | `a4_memory.py`, called by runner |
| A5 | Seal all whole-month arrivals: frozen, static memory, direct ×5, consolidation on/off ×5 | `a4_score.py` |
| A6 | Admit only frozen, finished, unedited, source/unit-checked dual-scorer successes; freeze ledger hashes | `a4_ledger.py` |
| A7 | Direct stage each month; cumulative consolidation at 2/4/6; incoming retention references and NLL | `a4_train.py` |
| A8 | After each stage: all arrived holdouts with archived contexts, on/off for consolidation; each general-v1 panel | `a4_score.py` |
| A9 | After month 6: every saved stage checkpoint, memory off, on all six holdouts | `a4_score.py` |
| A10 | Registered reader, paired transitions, retention bounds and costs | `a4_read.py` |
| A11 | Save/verify durable OUT; compact return 2 and archive readback | `a4_collect.py`, `a4_read.py --source` |

`a4_run.py` executes A3–A10 in order, with a separate inference process for each
score cell and a fresh trainer process per stage. All long commands below use
`run_stage`, so TERM reaches their children before the combined handoff saves.
Keep absolute paths unchanged across jobs. Re-run identical commands: completed
rows verify hashes and SKIP; partial trainers keep their evidence and restart
the same seed at step zero. A completed 40-update checkpoint is recovered and
exported without another dose. Weak scores never stop completion. Exit 2 is a
correctness stop; exit 3 is a missing prerequisite. A changed scorer stops 4A;
do not mix epochs or reuse an old collector/training ledger on a new scorer.
The registration requires a dated prospective repair for identity/scoring faults.

Bind local inputs after combined M9 (return 1). `B_SOURCE` is the complete
unreset B score tree or compact return accepted by B's reader. W is recomputed
with the same C §3 selector; reset/recovery exports cannot select it. The 4A
binding also pins all ten selected-recipe standalone/sequence B step-40 export
identities from their original scoring receipts. A bare CSV cannot bind a real
run; the original identity metadata must accompany it. These exports identify
the selection evidence; every 4A arm starts from the registered base.
The base
must be the local pinned Qwen3-8B snapshot
`b968826d9c46dd6066d109eabc6255188de91218`, with its registered model-file hashes.
General-v1's 300 items ship with the kit. 4A starts from base weights and has
no FinQA retention panel; the registered panels are arrivals, permanent
holdouts with their memory-on/off references, and general-v1 IFEval/knowledge/maths. All 92 permanent holdouts in the 395-question primary slice
stay out of memory, teacher demonstrations, replay and training.

```bash
export A4_OUT="$PARENT_OUT/A4"
export FOUR_A_SLICE=/durable/data/slice-4a-primary
export B_SOURCE=/durable/B-complete-unreset
export B_REPLAY="$B/replay"
export FOUR_A_TIMING=/durable/receipts/W-stage-timing.json
export FOUR_A_STAGE_GPU_HOURS=6.666666666666667
export QWEN3_8B_TOKENIZER="$BASE"
# Existing BASE, SDPO, TRAINER_PY, INFERENCE_PY and SCORE_GPU
# retain the combined kit's pinned paths/environments. No download is performed.
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# First command at entry into 4A, in EVERY job, after return 1; safe to repeat.
# This fsyncs the job ID + restart count + UTC marker and saves it in a verified durable generation.
"$TRAINER_PY" "$KIT/a4_allocation.py" --out "$A4_OUT" --start \
  --parent-out "$PARENT_OUT" --durable "$DURABLE_ROOT" || exit $?
run_stage "$INFERENCE_PY" "$KIT/a4_bind.py" --out "$A4_OUT" \
  --b-source "$B_SOURCE" --slice "$FOUR_A_SLICE" --base "$BASE" \
  --replay "$B_REPLAY" --gpu "$SCORE_GPU" \
  --timing-receipt "$FOUR_A_TIMING" \
  --stage-gpu-hours "$FOUR_A_STAGE_GPU_HOURS" \
  --trainer-python "$TRAINER_PY" \
  --allocation-gpus 8 --score-max-seconds 600 --nll-max-seconds 600 \
  --handoff-seconds 300 || exit $?
```

The timing receipt is `{"recipe":"D+replay","gpus":8,"seconds":3000}`
for a measured 50-minute D+replay stage; replace it with the actual measured W
receipt. Its recipe and seconds must match the bound selection and GPU-hours.
The example is a schema, not a new measurement. The kit pins
`approved_ceiling_gpu_hours` to **600 for F/F+replay and 800 for D/D+replay**.
Package B tier 2 selected W = D, so 800 applies. The fixed approval reference is
owner, 21 October: "Approved with 600 partner GPU-hours. If it is tight, make it more",
from [Approved budget](../docs/phase4/4a-registration-20261020.md#approved-budget).
The binding records this exact quote and the registration path. Optional
`--approved-ceiling` and `--approval-reference` inputs must equal the pinned
values; differing values stop with exit 2. Runtime budget checks re-validate
both fields. Any increase needs the owner's decision and a new kit from us.
`ceiling_gpu_hours` retains the registered
formula as a forecast only, even if that forecast exceeds approval. Record the kit commit and
hashes, Python/software versions, node, physical GPU UUID and allocation IDs with
the return. Binding records implementation/resource/model/data/receipt hashes
and physical identity before new model output.

The 21 Oct deviation records that recurrence aggregation was not frozen before
v0. The reader always labels the recurrence split **EXPLORATORY** and writes
`v6_trigger: null`. There is no receipt override. This does not alter the
registered primary or retention bars.

As the first job-script command, run the `job-start` action in the combined
job-boundary block, before restoring OUT. It fsyncs
`DURABLE/job-starts/<job-id>-r<restart-count>.json`; allocation start is
`SLURM_JOB_START_TIME`, else this earliest script record. Keep those records.
At section entry, `4a-starts/<job-id>-r<restart-count>.json` pins the job ID,
`SLURM_RESTART_COUNT` (default 0), Unix seconds and UTC marker time. The marker
is fsynced, saved durably before binding/work, and unchanged on repeated entry
in that run. Each requeued run writes a fresh marker.

The first job that ever enters 4A charges from **max(allocation start, its 4A
marker)**. Earlier return-1 work in that job costs 4A zero. If the restored tree
already has a marker from an earlier run (an earlier job, or an earlier run of
the same requeued job), this continuation run charges from
**its allocation start**, including restore, identity checks, SKIPs and any
re-scoring after a GPU change before the payload. Its new marker retains both
entry time and the durable allocation start. Jobs without any 4A entry charge
zero, and attempts require their own run's marker.
If a hard kill leaves only a durable marker, accounting reconstructs its
allocation receipt. `allocations/<job-id>-r<restart-count>.json` and attempt
receipts carry both job ID and restart count, plus window, last record and end.
Real runs require `SLURM_JOB_ID`.

Before a new attempt, earlier jobs need an actual end from their own terminal
Slurm record or an explicit end marker. Accounting first tries `scontrol`, then
`sacct -D -j ID -X -n -P -o JobIDRaw,Start,End,State` when `scontrol` has no record.
Accounting accepts a row only when JobIDRaw equals the job ID exactly and Start
matches that run's recorded allocation start.
`scontrol` forgets finished jobs after MinJobAge (300 seconds by default);
`sacct` keeps accounting history. RUNNING and COMPLETING are non-terminal;
their scheduled EndTime is never an actual end. An unknown end leaves the run
pending (exit 3). If neither scheduler source has the actual end, the manual
command below is the last resort before resuming. Inspect
`sacct -D -j ID -X -P -o JobIDRaw,Start,End,State` and pick the row whose
JobIDRaw matches exactly and whose Start matches the run's allocation start.
Convert that row's actual End to Unix seconds. Run every missing-end command
first, then resume 4A; each command saves only that end, and the next budget
check refreshes the accounting for all runs.

```bash
# Repeat for every missing end, using the matching run's actual Unix end time.
"$TRAINER_PY" "$KIT/a4_allocation.py" --out "$A4_OUT" \
  --job-id "$PREVIOUS_SLURM_JOB_ID" --restart-count "$PREVIOUS_SLURM_RESTART_COUNT" \
  --ended-at "$ACTUAL_ALLOCATION_END_SECONDS" --durable "$DURABLE_ROOT" || exit $?
```

For a requeue of the same job ID, Slurm now describes the later run. The earlier
run charges through its recorded end. If no end survived, use the later of its
last 4A record and each unfinished attempt's start plus duration limit, capped
at the next run's allocation start. An actual end from the manual command or
Slurm can replace that provisional end. A manual end past any later run's
allocation start is refused (exit 2), including a later run recorded only in
`DURABLE/job-starts` before restore. A run charges from its allocation start
when any earlier run (another job or the same requeued job) already started 4A;
otherwise it charges from max(its allocation start, its fresh marker). If
`SLURM_JOB_START_TIME` differs from an existing record for the same run key,
or `SLURM_RESTART_COUNT` differs from scontrol's `Restarts`, stop (exit 2).
Before retrying job-start and 4A entry, export the current restart count from
the `Restarts=` field of `scontrol show job $SLURM_JOB_ID`:

```bash
SLURM_RESTART_COUNT=$(scontrol show job -o "$SLURM_JOB_ID" | tr ' ' '\n' | sed -n 's/^Restarts=//p')
export SLURM_RESTART_COUNT
```

Completed
attempts plus each run's idle remainder charge exactly **8 × held hours in its
4A window**, including idle GPUs and handoff copies. **Queue gaps between jobs
or requeued runs cost zero**. Allocation and attempt receipts travel in the
return. `--allocation-gpus` must be eight. Legacy wall-clock journals require
explicit reconciliation before this implementation can resume them.

Each attempt has an enforced duration limit; training includes config resolution
and export within 1.5 times its measured stage duration. Set scoring/NLL/handoff
limits at bind time from safe upper bounds for the actual machine; the examples
are limits, not timings. Before starting, the remaining **approved** budget must
cover that full limit and a reserved handoff. A timeout stops the worker and its
launched process tree. Spending stops at the approved ceiling even when it
prevents the remaining rows from completing. Use `a4_handoff.py` for bounded,
journalled saves throughout 4A.

Execute A3–A10. `--through-month 1` through `6` can bound work at a calendar
boundary. The default is all six; the same invocation resumes in job 2. Save
between calendar boundaries to limit checkpoint-copy work before TERM.

```bash
for MONTH_NUMBER in 1 2 3 4 5 6; do
  run_stage "$TRAINER_PY" "$KIT/a4_run.py" --out "$A4_OUT" \
    --through-month "$MONTH_NUMBER" --score-gpu "$SCORE_GPU" \
    --gpu 0,1,2,3,4,5,6,7 --tokenizer "$BASE" --sdpo "$SDPO" \
    --trainer-python "$TRAINER_PY" --inference-python "$INFERENCE_PY" || exit $?
  "$TRAINER_PY" "$KIT/a4_handoff.py" --out "$A4_OUT" --parent-out "$PARENT_OUT" \
    --durable "$DURABLE_ROOT" --trainer-python "$TRAINER_PY" || exit $?
done
```

The stream's manifest hash is
`ed5a8b17357e3e9679f12b653e7a782dfab5af7ac9ddf14452183f26ae4cbcad`.
All 15 supplied slice/audit files are pinned in `resources/a4/slice.json`; the
original directory is read-only input. Its counts are 37/80/80/75/80/43, with
5/24/19/14/20/10 holdouts. The renderer enforces 10,240 input tokens and drops
lowest-ranked whole cases; it never truncates the current documents. It matches
the incumbent's family/Jaccard/form ranking, ID tie breaks, case formatting and
issuer/accession/required-filing exclusions. Recurrence uses the pinned label
tokenizer and only previously released ledger months. Earlier holdout prompts
are archived at arrival, including independent memory-off references.

The reviewed primary build invocation, for provenance only (do not rebuild or
replace the delivered audited slice during the experiment), was:

```bash
SEC_CACHE_GZIP=1 python scripts/p4_sec_slice.py build --version 4 \
  --purpose 4a-primary --months 2026-01,2026-02,2026-03,2026-04,2026-05,2026-06 \
  --families fcf_margin_change,cash_conversion_change --per-family 40 \
  --max-candidates 2400 --seed 4a0-slice-v1 --allow-shortfall \
  --amendment-rule restating --amendment-horizon 2026-Q3 \
  --out data/p4_sec/slice-4a-primary --cache "$SEC_CACHE_PATH"
```

The builder is our-side provenance code, outside the public partner export.
Gold/source checks are re-run against each frozen item's displayed-cell/XBRL
bindings at bind/admission. Accepted SFT targets and D teacher demonstrations are
the complete original collector responses; no gold rationale or 27B solution is
substituted. W retains full weights, 40 updates, batch 32, LR 1e-5, a ten-step
warmup starting at zero, response/EOS masks and fresh optimizer/EMA per stage.
Replay slots remain 8/16/24/32; D skips over-cap draws in the shuffled full generic
pool. The same frozen replay background is accessible to every arm; no arm gets
a new private pool or extra teaching. The entire pool is checked against 4A
and general-v1 evaluation prompts before binding. Training schedules have
1,280 exposures (1,120 task +160 generic for replay), with unique/repeat counts.
Only empty monthly/cumulative ledgers skip their respective stage. No replacement
stage is made up. Only stage-final step-40 weights are saved: nine full-weight
checkpoints per seed, all scored; none is selected by score.
The approximately 0.7 TB figure covers HF stage-final weights only. Native
optimizer checkpoints and failed partials are also retained and hash-listed;
their additional storage and copy time must fit the actual durable filesystem.

A10 writes `reading/summary.json` and Markdown. The primary is the unweighted
months 2–6 mean of consolidation minus static memory on permanent unseen-issuer
holdouts: ≥5 explicit-unit points, ≥4/5 positive seeds, ≥3/5 positive monthly
seed-means and a positive two-sided 95% paired seed-t interval (df 4). Retention
uses paired one-sided seed-t upper bounds ≤3 points for every registered margin,
including incoming and original general references. Crossing bounds mean **not
established**. Historical scores, issuer bootstrap, all-arrival/family/recurring-
issuer splits, paired transitions and acquired-then-lost IDs remain descriptive.
FinQA is returned with paired retention bounds as a diagnostic; §9 does not add
it to the registered joint-success bars.
Recovery/gain-kept ratios with zero denominators remain undefined. Equal-update
comparisons use 40/80/120 updates; equal-time tables report the bracketing saved
checkpoints without selecting a winner or claiming an exact time match.

Finish A11 and return 2 immediately:

```bash
run_stage "$TRAINER_PY" "$KIT/a4_read.py" --out "$A4_OUT" || exit $?
"$TRAINER_PY" "$KIT/a4_handoff.py" --out "$A4_OUT" --parent-out "$PARENT_OUT" \
  --durable "$DURABLE_ROOT" --trainer-python "$TRAINER_PY" || exit $?
COMMITTED_OUT=$("$TRAINER_PY" "$KIT/message_handoff.py" --out "$PARENT_OUT" \
  --durable "$DURABLE_ROOT" --action path) || exit $?
run_stage "$TRAINER_PY" "$KIT/a4_collect.py" --out "$A4_OUT" \
  --durable-root "$COMMITTED_OUT/A4" --archive "$PARENT_OUT/4a-return.tar.gz" || exit $?
run_stage "$TRAINER_PY" "$KIT/a4_read.py" --out "$A4_OUT" \
  --source "$PARENT_OUT/4a-return.tar.gz" || exit $?
"$TRAINER_PY" "$KIT/a4_handoff.py" --out "$A4_OUT" --parent-out "$PARENT_OUT" \
  --durable "$DURABLE_ROOT" --trainer-python "$TRAINER_PY" || exit $?
```

The return is capped at **28 MiB**: gzipped score CSVs, summaries, provenance and
a hash inventory locating omitted bulk evidence on the verified committed
durable tree. Raw prompts, responses, token journals, datasets, partials and
weights remain durable. Archive readback verifies member hashes and reproduces
the reader without loading weights or the private repository. Send the compact
return with the job IDs, failed/restarted attempts, allocation/wall/GPU-hour and
idle-reservation accounting, storage paths and unresolved inputs. Nothing in
this kit sends messages or submits jobs.

Measured pricing and the registration's formula (45 stages across seeds 101–105):

| W timing anchor | Training GPU-h | Training wall h | Rounded formula forecast |
|---|---:|---:|---:|
| F, 2.7–3.3 GPU-h/stage | 121.5–148.5 | 15.2–18.6 | 350–400 GPU-h |
| D+replay, 6.3–6.7 GPU-h/stage | 283.5–301.5 | 35.4–37.7 | 600 GPU-h |

Formula forecast = `ceil_to_50(1.5 × 45 × measured_stage_gpu_hours + 140)`.
The 140 is the registered 60 scoring reserve plus 80 startup/export/storage/
failure allowance. F+replay and D use their own measured W receipt at send;
until then their respective F and D+replay proxies apply. The combined C runbook
also records a slower 58-minute D+replay anchor (7.73 GPU-h), which would make
this same formula **700 GPU-h**. Bind actual measurements and the approved
**800 GPU-h** D-family ceiling. No stage timing for SEC long contexts is claimed.

There are **688 scientific score cells**: 102 arrival, 45 incoming, 270
post-stage, 270 final-grid and one base control. Without deterministic cache
reuse they contain **41,575 predictions**. Removing the 91 FinQA cells removes
104,377 predictions. Exact matching weights/prompts/physical-scorer generations
are reused under each arm/reference label. At the registration's approximate
1.22 s/question, the uncached registered panels take **14.1 one-GPU hours**,
versus 49.5 previously. NLL, engine loads and long-context throughput still need
measurement. The registered 60 scoring +80 overhead reserve and rounded ceiling
formula remain unchanged; unused reserve is not permission to add panels.

For an operational forecast, charge all eight reserved GPUs during those
14.1 scoring wall hours: **112.7 allocation GPU-h**. With training
121.5–301.5 GPU-h and the existing 80 GPU-h overhead allowance (10 wall hours
on eight GPUs), 4A is **314.2–494.2 allocation GPU-h and 39.3–61.8 wall hours**,
before NLL, slower contexts, or training contingency. Combined with return 1's
265.7–486.3 GPU-h and 70.8–119.2 wall hours, this is **579.9–980.5 GPU-h and
110.1–181.0 wall hours**, plus M3b's 2.6–4 GPU-h and ≤2 wall hours. These are
forecasts, not measured SEC throughput or a guarantee that every bound ceiling
will fit. NLL and longer startup/handoff timings consume the remaining reserve;
the runner refuses jobs that cannot fit their worst-case limits.

The owner-approved 4A ceilings are **600 GPU-h for F/F+replay** and **800 for
D/D+replay**, independent of the formula forecast. Combined with the matching
return-1 W totals, the budget envelope is **865.7–1,286.3 GPU-h**, plus M3b's
2.6–4 GPU-h. The slower 58-minute D receipt forecasts 700 under the formula;
its approved ceiling remains 800. Raw journals include
failed attempts, reserved idle time, merges, NLL and handoffs. Serving denominators
are six-month arrival correct counts. Collector generations are charged once
per deployable system; retention overhead and shared study spend are reported
separately. A >1.5× frozen output-token cost is labelled “at cost”, never excluded.
Undefined break-even stays undefined.

Offline full rehearsal, including all rows, interruption, cross-job restore,
hash/identity rejection, completed-row SKIPs and compact-return equivalence:

```bash
python kit/a4_dryrun.py --out /tmp/a4-full-D-standin --standin --recipe D+replay
python kit/a4_dryrun.py --out /tmp/a4-full-F-standin --standin --recipe F+replay
python -m pytest tests/test_a4_kit.py -q
```
