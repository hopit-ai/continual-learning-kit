# One combined next partner message

Use this order for the two follow-on jobs, with a seven-day cap per job. The
partner owns sbatch. No submission wrapper, network, push or paid API is needed.
The final 20 October registration amendments take precedence over plan v6's old
probe-before-C order: replay breadth now runs on C's FinQA→ContractNLI route.
B tier 2 is read as forgetting/recipe comparison only, limited by MATH headroom
(base 89/100). Complete unreset B selects W; recovery/reset evidence never does.

Use [README-next.md](README-next.md) for v0+/v3 and frozen pools, and
[README-c.md](README-c.md) for C plus its probe. Every script is resumable:
identical completed rows print SKIP, identity mismatches stop, partial training
preserves evidence and restarts the same registered seed from step 0. Never run
two copies of a row or overlap full-node training. Exit 2 means correctness stop,
3 means prerequisite pending; weak scores do not stop the core.

| Row | Ordered work | Concrete commands |
|---|---|---|
| M1 | Restore durable OUT; identify scoring host/GPU | Job boundary block below |
| M2 | Bind W/inputs and freeze HellaSwag before any new output | README-next N1–N3 |
| M3 | v0+ and v3 screen scoring, serial on one GPU | README-next N4–N5, N10 |
| M3b | Quick job: hard-MATH teacher coverage, two GPUs in a free gap | Quick-jobs block below |
| M4 | Freeze C membership; 27B targets, two GPUs in a free gap | README-c §§2–3 |
| M5 | Broad/earlier pools; checked base answers | README-next N6 |
| M6 | C base/incoming controls, then core full-node stages serially | README-c §4 |
| M7 | C probe reference + broad + earlier + both, full-node serial | README-c §7 |
| M8 | Read complete C/probe and screen tables | README-c §5/§7; README-next N10 |
| M9 | Durable save/verify; collect return 1 immediately | Return block below |
| M10 | Explicit optional 4A gate, only a present audited 4A kit | Gate block below |
| M11 | Return 2 after 4A, durable save/verify | Gate block below |
| M12 | Job end save/verify; next job repeats M1 and unfinished rows | Job boundary block |
| M13 | Manager verifies and reads compact returns | README-next N12; README-c §5 plus c_probe_read |

Bind paths from both supplements, then use one parent tree. Keep the same
absolute OUT/NEXT/C/NVME paths across jobs because receipts contain paths.
Inputs/models/v4 exports/B stay at their original durable paths. OUT contains
all newly produced scientific evidence, exports, journals and receipts.

```bash
export CHECKOUT=/absolute/reviewed-checkout
export KIT="$CHECKOUT/kit"
export OUT=/local/nvme/message
export NEXT="$OUT/NEXT"
export C_OUT="$OUT/C"
export NVME=/local/nvme/blends
export DURABLE=/fsx_efa/next-message
export TRAINER_PY=/venvs/trainer/bin/python
export INFERENCE_PY=/venvs/inference/bin/python
export SCORE_GPU=0
export RETURN="$OUT/next-return-1.tar.gz"
export RUN_4A=0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

When executing README-c commands, temporarily set `OUT="$C_OUT"`, then restore
the parent OUT for handoff/return commands. The job-boundary block captures
read-only `PARENT_OUT` and `DURABLE_ROOT` once; C commands must not change them. Set `V4_FINQA_PANEL="$FINQA"` for
c_score/c_targets; it is the validated 1,147-item FinQA dataset, while N6 uses
`FINQA_PANEL`, the exclusion JSONL. Keep the supplements' remaining pinned paths.
Execute C §§2–3 in M4, §4 in M6, §7 in M7, then §5 in M8. N3 must precede M3;
M4 may use two GPUs only during a gap with no full-node training and sufficient
host bandwidth. Scoring stays serial; no allocation overlap is assumed below.

Quick-jobs block before C (M3b). **Hard-MATH teacher coverage**: the pinned
Qwen3.6-27B answers the frozen 200-item Level-5 MATH screen panel with the 3C
teacher protocol (four registered attempts, thinking off, the 3C verifier) and
the frozen rule decides whether this panel can be the connected continuation's
third task: teacher accuracy at least 160/200 **and** at least 30 of the frozen
8B's 62 failures solved by the teacher. Inference venv (vLLM 0.18.0), **two
GPUs** (TP2), no other input than the kit's own frozen resources and the two
pinned snapshots from README-c (`TEACHER`, `BASE`; `BASE` supplies the 8B
tokenizer for 3C's target limits). It can run in any gap with no full-node
training, beside one-GPU scoring on other GPUs; nothing else reads its output,
so C never waits for it. Run it before C if a gap exists, and in any case before
return 1, so its directory travels with the return.

```bash
export TEACHER_GPUS=6,7   # two GPUs not used by any running job
run_stage "$INFERENCE_PY" "$KIT/math_hard_teacher.py" --out "$OUT" --model "$TEACHER" --tokenizer "$BASE" --gpu "$TEACHER_GPUS" || exit $?
run_stage "$TRAINER_PY" "$KIT/math_hard_teacher_read.py" --source "$OUT/math-hard-teacher" || exit $?
run_stage tar -czf "$OUT/math-hard-teacher-return.tar.gz" -C "$OUT" math-hard-teacher
```

Exit 0 prints QUALIFIES or DOES_NOT_QUALIFY; either is a finished job. Exit 3
(incomplete journal) or an interrupted allocation: rerun the same command, which
resumes the journalled attempts and never resamples them; a finished run prints
SKIP. Exit 2 means a frozen input, model or setting differs: stop and report.
Estimated cost from the 3C MATH teacher step's measured 1,548 s on two GPUs:
**about 1.3 h wall, 2.6 GPU-h; allow 2 h wall, 4 GPU-h**. Rehearsal on CPU:
`python "$KIT/math_hard_teacher.py" --out /tmp/mht-standin --standin`.

Job boundary commands (every job and requeue). Make `job-start` the first
executable job-script command, with the path/venv variables above already set:

```bash
"$TRAINER_PY" "$KIT/message_handoff.py" --out "$OUT" --durable "$DURABLE" --action job-start || exit $?
readonly PARENT_OUT="$OUT" DURABLE_ROOT="$DURABLE"
"$TRAINER_PY" "$KIT/message_handoff.py" --out "$PARENT_OUT" --durable "$DURABLE_ROOT" --action restore --move-aside || exit $?
"$TRAINER_PY" "$KIT/message_handoff.py" --out "$PARENT_OUT" --durable "$DURABLE_ROOT" --action identity --gpu "$SCORE_GPU" || exit $?
# Preserve OUT on exits; forward TERM promptly to the running stage.
export TERM_LEAD_SECONDS=1800 SAVE_ALLOWANCE_SECONDS=900
source "$KIT/message_job.sh"
```

At each job end, also run the save command explicitly and check exit status.
The helper locks the durable root with `flock` for save/restore. A second
writer stops with a clear busy message. It compares a size + nanosecond mtime
manifest to CURRENT, reuses cached hashes for unchanged files and skips an
unchanged OUT. New generations use `rsync -a --link-dest=<CURRENT tree>` for
unchanged hard links; only changed/new files are copied and hashed at source and
destination. The receipt reports `bytes_copied` from rsync and `changed_bytes`.
Same-size changes within one second are copied explicitly. Stop all OUT writers
before saving; a size/mtime change during the copy refuses publication. This
assumes completed checkpoints are immutable: deliberately preserving both size
and mtime on a content edit defeats the quick check.

Publication renames the verified generation and atomically replaces CURRENT
using unique pid/random writer temps. CURRENT alone is authoritative, including
if a dead writer left `CURRENT.json.tmp`. After publication, only CURRENT and its
previous committed generation remain; older/unpublished generations and stale
`.partial-*` are removed. Failed saves leave CURRENT untouched. A hard kill
cannot run a trap; the next restore uses the last committed generation.

Restore fully verifies CURRENT. It never deletes local evidence: with
`--move-aside`, differing/local-only entries are renamed into the sibling
`OUT.pre-restore-<UTC>-<unique>` tree before copying, and the receipt lists those
paths. Newer/local-only files print a prominent warning. Without that flag,
a differing local OUT is refused. Inspect preserved newer stages before resuming.
An old in-place durable layout still requires explicit inspection/migration.

Run **every long stage** through the background-child wrapper, including commands
from the supplements and optional command files:

```bash
# Set REGISTERED_C_ARGS to the exact argument array from README-c.
run_stage "$TRAINER_PY" "$KIT/c_train.py" "${REGISTERED_C_ARGS[@]}" || exit $?
# For an existing payload script: run_stage bash -e partner-stages.sh || exit $?
```

The wrapper uses `cmd & child=$!; wait "$child"`, so TERM interrupts the wait
immediately. Each stage gets its own process group. It forwards TERM to that
group (including trainer/inference workers), waits at most `TERM_LEAD_SECONDS -
SAVE_ALLOWANCE_SECONDS`, then kills an unresponsive child and saves on EXIT.
Never detach workers from that process group or launch separate writers into OUT. Save between stages to bound
changed checkpoint volume. Measure incremental copy + changed-file hash time
on the actual durable filesystem and set the allowance above its worst case
plus metadata/cleanup margin. The 1800 s example reserves 900 s for saving and
900 s for child shutdown; it is sufficient only if measured work fits those
budgets. The first save copies the entire starting OUT: do it before long work,
outside the TERM window. A 3 TB initial copy cannot rely on a 30-minute lead.

```bash
# Partner supplies allocation/account/partition and existing payload scripts.
# Request --nodelist=<host> and --dependency=afterany:<prev> for both jobs.
HOST=the-current-host
B_JOB_ID=the-current-B-job-id
J1=$(sbatch --parsable --time=7-00:00:00 --nodelist="$HOST" --dependency="afterany:$B_JOB_ID" --signal=B:TERM@1800 partner-message-job1.sh)
J2=$(sbatch --parsable --time=7-00:00:00 --nodelist="$HOST" --dependency="afterany:${J1}" --signal=B:TERM@1800 partner-message-job2.sh)
```

These are partner submission examples, not commands this kit executes. Job 2
runs the same ordered rows; completed scoring/training rows SKIP. `afterany`
also starts after a failed job, so inspect the saved failure/status note first.
If the scoring **machine, environment or physical GPU UUID differs**, the helper
moves old scores/SEC/blends/aliases into `OUT/scoring-history/<epoch>/` outside
active readers. Re-run M2–M3 (base and λ=0/1 references first), then C/probe
base and incoming/reference scores before any comparison; re-score every compared
checkpoint on this GPU. Finished training and frozen pools/targets remain reusable.
This full refresh avoids mixing epochs; receipts record UUID/host/environment,
and the old raw evidence stays durable. CUDA ordinal alone is insufficient.

Return 1 after C **and** probe, before 4A:

```bash
run_stage "$TRAINER_PY" "$KIT/next_read.py" --out "$NEXT/reading" --source "$NEXT" || exit $?
run_stage "$TRAINER_PY" "$KIT/c_read.py" --out "$C_OUT/reading" --source "$C_OUT" || exit $?
run_stage "$TRAINER_PY" "$KIT/c_probe_read.py" --out "$C_OUT/reading" --source "$C_OUT" || exit $?
"$TRAINER_PY" "$KIT/message_handoff.py" --out "$PARENT_OUT" --durable "$DURABLE_ROOT" --action save || exit $?
COMMITTED_OUT=$("$TRAINER_PY" "$KIT/message_handoff.py" --out "$PARENT_OUT" --durable "$DURABLE_ROOT" --action path) || exit $?
run_stage "$TRAINER_PY" "$KIT/next_collect.py" --out "$NEXT" --durable-root "$COMMITTED_OUT/NEXT" --archive "$RETURN" || exit $?
run_stage "$TRAINER_PY" "$KIT/c_collect.py" --out "$C_OUT" --archive "$PARENT_OUT/package-c-return.tar.gz" || exit $?
"$TRAINER_PY" "$KIT/message_handoff.py" --out "$PARENT_OUT" --durable "$DURABLE_ROOT" --action save || exit $?
# RETURN 1: both compact archives plus kit/env/GPU receipts, job IDs,
# failed/restarted attempts, wall/GPU times, durable path and unfinished inputs.
```

Optional 4A is now present: [README-4a.md](README-4a.md), plan-v6 rung v2,
with its audited primary-slice pins, collector, all arms, reader and compact
return. Historical k4a is unrelated. Complete return 1 first. Set the A2 input,
and measured-timing variables from README-4a before opening
this gate. The payload runs A2–A11 inside the existing process-group wrapper;
monthly durable saves and unchanged-row SKIPs support both chained jobs.

```bash
if [ "$RUN_4A" = 1 ]; then
  test -f "$KIT/README-4a.md" && test -s "$KIT/a4_payload.sh" || exit 3
  export B_REPLAY="${B_REPLAY:-$B/replay}"
  export PARENT_OUT DURABLE_ROOT KIT BASE FINQA SDPO B_REPLAY B_SOURCE
  export TRAINER_PY INFERENCE_PY SCORE_GPU FOUR_A_SLICE FOUR_A_TIMING
  export FOUR_A_STAGE_GPU_HOURS
  # Payload fsyncs this run's marker; continuation charging starts at job-start.
  # Optional bound limits: FOUR_A_ALLOCATION_GPUS, FOUR_A_SCORE_MAX_SECONDS,
  # FOUR_A_NLL_MAX_SECONDS, FOUR_A_HANDOFF_SECONDS (README-4a).
  run_stage bash -e "$KIT/a4_payload.sh" || exit $?
  "$TRAINER_PY" "$KIT/a4_handoff.py" --out "$PARENT_OUT/A4" --parent-out "$PARENT_OUT" \
    --durable "$DURABLE_ROOT" --trainer-python "$TRAINER_PY" || exit $?
  # RETURN 2: 4A kit compact archive, findings and measured costs. Never delay return 1.
fi
```

Operational totals from measured B stage timings: F/F+replay **20–26 minutes**
(2.67–3.47 GPU-h), D/D+replay **58 minutes** (7.73 GPU-h; D uses the replay proxy).
Answer-only is charged a full F slot. Sharing removes only the five probe generic
stages when W is already its replay sibling. No B finals are reused as legal runs.

| W | Core + new probe stages | Training GPU-h | Training wall h | Combined through return 1 GPU-h | Serial wall h |
|---|---:|---:|---:|---:|---:|
| F | 20 + 20 = 40 | 106.7–138.7 | 13.3–17.3 | 265.7–323.7 | 70.8–98.8 |
| F+replay | 30 + 15 = 45 | 120–156 | 15–19.5 | 279–341 | 72.5–101 |
| D | 30 + 20 = 50 | 285.3–301.3 | 35.7–37.7 | 444.3–486.3 | 93.2–119.2 |
| D+replay | 30 + 15 = 45 | 246.7–262.7 | 30.8–32.8 | 405.7–447.7 | 88.3–114.3 |

The M3b quick job adds 2.6–4 GPU-h (at most 2 wall h on two GPUs), not in the
table or totals below.
Combined adds once: 45 GPU-h for 27B targets on two GPUs (22.5 wall h allowance),
24 for C scoring/startup, 6 v0+, 10–20 v3 interiors, 6 fresh controls, 68–84
probe pools/scoring/startup: **159–185 GPU-h overhead**. Serial wall adds target
22.5 h, one-GPU scoring/preparation 24–48 h and 11 h startup/IO margin:
**57.5–81.5 h**. These are operational estimates based on measured B training,
not measured ContractNLI or per-panel timings. General legal inputs may be slower.
No allocation-equivalent GPU total is presented as elapsed time. Measure each
process, include failed attempts and idle reservation, and reprice at send.

Through return 1: **265.7–486.3 GPU-h, 70.8–119.2 serial wall hours**, depending
on W. At send, 4A binds the owner's 21 October approval:
**at least 600 partner GPU-h for F/F+replay; 800 for D/D+replay**, with the
owner's approval reference recorded in the binding. A higher explicitly approved
value is accepted. The registered formula
is a reported forecast only (350–400 for the F anchor, about 600 for the
50-minute D+replay anchor, 700 for the slower 58-minute D anchor).
The stop and worst-case preflight both use `approved_ceiling_gpu_hours`.
At these minimum approvals, combined with the matching return-1 W totals, the budget envelope through
return 2 is **865.7–1,286.3 GPU-h**, plus M3b's 2.6–4.

README-4a covers registered arrivals, permanent holdouts and general-v1:
**688 score cells, 41,575 uncached predictions**, removing 104,377 FinQA
predictions. At 1.22 s/question that is **14.1 scoring wall hours**; with eight
GPUs held, scoring costs **112.7 allocation GPU-h**. Including training and
the 80 GPU-h overhead allowance, the 4A operational forecast is
**314.2–494.2 GPU-h, 39.3–61.8 wall hours** at the 2.7–6.7 GPU-h stage anchors.
Combined through return 2: **579.9–980.5 GPU-h, 110.1–181.0 wall hours**, before
NLL, slower contexts, training contingency and M3b. These are forecasts.

The first job-script command durably records allocation start in
`DURABLE/job-starts/<job-id>-r<restart-count>.json`, before restore and identity
checks. At entry into 4A the payload fsyncs a marker keyed by Slurm job ID and
restart count (default 0), then saves it before binding. The first job that ever
starts 4A charges **8 × time from max(allocation start, marker) to job end**;
return-1 work before that marker costs 4A zero. A continuation whose restored
tree holds a 4A marker from any earlier run (an earlier job, or an earlier run
of the same requeued job) charges from **its own allocation start**:
restore, identity checks, SKIPs and re-scoring after a GPU change all serve 4A.
Use `SLURM_JOB_START_TIME`, else the durable first script record.

Allocation and attempt receipts distinguish requeued runs of the same job ID.
A requeue writes a fresh marker. It charges from its allocation start when any
earlier run already started 4A; otherwise from max(allocation start, its marker).
Its earlier run charges through its recorded end, or provisionally through the
later of its last 4A record and each unfinished attempt's start plus duration
limit, capped at the next run's allocation start. An actual end can replace
that estimate; **queue gaps cost zero**.
For an earlier job with unknown end, local `scontrol`, or `sacct` if `scontrol`
has no record, must report a terminal state and actual end, else the run is
pending (exit 3) before any new attempt.
RUNNING and COMPLETING are non-terminal; scheduled EndTime is never an end.
If scheduler history is unavailable, supply the actual end and restart count
with `a4_allocation.py` as in README-4a. Never charge through the next job's start.
Duration limits and worst-case preflight
enforce the approved ceiling. The recurrence split remains EXPLORATORY under
the verbatim 21 Oct registration deviation. C's registered ceiling remains 600;
no seeds, endpoints or attempts change.

Full offline rehearsal (fresh OUT, CPU fixtures only):

```bash
python "$KIT/message_dryrun.py" --out /tmp/message-standin --standin
```

The existing combined rehearsal retains its deliberately missing-kit gate
fixture and A/B/C/next paths unchanged. Rehearse the present M10/M11 payload too:

```bash
python "$KIT/a4_dryrun.py" --out /tmp/a4-message-D-standin --standin --recipe D+replay
python "$KIT/a4_dryrun.py" --out /tmp/a4-message-F-standin --standin --recipe F+replay
```

That runs every 4A row, 80 h of preceding return-1 work in the same job (excluded
from 4A), failure and cross-job resume paths, and compact archive
readback. Nothing sends partner messages.
