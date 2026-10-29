# FinQA main campaign: self-monitored host run

Plan v4 sections **15.21–15.22** control this package. Phase 0 v7 passed on job <redacted> (26/26 rows; 12.06 allocation GPU-hours). Qualification still runs first,
including the **full 1,441-question teacher corpus and rewrites** and initial
measurements. Main then starts automatically: **no PAUSE, proceed receipt or new
main-tag transition**. Read [FINQA-AGENT-GUIDE.md](FINQA-AGENT-GUIDE.md).

There are 23 forty-update standalone main slots: S/F/R/D at seeds 101–105 plus
D-R at 101–103. Primary seeds remain 101–103; 104–105 are replications. Models,
revisions, source data, panels, prompts, sampling, verifier, reward, scoring,
learning rates, batch, dose and arm definitions remain registered. D-R uses
verified frozen-initial-8B rewrites as privileged context. Qualification launches
all five arms at two updates. Main retains step-20 and step-40 exports, full
FinQA B/H and general-v1 scoring, and eight-trajectory mechanism probes.
D >= 58/1,147 and D minus S >= five points per primary seed are analysis bars;
they never gate execution. Low coverage warns; training uses surviving verified
intersection questions with the same exposure schedule and batch/dose.

## Storage, cache and limits

Choose TASK_ROOT on large shared Lustre visible from login and compute nodes.
**1,879 GiB is capacity advice, not a refusal threshold.** Check `quota -s`,
`lfs quota -u $USER` or `mmlsquota`. The capacity calculation includes all 61 BF16
exports (46 main, five qualification, ten restoration), old/new trainer-state
coexistence, protected metadata and cache allowances. Larger snapshots may need
more. Failed retirement retains states and warns, so actual space may exceed
this advisory estimate. Only insufficient free bytes for the next checkpoint
save/export stops that writer. ENOSPC is an actual disk-exhaustion failure.

All 61 exports and evidence are retained. Trainer tensors retire only after
unchanged identities and durable proof-bound conditions permit deletion. An
interrupted deletion replays its verified journal; an unsafe unlink never happens.
Initial full-state hashing and step-20 CPU merge/reload use the trainer row’s cap,
with no separate retirement deadline. Retirement reads at most the registered
8 GiB of evidence at an unmeasured conservative 64 MiB/s plus 32 s overhead:
160 s base, **1,600 s after the shared multiplier**. Cached state/export identities
reuse unchanged device/inode/size/mtime/ctime fingerprints and durable receipts.

`TIMING_MULTIPLIER=10` applies once to infrastructure limits, never elapsed
measurements, scientific settings or budget prices. Prepared-byte rehash allowance
is `10 × max(60, ceil(1.5 × measured seconds))`; 84 s becomes 1,260 s. A projection
that cannot fit the site window warns; it does not reject the campaign. v7 settings
reach main engines: eager teacher/rewriter/scorer, RPC 3,000 s, teacher/NCCL 6,000 s,
Gloo 18,000 s and training NCCL 6,000 s. Batch-invariance, determinism, metrics,
versions and scoring-environment findings are warnings. A genuinely failed
operation still leaves FAIL in its original attempt; independent work continues.

PIP_CACHE_DIR, TRITON_CACHE_DIR, VLLM_CACHE_ROOT and TORCHINDUCTOR_CACHE_DIR use
stable `TASK_ROOT/cache` (WORK parent/cache fallback). Fresh WORK reuses wheels
and compilation. TMPDIR remains per job; short Ray/vLLM sockets remain in job
`/tmp`. Caches are not evidence and are never archived. There is **no second download**
or phase-0 technical checkpoint used as a scientific initialisation.

The **1,000 GPU-hour ceiling** counts actual allocation width × wall time,
including phase-0 spend, idle time and failed operations. The historical planned
100-hour qualification block is advice. Projection fit and reservation size never
refuse a row or stage. Every paid row’s wall-clock backstop is clipped to remaining
actual ceiling and the live site allocation end. At actual ceiling, the flow stops.

## Prepare and submit once

The manager generates both manifests from the read-only phase-0 return:

```bash
python3 kit/v4_finqa_ingest.py --archive PHASE0_RETURN --local-diff PHASE0_DIFF --out EMPTY_DIRECTORY
```

Review source correspondence and stage the generated `campaigns/v4-*.yaml` in
the immutable main package. They bind C1’s measured two-step launch × 1.5,
registered floors and phase-0 actual spend. The same already-verified phase-0
model revisions/shard hashes supply main registration; no new model is selected.
Unbound engineering templates cannot establish a live source identity.

Use the native host venv route confirmed in the pilot. Preserve its exact
`export PATH=/home/<USER>/envs/train/bin:$PATH` line, not a container wrapper.
On the login node:

```bash
export PATH="$HOME/envs/train/bin:$PATH"
export TASK_ROOT=/YOUR_SHARED_LUSTRE/v4-finqa
mkdir -p "$TASK_ROOT"
export KIT="$TASK_ROOT/continual-learning-kit/kit"
export V4_KIT_TAG="$TAG" PYTHONPATH="$KIT/.." V4_FINQA_CAMPAIGN=1
export WORK="$(mktemp -d "$TASK_ROOT/work.XXXXXX")"
export ACCOUNT=ASSIGNED_ACCOUNT PARTITION=ASSIGNED_PARTITION QOS=ASSIGNED_QOS
python3 "$KIT/v4_phase0_carry.py" --work "$WORK" --archive "$PHASE0_ARCHIVE"
finqa_host_exports() {
    local host_exports
    host_exports="$(python3 "$KIT/v4_finqa_site.py" "$1" --work "$WORK")" || {
        printf '%s\n' 'STOP: host setup failed; correct it and retry.' >&2
        return 2
    }
    eval "$host_exports"
}
finqa_host_exports inherit && {
    source "$V4_TRAINER_ACTIVATE"
    python "$KIT/v4_prepare.py" --work "$WORK" --phase qualification
    python "$KIT/v4_prepare.py" --work "$WORK" --phase main
}
# Request the site MaxTime: 10,080 minutes. SITE_MINUTES is its captured value.
python "$KIT/v4_allocation.py" --work "$WORK" --phase qualification --stage qualification \
  --minutes 10080 --site-minutes "$SITE_MINUTES" --out "$WORK/allocation/flow.sh" \
  --partition "$PARTITION" --qos "$QOS" --account "$ACCOUNT" --activate "$V4_TRAINER_ACTIVATE"
```

Copy your same `k8b_pilot_run.sbatch` to `$WORK/allocation/flow.sbatch`. Replace
only its header with `flow.header.sh`; after the pilot’s unchanged host-venv lines,
run `bash "$WORK/allocation/flow.sh"` directly in the batch process. Submit once:

```bash
python "$KIT/v4_finqa_submission.py" --work "$WORK" \
  --plan "$WORK/allocation/flow.json" --script "$WORK/allocation/flow.sbatch"
```

**One native eight-GPU allocation runs qualification then all main stages.**
If the site window ends, `v4_finqa_flow.py` submits that same retained native wrapper
with `--dependency=afterany:$SLURM_JOB_ID`. Continuations use the same WORK,
retain all attempts/accounting, and skip rows already passed. No owner intervention
or proceed file is needed. Site MaxTime limits each job; the actual 1,000-hour
backstop limits cumulative work. Keep the copied wrapper at that path for chaining.
The initial plan is an observation; live accounting controls continuations.
Request **10,080 minutes**, the site's MaxTime, and watch progress reports for
actual spend, stalls and warnings. Scoring runs on **one GPU of the eight by
design**; the allocation continues charging all eight GPUs during scoring.

## Monitor, fix and return

Warnings print as `WARNING` and are retained under `$WORK/v4/warnings/*.json`.
Progress reports live in `v4/report-{qualification,main}/reading.json`, timings
in their `timings/`, runner attempts under `campaign/v4-<phase>/<row>/attempt-N/`,
and continuation receipts in `v4/report-flow/submissions/`. Return qualification
progress reports while main continues. Reports preserve failed operations and
analysis bars; telemetry or coverage gaps warn without disqualifying later work.

After a local fix, rerun the failed row in the SAME WORK:

```bash
python "$KIT/runner.py" run "$KIT/campaigns/v4-main.yaml" --row ROW_ID
# For a qualification row, substitute v4-qualification.yaml.
python "$KIT/v4_finqa_flow.py" --work "$WORK"
```

The runner creates a new append-only attempt without an owner-reviewed retry file.
It skips passed rows even after a dependency retry. Valid saved optimizer state
resumes; failed/partial state and logs are archived, never manually deleted.
Infrastructure fixes are free. Change science only if nothing else works, flag it
in the note, and let the owner decide whether that run counts. Never fabricate or
hand-edit outputs, metrics, verdicts or reports. Never delete evidence.

Collect progress/final evidence inside WORK, then return it with job IDs, warnings,
a note and the diff against the original tag:

```bash
python "$KIT/collect.py" --work "$WORK" --out "$WORK/main-return-UNIQUE.tar.gz"
git -C "$KIT/.." diff "$V4_KIT_TAG" > "$WORK/main-local.diff"
```

The collector excludes weights/checkpoints and caches; it preserves the bound
phase-0 source, warnings and retry evidence. Read with
`python scripts/read_v4_campaign.py ARCHIVE`. Only primary seeds determine the
claim; replication, D-R and general-v1 losses remain descriptive. The optional
teacher-depth hook remains omitted as registered; no hooks are installed.

CPU rehearsals use explicit scheduler/model/GPU stand-ins and cannot authorize
science. Run `python kit/simulate_v4_finqa_host.py --out EMPTY_DIRECTORY` for the
native flow rehearsal. Always delete every pytest `--basetemp` after a run.
