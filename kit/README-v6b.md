# v6b operator entry

v6b runs core M/X/S (including R/RA), PD, SP, 4A, their reference lanes,
collectors and the combined pasted-text report. v6a is already running:
**do not rerun v6a, K/CR training, M3-rescore or REF-m3**. Keep its local
K/CR root, base/incoming/final exports, assignments and complete attempt trees.
The v6b report reads that same root; SP reuses its retained parent exports.
Missing v6a outputs leave dependent readers PENDING; never retrain to fill them.

Unpack v6b into a separate code directory. Keep v6a's code and authority file
unchanged until its submitted jobs finish; point v6b at its existing K/CR data root.
Use a fresh shared `control_root` and `text_out` for v6b, visible on every host.

Use the full shipped kit. Scientific procedures and local input templates:
[core](README-message2.md), [combined](README-combined.md),
[PD](resources/pd/README.md), [SP](jobs/message3-sp/README.md),
[4A](README-4a-v6b.md). The 2 November 4A amendment uses seeds 101–103,
paired t df 2, 3/3 positive seeds and the unchanged 800 GPU-h ceiling.
4A is required; all 197 configured rows must appear in the inventory and return.

## Bind and inspect on CPU

Copy the authority hash supplied in the owner's tagged send message:

```bash
export KIT_V6_AUTHORITY_SHA='<KIT_V6_AUTHORITY_SHA from the send message>'
```

Never derive launch approval from the authority file itself. The manager seals
`resources/v6/infrastructure-update.json` last from the committed source.
Absent, wrong or stale seals refuse amended code. Retain this hash and check
that Part 1 prints the identical `T|AUTH|` value.

Prepare the core binding and frozen launch, PD's three plans and SP inventory
as in README-combined. Keep the existing K/CR manifest and root as input
provenance only. In combined settings, set `a4_config` to the absolute local
JSON described in README-4a-v6b, and `a4_nodes` to its storage hosts. Its durable
output must be separate from the core, K/CR, PD, SP and control roots.
Set `report_inputs` to a local JSON containing M/X/S, KCR, PD and SP reader
sources and their job/utilisation inputs as in README-combined. The dispatcher
adds 4A and all shared utilisation groups automatically. Retain v6a's receipts.
Set `v6b_send` to v6b's own record from the owner's tagged SEND message:
`status: "SEND"`, `date` (actual SEND date, YYYY-MM-DD), `review_file` (local
path to that v6b review) and `review_sha256` (its supplied SHA256). The dispatcher
checks that file's bytes and requires a valid date, with no 2 November cutoff.
The historical `tier_send` record does not authorize v6b PD/SP. Set these values
before building and tier preflight; changes require rebuilding and preflight.

```bash
python3 -I "$KIT/combined_send.py" build-v6b --settings "$SETTINGS" \
  --out "$SEND/v6b.json" --inventory "$SEND/v6b-inventory.md"
export COMBINED_MANIFEST="$SEND/v6b.json"
export JOB_PY="$INFERENCE_PY"
```

Inspect every resource and dependency. The inventory contains no K/CR or REF-m3
submission rows. `reused_v6a_requires` records removed scheduler edges whose
outputs must already exist; each consumer still checks its own real inputs.
Building submits nothing. Rebuild after changing bound settings/configs.

## Ordered manual submissions

The operator supplies sbatch, host constraints and job-ID dependencies. Submit
one row per allocation through `bash "$KIT/jobs/combined/run-task.sh" ROW`.
Every row requests one node. Use the row's CPUs exactly; CPU jobs omit `--gpus`
if the site rejects `--gpus=0`. Each resource line below is one row type:

```text
core CPU light stage/read:  --nodes=1 --cpus-per-task=4
core CPU preparation/export: --nodes=1 --cpus-per-task=8
core score/generation lane: --nodes=1 --gpus=1 --cpus-per-task=8
core TP2 teacher:           --nodes=1 --gpus=2 --cpus-per-task=8
core train/qualification:   --nodes=1 --gpus=8 --cpus-per-task=32
tier CPU preflight:         --nodes=1 --cpus-per-task=8
PD prep:                   --nodes=1 --cpus-per-task=8
PD pre/post score:          --nodes=1 --gpus=1 --cpus-per-task=8
PD train:                  --nodes=1 --gpus=8 --cpus-per-task=32
PD finalize/export:        --nodes=1 --cpus-per-task=16
PD collect/read:           --nodes=1 --cpus-per-task=4
SP verify/prepare/read:    --nodes=1 --cpus-per-task=8
SP panel lane:             --nodes=1 --gpus=1 --cpus-per-task=8
REF assign/prepare:        --nodes=1 --cpus-per-task=8
REF complete seed lane:    --nodes=1 --gpus=1 --cpus-per-task=8
4A CPU phases:             --nodes=1 --cpus-per-task=8
4A native train:           --nodes=1 --gpus=8 --cpus-per-task=32
4A complete score lane:    --nodes=1 --gpus=1 --cpus-per-task=8
REPORT-text:               --nodes=1 --cpus-per-task=8
```

1. Submit core and tier CPU preflights on their configured storage hosts.
2. Submit all core rows on their frozen slots, following `requires` and
   `dependency`. CPU preparation precedes each GPU phase; CPU export follows
   native training. Keep independent collectors on their afterany edges.
3. Record the core submissions (no scheduler call):

```bash
python3 -I "$KIT/combined_send.py" core-submitted --manifest "$COMBINED_MANIFEST"
```

4. **PD is ON HOLD in v6b.** Do not submit any PD row until we confirm your
   cluster's 70% rule (metric, window, start-up grace). PD's full-precision
   scorer measured about 38% busy on one H100 (one copy; two do not fit).
   When released, run unchanged from this kit:
   Submit PD prep → pre → train → finalize → post → collect for each seed,
   then PD-read. Keep each seed on its bound storage host and reserve spool
   space as in the PD runbook. A failed seed follows its full restart rules.
5. Submit SP's explicit afterany graph. Missing entries do not suppress the
   other ready entries. Keep its independent inventory verification receipts.
6. Submit 4A's monthly graph: context → arrival prepare/score → ledger →
   eligible prepare/train/export/score-prepare/score/finish → month closure.
   Then comparison-prepare → comparison → read (afterany) → collect.
7. Submit non-m3 REF assign/prepare and each complete seed scoring lane after
   its listed native checkpoint frontier. Pin all non-m3 REF phases to the first
   configured core host (`node` in each row). Reuse v6a REF-m3 without submission.
8. Submit REPORT-text afterany **all** its listed frontier IDs, including
   reference lanes, native collectors/readers, PD-read, SP-read and A4-collect.
   Omit PD-read while PD is held. Inventory lane K is a maximum (K=1 if
   commands depend on each other or the engine is unqualified).

Every dependent sbatch uses `--kill-on-invalid-dep=yes`, with afterok or
afterany exactly as listed.
For example: `--dependency=afterok:JOB_ID --kill-on-invalid-dep=yes`.
This cancels invalid descendants so afterany reporting can produce a partial
return. PENDING requires manual staging/resubmission and new dependency IDs.
No job polls for prerequisites while holding GPUs. Preserve failed attempts.

GPU leases and one watcher cover each allocation. Measured identity is required
within each complete contrast; a new job rescores the whole contrast. Qualified
8B vLLM lanes use K≤3, PD/reference K≤2, and 4A HF NLL K=1. CPU/inference may
share a host; eight-rank training owns its eight GPUs. Rolling utilisation
above 70%, startup/tail costs and memory fit still require real-node evidence.
PD's 63–98 GPU-h and 4A's 400–450 proxy are planning figures; charge failures
and full held allocation time against the applicable separate approvals.

## Pasted-text return

Each REPORT-text attempt writes `text_out/<nonce>/`; paste from the latest
completed attempt. Watchers write shared `control_root/utilisation/<group>/`;
the reporter includes every group and counts started GPU rows, including retries.
Keep those start markers and failed watcher attempts for partial reports.

Paste every numbered ESSENTIAL part from the latest report, including 4A and
Part 3 utilisation lines for all allocated families. Keep FULL parts and raw
receipts locally; paste requested supplemental parts if a checksum/reader asks.
Check the tag/commit, `T|AUTH|`, part counts and checksums. Partial/PENDING and
UNVERIFIED-HARDWARE results must remain visible. Do not omit 4A when inputs exist.

Also please paste Paper 1 Chemistry (3A) package A training logs for checkpoints
r1 and r2: per-step loss/reward and run configs, as text or a summary. No
checkpoints. Redact nothing on your side; we redact before any supplement.
