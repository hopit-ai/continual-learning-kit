# v6a: completed K/CR training, fresh scoring and text return

This staged send runs K/CR only. Core M/X/S, PD and SP follow in v6b.
No core files, submissions, bindings, SEND/date/review or PF-tier gates are needed.
Ship the full kit tree: the text vocabulary imports other families' calculators,
but none of their runtime inputs are required. Scientific rules remain unchanged.

Reuse the existing K/CR root, base binding, frozen assignment, HellaSwag freeze,
pools and completed C0/K/G/CR training exports/receipts (all stages, seeds 101–103).
Do not retrain; if intermediate curve exports were pruned, run anyway — primary
verdicts need only base, step 0 and step 40.
Keep tokenizer files, the FinQA panel root and local HellaSwag parquet cache staged.
CPU ready jobs record intermediate exports (10/20/30) as PRESENT or MISSING-EXPORT
and hash present inputs before GPU admission. Missing base/incoming/final exports
exit PENDING. Missing intermediates omit only their diagnostic scoring calls;
affected AUC/step-20 diagnostics are INCOMPLETE, with no interpolation.
Guards come only from REF rows; M3-fv/kv validation rows are not needed in v6a.
The near-tie and reference-scorer rulings stay unchanged.

Create local settings with only these required fields (use absolute local paths):

```json
{
  "kcr_manifest": "/staged/kcr/m3-launch.json",
  "kcr_root": "/staged/kcr",
  "control_root": "/staged/v6a-control",
  "inference_python": "/venv-inference/bin/python"
}
```

The existing K/CR manifest supplies nodes and the unchanged scoring commands.
Keep control_root separate from kcr_root; keep all jobs on nodes that can read
that restored root. Optional report_inputs/text_out override generated defaults.
Export the authority SHA supplied in the owner's message; never compute it locally:
The manager regenerates that authority from the final committed build before send.

```bash
export KIT=/path/to/kit JOB_PY=/venv-inference/bin/python
export KIT_V6_AUTHORITY_SHA=SHA_FROM_THE_OWNERS_MESSAGE
export COMBINED_MANIFEST=/staged/v6a-control/launch.json
"$JOB_PY" -I "$KIT/combined_send.py" build-v6a --settings settings.local.json \
  --out "$COMBINED_MANIFEST" --inventory /staged/v6a-control/inventory.md
```

Inspect that inventory. It contains exactly 14 rows, with dependencies below.
No script submits, polls or retries; the operator issues each sbatch explicitly.
Add partition, account, time limit and `-w NODE` from the K/CR node list to every
command. CPU rows omit the GPU flag for sites that reject `--gpus=0`.
Use the same exported environment in each allocation. Submit in this exact order;
replace job-ID variables with sbatch's returned numeric IDs (use --parsable).

| Row type | Count | Resource line | Dependency |
|---|---:|---|---|
| REF-m3-assign | 1 | `--nodes=1 --cpus-per-task=8` | none |
| M3-rescore-SEED-ready | 3 | `--nodes=1 --cpus-per-task=8` | none |
| REF-m3-SEED-prepare | 3 | `--nodes=1 --cpus-per-task=8` | afterok:assign |
| M3-rescore-SEED | 3 | `--nodes=1 --gpus=1 --cpus-per-task=8` | afterok:ready |
| REF-m3-SEED | 3 | `--nodes=1 --gpus=1 --cpus-per-task=8` | afterok:prepare |
| REPORT-text | 1 | `--nodes=1 --cpus-per-task=8` | afterany:all 6 GPU jobs |

Set KCR_NODE to a node from the K/CR list. For each SEED in 101, 102, 103:

```bash
sbatch --parsable -w "$KCR_NODE" --nodes=1 --cpus-per-task=8 \
  "$KIT/jobs/v6a/run.sh" REF-m3-assign
sbatch --parsable -w "$KCR_NODE" --nodes=1 --cpus-per-task=8 \
  "$KIT/jobs/v6a/run.sh" M3-rescore-101-ready
sbatch --parsable -w "$KCR_NODE" --nodes=1 --cpus-per-task=8 \
  --kill-on-invalid-dep=yes --dependency=afterok:$ASSIGN_JOB \
  "$KIT/jobs/v6a/run.sh" REF-m3-101-prepare
sbatch --parsable -w "$KCR_NODE" --nodes=1 --gpus=1 --cpus-per-task=8 \
  --kill-on-invalid-dep=yes --dependency=afterok:$READY_101 \
  "$KIT/jobs/v6a/run.sh" M3-rescore-101
sbatch --parsable -w "$KCR_NODE" --nodes=1 --gpus=1 --cpus-per-task=8 \
  --kill-on-invalid-dep=yes --dependency=afterok:$PREPARE_101 \
  "$KIT/jobs/v6a/run.sh" REF-m3-101
# Repeat the four seed rows for 102 and 103, recording their returned job IDs.
sbatch --parsable -w "$KCR_NODE" --nodes=1 --cpus-per-task=8 \
  --kill-on-invalid-dep=yes --dependency=afterany:$M101:$M102:$M103:$R101:$R102:$R103 \
  "$KIT/jobs/v6a/run.sh" REPORT-text
```

REF-m3-assign generates the exact checkpoint list: base plus four arms times
three stage finals at k40, per seed (13 processes/model loads, both panels each).
Each rescore job scores all four arms, all stages and all present panels (up to
198 calls). CPU ready fixes the command subset by export presence alone.
It scores the base five times per seed, including four stage-1 incoming calls;
this valid duplication costs extra GPU time. Measured allocation costs are unknown.
Charge loading, failed attempts, tails, queue/restore and retries separately.

One allocation uses one measured GPU and one nonce, with one shared lease/idle
check and one nonblocking watcher. Rescore uses K=3 at .31 only for the pinned
8B with maximum model length <=12288; otherwise K=1. HF reference uses K=2.
REF utilisation is unmeasured: merging panels saves loads but does not establish
the rolling 70% requirement. Keep the CPU-first model load for unchanged numerics;
paste back the watcher utilisation line in Part 3, including unknown telemetry.
Different seeds may reuse the same physical GPU in separate allocations.
Every attempt writes a fresh nonce tree; no cells are skipped across attempts.
Readers select the newest complete single-nonce, single-UUID attempt per seed;
older and partial attempts stay for audit. Mixed evidence is UNVERIFIED-HARDWARE.
No sealed attempt is PENDING; corrupt receipts are UNVERIFIED for that seed.

On PENDING (3/75) or failure, fix the missing input and manually resubmit that
row; rerun its CPU ready/prepare row if input metadata changed. Reconnect scheduler
dependencies to the replacement job IDs. Then submit a new REPORT-text attempt.
Every dependent sbatch uses `--kill-on-invalid-dep=yes`: failed CPU helpers must
cancel their afterok descendants instead of leaving DependencyNeverSatisfied jobs.
REPORT-text remains afterany on all six GPU jobs and produces a partial return.
No GPU job waits.
Watcher failures log a warning and run/continue the child unwatched; missing
telemetry is unknown and does not prove the rolling 70% utilisation requirement.

Paste all three RETURN-TEXT-*-of-3.md files from the reported fresh output tree,
located by text_out in control_root/state/REPORT-text.json; per-attempt records
are in control_root/state/attempts/ROW/NONCE.json. Paste the files
in order, including KIT headers and checksums. Part 1 carries K and CR verdicts;
its T|AUTH header must match the SHA supplied in the owner's message.
Part 2 the registered endpoints; Part 3 costs, job status and utilisation.
Keep FULL detail, local maps, old score attempts and watcher receipts on the node.
