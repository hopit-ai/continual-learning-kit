# Package B: plain sequence and recovery jobs

Science: `resources/p3b/registration.md`. One command is one job; supply your own
sbatch. No campaign, runner, submission or automatic continuation. Keep an
absolute OUT on durable storage; put training scratch/checkpoints on local NVMe
if desired, then copy the entire run back before releasing the node.

```bash
export KIT=/absolute/checkout/kit OUT=/absolute/p3b
export TRAINER_PY=/absolute/trainer-venv/bin/python
export INFERENCE_PY=/absolute/inference-venv/bin/python
export SDPO=/absolute/pinned-SDPO
export BASE=/absolute/Qwen3-8B-snapshot TEACHER=/absolute/Qwen3.6-27B-snapshot
export A=/absolute/package-A-OUT FINQA=/absolute/v4_finqa
export V4_FINQA_PANEL=/absolute/v4/report-inputs/finqa-panel.jsonl
```

Use SDPO `7c457fc1b1f636ae794eb0362ba37d4743b06fbc`, base revision
`b968826d9c46dd6066d109eabc6255188de91218`, teacher revision
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`. The trainer venv runs all training,
8B generation and scoring; the inference venv (vLLM 0.18) runs only the 27B.
Replay checking also needs `langdetect` (the pinned checker imports it),
pyarrow and huggingface_hub. The local reader needs numpy.

Tier 1 goes first. Only overlap with A on disjoint GPUs and with sufficient
host memory/IO; every full training/recovery job uses all eight GPUs. Teacher
preparation uses two GPUs; replay generation and each scoring job use one.
**Every scoring, including re-scored base/incoming checkpoints, uses the same
one physical GPU and software environment.** Record durable paths and GPU IDs.

1. Prepare replay on the partner's network-enabled compute node. A's frozen
inputs must exist, but its task decision need not be complete. `--finqa-panel`
is v4's frozen `v4/report-inputs/finqa-panel.jsonl` (all 1,147 native rows).
No reference answers are sent to generation. This records the pinned source
parquet hash on first use, exclusions, candidate IDs and a durable raw journal.

```bash
"$TRAINER_PY" "$KIT/p3b_replay_data.py" --out "$OUT" --package-a "$A" --finqa-panel "$V4_FINQA_PANEL" --model "$BASE" --gpu 0
```

`--parquet /cached/train.parquet` permits offline reuse of that pinned download.
First 64 accepted examples form `replay/refresher-64.jsonl`; first 16 are nested.
Remaining examples form replay. Fewer than 65 is exit 2; no additional pool.
Optional base refresher controls are **excluded by default**; decide to include
both sizes, seed 101, before observing any refreshed outcomes.

2. Recovery: one command per D/F seed 101–103 and size 16/64 (12 jobs).
MODEL is the exact original `fsx_efa/v4-finqa-keep/exports/` final, never a
recovery copy. RECEIPT is its original v4 `env/export-identity-step40.json`;
its recorded origin must identify `D-finqa-alone-s101/hf-step40` (or matching F/
seed). Relocation is allowed; every model-file hash must match that receipt.

```bash
"$TRAINER_PY" "$KIT/p3b_recover.py" --out "$OUT" --arm D --size 64 --seed 101 --model "$MODEL" --incoming-receipt "$RECEIPT" --tokenizer "$BASE" --trainer-python "$TRAINER_PY" --sdpo "$SDPO" --gpu 0,1,2,3,4,5,6,7
```

Four updates, batch 16, fresh F AdamW, LR 1e-5 from update one. Exports are under
`runs/recover-D-n64-s101/hf-step4`. Score incoming as arm `D`, step 40, and output
as `recover-D-n64`, step 4; F/other sizes analogously. Score base as `base`, step 0.

3. Prevention: D+replay seeds 101–105; F+replay seeds 101–103 (eight jobs).
SCHEDULE is v4's existing seed-matched scheduled `train.parquet`, with its
original MANIFEST. It contains the same verified teacher targets as v4.

```bash
"$TRAINER_PY" "$KIT/p3b_replay_train.py" --out "$OUT" --recipe D+replay --route finqa --seed 101 --model "$BASE" --tokenizer "$BASE" --task-schedule "$SCHEDULE" --schedule-manifest "$MANIFEST" --trainer-python "$TRAINER_PY" --sdpo "$SDPO" --gpu 0,1,2,3,4,5,6,7
```

Frozen v4 arguments are in `resources/p3b/v4-recipes.json`, with source notes.
These programs call `sdft_entry.py` / `sft_entry.py` directly and merge saved
models. Existing v4 jobs can still use `run_sdft.sh` / `run_sft_f.sh` unchanged
with their original MODEL_DIR/DATASET or TRAIN_FILE/VAL_FILE/DATA_MANIFEST/SEED.
Manual export recovery uses the same command as v4:
`$TRAINER_PY -m verl.model_merger merge --backend fsdp --local_dir CHECKPOINT --target_dir EXPORT`.
D checkpoints: `sdft/global_step_K/actor`; F: `train/global_step_K`.

4. Score one checkpoint/panel set on GPU 0. Run both commands at every stage
boundary (FinQA and three general panels). Non-base scores bind an export
receipt (`--checkpoint-receipt`; new run receipts are also auto-discovered).

```bash
"$TRAINER_PY" "$KIT/p3b_score.py" --out "$OUT" --model "$MODEL" --checkpoint-receipt "$RECEIPT" --arm finqa-D+replay --seed 101 --step 40 --panel-set finqa --panel-root "$FINQA" --gpu 0
"$TRAINER_PY" "$KIT/p3b_score.py" --out "$OUT" --model "$MODEL" --checkpoint-receipt "$RECEIPT" --arm finqa-D+replay --seed 101 --step 40 --panel-set general-v1 --gpu 0
```

Tier 2 waits for A's complete ordered PASS selection; neither qualifying means
SKIP. Pending means exit 3. Extend full-pool targets, reusing A's raw requests:

```bash
"$INFERENCE_PY" "$KIT/p3b_targets.py" --out "$OUT" --package-a "$A" --model "$TEACHER" --tokenizer "$BASE" --gpu 0,1
```

Use `targets/schedule-SEED.jsonl` and `targets/manifest.json` for **every** recipe
and start. Repeat training command with `--route standalone` from BASE, and
`--route sequence --incoming-receipt RECEIPT` from original matched FinQA finals,
for D, F, D+replay, each seeds 101–105 (30 stages). D+replay sequence starts from
`runs/finqa-D+replay-sSEED/hf-step40`, receipt `export-40.json`. Never use recovery
exports. All five D+replay finals proceed regardless of tier-1 scores.
Score selected task at 0/10/20/30/40 with `--panel-set code` (or `math_screen`)
and `--package-a "$A"`; use arms `standalone-D`, `sequence-D`, etc. Also score
the selected task once as arm `base`, step 0, for its serving-token denominator. Score FinQA
and general at 0/40. Step 40 is primary; curves never select a winner.

Our-side `p3b_read.py --source OUT_OR_CSV_OR_ARCHIVE --out READING` writes every
bar, per-seed/question interval, family/churn table, curve and reset trigger.
After its written trigger only, run `p3b_reset.py --out "$OUT" --trigger TRIGGER
--recipe RECIPE --seed SEED --model ORIGINAL_FINQA_FINAL --incoming-receipt RECEIPT
--base "$BASE"` for all five seeds. Use priority D+replay, D, F, one branch only.
Export is `resets/RECIPE-sSEED/hf-reset`, receipt `export.json`. Score immediately
post-reset as `reset-RECIPE`, step 0, against `sequence-RECIPE`, step 0 pre-reset.
Then train `--route reset` with reset export/receipt and the identical schedule;
score 10/20/30/40 plus FinQA/general boundaries. No trigger means no reset.

Rough reservation: replay 15 GPU-h; recovery 21.3; prevention 141.6; tier-2
training 531; remaining teacher 40; extra scoring 70; infrastructure 45; optional
controls 3.6; reset 88.5. One full eight-GPU stage is ~2.2 wall hours (17.7 GPU-h
all-in proxy); recovery ~13 minutes/job proxy, ignoring startup. Coding speed is
unmeasured. Maximum 956 GPU-h; leave 44 headroom. Flag revised estimates before
exceeding 1,000; never cut seeds/attempts to fit. No budget gate is implemented.

Rerun the identical command/OUT: completed inputs, jobs, scores and journals
skip. Identity changes stop. A training stage that crashed is restarted from
step 0 with the same seed; its partial state is kept under `partials/` (as for
an interrupted FinQA slot). Note every restart in your return note.
Logs append to OUT/logs. Exit 0 done/SKIP, 2 correctness, 3 NOT READY; other
failures propagate a traceback. Never run duplicate copies of the same job.

Do not change the science: panels, prompts, schedules, targets, decoding,
verifiers, seeds, attempts, objectives, caps or decision rules. If a scientific
change is unavoidable, preserve original evidence and flag the exact departure
for the owner; it is outside this registered run.

```bash
# #!/bin/bash
# #SBATCH --gres=gpu:8
# #SBATCH --time=<SITE_TIME>
# /absolute/trainer/bin/python /absolute/kit/p3b_recover.py <arguments above>
"$TRAINER_PY" "$KIT/p3b_collect.py" --out "$OUT" --archive /absolute/p3b-return.tar.gz
```

Send the archive plus `OUT/return/scores.csv.gz`, `metrics.csv.gz`, source diff,
job IDs and a short note on fixes, provenance, missing work and costs. No weights,
caches, private tests or rollouts return. Large evidence omissions are hash-listed;
archive is capped at 28 MiB. Keep all full raw evidence/checkpoints durably.
