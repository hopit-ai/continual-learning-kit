# 3A: Chemistry shortcut attribution — partner agent guide

This scores the base and two kept SDPO checkpoints on frozen original, letter
rotations, rank-balanced distractors and a rank-preserved redraw control. No
training. Registration: `resources/3a-shortcut/preregistration.md`; hashes and
pre-scoring validation live beside it. 51 affected / 159 unaffected questions;
12 scoring jobs, 3,744 total completions. Estimate 24 GPU-hours; allowance ceiling
50 including retries and idle allocation. Allocation requires owner approval
recorded by the manager in `docs/phase2/ledger.md` before launch.

## Prepare on CPU

Activate the working pilot scoring environment (vLLM, transformers, torch).
Set local absolute paths. Base is Qwen/Qwen3-8B revision
`b968826d9c46dd6066d109eabc6255188de91218`; use its pinned local snapshot.
The kept models are the final standalone Chemistry exports, not sequence exports.

```bash
export WORK=/absolute/path/to/new-3a-work
export CHEM3A_BASE=/absolute/path/to/pinned-Qwen3-8B
export CHEM3A_R1=/absolute/path/to/runs/sema-chem-r1-a1/hf-step40
export CHEM3A_R2=/absolute/path/to/runs/sema-chem-r2-a1/hf-step40
export KIT=/absolute/path/to/this-checkout/kit
python "$KIT/runner.py" plan "$KIT/campaigns/3a-shortcut.json"
python "$KIT/runner.py" prepare "$KIT/campaigns/3a-shortcut.json" --all
python "$KIT/runner.py" run "$KIT/campaigns/3a-shortcut.json" --row inputs
```

Inputs are shipped frozen JSONL in the authors' trainer schema; no parquet
conversion/download inside GPU time. `inputs` copies them to WORK. The wrapper
uses the shipped unchanged authors' MCQ scorer; no external SDPO checkout needed.
Base model download, if needed, is CPU preparation: `hf download Qwen/Qwen3-8B
--revision b968826d9c46dd6066d109eabc6255188de91218`. Set CHEM3A_BASE to the
resulting snapshot. Record how the local models map to these identities.

## Score after allocation approval

One physical GPU per row, deterministic eager, greedy, thinking off, B=2048,
context=12288. Same type/software across GPUs. No pilot or accuracy gates.
Sequential fallback on one allocated GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python "$KIT/runner.py" run "$KIT/campaigns/3a-shortcut.json" --all
```

With eight GPUs, run up to eight independent scoring rows concurrently using
`--row MODEL-PANEL`; MODEL is `base`, `r1`, `r2`, PANEL is `original`, `letters`,
`balanced`, `rank_preserved`. Prefer assigning all four rows of each model to
the same physical GPU (three concurrent lanes). Run rows sequentially within
a lane. The optional plain launcher `bash "$KIT/run_3a_lanes.sh"` now uses
OUT, TRAINER_PY and CHEM3A_BASE/R1/R2, as described in README-p3a.md. It runs
three lanes on GPUs 0,1,2; set CHEM3A_GPUS='0 1 2' for other visible devices.
Run the CPU reader separately after scoring.


## Find and fix

Runner command/log/verdict: `WORK/campaign/3a-shortcut/<row>/attempt-N/`.
Raw answers, token IDs, grades, model/machine identities and wall time:
`WORK/3a/eval/<model>-<panel>-aN/`. Frozen inputs and validation:
`WORK/3a/report-inputs/`. Table/warnings: `WORK/3a/report-reading/aN/`.

Fix infrastructure locally: paths, environments, GPU/scheduler plumbing, timeouts,
crashes. Rerun a failed row with the same command; runner keeps a new attempt.
Warnings are information, not disqualifying stops. No manager permission between
rows. Failed output remains in its original attempt; never delete evidence.
Do not change models, panel bytes, prompts, verifier, decoding, B or analysis
because of scores. If science must change to make anything work, retain the
originals, flag the exact departure and reason in the note for the owner.
Never hand-edit answers or metrics. Write outputs inside WORK. No tags/pushes,
publication or contacting anyone. CPU stand-in runs are technical only.

## Return

Rerun the reader after repairs, then collect (fresh archive name for each return):

```bash
python "$KIT/runner.py" run "$KIT/campaigns/3a-shortcut.json" --row report
python "$KIT/collect.py" --work "$WORK" --campaign "$KIT/campaigns/3a-shortcut.json" --out 3a-return.tar.gz
```

Send that archive plus a short note: scheduler/job IDs, actual charged GPU-hours
(including failures/idle time), model provenance, every local fix/science
change and any missing rows. Keep a source diff and untracked fixes in the note
folder `WORK/3a/report-local/` before collection. Weights stay on your node.
Reader from the returned archive alone (run from the repo, or extracted
`kit-snapshot` folder):

```bash
python -m kit.chemistry_shortcut_report --archive /absolute/path/3a-return.tar.gz --out /absolute/path/reading-3a
```

`reading.json` and `table.md` give paired gains and selective changes, question
bootstrap intervals, tokens per correct answer and descriptive lost-gain ratios.
The headline rank contrast and lost-gain fraction use Rank-preserved redraws as
the reference. Original-referenced changes also alter familiarity and band cues.
The reader reports training-gold band, median and string diagnostics by property;
surviving gains bound rank-cue dependence only, while the value-band cue remains
untested. Panel bytes stay frozen.
Missing rows warn and yield a partial report; complete the missing rows and rerun.
