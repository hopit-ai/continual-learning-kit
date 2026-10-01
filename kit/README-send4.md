# Send 4: three campaigns, one tag, two archives back

Tag `kit-batch3-v1`. This replaces send 3 (`kit-batch2-*`), which was never sent.

## Why this send

Your K1c archive showed that in eight of the ten 1.7B runs the answers had grown during training (the trainer allows
8,192 new tokens) and were cut at our 2,048-token scoring cap. A cut answer is scored as it stood, so that score
cannot separate an answer that was wrong from one that did not end. We then found that our plain-RL recipe (learning
rate 1e-5, minibatch 32) is the SDPO paper's weaker "on-policy" baseline; its tuned baseline is 1e-6 with minibatch 8.
So this send measures three things:

| # | campaign | what it measures | where | GPU-hours (estimate) |
|---|---|---|---|---|
| 1 | `campaigns/k1c-audit.yaml` | The ten K1c checkpoints generated once at 8,192 tokens with their token ids kept; the score at every shorter budget read from prefixes. Trains nothing. | `/work/k1c-work` | 4 to 8, on ONE GPU |
| 2 | `campaigns/k1c-control.yaml` | The 1.7B on GSM8K again, three runs each of: the lower learning rate; a 2,048-token training limit with reward only for answers that finish; both. | `/work/k1c-work` | 40 to 50 |
| 3 | `campaigns/k8b-pilot.yaml` | Qwen3-8B on the SDPO authors' Chemistry and ToolAlpaca tasks, each alone and each on top of the other, under four recipes (tuned GRPO, K1c's GRPO, SDPO, SDPO with a frozen teacher). 24 training runs. | `/work/k8b-work` (new) | 170 to 200 |

The GPU-hours are estimates from your K0 and K1c step times, not bounds: training rows hold all eight GPUs, scoring
rows hold one. **If campaign 3 has used 300 GPU-hours and is not finished, please stop it and send what there is.**

Each campaign opens with pilots that stop it if something is wrong. Nothing in this send asks you to choose, join
or interpret anything: the arms, the repeats and the order are fixed in the files. The four recipes of campaign 3
differ in learning rate, minibatch and teacher at once and have one or two runs each: it is a pilot that says where
to look, not a comparison of algorithms.

## Before the commands, once

```bash
export KIT=/work/continual-learning-kit/kit SDPO_DIR=/work/SDPO NGPU=8
export GSM8K_ROOT=/work/gsm8k FINQA_ROOT=/work/FinQA/dataset K0_REPORT=/work/k0-report/report.json
cd /work/continual-learning-kit && git fetch --tags && git checkout kit-batch3-v1
```

Same environment as K1c; no new package to install. Campaign 3 preprocesses Chemistry inside your SDPO checkout the
way K0 preprocessed ToolAlpaca (it writes two parquet files under `datasets/sciknoweval/chemistry/`).

**Please keep** the ten K1c Part B checkpoints (`/work/k1c-work/runs/{gsm8k,finqa}-seed*/hf-step40`),
`/work/k1c-work/models/Qwen3-1.7B` and `/work/k1c-work/data/gsm8k-1280/` until campaigns 1 and 2 have finished.

## The commands

```bash
WORK=/work/k1c-work python $KIT/runner.py batch $KIT/campaigns/k1c-audit.yaml
WORK=/work/k1c-work python $KIT/runner.py batch $KIT/campaigns/k1c-control.yaml
WORK=/work/k8b-work MODEL_DIR=/work/models/Qwen3-8B python $KIT/runner.py batch $KIT/campaigns/k8b-pilot.yaml
```

The three lines are independent of each other: any order, or different nodes (the first two share `/work/k1c-work`,
so run those two on the node that holds it).

**One job at a time on a node.** A training row takes all eight GPUs and a scoring row takes one. The runner above
runs rows one after another, so it never overlaps them. If your own dispatcher runs rows in parallel, it must not
start a training row while another training row or a scoring row is running on the same node.

**Scoring must stay on one physical GPU.** Every scoring row carries `CUDA_VISIBLE_DEVICES=0` and follows the scoring
before it, because the reports refuse to compare scorings whose machine fingerprints differ and the fingerprint
includes the GPU. Under Slurm, index 0 is the first GPU of the allocation, so all scoring rows of one campaign must
run in the same allocation on the same node. (In K1c your scorings ran on GPUs 0 to 7 with one machine id; if you
changed how the fingerprint is computed, tell us and we will match it.)

**Disk for campaign 3: about 560 GB.** Its first row refuses to start with less than 650 GB free under `$WORK` and
says so. Each run keeps its merged model (about 17.6 GB). The trainer's own checkpoint (about 33 GB a run) is
deleted as soon as a complete merged model is on disk; nothing here resumes training. A retried run adds a model.

**If a row fails.** After the pilots, a failed row costs only the rows that need it: the runner refuses those,
runs everything else, writes the reports with what there is, and exits non-zero. To retry, fix the cause and run
the same line again: rows that passed are skipped, the failed row runs as a new attempt, and every row that read
its output (its scorings, the report) is run again by itself. If a pilot fails, everything after it stops; send
the archive as it is.

## What to send back

```bash
python $KIT/collect.py --work /work/k1c-work --out send4-k1c.tar.gz
python $KIT/collect.py --work /work/k8b-work --out send4-k8b.tar.gz
```

Text only, no weights: answers, token ids, sweeps, reports, each run's command and metrics, and the log of every
row that did not pass. The first packs the K1c tree again with the new `k1c/cap8192/` and `k1c/control/` folders
(roughly 150 MB before compression); the second is new. If a line starting `INCOMPLETE:` is printed, a file was
over 50 MB and only its end is in the archive: please send that file separately. If an archive is too large to
move, send the `report-*` folders first.

## Running fewer seeds

`--seeds 1` on the third line runs only the first run of each recipe (16 of the 24 training runs) and still writes
the report. Say so when you send it. Please run the first two lines whole.
