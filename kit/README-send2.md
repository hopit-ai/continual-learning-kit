# Send 2: four packages, one command, one folder back

Everything below is already built, tested, and read by the reports; nothing needs a decision from you while it
runs. One tag, one `batch` command, one archive to send back.

## What runs, in order

| # | campaign | what it answers | GPU-hours (8xH100) |
|---|---|---|---|
| 1 | `campaigns/k3-dose-2.yaml` | Probe 2: does an 800-character length budget let stage A take more steps without the length drift probe 1 found? Four arms, three runs each. `README-k3-dose.md`, section "Probe 2" | about 30 to 36 |
| 2 | `campaigns/k3-replay.yaml` | K3 restarted at the reference dose, **ten seeds in-line**, every scoring on one node. Its gate is unchanged. `README-k3.md` | about 89 to 229, most likely 134 |
| 3 | `campaigns/k4-hints.yaml` | K4, hints on the stuck questions, now with its own `none` control arm; nothing borrowed from K3, ten seeds. `README-k4.md` | about 150 to 170 |
| 4 | `campaigns/k1c-grpo-baselines.yaml` | K1c, every job learned alone (8B ToolAlpaca full vs LoRA; 1.7B GSM8K and FinQA), ten seeds. `README-k1c.md` | about 155 to 305, most likely 195 |

Every package runs ten repeats of every arm and probe 2 runs three per arm: the effects we look for are about 5
points against a run-to-run spread of 3 to 4, and ten repeats halve the variance of every mean.

Roughly 425 to 740 GPU-hours in all, most likely about 520. Probe 2 is first because it is short and its answer may re-dose K3 later;
K3 does not wait for it.

## Before the command: environment, once

```bash
export KIT=/work/continual-learning-kit/kit SDPO_DIR=/work/SDPO
export SPIDER_ROOT=/work/spider_data GSM8K_ROOT=/work/gsm8k FINQA_ROOT=/work/FinQA/dataset
export MODEL_DIR=/work/models/Qwen3-8B NGPU=8                 # K1c Part A
export K0_REPORT=/work/k0-report/report.json                  # K1c reads your K0 report as its control
export HINT_BASE_URL=http://localhost:8000/v1 HINT_MODEL=<the model you serve>   # K4's hint rows only
```

Each campaign keeps its own work tree (`WORK`): the batch command below sets it per campaign, so the trees are
`/work/k3-work`, `/work/k4-work` and `/work/k1c-work` as the individual READMEs say. Spider and FinQA are on the
`data-v1` branch of this repository and on Hugging Face (`HopitAI/spider-finqa-data`) if the cluster still needs them.

**K4's hint rows need a large open model served on an OpenAI-compatible endpoint** (vLLM's `--served-model-name`
works; `README-k4.md` says which rows and how many requests, about 700 short ones). Start it before the batch
reaches K4, or run K4 separately afterwards with the same command shape; nothing else in the batch touches it.

**Disk.** K1c keeps a sharded checkpoint plus a merged copy per run; at twenty 8B runs that is about 1.1 TB if nothing is
deleted along the way (`README-k1c.md` says what can go once a run's scoring has passed). K3 and K4 are far smaller.

## The command

```bash
cd /work/continual-learning-kit && git fetch --tags && git checkout kit-batch1-v1
WORK=/work/k3-work  python $KIT/runner.py batch $KIT/campaigns/k3-dose-2.yaml $KIT/campaigns/k3-replay.yaml
WORK=/work/k4-work  python $KIT/runner.py batch $KIT/campaigns/k4-hints.yaml
WORK=/work/k1c-work python $KIT/runner.py batch $KIT/campaigns/k1c-grpo-baselines.yaml
```

`batch` prepares each campaign, runs it to the end, prints one PASS or FAIL line per campaign, and stops at the
first campaign that fails so nothing runs on a broken input. Rows whose inputs are produced by earlier GPU rows are
prepared right before they run, so the per-row workaround you used for K2 is not needed any more. `batch --plan`
prints every row and executes nothing.

**One rule: run each command as one job on one node**, so every scoring shares a machine fingerprint. If a report
shows two fingerprints, re-score both sides on one node with `--row` and re-run the report row.

## What to send back

```bash
python $KIT/collect.py --work /work/k3-work  --out send2-k3.tar.gz
python $KIT/collect.py --work /work/k4-work  --out send2-k4.tar.gz
python $KIT/collect.py --work /work/k1c-work --out send2-k1c.tar.gz
```

Each archive holds every report, scoring, delta, metrics log and run summary, with a hash manifest, and no
weights: text and a few MB of answers. Send the three archives; nothing else is needed.

## If something stops

`python $KIT/runner.py status <campaign>` names the row and the bar. A K3 gate that fails on `sql-improved-by-5`
alone, with `same-machine` at 1, is a real result at this dose and is what probe 2 is for; send the archive as it
is. Anything else, send the archive as it is too, with the status output: the archive carries `output.log` of
every attempt that did not pass.
