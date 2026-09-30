# Send 3: the dose-and-anchor test and K2b, one command each, one archive back each

Send 2's K1c showed that our reference dose (learning rate 1e-5; 20 steps on Spider, 40 on GSM8K and FinQA) makes
the 1.7B worse on every bed: its SQL style collapses on Spider and its answers stop parsing on GSM8K and FinQA. K4 is
held for that reason, and K3 is not rebuilt until a dose is found that does not break the model. This send finds
it, and runs K2b, which does not depend on the dose question.

## What runs

| # | campaign | what it answers | GPU-hours (8xH100) |
|---|---|---|---|
| 1 | `campaigns/k3-anchor.yaml` | Which dose does not break the model, and does a pull back towards the untrained model help at that dose? Eleven arms at three runs each on Spider and GSM8K. `README-k3-anchor.md` | about 90 to 120 |
| 2 | `campaigns/k2b-route.yaml` | Does the route of damage (reward-only RL against supervised fine-tuning) decide whether a lost ability comes back? Ten repeats per route. `README-k2b.md` | about 60 |

Run 1 first: its answer decides what K3 becomes. Run 2 when a node is free; it depends on nothing.

## Before the commands, once

```bash
export KIT=/work/continual-learning-kit/kit SDPO_DIR=/work/SDPO
export SPIDER_ROOT=/work/spider_data GSM8K_ROOT=/work/gsm8k
pip install qwen-vl-utils==0.0.14          # K2b's supervised route needs it; see README-k2b.md
```

## The commands

```bash
cd /work/continual-learning-kit && git fetch --tags && git checkout kit-batch2-v1
WORK=/work/k3-work  python $KIT/runner.py batch $KIT/campaigns/k3-anchor.yaml
WORK=/work/k2b-work python $KIT/runner.py batch $KIT/campaigns/k2b-route.yaml
```

Each as one job on one node, so every scoring shares a machine fingerprint. `batch` prepares and runs each campaign to
the end. Since this tag: every ordering between rows is an explicit `needs` edge (your dispatcher never has to infer
one, and the runner refuses a campaign that relies on file order); pilot dependencies are in `needs` too; a run that
fails stops the campaign only if a later row needs it, otherwise the campaign continues and the report records the
failed run; and `--seeds 0-4` on any command runs a reduced set and still writes a gated report.

## What to send back

```bash
python $KIT/collect.py --work /work/k3-work  --out send3-anchor.tar.gz
python $KIT/collect.py --work /work/k2b-work --out send3-k2b.tar.gz
```

Text and a few MB of answers; no weights.

## Running fewer seeds

`--seeds 0-4` on the batch command skips the rows of other seeds and everything that only they feed. The report row
still runs and records its verdict on what passed. A reduced run's error bars are wider; say so when you send it.
