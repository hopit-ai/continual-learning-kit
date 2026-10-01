# Send 3: the dose-and-anchor test, K2b and a K1c rescoring, one command each, one archive back each

Send 2's K1c showed that our reference dose (learning rate 1e-5; 20 steps on Spider, 40 on GSM8K and FinQA) makes
the 1.7B worse on every bed: its SQL style collapses on Spider and its answers stop parsing on GSM8K and FinQA. K4 is
held for that reason, and K3 is not rebuilt until a dose is found that does not break the model. This send finds
it, and runs K2b, which does not depend on the dose question.

Your K1c archive then showed what "stop parsing" is: the trained 1.7B's answers grew from about 250 to 1,500 to 8,000
tokens during training (the trainer allows 8,192), and the kit scores every model at 2,048 new tokens. In the
eight of ten Part B scorings that collapsed, the answers cut at that cap account for the "wrong format" count almost one for one, and most
answers that finished on their own were right (on GSM8K seed 0: 59 of the 61 that finished). The third campaign
scores the same ten checkpoints and the untrained model again at the trainer's 8,192 cap, bed and panel: is the
arithmetic intact behind the drift? Its answer does not change the bar of record (the cost bar fails either way);
it decides how we read the GSM8K half of the anchor test.

## What runs

| # | campaign | what it answers | GPU-hours (8xH100) |
|---|---|---|---|
| 1 | `campaigns/k3-anchor.yaml` | Which dose does not break the model, and does a pull back towards the untrained model help at that dose? Eleven arms at three runs each on Spider and GSM8K. `README-k3-anchor.md` | about 90 to 120 |
| 2 | `campaigns/k2b-route.yaml` | Does the route of damage (reward-only RL against supervised fine-tuning) decide whether a lost ability comes back? Ten repeats per route. `README-k2b.md` | about 60 |
| 3 | `campaigns/k1c-rescore.yaml` | Was the 1.7B's arithmetic intact behind the length drift? The ten finished K1c Part B checkpoints and the untrained model scored again at 8,192 new tokens on their bed and on the panel, and a second K1c report from those scorings. Trains nothing; needs the finished K1c tree in `/work/k1c-work`. | 3 to 6 in all, one GPU per row |

Run 1 first: its answer decides what K3 becomes. Run 3 whenever one GPU is free; it is small and depends on nothing
but the K1c tree already on disk. Run 2 when a node is free; it depends on nothing.

## Before the commands, once

```bash
export KIT=/work/continual-learning-kit/kit SDPO_DIR=/work/SDPO
export SPIDER_ROOT=/work/spider_data GSM8K_ROOT=/work/gsm8k
export FINQA_ROOT=/work/FinQA/dataset K0_REPORT=/work/k0-report/report.json   # the rescoring, as in K1c
pip install qwen-vl-utils==0.0.14          # K2b's supervised route needs it; see README-k2b.md
```

## The commands

```bash
cd /work/continual-learning-kit && git fetch --tags && git checkout kit-batch2-v2
WORK=/work/k3-work  python $KIT/runner.py batch $KIT/campaigns/k3-anchor.yaml
WORK=/work/k1c-work python $KIT/runner.py batch $KIT/campaigns/k1c-rescore.yaml
WORK=/work/k2b-work python $KIT/runner.py batch $KIT/campaigns/k2b-route.yaml
```

`kit-batch2-v2` is `kit-batch2-v1` plus the rescoring campaign and the `--max-new-tokens` option it uses; nothing
in the other two campaigns changed.

Each as one job on one node, so every scoring shares a machine fingerprint. `batch` prepares and runs each campaign to
the end. Since this tag: every ordering between rows is an explicit `needs` edge (your dispatcher never has to infer
one, and the runner refuses a campaign that relies on file order); pilot dependencies are in `needs` too; a run that
fails stops the campaign only if a later row needs it, otherwise the campaign continues and the report records the
failed run; and `--seeds 0-4` on any command runs a reduced set and still writes a gated report.

## What to send back

```bash
python $KIT/collect.py --work /work/k3-work  --out send3-anchor.tar.gz
python $KIT/collect.py --work /work/k1c-work --out send3-k1c-rescore.tar.gz   # packs the K1c tree again, with the new eval8k, forgetting8k and report8k folders
python $KIT/collect.py --work /work/k2b-work --out send3-k2b.tar.gz
```

Text and a few MB of answers; no weights.

## Running fewer seeds

`--seeds 0-4` on the batch command skips the rows of other seeds and everything that only they feed. The report row
still runs and records its verdict on what passed. A reduced run's error bars are wider; say so when you send it.
