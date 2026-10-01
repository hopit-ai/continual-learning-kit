> **RETIRED, NEVER SENT (1 October 2026).** This runbook and its tags `kit-batch2-v1` to `-v3` are superseded by
> `README-send4.md`. The rescoring campaign it names was replaced by `campaigns/k1c-audit.yaml`. Kept only as a record.

# Send 3: the dose-and-anchor test, K2b and a K1c rescoring, one command each, one archive back each

Send 2's K1c showed that our reference dose (learning rate 1e-5; 20 steps on Spider, 40 on GSM8K and FinQA) makes
the 1.7B worse on every bed: its SQL style collapses on Spider and its answers stop parsing on GSM8K and FinQA. K4 is
held for that reason, and K3 is not rebuilt until a dose is found that does not break the model. This send finds
it, and runs K2b, which does not depend on the dose question.

Your K1c archive then showed what most of "stop parsing" is. The kit scores every model at 2,048 new tokens and the
trainer allows 8,192. In eight of the ten Part B runs the answers grew during training, from about 250 tokens to
1,500 to 8,000 by step 40, and were cut at the scoring cap: on those eight the answers that were cut and wrong are
at least nine tenths of the "wrong format" count. FinQA seeds 0 and 2 did not collapse. A cut answer is scored as it
stood, so the count at 2,048 cannot separate an answer that was wrong from one that did not end. On GSM8K the cut
text already carries the right number after its last answer marker in about two thirds of cut answers, which is a
lower bound and not proof. The third campaign scores the same ten checkpoints and the untrained model again at the
trainer's 8,192 cap, bed and panel, to separate the two. Its answer does not change the bar of record (the cost bar
fails either way); it decides how we read the GSM8K half of the anchor test.

## What runs

| # | campaign | what it answers | GPU-hours (8xH100) |
|---|---|---|---|
| 1 | `campaigns/k3-anchor.yaml` | Which dose does not break the model, and does a pull back towards the untrained model help at that dose? Eleven arms at three runs each on Spider and GSM8K. `README-k3-anchor.md` | about 90 to 120 |
| 2 | `campaigns/k2b-route.yaml` | Does the route of damage (reward-only RL against supervised fine-tuning) decide whether a lost ability comes back? Ten repeats per route. `README-k2b.md` | about 60 |
| 3 | `campaigns/k1c-rescore.yaml` | Were the cut answers wrong, or only unfinished? The ten finished K1c Part B checkpoints and the untrained model scored again at 8,192 new tokens on their bed and on the panel, and a second K1c report from those scorings. Trains nothing; needs the finished K1c tree in `/work/k1c-work`. Opens with a two-minute pilot (eight questions at the long cap) that stops the campaign if the option fails on your engine. | 3 to 6, on ONE GPU |

Run 1 first: its answer decides what K3 becomes. Run 3 whenever one GPU is free; it depends on nothing but the K1c
tree already on disk. Run 2 when a node is free; it depends on nothing.

**Campaign 3 must be scored on one physical GPU.** Its report refuses scorings whose machine fingerprints differ,
and the fingerprint includes the visible GPU. Every row carries `CUDA_VISIBLE_DEVICES=0` and the rows are ordered one
after another, so a dispatcher that follows `needs` and `wants` runs them in sequence; if yours assigns GPUs itself,
pin this campaign to one GPU.

## Before the commands, once

```bash
export KIT=/work/continual-learning-kit/kit SDPO_DIR=/work/SDPO
export SPIDER_ROOT=/work/spider_data GSM8K_ROOT=/work/gsm8k
export FINQA_ROOT=/work/FinQA/dataset K0_REPORT=/work/k0-report/report.json   # the rescoring, as in K1c
pip install qwen-vl-utils==0.0.14          # K2b's supervised route needs it; see README-k2b.md
```

## The commands

```bash
cd /work/continual-learning-kit && git fetch --tags && git checkout kit-batch2-v3
WORK=/work/k3-work  python $KIT/runner.py batch $KIT/campaigns/k3-anchor.yaml
WORK=/work/k1c-work python $KIT/runner.py batch $KIT/campaigns/k1c-rescore.yaml
WORK=/work/k2b-work python $KIT/runner.py batch $KIT/campaigns/k2b-route.yaml
```

`kit-batch2-v3` is `kit-batch2-v1` plus the rescoring campaign and the `--max-new-tokens` option it uses; nothing
in the other two campaigns changed. (`kit-batch2-v2` was never sent; ignore it.)

Each as one job on one node, so every scoring shares a machine fingerprint. `batch` prepares and runs each campaign to
the end. Since this tag: every ordering between rows is an explicit `needs` edge (your dispatcher never has to infer
one, and the runner refuses a campaign that relies on file order); pilot dependencies are in `needs` too; a run that
fails stops the campaign only if a later row needs it, otherwise the campaign continues and the report records the
failed run; and `--seeds 0-4` on any command runs a reduced set and still writes a gated report.

## What to send back

```bash
python $KIT/collect.py --work /work/k3-work  --out send3-anchor.tar.gz
python $KIT/collect.py --work /work/k1c-work --out send3-k1c-rescore.tar.gz   # the whole K1c tree again (about 58 MB of text, 5 MB packed) plus the new k1c/cap8192 and k1c/report8k folders
python $KIT/collect.py --work /work/k2b-work --out send3-k2b.tar.gz
```

Text and a few MB of answers; no weights.

## Running fewer seeds

`--seeds 0-4` on the batch command skips the rows of other seeds and everything that only they feed. The report row
still runs and records its verdict on what passed. A reduced run's error bars are wider; say so when you send it.
