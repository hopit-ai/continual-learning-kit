# Package K2: the recovery test

**The question.** When a model seems to forget, has it lost the knowledge, or only picked up a bad habit of answering? This package takes two models we damaged in earlier work, gives each a short repair, and measures how much of the lost ability comes back, and how fast.

- **Subject 1** is Qwen3-8B after a training run that went wrong: its answers collapsed to a few tokens and its maths score fell from 91 to 34.
- **Subject 2** is Qwen2.5-7B-Instruct fine-tuned on bare SQL: it began answering physics questions with `SELECT`.

**The repair** is a small adapter trained for 300 steps on the *original* model's own answers to 1,000 everyday prompts. Those prompts contain no maths, no exam questions and no formatting rules, so they cannot re-teach what the test measures. If the ability returns anyway, it was never lost. A control applies the same repair to the healthy original, to show the repair is not itself harmful.

**It starts with a pilot you can watch.** The first 13 rows do the whole thing on a 0.6B model that the campaign damages on purpose, in about 30 minutes. The 8B rows are refused until that pilot has passed on your machine.

One GPU, about 4 to 6 hours, about 60 GB of disk for models. Same container and installs as K0 (`README-partner.md` sections 1 and 2); `peft` is already in that image.

## 1. Log in to Hugging Face (once)

Two of the damaged models are in private repositories you have been given read access to. Downloading them needs a login on this machine. Make a **read** token in your own Hugging Face settings, then:

```bash
hf auth login
```

Paste the token when asked. Nothing is uploaded by this package.

## 2. Run it

```bash
export KIT=/work/continual-learning-kit/kit WORK=/work/k2-work
```

```bash
python $KIT/runner.py plan $KIT/campaigns/k2-recovery-test.yaml
```

```bash
python $KIT/runner.py prepare $KIT/campaigns/k2-recovery-test.yaml --all
```

`prepare` needs no GPU. It downloads five things (Qwen3-0.6B, Qwen3-8B at a pinned revision, Qwen2.5-7B-Instruct, and the two damaged models) and writes the prompt file, so that no GPU waits on a download later. If your work folder is a network filesystem, tell us: the kit has a tool that copies models to local disk first.

```bash
python $KIT/runner.py run $KIT/campaigns/k2-recovery-test.yaml --all
```

It uses GPU 0 only, on purpose: scores are comparable only when made on the same GPU in the same mode. If a row fails, the rest are refused; fix and run the same command again, it skips what already passed.

## 3. What to send back

Two small folders of text: `$WORK/k2/report-small-a1/` (the pilot) and `$WORK/k2/report-a1/` (the real subjects). If anything is refused or fails:

```bash
python $KIT/runner.py status $KIT/campaigns/k2-recovery-test.yaml
```

and send that output, plus the `output.log` of the failed row from `$WORK/campaign/k2-recovery-test/<row>/attempt-1/`.

## 4. What we expect, so you can tell if something is off

| Row | What should happen |
|---|---|
| `small-repeatable` | two scorings of one model agree on at least 295 of 300 answers with no changed verdict (ours: 300 of 300) |
| `small-damage-check` | the model we damaged on purpose has lost at least 15 points of 300 |
| `spider60-damage-check` | subject 1 shows its damage on your machine too: at least 15 points lost (on ours, maths alone fell 57) |
| `report` | for each subject, a table of scores before the repair and after 50, 100, 200 and 300 steps, the share recovered, and how far the weights moved |

We do not know what the recovery will be. That is the experiment. Either answer is a result.
