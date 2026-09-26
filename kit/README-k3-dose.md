# K3 stage A, the dose probe: which change makes the model learn SQL?

**Your K3 pilot stopped the package at bar 1, and it was right to.** The untrained Qwen3-1.7B scores **70 of 100** on the Spider held-out set. After stage A — 20 GRPO steps over the 640 frozen Spider training questions, one pass, lr 1e-5 with 10 warm-up steps, batch 32 × 8 samples — it scored **62**, and the bar was +5. Your paired analysis: **21 right → wrong, 13 wrong → right, 66 unchanged, exact McNemar p = 0.229**; the training reward flat around 0.63 with no trend; one answer truncated at 2,048 tokens.

That is not a broken harness and it is not, as far as we can tell, a damaged model. It is a dose that did nothing much in either direction.

## Why we think the dose is the suspect

Three things point the same way.

1. **Most of stage A's questions carry no gradient.** We measured the same 640 Spider training questions with the untrained model at the trainer's own sampling — 8 attempts, temperature 1.0 (`docs/phase2/evidence/k4-gate`, receipt 227). It solves **300 of them every single time** and **171 of them never**. A question solved 8 times out of 8 gives GRPO a group in which every sample scores 1, so its advantage is zero and it contributes nothing. Only about 170 questions in the middle can move the weights at all, and they are scattered through a file the trainer reads once, in order.
2. **Half the run is warm-up.** `kit/run_grpo.sh` sets `lr_warmup_steps=10`, which is the reference command's own value. At 20 steps that is 10 steps spent getting to the learning rate and 10 at it.
3. **The learning rate is the reference's 1e-5**, chosen for 40- and 60-step runs, not for 20.

So: too few useful questions, too few steps at full speed, or too small a step. This probe changes one of those at a time.

## The four arms

Every arm is **one run at seed 0**, from the untrained model, on the same machine, scored the same way.

| Arm | The dose | What it isolates |
|---|---|---|
| `ref20` | 640 rows, 20 steps, lr 1e-5 | **Nothing.** It is your stage A rerun here. It measures what one run against one run is worth in this pipeline — the noise every other arm's gap must be read against. |
| `steps60` | 1,920 rows (three passes), 60 steps, lr 1e-5 | More steps at the same rate on the same questions. Warm-up becomes 10 of 60. |
| `lr3x` | 640 rows, 20 steps, **lr 3e-5** | A bigger step, same number of steps, same questions. |
| `hard20` | 640 rows drawn from the questions the model does **not** already solve every time, 20 steps, lr 1e-5 | The same 20 steps spent only on questions that carry gradient — about two passes over them. |

Two things about the launcher this design had to respect. `kit/run_grpo.sh` passes `data.shuffle=False` and `trainer.total_epochs=1`, so a run can take only as many steps as its training file has batches: **"more steps" means a longer file**, which `kit/sequence.py pool` writes deterministically (it repeats rows only when the input runs out, and records what it did in a manifest). And the learning rate is now the `LR` environment knob, recorded as `lr` in each run's `train-summary.json`. **Warm-up stays at 10 steps on every arm** — it is not a knob here, it is part of what is being measured.

## How to run it

The same environment as K3, and **`WORK` may be your K3 work tree**: the model and the Spider bed live at the paths K3 already uses (`$WORK/models/Qwen3-1.7B`, `$WORK/data/spider`), so `prepare` will find them and download nothing. Everything this probe writes goes under `$WORK/k3dose/` and `$WORK/runs/`.

```bash
export KIT=/work/continual-learning-kit/kit WORK=/work/k3-work
export SDPO_DIR=/work/SDPO SPIDER_ROOT=/work/spider_data
```

```bash
python $KIT/runner.py plan    $KIT/campaigns/k3-dose-probe.yaml
python $KIT/runner.py prepare $KIT/campaigns/k3-dose-probe.yaml --all   # no GPU
python $KIT/runner.py run     $KIT/campaigns/k3-dose-probe.yaml --all   # 8 GPUs; GPU 0 for scoring
python $KIT/runner.py status  $KIT/campaigns/k3-dose-probe.yaml
```

Before the training rows, two short rows prepare `hard20`'s file: `stuck` samples all 640 training questions 8 times at temperature 1.0 on **one GPU** (a couple of minutes) and `subset` drops the always-solved ones on a CPU and pools the rest back up to 640 rows. Both write manifests; neither touches a prompt, an answer or a checker.

## One node for every scoring, or the numbers cannot be compared

Your stage-A gate failed **two** bars, and the second is the one to fix first: `base-spider-a1` and
`a-seed0-spider-a1` were scored on different GPUs (fingerprints `87e0c32c…` and `6c7e8668…`), so `same-machine`
was 0. We measured what that costs: two nodes scoring the SAME model greedily disagree by about 3 answers in 100.
Your 70 to 62 is therefore "70 ± 3 against 62 ± 3", and part of the 34-question churn is two machines disagreeing,
not the model moving. Nothing here can be read until both sides of a difference come from one node.

So for this probe: **run the five `*-spider` scoring rows and `base-spider` on the same node**, in one job if your
scheduler allows it (they are a few minutes each on one GPU). If a `*-delta` row reports `same-machine 0`, do not
read its delta: re-score the untrained model and that arm on one node with `--row base-spider --row <arm>-spider`
and run the delta row again. The report prints every scoring's fingerprint so this is visible at a glance.

For the record, your grad_norm column settles the other question: gradients of order 1 at every step, so stage A
had real signal and a tiny dose, ten effective updates at full learning rate after the ten warm-up steps.

## What it costs

**About 4 to 5 GPU-hours on the 8×H100 node, plus five scorings.** 120 training steps in total (20 + 60 + 20 + 20) at roughly 15 s a step on 8 GPUs is about 30 minutes of wall clock, so 4 GPU-hours, plus four model loads, vLLM starts and checkpoint merges at about 5 minutes each. The five Spider scorings (100 questions each, GPU 0, deterministic mode) and the one-GPU stuck row add under an hour on one GPU. Expect **under two hours of wall clock** for the whole probe.

## What to send back

- `$WORK/k3dose/report/` — `report.md` and `report.json`, which is the readout;
- `$WORK/k3dose/eval/` — the five scoring folders (`bed-score.json` and `responses.jsonl`), which are what let us re-check any number without re-running anything;
- `$WORK/k3dose/stuck/` — `stuck.json` and `attempts.jsonl`;
- `$WORK/k3dose/deltas/` — the four difference files;
- every `$WORK/runs/<arm>-seed0-a*/metrics.jsonl` and `train-summary.json`.

**No checkpoints.** Nothing above is large; the whole set is text plus a few MB of answers.

## Two things to be clear about

**The four `*-delta` rows are readouts, not gates.** Each writes the paired difference between the untrained model and one arm to a file and checks only that the number exists and that both scorings came from the same machine. The +5 verdict is printed by the report, not by the runner, so an arm that misses +5 stops nothing: `runner.py run --all` runs every row to the end. Nothing in the campaign needs a delta row.

**This probe is one seed.** An arm that clears +5 here is a **candidate dose to confirm at K3's five seeds**, not a result. `ref20` is in the probe precisely so we can see how much of any gap could be the pipeline repeating itself. If an arm wins, what happens next is that stage A is re-dosed to that arm's dose and K3 restarts from its gate — and that is the only decision this probe is allowed to inform.

If **no** arm clears +5, that is also an answer, and a cheap one: it says the problem is not the size of the dose, and we go looking somewhere else before spending another 41 GPU-hours on K3.
