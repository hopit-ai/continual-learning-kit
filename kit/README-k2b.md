# Package K2b: does the route of damage decide whether an ability comes back?

**The question.** Your K2 run gave us our strongest lead. The Qwen3-8B damaged by reinforcement learning got its maths back from a 50-step repair while its weights barely moved: 96 percent of what it lost, against the repair's own control. The Qwen2.5-7B damaged by supervised fine-tuning got 54 percent back. If reinforcement learning *hides* an ability behind an answering habit while supervised fine-tuning *overwrites* it, that is a finding. But the two subjects differed in almost everything: base model, size, the job that damaged them, how deep the damage went, and LoRA against full training. Any one of those could be the whole effect.

K2b matches everything except the route. One base model learns one new job two ways, ten times each, and each result gets K2's repair.

## The design

- **The healthy original is the untrained Qwen3-1.7B.** The abilities at stake are the ones it already has: Spider held-out (it scores about 70 of 100) and the 300-question general panel.
- **There is no stage A.** We first planned to teach the model SQL before the new job. Our analysis of your K3 answers (receipt 238) showed that 20 GRPO steps on Spider do not teach the 1.7B SQL; they collapse its style (joins fell from 41 answers of 100 to between 0 and 9). A stage A would have handed K2b a collapsed model, not a healthy one.
- **The new job is GSM8K**, the same 1,280 questions K3's stage B used, 40 steps of 32, learning rate 1e-5, all parameters trained, by one of two routes:

| Route | Launcher | What the model learns from |
|---|---|---|
| `rl` | `kit/run_grpo.sh` (GRPO, as in K3) | a reward: 8 of its own answers per question, scored right or wrong |
| `sft` | `kit/run_sft.sh` (new: the pinned SDPO fork's own SFT trainer) | the dataset's worked solution to each question, ending in the `Answer: <number>` line the scorer reads |

Both use the same stack, the same optimiser settings and the same learning-rate schedule (10 warm-up steps, then constant). `kit/run_sft.sh` explains, key by key, where it overrides the SFT trainer's defaults to match GRPO's. `kit/beds/gsm8k.py sft-export` writes the SFT file from the same prepared rows, with the same prompt text, and refuses to write it unless the GSM8K scorer marks every target answer correct.

- **Ten repeats per route** (`rl-seed0` … `rl-seed9`, `sft-seed0` … `sft-seed9`). Each SFT repeat trains on the same 1,280 questions in its own order (`sft-seedN-data` writes it with `sft-export --seed N`), because supervised training samples nothing and its repeats would otherwise be near-copies of one run; GRPO's repeats share one file and differ through sampling.
- **After training**, each model is scored on Spider held-out, GSM8K held-out and the panel.
- **The repair is K2's, unchanged:** a small adapter trained for 300 steps on the *original's* own answers to 1,000 everyday prompts, merged and scored at 50, 100, 200 and 300 steps. This time Spider is scored at every step as well as the panel. The **control** is the same repair applied to the untrained model. It runs once and is shared by both routes and all repeats.


## One package to install, and one declared difference

**Install `qwen-vl-utils==0.0.14` before running.** The pinned trainer's supervised path imports it, but its
`setup.py` lists it only under the optional "geo" extra, so a plain `pip install -e /work/SDPO` lacks it. Our smoke
failed on exactly this.

```bash
pip install qwen-vl-utils==0.0.14
```

**The SFT route runs through `kit/sft_entry.py`**, which turns off pinned host memory in the trainer's data loader
before starting it. With the pinned stack (torchdata 0.11.0 under torch 2.9) the trainer's pin-memory thread crashes
on its first batch. Turning it off changes how batches are copied to the GPU, not the data, order, loss or
arithmetic; the authors' files are not edited. The GRPO route never takes that path.

## It starts with a pilot

The first rows score the untrained model twice (the two scorings must agree), train each route once at seed 0, and score both. **Each route must lower Spider held-out by at least 5 points against the untrained model** (`pilot-rl-damaged-spider`, `pilot-sft-damaged-spider`). If a route does not damage Spider, there is nothing to recover. Every later row is then refused, and the package stops. That is an answer, not a fault: it would tell us that learning maths this way does not cost this model its SQL. GSM8K's rise under each route is also written at this point (`pilot-*-gsm8k-gain`). It is not a gate.

## The one command

The same container and installs as K3. `WORK` may be your K3 work tree: the model and the GSM8K file live where K3 put them, so nothing is downloaded twice.

```bash
export KIT=/work/continual-learning-kit/kit WORK=/work/k3-work
export SDPO_DIR=/work/SDPO SPIDER_ROOT=/work/spider_data GSM8K_ROOT=/work/gsm8k
```

```bash
python $KIT/runner.py batch $KIT/campaigns/k2b-route.yaml
```

Run it as **one job on one node**. Training uses all 8 GPUs. Every scoring and every repair pins GPU 0, because scores are comparable only when they come from one GPU in one mode: two nodes scoring the same model disagree by about 3 answers in 100, which is the size of the pilot's bar. If anything fails, the rest is refused. Fix it and run the same command again: it skips what already passed. `python $KIT/runner.py plan $KIT/campaigns/k2b-route.yaml` prints all 242 rows without running anything.

## Cost

**About 60 GPU-hours of work, and about a day and a half on one node.** How we estimated it, per repeat (both routes):

| Step | GPU-hours | Basis |
|---|---|---|
| GRPO, 40 steps on 8 GPUs | 1.3 | your K3 stage-B runs, same dose, same model |
| SFT, 40 steps on 8 GPUs | 0.5 | a forward and backward pass over the same 1,280 rows, with no generation |
| scoring both trained models (Spider, GSM8K, panel) | 1 | your K3 scorings of the 1.7B |
| two repairs, each with 4 merges and 4 scorings of the panel and Spider | 2 to 3 | your K2 repairs, scaled from 8B to 1.7B |

That is about 5 to 6 GPU-hours per repeat, so 50 to 58 for ten, plus about 2 once for the untrained model's scorings, the repair targets and the control. The scorings and repairs run on GPU 0 while the other seven GPUs wait, so wall-clock time is set by GPU 0: about 3 hours per repeat, or about 30 hours in all, plus about 2.5 hours of training. If your scheduler bills the whole node, that is about 8 × 33 GPU-hours reserved.

Disk: each trained model keeps its checkpoint shards and its merged copy, about 10 to 14 GB per run and up to about 300 GB for all 20 runs. Once a run's `hf-step40/config.json` exists, its `train/global_step_40/` folder may be deleted. Repaired models are merged into a temporary folder, scored and then deleted.

## What to send back

```bash
python $KIT/collect.py --work $WORK --out k2b-return.tar.gz --campaign $KIT/campaigns/k2b-route.yaml
```

This packs text only: the report (`$WORK/k2b/report-a1/`), every scoring, the per-step training metrics and the runner's records. It contains no weights. If a row fails or is refused, send the same archive. It includes the failed row's `output.log`.

## What we expect, so you can tell if something is off

| Row | What should happen |
|---|---|
| `q17-repeatable` | two scorings of the untrained model agree on at least 295 of 300 answers, with no changed verdict |
| `rl-seed0`, `sft-seed0` | `train-summary.json` shows `returncode 0`, `merged 1`, `steps 40`. The SFT run's `metrics.jsonl` shows its training loss falling |
| `pilot-*-gsm8k-gain` | GSM8K held-out rises under both routes. If it does not, tell us before reading anything else |
| `pilot-*-damaged-spider` | Spider held-out falls by at least 5 under each route. This is the bar |
| `report` | `k2b-report.md`: per route, the mean and spread over ten repeats of the damage, the share recovered at 50 and 300 repair steps against the original and against the control, the weight distance and the GSM8K gain, followed by the verdict |

**The verdict was fixed before any number existed.** The claim "the route decides" holds if, on Spider held-out, the mean share recovered by step 300 against the control is at least 80 percent for RL, below 80 percent for SFT, and at least 20 points higher for RL than for SFT. The panel gets the same rule, printed beside it. Every outcome is a result:

- RL hides and SFT overwrites: the lead holds.
- Both hide: K2's difference came from the base model or the job.
- Neither hides: the effect needs a bigger model.
- SFT hides and RL does not: the lead reverses.

We do not know which it will be. That is the experiment.
