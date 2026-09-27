# Package K3: does rehearsing the old job protect it?

**The question.** A model learns one job, then learns a second job, and gets worse at the first. The cheapest thing anyone can try is to keep asking the old job's **questions** while the new one is being learned. Does that protect the old job, and at what cost to the new one?

- **Job A** is Spider text-to-SQL: 640 training questions, 100 held out. Answers are graded by executing the SQL against the question's own database.
- **Job B** is GSM8K grade-school maths: 1,280 training rows, 300 held out.
- **The four stage-B arms** are `none` (maths only), `rehearse10` and `rehearse30` (10 or 30 percent of the stage-B rows are Spider questions), and `kl` (maths only, with the trainer's KL-to-reference loss switched on).

The rehearsal is **on-policy**: the Spider rows carry questions only. The model being trained answers them afresh at every step and Spider's own checker rewards those answers. Nothing is replayed from a stored answer, and no gold SQL is ever shown to the model or carried in any feedback.

Everything trains with the authors' code — `lasgroup/SDPO` at the pinned commit, `--config-name baseline_grpo`, unmodified. The arms differ only in their data file and, for `kl`, three loss settings. **Ten seeds, all in this one campaign**: stage A at seeds 0 to 9, then every stage-B arm at seeds 0 to 9. All ten run the reference dose: 20 steps at lr 1e-5 for stage A (exactly one pass over Spider's 640 questions) and 40 steps × 32 questions = 1,280 rows for every stage B, so rehearsal **displaces** maths rows rather than buying extra steps.

Seeds 3 to 9 run here. The separate seeds follow-up package (tags `kit-seeds-v1` and `v2`) is retired: ten seeds are in-line in K3, K4 and K1c from this tag on.

**Stage B starts from stage A's merged checkpoint as a weight-initialised restart**: fresh optimizer state, fresh warm-up, no resume — the same for every arm.

## It starts with a pilot, and the pilot can stop the package

The first rows, at seed 0, are the gate:

1. the untrained model is scored twice and the two scorings must agree;
2. stage A is trained; **bar 1: Spider held-out correct must rise by at least 5**;
3. stage B arm `none` is trained; **bar 2: Spider held-out correct must then fall by at least 5**.

If bar 1 fails, the model never learned job A. If bar 2 fails, nothing was damaged, so there is nothing for rehearsal to protect and any difference between the arms would be noise. Either way every later row stays **refused**, on purpose. Tell us and stop; that is a result, and it costs one stage A and one stage B instead of fifty.

## What you need

- **One node with 8 GPUs** for the training rows. Scoring rows use **GPU 0 only** and are pinned to it: counts made on two different GPUs are not comparable, which is what the `base-repeatable` row checks before anything else runs.
- **The container and the pinned SDPO install** exactly as in `README-partner.md` sections 1 and 2 (`nvcr.io/nvidia/vllm:25.12.post1-py3`, `lasgroup/SDPO` at `7c457fc1b1f636ae794eb0362ba37d4743b06fbc`, `pip install -e /work/SDPO`).
- **Your own Spider 1.0 download** (`spider_data.zip` from yale-lily.github.io/spider), unzipped. `SPIDER_ROOT` points at the `spider_data` directory itself — the one holding `database/`, `train_spider.json` and `train_others.json`. The kit ships no Spider text: it carries the question ids and the sha256 of every prompt and every database file, and refuses to run if your copy is a different release.
- **GSM8K**, downloaded by you:

  ```bash
  hf download openai/gsm8k --repo-type dataset --local-dir /work/gsm8k
  ```

  `GSM8K_ROOT` points at `/work/gsm8k`. The kit ships no GSM8K text either. 100 of the 300 questions on our general panel are GSM8K test questions, so the bed removes any panel question from the training file and refuses to write a file in which one remains.
- **About 180 GB of disk**: the 1.7B base is about 3.4 GB, and each of the 25 runs leaves a sharded trainer checkpoint plus the merged copy the scorers read, about 7 GB a run. If space is tight, delete `$WORK/runs/<run>/train/global_step_*/actor` once that run's `hf-step*/config.json` exists — nothing downstream reads the shards, and the scoring folders you send back are text.

Nothing is uploaded. No Hugging Face login is needed: the only model downloaded is the public `Qwen/Qwen3-1.7B`.

## What "seed" means here

`SEED` names a run and is passed to the trainer as `data.seed`, which the trainer ignores because the training order
is fixed (`data.shuffle=False`), and the sampler is not seeded. So the ten "seeds" of every arm are ten
independent repeats of one recipe, not ten controlled random streams. That is what the spread is for, a
run-to-run spread, and it is stated here so nobody reads "seed 3" as reproducible to the token.

## One node for every scoring, or the numbers cannot be compared

Your first K3 gate failed on this, not on the model: `base-spider` and `a-seed0-spider` were scored on
different nodes, and two nodes scoring the SAME model greedily disagree by about 3 answers in 100 — the size
of the pilot's bars. Rerun on one node, the same dose gave +5, exactly the bar. Nothing here can be read
until both sides of a difference come from one node.

So: **run the whole campaign as one job on one node.** Every scoring row pins GPU 0
(`CUDA_VISIBLE_DEVICES=0`), and the two pilot deltas and the report check that every scoring carries one
machine fingerprint. If a delta row reports `same-machine 0`, or the report refuses because it found two
fingerprints, do not read the number: re-score both sides on one node with `--row` (for the first pilot,
`--row base-spider` and then `--row a-seed0-spider`) and run the delta row again. The report prints every
fingerprint it read, so this is visible at a glance.

## The four commands

```bash
export KIT=/work/continual-learning-kit/kit WORK=/work/k3-work
export SDPO_DIR=/work/SDPO SPIDER_ROOT=/work/spider_data GSM8K_ROOT=/work/gsm8k
```

```bash
python $KIT/runner.py plan $KIT/campaigns/k3-replay.yaml
```

```bash
python $KIT/runner.py prepare $KIT/campaigns/k3-replay.yaml --all
```

`prepare` needs **no GPU**: it downloads the model, builds the bed files from your Spider and GSM8K copies, and writes the ten rehearsal mixtures, so that no GPU ever waits on a download or a conversion.

```bash
python $KIT/runner.py run $KIT/campaigns/k3-replay.yaml --all
```

```bash
python $KIT/runner.py status $KIT/campaigns/k3-replay.yaml
```

`run` stops at the first refusal or failure; fixing the cause and running the same command again skips everything that already passed. A retry of one row is a new attempt in its own directory — nothing is ever overwritten.

## How long this takes, and why that number is a guess

**Estimated 89 to 229 GPU-hours, most likely about 134, and 24 to 59 hours of wall clock.** That is the five-seed estimate this package carried (43 to 117, most likely 68, 12 to 30 hours) scaled by 2 for ten seeds — every run and every scoring of a trained model doubles, the untrained model's scorings do not — and it is ten because every effect this package looks for is about 5 points against a run-to-run spread of 3 to 4, and ten repeats halve the variance of every mean (the error bar narrows by about 30 percent). It is an *estimate*, not a measurement, and it stays one until your measured step times replace it. The arithmetic, so you can correct it yourself:

| Part | Work | Rate assumed | Result |
|---|---|---|---|
| Training | 10 stage A × 20 steps + 40 stage B × 40 steps = **1,800 steps** on 8 GPUs | 20 s/step (band 10 to 40) — scaled down from our only measurement, Qwen3-8B SQL at 82 s/step on 4 × H200, for a model 4.7× smaller, twice the GPUs and no teacher forward | 10.0 h (5.0 to 20.0) |
| Run overhead | 50 runs × model load, vLLM engine start, checkpoint merge | 5 min a run | 4.2 h |
| Scoring | 51 points × (100 Spider + 300 GSM8K + 300 panel) + one repeat = **36,000 answers** on GPU 0 | 1.5 s an answer (band 1 to 3); the scorer runs eager with CUDA graphs off, which is slower on purpose and is what makes two scorings agree | 15.0 h (10.0 to 30.0) |
| Scoring overhead | 154 scorings × engine start | 2 min each | 5.1 h |

Training occupies all 8 GPUs (14.2 h × 8 = 114 GPU-hours); scoring occupies one (20.1 GPU-hours).

If the pilot stops the package at bar 1 or bar 2, you will have spent about 1.5 hours of that.

## What to send back

One folder: `$WORK/k3/report-a1/` (`k3-report.md` and `k3-report.json`). If anything was refused or failed, also send the output of `status` and the `output.log` of the failed row from `$WORK/campaign/k3-replay/<row>/attempt-1/`.

If you can spare the space, the per-point scoring folders under `$WORK/k3/eval/` and `$WORK/k3/forgetting/` are what let us re-check any number without re-running anything; they are small text files, plus one `responses.jsonl` per scoring.

## What we expect, so you can tell if something is off

| Row | What should happen |
|---|---|
| `base-repeatable` | two scorings of one model agree on at least 295 of 300 answers with no changed verdict (on our machines: 300 of 300) |
| `base-spider` | the untrained 1.7B gets some Spider questions right and writes SQL in a ```sql fence; a score of 0 means the prompt or the databases are wrong, not the model |
| `a-seed0` | the trainer exits cleanly, writes a checkpoint at step 20 and merges it to `hf-step20/` |
| `pilot-learned-sql` | Spider held-out correct is at least 5 higher than the untrained model's |
| `pilot-damaged-sql` | after maths with no protection, Spider held-out correct is at least 5 lower than it was after stage A |
| `report` | a table, per arm, of what stage B did to Spider and where GSM8K landed, averaged over ten seeds |

**We do not know what the rehearsal arms will show.** That is the experiment. Rehearsal protecting job A, rehearsal doing nothing, and rehearsal costing job B are all results, and all three are worth the same to us. Please send the report whichever it is.

One thing to say in advance about the ten seeds. The trainer reads every training file in its written order (`data.shuffle=False`, as the reference SQL runs did), so its own data seed changes nothing. For `rehearse10` and `rehearse30` a seed changes which Spider questions are rehearsed and where they fall. For stage A, `none` and `kl` the ten seeds read the same file in the same order and differ only because the reference command leaves rollout sampling unseeded. They are ten independent repeats, not ten data orders. The report shows each seed as well as the mean, so you can see the spread rather than take our word for it.
