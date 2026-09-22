# Package K4a: the questions SDPO cannot learn from

**The question.** Your K0 runs showed something we did not expect. On this tool-use task, about
**63 out of every 100 questions in a training batch have no successful attempt among the 8** the
model makes, so self-distillation has nothing to copy from and those questions teach it nothing.
The questions it can do, it does about 85 times in 100. The gain you measured (+5.5 points at 17
steps, +7.8 at 40) comes from barely a third of the data.

This package asks: **can we make the stuck questions teach something, without spoiling the ones that
already work?** Three candidate fixes, three seeds each. Nothing else changes.

**What you run is still not our code.** It is `lasgroup/SDPO` at the same pinned commit,
unmodified, launched by the same `kit/run_sdpo_toolalpaca.sh` you already used. Each arm is **one**
extra environment variable, which becomes **one** extra override in the trainer's command line. The
script refuses to run two arms at once.

## The three arms

| arm | the one change | what it is supposed to do |
|---|---|---|
| **feedback** | `FEEDBACK=1` → `...self_distillation.include_environment_feedback=True` | The authors' own switch. When no attempt succeeded, the **teacher** is shown the checker's complaint — which names the expected tool calls and arguments — and the student learns from the teacher's answer. |
| **soft** | `SOFT=1` → `custom_reward_function.path=$KIT/beds/tooluse_soft.py` | A partial-credit **training** reward. An attempt that calls exactly the right tools and gets at least half of the arguments right is offered to the teacher as a worked example, instead of nothing. Calling only some of the right tools never qualifies, however good its arguments. |
| **variation** | `TEMP=1.2` → `+actor_rollout_ref.rollout.temperature=1.2` | Hotter sampling during training, so that the 8 attempts are 8 real tries rather than 8 near-copies. |

Three things we want to be plain about, because they are the whole point of reading the numbers
afterwards:

- In **feedback**, the teacher is shown the right answer. The student is not, and validation is not.
  What we are measuring is whether a student taught by an answer-conditioned teacher can then answer
  on its own.
- In **soft**, an attempt promoted by partial credit is still a **wrong** answer being shown as a
  worked example — and SDPO takes the *first* eligible attempt, not the best one, so this arm can
  also change what happens on questions that already worked. That is why the report splits the
  held-out questions into the ones your K0 run solved and the ones it did not.
- In **variation**, temperature does two things at once in this trainer: it makes sampling more
  varied *and* it softens the distribution the distillation loss is computed over. One knob, two
  effects, both declared. (We also considered raising the number of attempts from 8 to 16. Your own
  K0 per-question data says that would move the share of questions with at least one success by
  about **one point**, because what the model can do it almost always does. It would have cost
  double the time for that one point, so we did not ask for it.)

**The score does not change.** Every accuracy in this package is the same number K0 reports:
`val-core/tooluse/acc/mean@16`, the authors' strict all-or-nothing tool-use score. The soft reward
touches training only; the trainer computes the validation metric from a different field
(`acc`), which our reward function copies from the authors' checker without changing it. The two-step
pilots check this on your machine: if a soft reward had leaked into the measurement, the untrained
model's score would move out of the calibration band, and the campaign would stop.

**Your finished K0 runs are the control.** There are no new control runs: arm seed 42 is compared
with your `dose40-seed42`, 43 with 43, 44 with 44.

## What you need

Everything from `README-partner.md` sections 0 to 4, unchanged and already on disk: the container,
the pinned `lasgroup/SDPO` checkout, the converted `datasets/tooluse` parquet files, and the pinned
Qwen3-8B snapshot. Plus two things:

- **your K0 `report.json`** (the file you sent us), anywhere on disk;
- **the K0 checkpoints are not needed.** You can delete them if you have not already.

Please pull the kit again and check out the tag for this package, then start the container as before.

## Run it

Inside the same container and environment as K0 (the authors' code installed, as in
`README-partner.md` sections 1 and 2):

```bash
export KIT=/work/continual-learning-kit/kit WORK=/work/sdpo-work
export SDPO_DIR=/work/SDPO MODEL_DIR=/work/models/Qwen3-8B NGPU=8
export K0_REPORT=/work/k0-report/report.json
```

```bash
python $KIT/runner.py plan $KIT/campaigns/k4a-stuck-problems.yaml
```

```bash
python $KIT/runner.py prepare $KIT/campaigns/k4a-stuck-problems.yaml --all
```

`prepare` needs no GPU. It checks the data files and the model are where the campaign expects them,
checks your `K0_REPORT` exists, and runs the soft reward function against the authors' own checker on
five worked examples. If any of that is wrong, nothing is wasted.

```bash
python $KIT/runner.py run $KIT/campaigns/k4a-stuck-problems.yaml --all
```

That runs everything in order and stops at the first refusal. The three training pilots come first.
Nothing else starts until all three have passed.

The three `base-*` rows near the end score the untrained model and need no training, so if it suits
your queue you can run them at any point after the pilots:

```bash
python $KIT/runner.py run $KIT/campaigns/k4a-stuck-problems.yaml --row base-1
```

If a row fails, this prints the state of everything:

```bash
python $KIT/runner.py status $KIT/campaigns/k4a-stuck-problems.yaml
```

## Time

From your own K0 numbers: 50 to 68 seconds a step on 8 × H100, and 0.73 to 1.01 hours of wall clock
for a complete 40-step run including its nine validations. None of the three arms changes the batch
shape or the number of attempts, so none of them should change the step time much; `feedback` does
one extra teacher forward on samples that previously had none, which may add a little.

| block | rows | about |
|---|---|---|
| the three pilots | 3 | 1 hour |
| the grid, 3 arms × 3 seeds at 40 steps | 9 | 7 to 9 hours |
| forgetting, all on GPU 0 | 12 | 45 minutes |
| the report | 1 | seconds, no GPU |

**Roughly 9 to 11 hours of wall clock and 65 to 80 GPU-hours.** The grid rows are independent of
each other: if you have a second node, the arms can be run in parallel with `--row`, as long as
every forgetting scoring later happens on **one** machine's GPU 0.

## The stop rule

Stop and tell us, rather than working around it, if any of these happens:

- **A pilot fails.** Send `$WORK/campaign/k4a-stuck-problems/pilot-*/attempt-*/verdict.json` and the
  run's `metrics.jsonl`. Every pilot bar is a number we wrote down in advance; if one is missed we
  would rather drop that arm than spend 8 hours on it. A failed pilot stops **all three** arms,
  because the runner requires every pilot before every later row — which is also why we gate on as
  little as we can: exit code, a merged model, response length, the untrained calibration band, and
  for `feedback` only, that the batch now has something to learn from. Whether the `soft` and
  `variation` knobs moved what they were meant to move is **printed in the report**, beside your K0
  runs' own first two steps, and gates nothing: over two steps those numbers swing too widely
  (your K0 per-step success share ranges 0.19 to 0.62) to stop a package on.
- **The untrained validation is outside 0.555 to 0.600.** Same rule as K0: the environments differ
  and no later number would be comparable.
- **`response_length/mean` in `metrics.jsonl` falls below 16 tokens and stays there**, or a GPU has
  under 2 GB free. Stop the run, keep the directory. Your K0 runs peaked at 78.8 GB of 80, so there
  is very little headroom; `variation` in particular has never been run by us at all.
- **Anything needs a setting changed to fit.** Batch size, attempts per question, learning rate,
  warm-up and step counts are what make these numbers comparable with your K0 runs. Tell us instead
  of shrinking one.

We would rather have three arms and an honest stop than four arms and a number we cannot read.

## What to send back

One folder of small text files:

```
$WORK/k4a/report-a1/          k4a-report.md and k4a-report.json
```

plus, if anything was refused,

```bash
python $KIT/runner.py status $KIT/campaigns/k4a-stuck-problems.yaml
```

The report holds, for every arm and seed beside your K0 run at the same seed: the held-out score at
every validation, the share of each batch with no successful attempt (both as the trainer counted it
and recomputed strictly, so the arms can be compared), the success share, response length, entropy,
peak memory, the three general panels, and the held-out questions split into the ones your K0 run
could already do and the ones it could not. It is recomputed from the raw files every time and needs
only the Python standard library.

Please keep `$WORK/runs/` until we have read the report. Anything you noticed that the files would
not show — a restart, a node change, a queue wait — belongs in the email, in your own words.
