# Package K5: does the recipe hold over four jobs in a row? (DRAFT)

> **Draft.** The arms are not fixed yet — they are settled once K3 and K4 report, and this campaign is
> regenerated with the final list before anything is sent. Everything else here is finished: the
> sequence, the doses, the scoring, the pilot, the tools and the tests. Two things are still open and
> both are named at the bottom, under *What is not finished*.

**The question.** Everything measured so far has been one job, or two. The claim the programme wants
to make is about a *sequence*: a small model keeps learning new jobs and does not forget the old
ones. So: four jobs, learned one after another, and at every point along the way every job is scored
— including the ones learned three stages ago, and 300 general questions that belong to no job at
all.

- **The jobs.** SQL (Spider, 100 held out), coding (51 held out), maths (GSM8K, 300 held out), FinQA
  (the whole 1,147-question test split). These are the four jobs the frontier yardstick was measured
  on (receipts 220 to 222), so "within 5 to 15 percent of a frontier model" can be read job by job.
- **The two orders.** `sqlfirst` is SQL → coding → maths → FinQA. `mathsfirst` is maths → FinQA →
  coding → SQL. The second is not a robustness check for its own sake: it puts maths first **and**
  lands SQL in position 4, and SQL is in position 1 of the first order, so the same job is measured
  at both ends of a sequence under the same seed. That is exactly the pair the plasticity probe
  compares (plan 4b, research target Q11).
- **The arms.** Today: `none` (plain GRPO on the current job alone) and `rehearse10` (10 percent of
  every stage's rows are questions from the jobs already learned, drawn equally from each). `kl` (the
  trainer's KL-to-reference loss, anchored to the previous stage) is built and simply not in the
  list. `sdft` and `isdft` are **refused** by the generator, loudly, because the kit has no launcher
  for them — see *The arms that are not here*.
- **Five seeds** for every arm, both orders.

Everything trains with the authors' code — `lasgroup/SDPO` at the pinned commit, `--config-name
baseline_grpo`, unmodified, through `kit/run_grpo.sh`. The arms differ only in their data file and,
for `kl`, three loss settings.

## The shape of one sequence

Every stage is **40 steps × 32 questions = 1,280 rows**, which is K3's stage-B dose and K1c's dose,
so a K5 stage reads directly against both. Each stage starts from the previous stage's **merged**
checkpoint (`MODEL_DIR`) as a weight-initialised restart: fresh optimizer state, fresh warm-up, no
resume, the same for every arm.

**After every stage, all four held-out sets and the 300-question forgetting panel are scored**, plus
one plasticity probe of the checkpoint. That is the whole point: the scorecard's grid (`kit/scorecard.py`)
is a (T+1) × T table — every job after every stage, and the untrained model as stage 0 — and it is
filled in as the sequence runs, never reconstructed afterwards from whatever happens to be on disk.

**Stage 1 is one run per order and seed, shared by every arm.** There is nothing to rehearse and
nothing to anchor to before a job has been learned, so every arm's stage 1 *is* the same run; it is
trained once, under the arm name `shared`, and every arm's stage 2 starts from it. This is what K3
did with its stage A, and it saves ten trainings and fifty scorings.

Two doses need saying out loud, because both are departures from "one pass over the bed":

- **Spider has 640 training questions and a stage consumes 1,280 rows**, so the SQL stage file holds
  each question **twice**, in two differently-ordered passes, written by `kit/sequence.py pool`. The
  alternative was a 20-step SQL stage, which would make SQL's dose different from every other job's
  and would make the position-1-versus-position-4 plasticity comparison meaningless. The file's
  manifest records exactly how many rows came from where.
- **FinQA and coding files are written 1.25× long** (1,600 rows for a 1,280-row stage) because verl
  silently drops prompts over 2,048 tokens and 1.5 percent of FinQA's are longer than that. A file
  that is one row short of the dose ends at step 39 with no checkpoint and no error (K1c's
  `FINQA_TRAIN_ROWS`). The 1.25 for coding is *copied* from FinQA and has not been measured on the
  coding bed — see *What is not finished*.

## It starts with a pilot, and the pilot can stop the package

The first rows are the gate, and they are K3's two bars on K3's pair of jobs at K5's dose:

1. the untrained model is scored twice and the two scorings must agree (at least 295 of 300 answers
   identical, no changed verdict, same machine);
2. SQL is learned at seed 0 with plain GRPO — **bar 1: Spider held-out correct must rise by at
   least 5**;
3. maths is learned on top of it — **bar 2: Spider held-out correct must then fall by at least 5**.

If bar 1 fails, the model never learned the job. If bar 2 fails, nothing was lost, so no protective
arm could show anything but noise. Either way every later row stays **refused**, on purpose; tell us
and stop. That costs two trainings and a handful of scorings instead of the whole package.

The pilot is a chain of its own rather than the first two stages of an order, for one reason: the
headline order learns **coding** second, and the coding bed is being written separately. A gate must
not be able to wait on a file that does not exist yet. The cost of that decision is one extra
training (about 1.3 GPU-hours), and it is the only duplicated run in the package.

## What you need

- **One node with 8 GPUs** for the training rows. Every scoring row is pinned to **GPU 0**, because a
  count made on one GPU cannot be subtracted from a count made on another — that is what the
  `base-repeatable` row checks before anything else runs, and it is why the scoring is serial.
- **The container and the pinned SDPO install** exactly as in `README-partner.md` sections 1 and 2.
- **Your own copies of the four beds**, with these environment variables pointing at them:
  `SPIDER_ROOT`, `GSM8K_ROOT`, `FINQA_ROOT`, `LCB_ROOT` (the LiveCodeBench `test*.jsonl` directory —
  see `README-code-bed.md`). The kit ships no dataset text; each bed checks the copy it is given and
  refuses a different release. The coding rows also set `CODE_TESTS` themselves, to the tests file
  the coding bed's `prepare` wrote.
- **About 250 GB of disk at any one time.** Each run leaves a sharded trainer checkpoint plus the
  merged copy the scorers read, about 7 GB a run, and there are 72 runs — 500 GB if nothing is ever
  deleted. The rule that keeps it under control: once a point's five scorings **and** its plasticity
  probe have passed, its sharded `train/global_step_*/actor` directory can be deleted, and once the
  *next* stage of that chain has merged, the whole merged directory can go too. Nothing downstream
  reads either; the folders you send back are text.

Nothing is uploaded, and no Hugging Face login is needed: the only model downloaded is the public
`Qwen/Qwen3-1.7B`.

## The commands

```bash
export KIT=/work/continual-learning-kit/kit WORK=/work/k5-work
export SDPO_DIR=/work/SDPO SPIDER_ROOT=/work/spider_data GSM8K_ROOT=/work/gsm8k
export FINQA_ROOT=/work/FinQA/dataset LCB_ROOT=/work/code_generation_lite
```

```bash
python $KIT/runner.py plan $KIT/campaigns/k5-sequence.yaml          # prints 533 rows, runs nothing
python $KIT/sequence.py estimate --campaign $KIT/campaigns/k5-sequence.yaml
python $KIT/runner.py prepare $KIT/campaigns/k5-sequence.yaml --all  # no GPU: model, beds, pools, mixtures
python $KIT/runner.py run $KIT/campaigns/k5-sequence.yaml --all      # 8 GPUs for training, GPU 0 for scoring
python $KIT/runner.py status $KIT/campaigns/k5-sequence.yaml
```

`run` stops at the first refusal or failure; fixing the cause and running the same command again
skips everything that already passed. A retry of one row is a new attempt in its own directory —
nothing is ever overwritten.

## How long this takes, and why that number is a guess

`python $KIT/sequence.py estimate` adds up the campaign's own arithmetic. It counts **compute only**:

| what | rows | GPU-hours |
|---|---|---|
| training | 72 runs × 40 steps × 8 GPUs, at 15 s a step | 96 |
| bed scorings | 286 scorings, 1 GPU, at 1.5 s an answer | 47 |
| forgetting panels | 72 × 300 questions | 9 |
| plasticity probes | 71 × 200 prompts, forward passes only | 6 |
| **total** | | **158** |

Add the overhead the tool does not model — model load, vLLM engine start and checkpoint merge, about
5 minutes a run and 2 minutes a scoring — and it comes to roughly **220 GPU-hours**: about 18 hours
of wall clock on all 8 GPUs for the training, plus about **80 hours on GPU 0 for the scoring**, which
is serial by design. Only one number in this is measured (15 s a step, from K3 on the 1.7B on 8 ×
H100); the other two are assumptions and the tool prints them as assumptions. Correct them with
`--seconds-per-answer` and re-run the estimate.

**The one lever worth knowing about.** FinQA is 1,147 of the 1,598 held-out questions scored at every
one of the 71 points — about 34 of the 56 scoring hours. Scoring it on a fixed 400-question slice
would cut roughly 24 GPU-hours and 24 hours of wall clock. We have **not** done that, because the
frontier yardstick (948 of 1,147) and K1c were measured on the whole split, and shrinking a
comparison-bearing set to fit a budget is how a package stops being comparable with the work it is
supposed to be read beside. If the owner wants the slice, the frontier line has to be re-scored on
the same slice first.

If the pilot stops the package, you will have spent about 5 of those GPU-hours (two trainings, the
untrained model's six scorings and two more of Spider) — under two hours of wall clock.

## What to send back

One folder: `$WORK/k5/report-a1/` (`k5-report.md` and `k5-report.json`), plus
`$WORK/k5/scorecards/` and `$WORK/k5/plasticity-compare/`. If anything was refused or failed, also
send the output of `status` and the `output.log` of the failed row from
`$WORK/campaign/k5-sequence/<row>/attempt-1/`.

If you can spare the space, the per-point scoring folders under `$WORK/k5/eval/`,
`$WORK/k5/forgetting/` and `$WORK/k5/plasticity/` are what let us re-check any number without
re-running anything. They are small text files, plus one `responses.jsonl` per scoring.

## What we expect, so you can tell if something is off

| Row | What should happen |
|---|---|
| `base-repeatable` | two scorings of one model agree on at least 295 of 300 answers, no changed verdict |
| `base-spider`, `base-gsm8k`, `base-finqa` | the untrained 1.7B scores something on each; a 0 means the prompts or the data root are wrong, not the model |
| `pilot-learned-sql` | Spider held-out correct at least 5 higher than the untrained model's |
| `pilot-damaged-sql` | after maths with no protection, at least 5 lower than after the SQL stage |
| any `...-stage<N>` | the trainer exits cleanly, writes a checkpoint at step 40 and merges it to `hf-step40/` |
| `...-manifest` | names every one of the 100 cells of that order and arm; it **refuses** if one is missing |
| `...-scorecard` | average accuracy, backward transfer, forward transfer and the general delta, over five seeds |
| `report` | one table per order and arm: what each job scored when it was learned, what it kept to the end, and where the panels ended |

**We do not know what the arms will show.** That is the experiment. Rehearsal protecting the old
jobs, rehearsal doing nothing, and rehearsal costing the new job are all results and all three are
worth the same to us.

## The arms that are not here

`sdft` and `isdft` are in the generator's arm table with the reason they cannot be built: the kit's
only launcher is `kit/run_grpo.sh`, which runs GRPO. Our iSDFT pilot (receipts 223 and 224) ran the
authors' trainer inside the partner's container, not from the kit. Asking the generator for either
arm **refuses and says so**, and writes nothing:

```
$ python scripts/make_k5_campaign.py --arms none isdft
REFUSED: refusing to generate the isdft arm: iSDFT (Khamis et al., 21 September 2026) ...
the kit has no iSDFT launcher. ... Add that launcher to the kit, give it a run-summary with the
same keys run_grpo.sh writes, and then add the arm here.
```

`--refusing-rows` writes such an arm as a row that fails with that message instead, for the case
where the campaign has to carry the fact that an arm was asked for and could not run.

## What is not finished

1. **The coding bed exists, but the kit cannot yet reach it.** `kit/beds/code.py` (LiveCodeBench,
   run in `kit/sandbox.py`) landed while this package was being written, and the campaign is written
   against its real interface: `prepare --lcb-root ... --out ...`, `LCB_ROOT` for the problems and
   `CODE_TESTS` for the tests file — the training subset on a training row, every test on a panel
   row. Two one-line entries in two files that belong to neither this package nor the bed are still
   missing, and until they land every coding row fails:

   - `kit/eval_bed.py` needs `code` in `BED_FILES`, `DEFAULT_SPLIT`, `SPLITS` and `MAX_NEW_TOKENS`,
     and a branch in `items_of`/`score_one` for its prompts and its sandboxed checker — the campaign
     scores every bed through `eval_bed.py generate`, which is what records the machine-and-mode
     fingerprint that makes two counts comparable at all;
   - `kit/beds/rewards.py` needs `code` in its `BED_FILES`, or a coding row raises
     `UnknownDataSource('livecodebench')` inside the trainer.

   `tests/test_kit_k5.py` carries a tripwire that skips with exactly those two sentences and starts
   checking the wiring the day it lands. **Every row that needs the bed is tagged `code`** — 106 of
   the 533, because a rehearsal stage after the coding stage mixes coding questions in too. Until the
   wiring lands, the runnable campaign is the three-job fallback:

   ```bash
   python scripts/make_k5_campaign.py --without-code > /somewhere/k5-three-jobs.yaml
   ```

   The coding bed is scored **last** at every point for the same reason, so that a campaign stopped
   at a coding row has already written everything a coding-free readout needs.
2. **Plan 4b's bar is only half measurable as this campaign stands.** `kit/plasticity.py compare`
   has two halves: internal signals from the probes (measured here, at every checkpoint, against the
   untrained model) and a learning-curve half — "does a job take 25 percent more steps to reach half
   its final gain in position 4 than in position 1" — which is read from an **in-trainer validation
   series** that this campaign does not produce. Every run sets `TEST_FREQ=-1`, which is K3's
   setting and the reason every comparison in this programme comes from `kit/eval_bed.py` afterwards
   instead. Turning validation on would cost generation time nobody has priced: at `TEST_FREQ=5`
   there are 8 validations a run, 72 runs, and the cost of one depends on the validation set and on
   the trainer's `val_kwargs.n`, which we have not measured. So the probes are collected, the
   comparison runs with the learning half marked *not measured*, and the tool reports `INCOMPLETE`
   rather than a verdict — which is what it is designed to do, and better than a verdict built on an
   unpriced departure. The run directories are already named `<job>-pos<POSITION>-seed<SEED>`, which
   is the name the learning half needs, so turning validation on later is a one-line change to the
   generator and no change anywhere else.
