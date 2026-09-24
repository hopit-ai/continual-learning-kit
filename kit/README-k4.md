# Package K4: a hint on the questions the model never solves

**The question.** On both of these jobs your untrained Qwen3-1.7B is **either right every time or never
right**. We sampled 8 attempts a question at temperature 1.0 — the trainer's own sampling — on 640
training questions of each bed:

| bed | never right in 8 | right all 8 times | a frontier model on the never-right ones |
|---|---|---|---|
| Spider (SQL) | 171 of 640 | 300 | 85 of 120 |
| FinQA (report arithmetic) | 200 of 640 | 200 | 62 of 120 |
| GSM8K | 33 of 640 | 394 | 22 of 33 — **too few; dropped** |

So a quarter to a third of the training data teaches this model nothing, and those answers **are**
inferable from the question: a frontier model gets most of them from the prompt alone. This package
asks whether **a short plan from a large model, given during training, teaches the student to answer
them on its own afterwards.** Every number we report is the student answering with **no hint in the
prompt**.

## The five arms, five seeds each, one bed at a time

A hint can reach the model on two routes — the **student** reads it, or only the **teacher** does —
and each route here carries its own no-hint control, so each route's number is a difference within one
trainer:

| arm | route | what changes | what it answers |
|---|---|---|---|
| **none** | GRPO | **nothing, and nothing is rerun.** Your finished K3 stage A is this arm for Spider; your finished K1c part B is it for FinQA | the baseline |
| **hint** | GRPO | the hint is appended to the prompt of every never-solved question, for the whole run | does scaffolding during training transfer to unaided answers? |
| **hint-faded** | GRPO | as `hint` for the first half of the run, then removed — a restart from the halfway checkpoint onto the same questions with no hints in them | does the model keep what the hint taught once the hint is gone? |
| **teacher-none** | SDPO | nothing but the trainer: the same command as `teacher-hint` with the feedback switch **off**, on the same training file, at the same dose and seeds | the SDPO route's control |
| **teacher-hint** | SDPO | the hint is returned as the reward function's `feedback` string, so **only the teacher** sees it. The student's training file is the control's own, unchanged | was the tool-use null result (K4a) the bed or the mechanism? |

**So each number belongs to a route, and the report says which:** `hint` minus `none` is the
GRPO-route effect and is the verdict; `teacher-hint` minus `teacher-none` is the SDPO-route effect,
and those two runs are **one trainer key apart**. `teacher-hint` minus `none` would be the hint and
the change of trainer together, so the report prints it for completeness and reads it as neither.

**What you run is still not our code.** `hint` and `hint-faded` are `kit/run_grpo.sh`, the launcher
your K3 and K1c runs already used, with a different training file and nothing else. `teacher-hint`
and `teacher-none` are `kit/run_sdpo_bed.sh`, which is the `kit/run_sdpo_toolalpaca.sh` you ran for K0
and K4a with this bed's data and reward file and the authors' own `FEEDBACK` switch as its one knob —
our tests compare the two commands key for key **at both switch positions**, and allow a difference
only in the data files, the reward file, `vars.task`, the group name and the output folder. The two
SDPO arms' own commands differ in `include_environment_feedback` and in nothing else, and each row
gates on the switch position it was meant to run at.

**Two things we want to be plain about**, because they are what a reader could otherwise misread:

- **The SDPO arms are read against each other, never against `none`.** `teacher-hint` changes the
  trainer (SDPO, with a teacher) as well as the hint, so it is `teacher-none` that makes it readable:
  both are SDPO, both serve the same hints through the same reward function, and only the trainer's
  use of the `feedback` string differs. The verdict is still `hint` minus `none`, which uses neither.
  The report also prints `teacher-none` minus `none` — the change of trainer with no hint on either
  side — because that is a trainer effect and it must not be mistaken for a hint effect.
- **Spider runs 20 steps, not 40.** Spider's frozen training split is 640 questions and the trainer
  makes one pass, so 20 updates × 32 questions is all there is — and your K3 stage A, which is this
  bed's `none` arm, is exactly that run. FinQA runs 40 steps against K1c's 40. Each arm is at its own
  control's dose, because a double dose against a single-dose control would measure the dose.

## The hint, and why you can believe it is not the answer

A large open-weight model **you** serve with vLLM on one node is asked, per never-solved question, for
a three-to-five-line plan. The request text is fixed in `kit/hints.py` and its sha256 is recorded with
every run. Then **every hint passes a leakage filter before it is used**:

- on Spider: it may name the gold query's tables and columns, but it may not contain the gold query,
  and it may not contain a complete `SELECT` statement of its own (upper-case SQL, SQL in backticks, or a
  code-shaped `select ... from <table>`; an English step such as "Select the titles from the publication
  table" is a plan, not a statement);
- on FinQA: it may not contain the gold number, nor any number within 1 percent of it at any of the
  three scales the bed accepts, nor state a yes/no answer;
- on either: it may not write an `Answer:` line, and it must be three to five lines.

Dropped hints are counted **by reason** and committed, so you and we can read every one. The pilot bar
is that at least **90** percent of the never-solved set still has a hint afterwards.

At test time nothing has to remove the hint: `kit/eval_bed.py` builds its prompts from the bed's own
data, so no measurement in this package can contain one.

## What you need

Everything from `README-partner.md` sections 0 to 4, plus:

- **your finished K3 and K1c package directories** (`$K3_ROOT`, `$K1C_ROOT`) — the folders holding
  `eval/` and `forgetting/`. These are the `none` arm. Nothing else is a substitute, and we would
  rather stop than invent one. **K4 therefore comes after both of those packages**, and after the K3
  seeds follow-up (`kit-seeds-v1`), which is what takes K3's stage A from three seeds to five. If you
  have only K3's first three seeds when you get here, run K4 at those three
  (`--seeds 0,1,2` on the report row) and tell us: three seeds against a 3-to-4-point seed spread is
  worth less, and we would rather know which it was than guess from the numbers.
- **a large open model served on an OpenAI-compatible endpoint** (vLLM's own `--served-model-name`
  works). Which model is yours to choose from what your cluster holds — Qwen3-235B-A22B, or the
  largest Llama you have. Tell us which, because it goes in the write-up. It only has to be up for the
  two `*-hints` rows, about 700 short requests in total. A reasoning model (Qwen3, R1) is fine: the
  request switches its thinking off (`chat_template_kwargs.enable_thinking=false`, which vLLM
  understands) and any thinking block that still comes back is stripped before the filter sees the
  plan. The generate manifest counts how many replies thought anyway.
- **Spider and FinQA on disk**, as in K3 and K1c.

**Run it in the same container, on the same machine, as K3 and K1c.** The `none` arm was scored there,
and across machines a panel moves by up to 3 points before any training — the size of the effect we
are measuring. The report refuses to mix two machine fingerprints.

## Run it

```bash
export KIT=/work/continual-learning-kit/kit WORK=/work/k4-work
export SDPO_DIR=/work/SDPO SPIDER_ROOT=/work/spider_data FINQA_ROOT=/work/FinQA/dataset
export HINT_BASE_URL=http://localhost:8000/v1 HINT_MODEL=<the model you served>
export K3_ROOT=/work/k3-work/k3 K1C_ROOT=/work/k1c-work/k1c
```

```bash
python $KIT/runner.py plan $KIT/campaigns/k4-hints.yaml
```

```bash
python $KIT/runner.py prepare $KIT/campaigns/k4-hints.yaml --all
```

`prepare` needs no GPU. It writes both beds' training files, checks the row counts the doses need,
asks your hint endpoint for its model list, and checks `$K3_ROOT` and `$K1C_ROOT` are on disk. Start
the hint-giver **before** this step. If any of it is wrong, nothing is wasted.

```bash
python $KIT/runner.py run $KIT/campaigns/k4-hints.yaml --all
```

That runs everything in order and stops at the first refusal. The order is: the untrained model and
the machine check, then Spider completely, then FinQA, then the report.

Per bed the hint work comes first — the never-solved set (one GPU, under two minutes), the hints
(your served model), the filter and the hinted files (no GPU at all) — and then four two-step pilots,
one an arm. Nothing large starts until those have passed. You can run a single row at any time:

```bash
python $KIT/runner.py run $KIT/campaigns/k4-hints.yaml --row spider-stuck
```

If a row fails, this prints the state of everything:

```bash
python $KIT/runner.py status $KIT/campaigns/k4-hints.yaml
```

## Time

At 15 seconds a step on 8 × H100 (your K3 numbers), and your measured scoring times:

| block | rows | about |
|---|---|---|
| the untrained model, twice, plus both beds' held-out sets | 5 | 1 hour |
| Spider: the never-solved set, the hints, the filter, the files | 5 | 30 minutes, at most one GPU |
| Spider: 4 pilots | 4 | 25 minutes |
| Spider: 20 runs at 20 steps (the faded arm is 10 + 10) | 25 | 2 hours wall, ~16 GPU-hours |
| Spider: 20 held-out scorings and 20 panels, GPU 0 | 40 | 4 hours |
| FinQA: the never-solved set, the hints, the filter, the files | 5 | 1 hour, at most one GPU |
| FinQA: 4 pilots | 4 | 25 minutes |
| FinQA: 20 runs at 40 steps | 25 | 3.5 hours wall, ~27 GPU-hours |
| FinQA: 20 held-out scorings (1,147 questions each) and 20 panels, GPU 0 | 40 | 9.5 hours |
| the report | 1 | seconds, no GPU |

**Roughly 22 hours of wall clock and 65 to 75 GPU-hours**, of which about 43 are the training. The
`teacher-none` arm is 11 of those GPU-hours (about 4 on Spider and 7 on FinQA) and is what makes the
SDPO route readable at all. The grid rows are independent: if you have a second node the two beds can
run in parallel with `--row`, as long as **every** scoring happens on one machine's GPU 0.

## The stop rule

Stop and tell us, rather than working around it, if any of these happens:

- **The hint coverage bar fails** (under 90 percent of the never-solved set has a usable hint). Send
  `hints-dropped.jsonl` and `filter.json`. The fix is our request text or our filter — never a lower
  bar.
- **The never-solved share is outside its band** (Spider 0.20 to 0.35, FinQA 0.24 to 0.39). Our own
  measurement was 0.27 and 0.31; far outside means this is not the model or the data we measured, and
  the hints would be for the wrong questions.
- **The untrained held-out score is outside its band** (Spider 55 to 85 of 100, FinQA 520 to 760 of
  1,147). These are derived from the frontier comparison, not measured by us on your machine, so the
  bands are wide on purpose — outside them the setup differs and no later number would be comparable.
- **`kit/hints.py apply` refuses because a hinted prompt would exceed 2,048 tokens.** verl silently
  drops a prompt over that limit, which would shorten the dose and leave no checkpoint at all, so the
  refusal is deliberate — but it should never fire: we tokenized both beds with the pinned Qwen3
  tokenizer first. Spider's longest training prompt is 1,094 tokens (954 of headroom) and FinQA keeps
  1,535 of its 1,600 rows even with a 400-token hint on every one, against the 1,280 the dose needs.
  If it fires anyway, something differs from our copy of the data: send us `apply.manifest.json`
  rather than working around it.
- **A pilot fails.** Send the row's `verdict.json` and the run's `metrics.jsonl`. A failed pilot stops
  every later row, including the other bed's — that is the runner's design. Spider runs first and
  completely, so a FinQA failure costs no Spider row; a Spider failure does block FinQA, and once we
  have answered you can run the FinQA rows with `--row`.
- **`response_length/mean` falls below 16 tokens and stays there**, or a GPU has under 2 GB free. The
  two SDPO arms are the first time SDPO has run on either of these beds, and SDPO holds a teacher copy
  as well as the actor. Their pilots run in that order — `teacher-none` first — so a memory or a
  length failure shows up on the control, before the treatment is paid for.
- **Anything needs a setting changed to fit.** Batch size, attempts per question, learning rate,
  warm-up, the step counts and the 2,048-token prompt limit are what make these numbers comparable
  with K3 and K1c. Tell us instead of shrinking one.

We would rather have one bed and an honest stop than two beds and a number we cannot read.

## What to send back

One folder of small text files:

```
$WORK/k4/report-a1/        k4-report.md and k4-report.json
$WORK/k4/hints/            every stuck.json, hints.jsonl, filter.json and apply.manifest.json
```

The hints themselves matter as much as the scores: they are the treatment, and we want to read them.

The report holds, per bed and arm: the unaided held-out score at five seeds with its mean and spread,
the paired gain over your `none` runs seed for seed, the held-out questions split into the ones the
untrained model already answered and the ones it did not, the three general panels, and one verdict
line a bed against the 3-point bar we wrote down before any of these numbers existed. Above those it
prints one line per route — `hint` minus `none`, `teacher-hint` minus `teacher-none`, and the change
of trainer on its own — so that every number in it is attributable to one change. FinQA's items
whose own published answer disagrees with the program that produced it (about 8 percent) are excluded,
as in every FinQA number we have sent you. It is recomputed from the raw files every time and needs
only the Python standard library.

Please keep `$WORK/runs/` until we have read it. Anything you noticed that the files would not show —
a restart, a node change, which model you served, a queue wait — belongs in the email, in your own
words.
