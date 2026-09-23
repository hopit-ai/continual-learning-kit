# Two more seeds for K3 and for K4a

**What this is.** Each of these two campaigns is a **follow-up**: it adds two seeds to a package you
have already run, and changes nothing else. No new arm, no new setting, no new control, and no new
pilot.

| follow-up | adds | to |
|---|---|---|
| `campaigns/k3-seeds-3-4.yaml` | seeds **3 and 4** | `campaigns/k3-replay.yaml`, which ran seeds 0, 1, 2 |
| `campaigns/k4a-seeds-45-46.yaml` | seeds **45 and 46** | `campaigns/k4a-stuck-problems.yaml`, which ran seeds 42, 43, 44 |

**Why.** Every one of these packages is read as a difference between arms, and in your K0 runs two
seeds of the *same* method landed 3 to 4 points apart. Three seeds cannot tell a 3-point effect from
that spread; five can begin to. So we are asking for the two seeds we are missing, not for anything
new.

**They run in the same `$WORK` as the original**, with the original's own names for every run and
every folder — `$WORK/runs/a-seed3-a1`, `$WORK/k4a/forgetting/soft-seed46-a1`. That is deliberate:
the report tools you already ran then read all five seeds without being changed at all. Please do
**not** point them at a fresh work directory.

**And on the same machine as the original.** Every count these produce is compared with a count from
the original run, and a panel moves by up to 3 points between machines before any training — the
size of the effect we are looking for. The report refuses to mix two machines rather than quietly
average over them. If the original ran somewhere you no longer have, tell us and stop; that is not a
thing to work around.

## The one condition

The first row of each follow-up is its only pilot, and it trains nothing. It reads the verdicts the
runner already wrote for the **original campaign's pilots in this same `$WORK`** and requires every
one of them to have passed:

```
$WORK/campaign/k3-replay/<pilot row>/attempt-N/verdict.json
$WORK/campaign/k4a-stuck-problems/<pilot row>/attempt-N/verdict.json
```

If the original never ran here, or one of its pilots failed, that row **fails and every row below it
is refused** — nothing is trained and no GPU is held. The follow-up cannot repeat a pilot or invent
one of its own, because a pilot is what decided the package was worth running, and that decision has
already been made in this directory.

So: run the original first, let it finish, then run the follow-up **in the same place**.

## Run it from the same tag as the original

A follow-up extends a run, so it must use exactly the code that run used: check out the **same tag** you ran the original from (`kit-k3-v1` or `kit-k4a-v1`, or whichever it was), not a newer one. A newer tag may carry a changed scorer or reward, and five seeds scored by two rules are not five seeds of one experiment. If you are starting an original and its follow-up together, use one tag for both.

## What you need

Nothing new. Exactly the environment the original package used, with the same variables exported:

```bash
# K3 follow-up — the same environment as README-k3.md
export KIT=/work/continual-learning-kit/kit WORK=/work/k3-work
export SDPO_DIR=/work/SDPO SPIDER_ROOT=/work/spider_data GSM8K_ROOT=/work/gsm8k
```

```bash
# K4a follow-up — the same environment as README-k4a.md
export KIT=/work/continual-learning-kit/kit WORK=/work/sdpo-work
export SDPO_DIR=/work/SDPO MODEL_DIR=/work/models/Qwen3-8B NGPU=8
export K0_REPORT=/work/k0-report/report.json
```

The K3 follow-up needs about **70 GB more disk** (10 runs, a sharded checkpoint plus the merged copy
each). The K4a follow-up needs room for 6 more runs. As before, `$WORK/runs/<run>/train/global_step_*/actor`
can be deleted once that run's `hf-step*/config.json` exists.

## The commands

```bash
python $KIT/runner.py plan    $KIT/campaigns/k3-seeds-3-4.yaml
python $KIT/runner.py prepare $KIT/campaigns/k3-seeds-3-4.yaml --all
python $KIT/runner.py run     $KIT/campaigns/k3-seeds-3-4.yaml --all
python $KIT/runner.py status  $KIT/campaigns/k3-seeds-3-4.yaml
```

```bash
python $KIT/runner.py plan    $KIT/campaigns/k4a-seeds-45-46.yaml
python $KIT/runner.py prepare $KIT/campaigns/k4a-seeds-45-46.yaml --all
python $KIT/runner.py run     $KIT/campaigns/k4a-seeds-45-46.yaml --all
python $KIT/runner.py status  $KIT/campaigns/k4a-seeds-45-46.yaml
```

`prepare` needs **no GPU**. For K3 it writes the four new rehearsal mixtures (`rehearse10` and
`rehearse30` at seeds 3 and 4) and checks the model and bed files the original already built are
still there. For K4a it has nothing to build.

`run` stops at the first refusal or failure; fixing the cause and running the same command again
skips everything that already passed, and a retry is a new attempt in its own directory. Nothing is
ever overwritten — including the original's report, which stays exactly as it was.

## How long

**K3, seeds 3 and 4: about 27 GPU-hours (band 18 to 46), roughly 7 hours of wall clock.** Two thirds
of the original package, on the same arithmetic as `README-k3.md`, and still an estimate rather than
a measurement:

| Part | Work | Rate assumed | Result |
|---|---|---|---|
| Training | 2 stage A × 20 steps + 8 stage B × 40 steps = **360 steps** on 8 GPUs | 20 s/step (band 10 to 40) | 2.0 h (1.0 to 4.0) |
| Run overhead | 10 runs × model load, engine start, checkpoint merge | 5 min a run | 0.8 h |
| Scoring | 10 points × (100 Spider + 300 GSM8K + 300 panel) = **7,000 answers** on GPU 0 | 1.5 s an answer (band 1 to 3) | 2.9 h (1.9 to 5.8) |
| Scoring overhead | 30 scorings × engine start | 2 min each | 1.0 h |

**K4a, seeds 45 and 46: about 35 to 49 GPU-hours, roughly 5 to 7 hours of wall clock.** From your
own K0 step times, as in `README-k4a.md`:

| block | rows | about |
|---|---|---|
| the gate row | 1 | seconds, no GPU |
| the grid, 3 arms × 2 seeds at 40 steps | 6 | 4.4 to 6.1 hours on 8 GPUs |
| forgetting, all on GPU 0 | 6 | 25 minutes |
| the report | 1 | seconds, no GPU |

The grid rows are independent: with a second node the arms can be run in parallel with `--row`, as
long as every forgetting scoring happens on **one** machine's GPU 0 — the same one as before.

## What to send back

**The same report folders as the original, now holding five seeds**, written beside the originals and
not over them:

```
$WORK/k3/report-5seeds-a1/      k3-report.md and k3-report.json     (K3: seeds 0, 1, 2, 3, 4)
$WORK/k4a/report-5seeds-a1/     k4a-report.md and k4a-report.json   (K4a: seeds 42 to 46)
```

Each is the original report tool run over the whole tree, so it holds the old seeds and the new ones
in one table. Please keep the original `report-a1/` folders as well and send both.

If anything was refused or failed, send the output of `status` and the `output.log` of the failed row
from `$WORK/campaign/<campaign name>/<row>/attempt-1/`. If the gate row is what failed, the file that
explains it is `$WORK/k3/pilots-a1/pilots-passed.json` (or `$WORK/k4a/pilots-a1/`): it names each
pilot of the original and what its latest verdict here was.

As before, the per-point scoring folders under `$WORK/k3/eval/` and `$WORK/k3/forgetting/` are what
let us re-check a number without re-running anything, if you can spare the space.

## One thing these two seeds cannot buy, said plainly

Each **K4a** arm is paired seed for seed with one of your finished K0 runs, and K0 ran
`dose40-seed42`, `43` and `44` only. Seeds 45 and 46 have no K0 twin. They take each arm's **own**
score to five seeds, which is what tells us how much of an arm's result is seed noise; the paired
**arm-minus-control** difference stays a comparison over three seeds. The report prints both and
leaves the control column empty for the two new seeds rather than filling it with a run that does not
share their seed. We are not asking you to run new K0 controls.

K3 has no such gap: its comparison is between its own arms at the same seed, and all five seeds carry
every arm.
