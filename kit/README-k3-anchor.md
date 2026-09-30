# K3, the dose-and-anchor test: which dose does not break the model, and does an anchor help at that dose?

## What we know so far

**Twenty GRPO steps on Spider do not teach the untrained Qwen3-1.7B SQL.** The reference dose (640 training questions, 20 steps, lr 1e-5) has been run **nine times**, each run on a single node, and the nine runs average **about −7** on the 100 held-out questions. Receipt 238 found why by reading the answers instead of counting them. The untrained model writes a JOIN in **41 of its 100 answers**. After training it writes a JOIN in **0 to 9** of them and nests subqueries instead, and the questions it loses are the ones whose gold answer joins tables. The one run that read +5 had the same collapse.

**Your K1c Part B found the same thing on every other bed.** At K3's stage-B dose (40 steps of 32, lr 1e-5), GSM8K fell from **245 to 137 of 300**. That is not a model doing worse arithmetic: **the 40-step GSM8K dose breaks the model's answer format.** Its answers stop parsing, with **207 of 300 malformed** on one seed. FinQA fell from 633 to 414 of 1147, with 1017 malformed on one seed.

**Every one of those runs used lr 1e-5.** That is the value the authors used for an 8B model. The trainer's own documentation uses **1e-6**, and even the 8B collapsed on one seed of five at 1e-5 under plain GRPO. So the question is no longer "which dose gains 5". It is **"which dose does not break the model, and does an anchor help at that dose?"**, asked on Spider and on GSM8K.

**If a GSM8K arm clears, this test replaces K3's stage-B dose.** K3's stage B is the 40-step, lr 1e-5 GSM8K dose that K1c Part B found breaks the format. If a Spider arm clears, its dose is the Spider dose to carry forward.

## The knobs

All of them are knobs of `kit/run_grpo.sh`, and each run's `train-summary.json` records them.

- **The learning rate** (`LR`, recorded as `lr`).
- **The KL anchor** (`KL=1`, `KL_COEF`): the trainer adds a penalty for drifting away from a frozen copy of the model it started from, here the untrained one, so every step is pulled back toward the answers it already gave.
- **The entropy bonus** (`ENTROPY_COEF`): the trainer rewards the policy for keeping several ways of answering open, so it cannot collapse onto a single style.

**The KL arms build a frozen reference copy of the model next to the one being trained, so they use more GPU memory than plain GRPO.** A 1.7B model fits on 8×H100 with room to spare, and `OFFLOAD` stays off. If a KL row does run out of memory, send us its `output.log` and do not change the arm.

## The eleven arms, three runs each

Every run starts from the untrained model and runs **three times** (`<arm>-r1`, `-r2`, `-r3`; `SEED` 1, 2, 3 is only a label, as in probe 2). Every run uses all 8 GPUs and `FILE_LOG=1`.

**Spider**: 20 steps on the 640-row file, scored on the Spider held-out 100.

| Arm | lr | Anchor | What it asks |
|---|---|---|---|
| `ref20` | 1e-5 | none | The control: three more draws of the recipe that collapses. |
| `kl01` | 1e-5 | `KL=1`, `KL_COEF=0.01` | Does a light pull toward the starting model keep the joins at the reference lr? |
| `kl10` | 1e-5 | `KL=1`, `KL_COEF=0.1` | The same pull, ten times stronger. |
| `ent01` | 1e-5 | `ENTROPY_COEF=0.01`, no KL | Does keeping the policy spread out stop the collapse? |
| `lr06` | 1e-6 | none | Does the trainer's documented lr alone stop the fall? |
| `lr36` | 3e-6 | none | Between the two. |
| `lr06kl` | 1e-6 | `KL=1`, `KL_COEF=0.01` | Does the anchor help at the small lr? |

**GSM8K**: 40 steps on the 1,280-row GSM8K file (K3's stage-B dose), scored on the GSM8K held-out 300. The general forgetting panel is not scored here.

| Arm | lr | Anchor | What it asks |
|---|---|---|---|
| `g-ref` | 1e-5 | none | The control: K1c Part B's recipe, the one that breaks the format. |
| `g-lr06` | 1e-6 | none | Does the documented lr alone keep the format? |
| `g-lr06kl` | 1e-6 | `KL=1`, `KL_COEF=0.01` | Does the anchor help at the small lr? |
| `g-kl01` | 1e-5 | `KL=1`, `KL_COEF=0.01` | Does the anchor alone rescue the reference lr? |

Each Spider run's environment is exactly probe 2's `ref20` runs' except for `NAME`, `SEED` and the arm's own knobs. Each GSM8K run's environment is the Spider `ref20` run's except for those, `STEPS`/`SAVE_FREQ` (40), and the GSM8K training and validation files. The GSM8K file is `$WORK/data/gsm8k-1280`, built by the same `beds/gsm8k.py prepare --limit 1280` line as K3 and K1c use, so an existing K3 work tree already has it. `TASK` is left at the launcher's default, as it was in K3's stage B and K1c Part B.

Run directories are `$WORK/runs/anchor-<arm>-rN-seedN-a*`, so they never collide with probe 2's runs if `WORK` is the same tree. Everything else goes under `$WORK/k3anchor/`. Every KL run also checks that `train-summary.json` recorded `kl: 1`, so a run where the launcher silently dropped KL fails its row instead of passing as anchored.

That is **33 training runs** (21 at 20 steps, 12 at 40 steps), 33 held-out scorings plus the untrained model scored once on each bed (**35 scorings**), 33 delta rows and one report: **102 rows**.

## The pass rule, per bed

**On both beds, an arm clears only if all of these hold:**

1. its **mean** delta over its three runs is **at least 0** (it stops falling);
2. at least **2 of its 3 runs** are individually at 0 or better;
3. its mean tokens per correct answer are within **1.5×** the untrained model's.

**On Spider it must also keep its joins:** averaged across its runs, **at least 30% of its answers write a JOIN**. The untrained model's share is 41%.

**On GSM8K it must also keep its answer format:** averaged across its runs, the share of answers the bed cannot parse (`incorrect_format` in `bed-score.json`) may be **at most 5 percentage points of the 300 above the untrained model's share**. That is the number K1c Part B saw explode.

The rules live in the campaign file's `decision:` block, one per bed, and the report prints them at the top. A recipe that ends at 0 with its repertoire intact has fixed the thing that is broken. A recipe that gains a few questions by the same collapse has not.

The report (`kit/k3_dose_report.py --campaign kit/campaigns/k3-anchor.yaml`) groups the arms by bed. For each bed it prints:

- a **held-out table**: each run's churn and McNemar p, and a bold mean row per arm with its verdict;
- a **"What the answers look like" table**. On Spider this is the share of answers writing a JOIN, the share nesting a subquery, and the style shift, from `kit/repertoire.py`. On GSM8K it is the format failures (count, share, and points against the untrained model), the median answer length in tokens, and the truncation count.

After the tables, **"What to do next"** names, per bed, the arms that clear and the best-clearing arm by mean delta. The training signals and the cost of a correct answer are printed for every run, as in probe 2. You can read the SQL constructs of any two Spider scorings yourself:

```bash
python $KIT/repertoire.py compare --reference $WORK/k3anchor/eval/base-spider-a1 \
    --trained $WORK/k3anchor/eval/lr06kl-r1-spider-a1
```

## How to run it

The environment and `WORK` are the same as for K3, K1c and probes 1 and 2. The model, the Spider bed and the GSM8K file are read from where K3 put them. Nothing is downloaded if they are already there.

```bash
export KIT=/work/continual-learning-kit/kit WORK=/work/k3-work
export SDPO_DIR=/work/SDPO SPIDER_ROOT=/work/spider_data GSM8K_ROOT=/work/gsm8k
python $KIT/runner.py plan  $KIT/campaigns/k3-anchor.yaml     # prints all 102 rows, runs nothing
python $KIT/runner.py batch $KIT/campaigns/k3-anchor.yaml
```

**This is one job on one node.** `batch` runs `prepare --all`, which needs no GPU, and then `run --all`. Training uses all 8 GPUs, and every scoring runs on GPU 0.

**All 35 scorings must come from the same node**, as in probes 1 and 2. If a `*-delta` row reports `same-machine 0`, do not read that delta.

## What it costs

**About 11 to 15 hours of wall clock on the 8×H100 node, so roughly 90 to 120 GPU-hours of node time.** These figures come from your own run times, not guesses:

| Part | How many | Time each | Total |
|---|---|---|---|
| Spider runs, 20 steps | 21 | 12 to 16 min (probe 1: 716 s to 975 s end to end, including load, vLLM start, training and merge) | 4.2 to 5.6 h |
| GSM8K runs, 40 steps | 12 | about twice a Spider run, so 24 to 32 min | 4.8 to 6.4 h |
| Scorings on GPU 0 | 35 | a few minutes each (thirteen Spider scorings took under an hour) | about 2 to 3 h |

Training alone is 9 to 12 hours. The fifteen KL runs (`kl01`, `kl10`, `lr06kl` on Spider; `g-lr06kl`, `g-kl01` on GSM8K) also compute the reference model's log-probabilities at every step, so allow a little more for them. If the node cannot be held that long, tell us before splitting the job: every scoring must still come from the same node.

## What to send back

```bash
python $KIT/collect.py --work $WORK --out k3anchor.tar.gz
```

That one archive holds:

- the report;
- the 35 scoring folders (`bed-score.json` and `responses.jsonl`, which the JOIN and format tables are computed from);
- the 33 delta files;
- every run's `metrics.jsonl` and `train-summary.json`;
- the runner's records.

It contains **no checkpoints**: `collect.py` leaves weights and `hf-step*` folders out and lists in its manifest what it left out.

## Two things to be clear about

**The 33 `*-delta` rows are readouts, not gates.** They record each run's paired difference and check that both scorings came from one machine. A run that falls does not stop `run --all`. The verdict comes from the report.

**What a clearing arm means.** If a GSM8K arm clears, its dose replaces K3's stage-B dose, and the best-clearing arm by mean delta is the one named. If a Spider arm clears, stage A has a recipe that keeps the model's SQL repertoire. Looking for a gain *with* that recipe is a separate package.

If no arm clears on a bed, the report names that bed's best mean, and its answers table shows whether any dose or anchor slowed the collapse. That is also an answer. It says that neither a smaller step nor staying near the start keeps this model intact at this dose, and we look elsewhere (the 8B student) before spending more node time on it.
