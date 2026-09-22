# Package K1c: what does a job look like when it is learned on its own?

**The question.** Every later experiment in this programme compares a model that learned two jobs with one that learned one. That comparison needs the "learned one" row, measured the same way, and we do not have it. K1c measures it.

Every job here is learned **alone, from the untrained model, with reward-only RL** — the SDPO authors' own `--config-name baseline_grpo`, unmodified, with no teacher and no self-distillation. The model is told only whether each answer was right.

There are two parts.

- **Part A, Qwen3-8B on the tool-use task** (ToolAlpaca, the same `datasets/tooluse` as K0). Two arms: **`full`**, which updates every weight, and **`lora`**, which trains a rank-64 adapter on every linear layer with a ten-times learning rate. Five seeds each (42 to 46), 40 steps — K0's dose exactly, so seeds 42, 43 and 44 sit beside the K0 SDPO run with the same seed; 45 and 46 have no K0 partner and are there because K0 showed runs differ by 3 to 4 points by seed alone.
- **Part B, Qwen3-1.7B** (the K3 model). Two jobs learned alone: **GSM8K** grade-school maths and **FinQA** annual-report arithmetic. Five seeds each, 40 steps × 32 questions, which is K3's stage-B dose.

**SQL learned alone is not re-run.** K3's stage A already is exactly that at 1.7B. If you have finished K3, point `K3_REPORT` at its report and the row appears in the readout; otherwise the readout says where to find it.

**Coding is out of scope.** The kit has no code sandbox that runs without Modal, so there is no bed on which a coding task can be learned alone. That row is missing on purpose, and it is the obvious gap in this package.

## Why GRPO, and why this is the package that can ask about adapters

Two reasons, and the second one is the more interesting.

1. A "learned alone" row has to be the simplest thing that learns. Reward-only RL is that.
2. **Under SDPO, LoRA quietly changes the method.** SDPO's teacher is a second copy of the model held inside the trainer and moved slowly towards the student. With LoRA switched on, the pinned trainer gives that teacher *its own, differently random* adapter, so for the first half of a run the distillation target points somewhere else entirely. We found that reading the code, and it is why the LoRA-under-SDPO package was dropped. Under `baseline_grpo` with the KL loss off, **no reference model is built at all** — so there is no teacher for LoRA to corrupt, and the adapter question can finally be asked cleanly.

The LoRA arm's settings are not ours: rank 64 on all linear layers, alpha 32, and a learning rate ten times the full-training one, following *LoRA Without Regret* (Schulman et al., Thinking Machines, 2025).

## The one thing in this package that can go silently wrong

`python -m verl.model_merger merge` **does not fold a LoRA adapter into the model.** It saves the *untrained base weights* as the model and writes the adapter into a subfolder beside them — with `lora_alpha` hard-coded to `0`, which scales every adapter weight to nothing. No error is raised.

If that were missed, the forgetting panels would be scored on the untrained model, they would come out unchanged, and the package would report **"LoRA forgets nothing"** — the very answer it exists to test — with no LoRA weight ever scored. Every file is present, so an ordinary "did the merge produce a model?" check passes.

So each LoRA run ends with an explicit fold step (`kit/fold_lora.py`) that:

- corrects `lora_alpha` to 32 in a copy of the adapter, never touching the original;
- recomputes every module's delta itself, in full precision, and **refuses if any of them is zero** — which is what the merger's `lora_alpha: 0` makes all of them;
- cross-checks the alpha against the adapter the trainer wrote beside its own checkpoint, when that file is there, and refuses a disagreement rather than preferring one side;
- merges, then checks that every adapted weight moved and that **nothing else did**;
- writes nothing at all if any of that fails, and exits non-zero.

`hf-step40/` therefore means the same thing in both arms: a full model the scorers can read. `hf-step40/fold.json` is the receipt, and the readout prints it.

## It starts with pilots, and a pilot can stop the package

In order, and nothing below a pilot runs until it has passed:

1. **the untrained 8B is scored twice** on the 300-question panel and the two scorings must agree (295 of 300 answers identical, no changed verdict, same machine). This costs 25 minutes on one GPU and needs no training. If one machine cannot score one model the same way twice, no difference this package measures means anything.
2. **two steps of each Part A arm.** Each must exit cleanly, leave a merged model, keep responses above 16 tokens, and land the *untrained* validation score inside 0.555–0.600 — the band your own K0 runs measured (five readings, 0.571 to 0.586). The LoRA pilot must also have folded its adapter and actually changed weights.
3. **two steps of each Part B job**, after Part A's grid, so a Part B problem does not cost Part A's GPU-hours. Same bars, minus the validation band: we have no measurement of our own for the untrained 1.7B on either bed, and a band we invented would be gating on a guess.

**A failed pilot stops the campaign, including rows in the other part**, because the runner requires every earlier pilot before every later row. That is the design. Send us the pilot's `output.log` and `metrics.jsonl` and we will drop or re-specify that arm; the rest can then be run with `--row`.

Every bar here is something a clean run cannot fail by luck: an exit code, a model on disk, a length floor, a score measured *before* any training, and whether an adapter reached the weights. Nothing is gated on a quantity two steps cannot measure — those are printed in the readout instead.

## What you need

- **One node with 8 GPUs** for the training rows. Scoring rows use **GPU 0 only** and are pinned to it: counts made on two different GPUs are not comparable, which is what the first three rows check before anything else runs.
- **The container and the pinned SDPO install** exactly as in `README-partner.md` sections 1 and 2 (`nvcr.io/nvidia/vllm:25.12.post1-py3`, `lasgroup/SDPO` at `7c457fc1b1f636ae794eb0362ba37d4743b06fbc`, `pip install -e /work/SDPO`).
- **Your Qwen3-8B snapshot** at `MODEL_DIR`, the same one K0 used. The 1.7B is downloaded by `prepare`.
- **Your finished K0 report** at `K0_REPORT`. Part A is read seed for seed beside its dose-40 runs.
- **GSM8K**, downloaded by you:

  ```bash
  hf download openai/gsm8k --repo-type dataset --local-dir /work/gsm8k
  ```

  `GSM8K_ROOT` points at `/work/gsm8k`. 100 of the 300 questions on our general panel are GSM8K test questions, so the bed removes any panel question from the training file and refuses to write a file in which one remains.
- **FinQA**, downloaded by you: clone `github.com/czyssrs/FinQA` and point `FINQA_ROOT` at its `dataset/` directory (the one holding `train.json`, `dev.json`, `test.json`). The kit ships no FinQA text.
- **About 260 GB of disk.** The 8B base is about 16 GB and each of the six Part A runs leaves a sharded trainer checkpoint plus the merged copy (about 35 GB a run); a LoRA run leaves **two** merged copies, the merger's and the fold's, so about 51 GB. The 1.7B runs are about 7 GB each. If space is tight, delete `$WORK/runs/<run>/tool-grpo/global_step_*/actor` and `$WORK/runs/<run>/merged-step40` once that run's `hf-step40/config.json` and `hf-step40/fold.json` exist — nothing downstream reads either, and the scoring folders you send back are text.

Nothing is uploaded. No Hugging Face login is needed: the only model downloaded is the public `Qwen/Qwen3-1.7B`.

## The four commands

```bash
export KIT=/work/continual-learning-kit/kit WORK=/work/k1c-work
export SDPO_DIR=/work/SDPO MODEL_DIR=/work/models/Qwen3-8B NGPU=8
export GSM8K_ROOT=/work/gsm8k FINQA_ROOT=/work/FinQA/dataset
export K0_REPORT=/work/k0-report/report.json
export K3_REPORT=/work/k3/report-a1/k3-report.json     # optional, adds the SQL-alone row
```

```bash
python $KIT/runner.py plan $KIT/campaigns/k1c-grpo-baselines.yaml
```

```bash
python $KIT/runner.py prepare $KIT/campaigns/k1c-grpo-baselines.yaml --all
```

`prepare` needs **no GPU**: it converts the ToolAlpaca parquet with the reference's own script, downloads the 1.7B, builds both bed files, checks that the fold step's dependencies import, and counts the rows in each training file. No GPU ever waits on a download.

```bash
python $KIT/runner.py run $KIT/campaigns/k1c-grpo-baselines.yaml --all
```

```bash
python $KIT/runner.py status $KIT/campaigns/k1c-grpo-baselines.yaml
```

`run` stops at the first refusal or failure; fixing the cause and running the same command again skips everything that already passed. A retry of one row is a new attempt in its own directory — nothing is ever overwritten.

## How long this takes, and exactly what the numbers rest on

**Estimated 80 to 155 GPU-hours, most likely about 100, and 20 to 30 hours of wall clock.** Five seeds per arm instead of three is most of that: K0 showed that three seeds cannot separate methods whose results differ by less than the seed-to-seed spread.

Part A's arithmetic rests on a **measurement**, not a guess: your own K0 runs recorded a per-step time on 8 × H100 with a median of **51 seconds** over 149 measured steps (mean 56, range 45 to 116, the high end being the steps that also ran a validation). Part B's rests on an **estimate** scaled from that, and says so.

| Part | Work | Rate | Result |
|---|---|---|---|
| A, training | 2 pilots × 2 steps + 10 runs × 40 steps = **404 steps** on 8 GPUs, plus 9 validations a run | 51 s/step measured (band 45 to 65); a validation of 68 questions × 16 samples adds about 60 s | 9.5 h wall (8.0 to 12.0) = **76 GPU-h** |
| A, scoring | 12 panel scorings × 300 questions = 3,600 answers on GPU 0 | 1.5 s an answer (band 1 to 3); the scorer runs eager with CUDA graphs off, which is slower on purpose and is what makes two scorings agree | 1.5 h + 12 × 2 min engine start = **1.9 GPU-h** |
| A, folding | 5 LoRA runs, on the CPU | about 5 min a run: load 16 GB, merge, write 16 GB | no GPU |
| B, training | 2 pilots × 2 steps + 10 runs × 40 steps = **404 steps** on 8 GPUs | **15 s/step, ESTIMATED** (band 8 to 30) | 1.7 h wall (0.9 to 3.4) = **13 GPU-h** (7 to 27) |
| B, scoring | 12 bed scorings (5 × 300 GSM8K, 5 × 1,147 FinQA, plus the untrained model on both) + 11 panel scorings = **11,982 answers** on GPU 0 | 1.5 s an answer (band 1 to 3) | 5.0 h + 23 × 2 min = **5.8 GPU-h** (4.1 to 10.8) |
| Run overhead | 24 runs × model load, vLLM engine start, checkpoint merge | 5 min a run for the 8B, 3 min for the 1.7B | 1.6 h |

**Where the 1.7B's 15 s/step comes from, and why it is only an estimate.** It is K0's measured 51 s/step for an 8B, scaled down for a model 4.7 times smaller on the same 8 GPUs, with the SDPO teacher's extra forward pass removed. Step time here is dominated by generation, which scales sub-linearly with model size, so the scaling is not 4.7× and the band is wide. Our only direct 1.7B measurements are two-step smokes on a *single* GPU that include engine start, model load and a merge in the total (626 s for the K3 plumbing smoke on 20 September), which bounds this loosely at best. **Your Part B pilot is the first real measurement, and we would like the number.**

If the pilots stop the package, you will have spent about 1.5 hours of the above.

## What to send back

One folder: `$WORK/k1c/report-a1/` (`k1c-report.md` and `k1c-report.json`). If anything was refused or failed, also send `status` and the `output.log` of the failed row from `$WORK/campaign/k1c-grpo-baselines/<row>/attempt-1/`.

If you can spare the space, these let us re-check any number without re-running anything, and are small text files plus one `responses.jsonl` per scoring:

- `$WORK/k1c/eval/` and `$WORK/k1c/forgetting/` — the per-point scorings;
- `$WORK/runs/*/metrics.jsonl`, `run-summary.json`, `train-summary.json`, `env/argv.txt`;
- `$WORK/runs/*/hf-step40/fold.json` — the LoRA receipts;
- `$WORK/runs/*/validation/` and `$WORK/runs/*/rollouts/` for the Part A runs, from which every accuracy in the readout is recomputed per question.

## What we expect, so you can tell if something is off

| Row | What should happen |
|---|---|
| `base8b-1` | the untrained Qwen3-8B gets 245 to 257 of 300 panel questions right; ours is 251, two other machines gave 250 and 251 |
| `repeatable` | the two scorings agree on at least 295 of 300 answers with no changed verdict (on our machines: 300 of 300) |
| `pilot-full`, `pilot-lora` | both exit clean, merge, and report an untrained validation of 0.555 to 0.600 — the same band in both arms, because at step 0 a LoRA model *is* the base model |
| `pilot-lora` | `fold.json` says every adapted module changed, and `max_abs_delta` is not tiny |
| `full-seed*`, `lora-seed*` | the trainer exits cleanly, writes a checkpoint at step 40, and merges (and folds) it |
| `base17b-finqa` | the untrained 1.7B gets *some* FinQA questions right; near zero means the prompt or the data root is wrong, not the model. A system with no reasoning at all scores 5.3 percent |
| `finqa-seed*` | a checkpoint at step 40. **If this one fails to merge, the likely cause is the training file**: verl silently drops every prompt over 2,048 tokens, and about 1.5 percent of FinQA's are, so the file is written with 1,600 rows to leave 1,280 after the filter. Send us the row count `prepare` printed |
| `report` | both parts in one table, each arm beside the K0 SDPO run at its seed, and a fold receipt per LoRA run |

**We do not know what any of this will show.** GRPO beating SDPO, losing to it, or not moving at all are three results and all three are worth the same to us; so are LoRA matching full training, forgetting less, or failing to learn. The one thing we would not learn anything from is a number we cannot trust, which is what all those bars are for. Please send the report whichever way it comes out.

One thing to say in advance about the seeds. A seed here sets the data order and the parameter initialisation, but the reference leaves vLLM's sampling unseeded, so two runs at one seed still differ in their rollouts. Part A's five seeds are five independent repeats of the same dose, and the readout shows every seed as well as the mean, so you can see the spread rather than take our word for it. Part B's launcher reads its training file in written order (`data.shuffle=False`, as the reference's own SQL runs did), so there a seed changes only the sampling.
