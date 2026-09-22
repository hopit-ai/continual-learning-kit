# Plasticity: has the model used up its room to learn?

**The question** (research target Q11, plan 4b). A model learns one job, then a second, then a third. By the time it starts the fourth, is it *slower to learn* than it was at the first — not worse at the old jobs, but less able to take on a new one? That is what people mean by "losing plasticity". `kit/plasticity.py` measures it. It does not fix it, and nothing in it is an argument that any fix would work.

It **trains nothing and changes no weight**. It reads checkpoints you already have and `metrics.jsonl` files your runs already wrote. One GPU, forward passes only, no vLLM; a tiny model runs on a laptop CPU.

## The two commands

```bash
# 1. the untrained model first: it is the reference every later probe is measured against
python $KIT/plasticity.py probe --model /work/models/Qwen3-8B --device cuda --dtype bfloat16 \
                                --out $WORK/plasticity/untrained

# 2. every checkpoint in the sequence, against that reference and that base
python $KIT/plasticity.py probe --model $WORK/runs/stage4/hf-step40 --device cuda --dtype bfloat16 \
                                --base /work/models/Qwen3-8B \
                                --reference $WORK/plasticity/untrained/plasticity.json \
                                --out $WORK/plasticity/stage4

# 3. the readout over the chain, with the training curves
python $KIT/plasticity.py compare --chain $WORK/plasticity/untrained/plasticity.json \
                                          $WORK/plasticity/stage1/plasticity.json \
                                          $WORK/plasticity/stage4/plasticity.json \
                                  --metrics $WORK/runs --out $WORK/plasticity/report-a1
```

`probe` writes one `plasticity.json`. `compare` writes `plasticity-compare.json` and `plasticity-compare.md` and prints the second. Neither ever overwrites: an existing `--out` is refused, and a retry is a new directory.

## What is measured, and what each number means

Every probe uses **the same 200 prompts**: 200 of the 300 questions on the frozen general panel (`kit/panels/general-v1.jsonl`), taken one panel at a time in a fixed rotation over the panels in sorted order, each panel sorted by id — 67 maths, 67 knowledge, 66 instruction-following. No random seed is involved; every machine draws the same 200 in the same order, and the result carries a sha256 of exactly which ones, so two probes can be *proved* to have measured the same thing. The model is never asked to generate: each prompt is a single forward pass and what is recorded is what happened inside.

| Number | In plain language | Which way is worse |
|---|---|---|
| **dormant share** | The share of the MLP's hidden units whose activation never once reaches 0.001 in size, on any token of any of the 200 prompts. A unit that never fires is carrying nothing; whatever capacity it represented is not available to the next job. | **up** |
| **saturated share** | The share of measured activations that are larger than the *untrained* model's own 99.9th percentile. On the untrained model this is 0.001 by construction (that is what a 99.9th percentile is), so the only informative thing is how far above 0.001 a later checkpoint sits: activations have grown out of the range the model was calibrated in. | **up** |
| **effective rank** | Per layer, of the residual stream: take the activations collected at 8 evenly spaced token positions per prompt, take their singular values, normalise them to sum to 1, take the entropy of that, and exponentiate. It is "how many directions is this layer really using". A layer whose activations have collapsed onto one direction scores near 1. | **down** |
| **weight norm** and **relative distance** | The Frobenius norm of every floating-point weight, and — with `--base` — how far the weights have moved from the untrained model as a fraction of the untrained model's own norm, overall and per layer. It says how much has changed, not whether that change cost anything. | neither; it is context |

Two smaller numbers sit beside those, and they are reported because the headline ones can mislead on their own:

- **saturated units share** — the share of *units* that ever exceeded the threshold, rather than the share of *values*. A handful of units firing hard looks the same as many units drifting up, in the share of values; these two numbers apart tell you which happened.
- **effective rank (centred)** — the same calculation after subtracting the mean activation. The uncentred number, which is the one plan 4b's bar reads, is often dominated by a single large mean direction and so moves very little; the centred one shows the spread around it. **If the two ever disagree about the direction of travel, say so in the readout rather than picking one.**

### Things it is honest about

- **The saturated share is measured from a sample.** Keeping every activation of an 8B model on 200 prompts is not possible, so each layer keeps 500 values per prompt — 100,000 per layer — chosen by a fixed seed, and the percentile and the share are computed on those. The dormant share and the saturated *units* share are exact: they come from a running maximum per unit, which costs nothing to keep.
- **An activation is not a count.** It moves with the GPU, the kernel and the dtype. Every probe records a machine-and-dtype fingerprint, and `compare` **refuses** a chain that mixes two of them unless you pass `--allow-different-machines`, which is recorded in the report as the departure it is. This is the same rule the scoring tools enforce, for the same reason (receipts 204 and 209).
- **Only prompts, never answers.** The model is not asked to generate, so none of this depends on decoding, sampling or a scorer.
- **`--prompts`, `--tokens-per-prompt` and `--values-per-prompt` exist so a laptop can run a small version.** Changing any of them changes the numbers; a probe drawn from a different prompt set is refused as a reference, and a chain that mixes prompt sets is refused outright.

## The bar, written before any of these numbers existed

From plan 4b:

> Plasticity loss is **present** only if a job takes **at least 25 percent more steps to reach half its final gain in position 4 than in position 1**, on **at least 2 of 3 seeds**, **and** at least one internal signal moves the same way.

`compare` applies exactly that and **reports the two halves separately**. It never answers PRESENT on one of them:

- a slowdown with no internal movement is a fact about a training curve, with nothing inside the model to explain it;
- an internal movement with no slowdown is a measurement with no consequence.

**Half one, the learning curve**, is read from each run's `metrics.jsonl` — the validation series the trainer already logs. "Steps to half the final gain" is the first logged step at which the run reached halfway between its first validation and its last. A run that ended no higher than it started has no gain to halve and is reported as *null*, never as zero. The runs must be named

```
<job>-pos<POSITION>-seed<SEED>          e.g.  sql-pos1-seed42,  sql-pos4-seed42
```

(an `-aN` attempt suffix is allowed and the highest attempt wins), because the comparison is **one job against itself**: the same job, learned first in one ordering and fourth in another, under the same seed. Without both sides under one seed there is nothing to divide. With fewer than three seeds compared, the half is reported as **not measured** — not as a pass and not as a fail. With *more* than three, the "2" is applied unchanged and the report says so in a note: plan 4b wrote "2 of 3" and never said what 2 of 5 means, and this tool does not decide that on its behalf. If K5 runs five seeds, re-state the bar before the numbers exist.

Because K5's validations land every 5 steps, the ratio of two step counts is coarse: 10 against 15 is a ratio of 1.5, and there is no value between. The table therefore also prints an interpolated step count beside each one, so you can see whether a call was close. **The bar itself reads the logged step, not the interpolated one.**

**Half two, the internal signals**, compares the first probe in the chain with the last and asks whether dormant share went up, saturated share went up, or effective rank went down. plan 4b set a *size* for half one (25 percent) and **no size at all** for half two, so any movement in the right direction counts. That is weak on its own, and it is the reason the bar needs both halves. The report says so on its face, and carries `size_of_move_was_never_pre_set: 1`.

The verdict is one of:

- **PRESENT** — both halves met.
- **NOT_PRESENT** — both halves were measurable and at least one was not met.
- **INCOMPLETE** — a half could not be measured (no `--metrics`, fewer than three seeds, a chain of one).

`compare` **exits 0 whatever the verdict**. This is a measurement, not a gate, and nothing downstream should refuse to run because of it.

## What this is not

It is **not a fix**. Nothing here says that resetting layers, re-initialising dormant units, adding a regulariser or lowering a learning rate would help, and nothing here should be cited as evidence that they would. plan 4b is explicit: *measure first; a fix only if the pre-set bar is met.* If the bar is not met, the answer to Q11 is "not at this scale, on this sequence, by this measure" — which is a result, and the cheapest one available.

It is also **not a claim about forgetting**. A model can lose room to learn while forgetting nothing, and forget badly while learning the next job as fast as ever. Forgetting is `kit/score_forgetting.py` and the K5 scorecard; this file answers a different question.

## What it needs

- `torch`, `transformers`, `safetensors` — all already in the kit's runtime. **No vLLM**, no trainer, no network.
- One GPU for an 8B checkpoint (`--device cuda --dtype bfloat16`); a few minutes per probe, since it is 200 forward passes of a prompt each and no generation.
- The weights are read one tensor at a time straight from the `.safetensors` files, so comparing a checkpoint with the untrained model never needs a second model in memory.
- `compare` needs nothing but the Python standard library and the files the probes wrote.
