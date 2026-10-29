# Package B: amended 3C sequence and recovery probe

**REGISTERED 14 October 2026 — owner: “Okay basis what we have, lets plan upto 1000 GPU hours”.** Design only. **1,000 partner GPU-hours is a ceiling, not a target.** Deliver plain scripts; the partner supplies his own `sbatch`. No runner, campaign files, allocation accounting or performance gates. This registration replaces the execution sketch in [programme-check-astra-20261014.md](programme-check-astra-20261014.md). [3B and standalone 3D](3b-3d-registration-20261014.md) remain registered but deferred. Their files are unchanged.

The question is whether teacher training can learn a second task after FinQA while retaining FinQA and general abilities. A small recovery probe asks whether v4's instruction-following loss is readily reversible. [V4 results, final section](../phase2/v4-finqa-results-20261014.md) and [family evidence](../phase2/evidence/v4-finqa-return/ifeval_failures.txt) show 26–50% losses on base-correct case, punctuation, length and keyword prompts; D loses little on content and start/end instructions. This supports testing alignment drift, not declaring that forgetting never happens.

## 1. Shared inputs and recipes

**D is SDFT; F is SFT on the same verified 27B solutions**, as defined in [v4-arm-index.md](../phase2/v4-arm-index.md). Reuse Qwen3-8B `b968826d9c46dd6066d109eabc6255188de91218`, Qwen3.6-27B `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, v4's FinQA pool/targets and seed schedules. Bind exact v4 D/F step-40 exports and hashes from partner `shared-storage/v4-finqa-keep/exports/`; never substitute a different checkpoint. Recovery exports are separate copies, never sequence starts.

Ordinary stages retain [v4 §§3 and 13](../plan-v4-teacher-20261006.md): full weights, 40 updates, 32 prompts/update, 2,048 response tokens, fresh optimiser/schedule; D also starts a fresh EMA teaching copy from the incoming student. LR `1e-5`, ten-step warmup then constant, including v4's zero-LR first update. Preserve D's full-vocabulary forward KL, masking, importance correction and EMA rate .01; preserve F's response-only, EOS-inclusive loss. Reuse [run_sdft.sh](../../kit/run_sdft.sh), [run_sft_f.sh](../../kit/run_sft_f.sh) and scorers unchanged. Probe and mixed-loss entry points must explicitly implement the departures below, not label scientific runs as technical smokes.

**Replay source (amended by the manager, 14 October, before any data was read):** `allenai/RLVR-IFeval` @ `47c03c73621c4aab2b824b7818681117d662770e` (ODC-BY; one train parquet). Astra named `allenai/tulu-3-sft-personas-instruction-following`, but on inspection its constraints are text labels only (e.g. `keywords:exclude words`), with no executable arguments, which is the incompatibility this registration anticipated. RLVR-IFeval carries an executable `ground_truth` per row (`func_name` plus arguments) covering the 25 IFEval constraint types. The checker is open-instruct `open_instruct/if_functions.py` @ `11826255077617a46919ce75cadf1f3d53f30dac` (Apache-2.0), vendored unchanged at `kit/resources/p3b/if_functions.py` with its SHA-256 recorded, and dispatched through its `IF_FUNCTIONS_MAP`. **Limit:** these constraints are the same kinds IFEval measures. Replay and the refresher are therefore in-distribution for the IFEval panel (prompt texts are still de-duplicated as below), so an IFEval recovery shows that this skill is recoverable, not that unrelated general skills return.

Remove shared source IDs and exact/near-duplicate prompts against **all** general-v1 prompts, including IFEval, and the task evaluation panels. Near-duplicate means Unicode NFKC + case-fold + whitespace-normalised exact equality, or word 5-shingle Jaccard ≥.8, or normalised edit similarity ≥.9; compare short prompts with the last rule. Deduplicate the training source the same way. Preserve exclusions and hashes; no evaluation outputs inform selection. Sort eligible rows by `sha256("package-b-replay-v1|" + id)`, then ID; use the first 2,048 as a fixed candidate pool.

Generate one response per candidate using the **frozen base 8B**, greedy, thinking off, 2,048 tokens, seed 0. Keep only complete responses passing **every** source constraint under its pinned checker. Discard source-written answers; no 27B text enters generic replay. Freeze all accepted IDs, outputs and checker results once. Use the first 64 accepted examples for the refresher, with the first 16 nested inside them; use the remaining accepted examples for replay. Fewer than 65 accepted examples makes the specified inputs incomplete; no adaptive extra generation. Use the same data for every arm and seed, without selecting families from observed failures.

**Replay objective:** replace fixed positions 8, 16, 24, 32 of each batch: **28 task + 4 generic prompts (12.5%)**, during both FinQA and stage 2. Cycle a seed-shuffled replay order; keep the other 28 positions from the ordinary task schedule. D+replay uses D loss on task items and **SFT cross-entropy on the frozen base responses** for replay items: `28/32 × mean task loss + 4/32 × mean replay loss`, each item's loss normalised over its valid response tokens. Replay SFT includes EOS; D's reduction/masks remain unchanged. This provides an explicit base anchor; no-extra-context KL against the moving EMA would not train toward the checked base response. F+replay uses F's unchanged SFT reduction across all 32 items. Report the different loss scales and actual tokens. Reduced task exposure (1,120 versus 1,280 prompts) is part of this recipe, not a matched-exposure causal test.

## 2. Tier 1: unconditional on package A

**(a) Recovery, 12 short runs.** Independently start from each v4 step-40 **D and F, seeds 101–103**, at both **16 and 64 unique examples**. Run exactly **four full-weight SFT updates**, global batch 16: four passes over 16 examples or one pass over 64, with seed-fixed order. Use F's AdamW (`betas=.9/.999`, weight decay .01, clip 1), EOS-inclusive loss, caps and fresh optimiser. Fix LR at **`1e-5` from update one, no warmup**: v4's ten-step ramp would consume the entire short intervention. Both sizes therefore have 64 example exposures; this tests diversity at fixed updates, not a dose curve. Score incoming and refreshed checkpoints on IFEval, knowledge, maths and all 1,147 FinQA questions. No second attempt, LR search or best-checkpoint selection. Optional base→refresher controls at both sizes, seed 101, measure ordinary instruction-training gains; decide to include them before refreshed outcomes.

Let `I0` be base IFEval, `Ipre` the incoming checkpoint and `Ipost` the refreshed score, all in points. A useful recovery has **gain ≥5 points and restores ≥50% of `I0−Ipre`**, while FinQA loses **≤1 point** and knowledge/maths each lose **≤1 point versus the incoming checkpoint**. Near-base recovery additionally requires IFEval within **3 points of base**. Apply bars to each arm/size's three-seed mean; test half-recovery using the paired contrast `(Ipost−Ipre)−.5×(I0−Ipre)`. A nonpositive initial loss makes the recovery fraction undefined; report raw changes. Report every seed, with the 64-example comparison primary and 16-example comparison secondary; no pooling to rescue a failure.

| Recovery outcome, fixed now | Interpretation |
|---|---|
| Useful recovery; also within 3 IFEval points of base | Substantial, inexpensive recovery consistent with reversible alignment drift; FinQA survives this intervention. |
| Useful recovery, still below the near-base bar | Partly recoverable loss; report the residual and family differences. |
| IFEval improves but FinQA or other-panel protection fails | Recovery has a measured trade-off, not joint retention. |
| Recovery bars fail, or intervals cross the margins | No demonstrated useful recovery at this fixed dose, or inconclusive; not proof of erased knowledge. |

**(b) FinQA prevention, eight full stages.** Run **D+replay seeds 101–105** and **F+replay seeds 101–103** from base. Three F controls cost 53.1 GPU-h under the slot proxy and preserve preparation/reset headroom; they are a limited control, explicitly overriding the older general five-seed preference for this comparison. Compare F+replay with F and D+replay on those same three seeds; show all five D+replay seeds separately. No F+replay stage-2 branch is registered. Score FinQA/general-v1 at 0 and 40. All five D+replay finals proceed to tier 2 when a task is selected, regardless of recovery or prevention scores.

## 3. Tier 2: second-task acquisition and sequence

[Package A](../../kit/README-p3a.md), sent as **`kit-p3a-plain-v1`**, decides the task using the unchanged [3C screen](3c-second-task-screen-preregistration.md): LiveCodeBench first, registered MATH fallback only after completed coding failure; incomplete is pending. Neither qualifies means no tier 2. Coding uses the frozen **223 training / 51 evaluation** problems; MATH uses the registered **500 / 100** rational-answer subset. No coding/frontier claim follows from MATH.

Extend the **3D bounded target protocol** over the selected frozen training pool, reusing compatible archived teacher attempts: pinned 27B, seeds 27101–27104, temperatures 0/.3/.6/.8, top-p 1, thinking off, at most four attempts, 2,560 teacher tokens. Select first complete verified target fitting the EOS-inclusive 2,048 student limit. Code verification uses all released private tests; MATH uses its frozen exact-rational verifier. Generation never sees tests, expected outputs or reference answers. Archive failures; use the same surviving target pool for every recipe/start. Rewrites are not a further admission condition. Cycle seed-shuffled IDs to fill schedules and report repetition/coverage. **The sketch's later ≥80% coverage and standalone-gain continuation gates are removed:** coverage is descriptive; a nonempty valid pool suffices. Zero valid targets leaves the task undefined. Package A's already registered selection rules remain intact.

| Route | Recipes and starts | Seeds | New 40-update stages |
|---|---|---|---:|
| Standalone second task | Base→D, base→F, base→D+replay | 101–105 each | 15 |
| FinQA→second task | v4 D→D; v4 F→F; tier-1 D+replay→D+replay | Matched 101–105 each | 15 |

Score second-task curves at **0/10/20/30/40**; FinQA and each general panel at stage boundaries. Pair task schedules and stage-2 seeds across base/FinQA starts within each recipe. Primary learnability measure, inherited from 3D: trapezoidal area under `a(k)−a(0)`, divided by 40 (**gain-AUC**, percentage points). Also show raw AUC, final score/gain, initial competence and initial-success/failure strata. This embeds 3D's behavioural question, but does not reproduce its S-versus-D starts under common SFT or its two adaptation repeats.

**Written reset trigger.** For a recipe whose base-start acquisition clears the bar below, require (i) FinQA-start minus base-start gain-AUC **≤−3 points**, negative in at least **4/5 seeds**, with a one-sided 95% seed-t upper bound below zero; and (ii) a mean deficit **≤−3 points** also on the within-seed questions both starts got wrong at step 0. Report those subset sizes; an undefined comparison cannot trigger. The second condition guards against higher initial competence merely leaving less headroom. Finish rates and raw curves accompany the decision.

If any recipe triggers, run **one five-seed reset branch**, fixed priority **D+replay, then D, then F**, from its original FinQA finals. Otherwise no reset. Reset **1% of MLP intermediate neurons per layer**, rounded down, to original base values before stage 2: associated gate/up projection rows and down projection columns together. Select indices by fixed hash of `package-b-reset-v1|layer|index`, identical across seeds; leave attention/norms untouched. Fresh optimiser and EMA where applicable; same 40-update recipe/targets. Score immediately before and after reset and at 10/20/30/40, including immediate FinQA/general damage. Cost: **5 × 17.7 = 88.5 GPU-h**, plus scoring below. This explicitly defined base-reset variant is inspired by Liu et al.'s ~1% reset; it is not a replication of their combined weight-fusion method or Dohare's continual random reinitialisation. Compare with its five unreset counterparts; triggered results are conditional diagnostics.

## 4. Measures, analysis bars and limits

Use unchanged [eval_bed.py](../../kit/eval_bed.py), frozen panels/scorers, one physical scoring GPU, greedy decoding, thinking off, seed 0 and 2,048 output tokens. Re-score relevant base/incoming checkpoints on that GPU; record identities. Score truncated answers as returned and report finish rates. Report counts, percentage-point changes, paired gained/lost IDs, and retention of FinQA successes acquired after base. Report IFEval **per instruction family**, including denominators, losses and recovered IDs; multi-family prompts overlap. Knowledge and maths remain separate, never hidden in a general total.

**Analysis bars, never stops:** standalone coding mean gain **≥3/51**, positive in ≥4/5 seeds and one-sided 95% seed-t lower bound above zero; MATH fallback uses **≥6/100** with the same seed conditions. A successful sequence ends within **2 points** of its same-recipe standalone route, loses **≤1 FinQA point** from stage 1, and stays within **3 points of original base on each general panel at both stages**. FinQA acquisition itself must have a positive one-sided 95% seed-t lower bound. Use seed-paired one-sided 95% bounds against each retention margin; a crossing interval is inconclusive, even if the mean clears it. Recovery uses the same uncertainty discipline on its three paired seeds. Show question-paired bootstrap intervals separately from seed intervals; neither curve points nor questions are extra seeds.

Deployable recipes also require **≤1.5× base serving tokens per correct** on FinQA and the selected task separately, counting output tokens from failed answers; zero correct is undefined/infinite. Show preparation/training costs separately. Accuracy/retention success above the cost bar is “at cost”. Step 40 is primary; intermediate points are curve diagnostics, never a retrospectively chosen winner. Coding **34/51** is 85% of the archived 40/51 frontier reference, not a current-frontier claim.

This can show recoverability of a base skill, prevention with one replay share, and acquired-task retention/behavioural learnability in one two-task order. It cannot establish universal non-forgetting, structural plasticity loss, equal-compute superiority, a replay ratio law or repeated learning from arriving traces. The three-seed F+replay control supports FinQA comparisons only; stage-2 superiority over SFT+replay remains untested. Prior benchmark exposure and small panels limit generalisation. [Research targets L1/L2/L3](../phase2/research-targets.md) and [literature readers D/E/H](../phase2/plan-v3-evidence/literature-findings-20261001.md) already record recovery/replay/reset prior art. **SEAL stays in 4B, waiting for a working goal-2 stream.**

## 5. Budget, order and builder handoff

[V4 timing](3b-3d-registration-20261014.md) is **407.4 GPU-h / 23 slots ≈17.7 GPU-h/slot all-in**, not measured training-only time. Use it for every new full stage. Short probes use a step-proportional proxy, conservatively ignoring their smaller batch and cheaper SFT; startup is not proportional, so retain separate allowances. Coding throughput remains unmeasured. Extra scoring/preparation allowances deliberately overlap some historical all-in overhead.

| Partner work | Derivation | GPU-h reservation |
|---|---|---:|
| Tier 1 recovery | 12 × (4/40) × 17.7, rounded up | 21.3 |
| Tier 1 FinQA D+replay / F+replay | (5 + 3) × 17.7 | 141.6 |
| Tier 1 replay generation/checking | Frozen base responses; startup included | 15.0 |
| Tier 2 standalone + sequence | 30 × 17.7 | 531.0 |
| Tier 2 remaining 27B targets | Selected task only, bounded attempts | 40.0 |
| Additional scoring | Recovery/boundaries/curves/reset; shared bases once | 70.0 |
| Startup/export/infrastructure allowance | No scientific retries or extra seeds | 45.0 |
| Optional base refresher controls | 2 × (4/40) × 17.7, rounded up | 3.6 |
| Conditional reset branch | 5 × 17.7 | 88.5 |
| **Maximum planned reservation** | **863.9 core; 867.5 with base controls; 956.0 including reset** | **956.0** |

The remaining **44 GPU-h** is ceiling headroom, not work to fill. Package A and completed v4 work are excluded; never charge reused targets twice. If operational estimates cease to fit, report unfinished work and a revised plan before exceeding 1,000; do not silently cut seeds, increase attempts or build allocation-accounting machinery. **Our-side replay preparation: $0 planned**, using partner generation and local CPU preparation. An optional Modal base-response smoke has a **$10 planning allowance**, not a paid-run instruction; bulk generation stays with the partner. No API teacher or outside-model audit is needed for executable constraints.

Tier 1 goes first: use GPUs package A leaves free only with disjoint devices and sufficient host memory/storage bandwidth; otherwise start immediately after A. Stage 2 waits for A's selected task and frozen targets. Keep WORK on node-local NVMe and copy checkpoints/raw evidence to durable `shared-storage`. Partner scheduling is manual. Wrong identities, leaked/malformed data, invalid admitted targets and broken scoring are correctness stops; ordinary weak results do not stop registered work.

**Plain scripts to build (names proposed; no code in this registration):**

- `p3b_replay_data.py` — bind historical source/checker revisions, exclude overlap, generate/check base responses and freeze probe/replay data.
- `p3b_targets.py` — extend A's bounded teacher requests and write common task schedules without regenerating completed requests.
- `p3b_recover.py` — run one checkpoint/size's four-update SFT refresher with the explicit short-probe settings.
- `p3b_replay_train.py` — run one mixed stage, reusing v4 objective/optimiser routines with the registered replay routing.
- `p3b_reset.py` — write one base-anchored neuron-reset export and exact changed-index record after the written trigger.
- `p3b_read.py` — read unchanged scorer outputs and training evidence into curves, family tables, bars, trigger decisions and cost summaries; never submit jobs.

The runbook supplies direct commands for unchanged v4 launchers, checkpoint export and scorers. No automatic continuation, scheduler wrapper, campaign configuration or new approval gates.
