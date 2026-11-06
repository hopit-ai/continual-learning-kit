# Bottom-three-layer freeze — DRAFT, 2 November 2026

**DRAFT — owner approval and an Opus cross-family review before any build. No launch or spending authority.** Goal 4 / P2; replan §3 row 8. Apply the [shared registration](README.md), including fresh controls, exact incumbent/ancestry binding and paired seeds **101–103** only.

## Question and one change

Does freezing the bottom three decoder blocks at the FinQA→ContractNLI boundary protect existing capabilities without impairing legal acquisition? [Spurious Forgetting, 2501.13453](https://arxiv.org/abs/2501.13453) motivates shallow-layer freezing as a parameter-isolation baseline. This is our dense-Qwen3-8B boundary intervention, not a replication of that paper's full task suite or a demonstration that measured forgetting is merely spurious.

Only the **student's trainable-parameter mask** changes. Both arms retain the bound incumbent's D objective, replay domains/slots, demonstrations, dose and compatible base regulariser. No adapters, reset, weight interpolation, output-head change, special LR or altered teacher source.

## Arms, starts and binding

**Correction M3:** resolve K/CR with v6a; P2/core stacking are omitted under K/CR, so freeze does not wait for their v6b return. Only I0 waits for v6b M8/M9, X, P2 and required compatible stacking, preserving README’s frozen order. Fork fresh **I** and **I+freeze3** legal stages from each exact bound FinQA final, base revision `b968826d9c46dd6066d109eabc6255188de91218`. Freeze tensor hashes, parent chain/seed/step, replay manifests and coefficient identity. Neither a later-stage final nor another dispatch challenger supplies a parent.

Freeze exactly zero-indexed decoder blocks **0, 1 and 2**, all their attention projections, MLP parameters and within-block norms. Bind architecture paths and the exhaustive parameter-name/shape/count list before training; in the native Qwen module these are the first three elements of `model.layers`. Embeddings, final norm, LM head and decoder blocks ≥3 remain trainable. **Declared deviation from the motivating evidence:** Spurious Forgetting §5.2 and Appendix G.3 include input embeddings in the bottom-layer freezing strategy ([primary text](https://arxiv.org/html/2501.13453v1#S5.SS2)); its “+3” result therefore does not establish benefit for this trainable-embeddings mask. Keep our decoder-blocks-only intervention fixed; do not relabel it a reproduction or expand the mask after results. Count tied tensors once; if a tensor spans a frozen and unfrozen role, stop qualification rather than guess a mask. No trainable bias or auxiliary parameter may be silently omitted from the audit.

Frozen tensors retain the **incoming FinQA values**, not restored base values. Exclude them from optimizer updates and weight decay; assert bit-identical at every export. Initial logits and teacher-forced outputs must agree with control before training under the same deterministic path. The EMA teacher starts from identical incoming weights in both arms and follows .99/.01 normally; its bottom blocks stay constant because the student blocks are constant. Other EMA changes are an indirect consequence of training, not another intervention.

If compatible P2 is present, retain its same λ and original base anchor in both arms, without recalibration. Keep its scalar penalty definition over the bound original parameter set for logging; frozen-block contribution is constant and has no gradient because those parameters are frozen. Apply the identical mask to **all** gradient sources, including regularisers; no regulariser or weight decay can move frozen tensors. Keep K's replay KL unchanged if K is bound. This implements the single trainability change without silently dropping or retuning an admitted component.

## Dose, data and qualification

Forty updates per arm/seed, full-weight recipe on eligible tensors, effective schedule exactly I's. LR 1e−5, ten-update warmup including initial zero LR, fresh AdamW/scheduler and EMA, one on-policy completion per task prompt, D's verified 27B context/full-vocabulary loss/importance correction/three-token mask unchanged. I0/K retain their bound 32-row schedules; CR keeps stage-2 32 legal+16 replay with denominator 32. Do not divide the task loss by the remaining trainable fraction or compensate with larger LR, batch or more updates. Export 0/10/20/30/40. Stages are serial and there is no hard-MATH continuation in this registration.

Use frozen ContractNLI training documents and verified targets, plus I's exact replay pools. Retain candidate-superset checks, all legal document/group exclusions, FinQA-1,147 exclusion, general/HellaSwag/PD/SP and monitor/probe exclusions. Normalized equality, five-shingle Jaccard ≥.8 and edit similarity ≥.9 are the shared duplicate rules. No evaluation panel may select layers or train the model. No SEC months are required; January–June remains 4A's and October–November untouched. No outcome-driven row replacement or prompt truncation. Both arms obey README’s admitted-caps/draws rule, including I0’s inherited replay filtering and K/CR’s unfiltered schedule. Verify returned student/teacher limits and masks; overlength is PENDING, not permission to raise caps or remove rows.

Preflight freezes the name list and parameter counts; checks bit equality before update, zero/absent frozen gradients and no optimizer state/decay on frozen tensors, finite nonzero gradients and movement in representative unfrozen tensors after nonzero-LR updates, finite EMA and exact update/exposure counts. Test mask persistence after export and **native resume**, including optimizer parameter-group mapping. Compare the no-mask path against I and global loss reductions at world sizes 1/2/8 (relative gradient error ≤1e−4 on unfrozen coordinates). Use the longest admitted legal input with materialised optimizer state. Any mask drift, unintended reset, stale optimizer mapping, numerical failure or mismatch blocks the comparison. Timing and saved memory are measured, never efficacy gates.

## Panels, endpoint and bars

**Primary:** freeze3−I difference in equal-weight retained accuracy across **instruction, maths, knowledge and FinQA** at ContractNLI step 40. General retention references original base; FinQA references the common incoming acquisition final. Freeze all four domains regardless of admitted replay. Report both retention and absolute final scores so a weak acquisition cannot inflate retention. Each seed supplies one mean; items and domains are not independent seeds.

Require mean ≥2 points, positive 3/3 and paired two-sided 95% lower bound >0 using **t=4.302653**, df=2. Every individual primary panel's absolute final and retention contrast, HellaSwag-100, legal-test-100 tolerant final, gain and normalised gain-AUC must have lower bound **≥−1**. Gain-AUC uses trapezoids at 0/10/20/30/40 divided by 40 minus the common untouched incoming accuracy. No post-freeze baseline adjustment: freezing changes no initial tensor. Preserve observed losses even if final catch-up occurs.

Score base/incoming/final on all panels and legal at every curve export; step-20 retention is diagnostic. Knowledge/HellaSwag use full-precision **REFERENCE-SCORED** authority. All decision references, controls/challengers and legal curve points must share their per-(seed,panel) physical GPU/environment in one complete lane. Legal diagnostic panel remains separate from legal test. General instruction keeps strict all-constraint accuracy. Report FinQA acquired-answer survival, answer transitions and per-panel deltas; bootstrap diagnostics never replace seed bounds.

Inherited package summaries remain distinct: positive FinQA acquisition; legal ≥69.7 with positive gain; general ≥base−3 and FinQA ≥acquisition−1, using **one-sided t=2.919986** where applicable. These are reported rather than added as new absolute gates. Report token/correct, full training cost and serving >1.5× base as “at cost”. A lower trainable count does not establish a runtime/serving advantage.

## Cost and outcome handling

**Repriced M1, all three seeds:** **K 121.24–221.12; CR 167.82–287.52; I0 117.54–204.68 GPU-h**, using README’s `6T+2Q+15–33` at the selected legal caps. Six stages alone cost K 96.36–168.54, CR 139.44–230.04, I0 92.94–153.36. **No speed credit** is assumed for freezing three blocks; embeddings stay trainable and gradients still propagate through them. Measure the paired production path and charge all eight reserved training GPUs. Separate fresh controls, per-arm qualification, scoring and scientific IO are included; shared integration is charged once in README. CPU work zero GPUs; training eight; scorer lane one with qualified concurrency 3. No paid execution authorized. Wall time UNMEASURED; flag >24 h per arm/seed or combined lane.

PENDING/INCOMPLETE for missing parents, prerequisite verdicts, masks, qualification or score cells; UNVERIFIED for provenance/overlap/parameter identity failures; UNVERIFIED-HARDWARE for mixed allocations. Valid evidence missing a bar is INCONCLUSIVE and leaves I unchanged. Never search freeze depth, switch to embeddings-only, add a fourth seed or rerun at another LR after outcomes. Passing supports this exact mask on this incumbent at this boundary; combining with another new winner needs a separately registered test and v6 stack validation.

## What this does NOT claim

It does not show that knowledge is physically confined to layers 0–2, that all apparent forgetting is recoverable, that adapters are unnecessary, or that acquisition is protected on a longer chain. Preserving general performance while losing legal learning fails the guards. Three frozen blocks at one boundary do not establish optimal depth, universal non-forgetting, restored plasticity or stream confirmation; October–November remains reserved.

**Compatible P2 cost overlay:** if carried on exact core ancestry in both arms, add **2–6 GPU-h** penalty/shard work to this row (zero if omitted). README includes the resulting conditional package reserve. K/CR cannot carry P2; only a compatible core OPR benchmark can incur this reserve under I=K.

## Review disposition

Response to [Opus round 1](review-opus-r1.md) and [round 2](review-opus-r2.md); round-3 manager decisions are fixed before any data, with cross-family re-review pending. Common findings apply through README; this table records row-specific changes.

| Finding | Applied/rejected | One-line note |
|---|---|---|
| M1 | Applied | Selected-ancestry cap-based pricing, qualification and per-branch budget in README; replace timing allowances before send. |
| M2 | Applied | README admitted caps/draws rule; source config verified, actual returned binding still required. |
| M3/M4 | Applied / preserved | Prerequisite timing corrected where applicable; dispatch/nomination precedence pinned at 214dbe6f and 7efbb6cb (timing-only clarification); v2 thresholds and round-3 decisions frozen in README before any return. |
| MINOR: controls | Applied decision | Separate fresh controls retained for independent release/recovery; costs and reason in README. |
| MINOR: house style | Applied | Review disposition and prospective Builder deltas tables added; no implementation or launch. |
| MINOR: embeddings | Applied | Primary paper includes embeddings; our trainable-embeddings/decoder-blocks-only mask is an explicit deviation. |

### Round-3 disposition (manager-fixed before data)

Shared rules apply through the [README](README.md); allocation and cross-family re-review remain pending.

| Finding | Disposition | Fixed action |
|---|---|---|
| r3-m1 | Applied | Pin 214dbe6f/7efbb6cb and timing-only precedence clarification; freeze v2 thresholds and all round-3 decisions before returns. |
| r3-m2 | Applied | Staggered reads allowed; later frozen specifications cannot change after earlier reads. |
| r3-m5 | Applied; owner approval PENDING | Explicit per-branch allocation decision versus rough 290–420 GPU-h; scope approval is not allocation approval. |
| r3-m7 | Applied | Decimal half-up component rounding, summed row/branch totals and ceiled owner allowances; iSDFT and GRPO discrepancies corrected. |

## Builder deltas

Prospective handoff only; owner approval and cross-family review still precede any build.

| Area | Required delta |
|---|---|
| Binding/mask | Seal admitted caps/draws and exhaustive blocks 0–2 mask; declare embeddings trainable and distinct from paper mask. |
| Runner | Exclude frozen tensors from all gradients, optimizer state and weight decay; preserve incoming values and verify export/resume bit equality. |
| Reader/cost | Preserve primary/guards; no assumed compute saving; publish trainable counts, branch budget and actual timings. |
