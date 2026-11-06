# On-policy earlier-task replay — DRAFT, 2 November 2026

**DRAFT — owner approval and an Opus cross-family review before any build. No run or spending authority.** Goal 4 / P2; replan §3 row 5. Inherit the [shared registration](README.md), including incumbent precedence, reference-scorer authority and exactly paired seeds **101–103**.

## Question, trigger and one change

Does replaying the current checkpoint's own verified FinQA solutions retain acquired FinQA skill better than replaying stored solutions, at fixed earlier-task prompt exposure? [Chen et al., On-Policy Replay, 2605.29495](https://arxiv.org/abs/2605.29495) motivates changing the data source while retaining ordinary replay CE. This adapts task-boundary refresh to D's replay slots; it is not a replication of TRACE or continuous regeneration during the stage.

**Read trigger:** an earlier-task pool is nominated in the **v6b return (M8/M9)** or **CR is admitted in the v6a return**. If both possibilities resolve false, label ARCHIVED UNREAD and do not interpret efficacy; do not generate/train by default. If required nomination or CR evidence is unresolved, PENDING, not false. The trigger alone does not admit a pool. Select the comparator by README: I if it contains FinQA slots, otherwise admitted CR, otherwise the nominated stored-replay recipe. A non-incumbent selection is explicitly a recipe benchmark and cannot change I.

Only **FinQA replay target text** changes. Keep the same FinQA prompt IDs, earlier-task slots, losses/weights, other replay domains, current ContractNLI data and dose. Do not replace K's base-KL generic targets with newly generated acquired-task responses or add earlier-task slots to K without a registered combination.

## Arms, starts and dose

Per seed, fork two fresh ContractNLI runs from the exact compatible **FinQA step-40 final**. Bind parent tensors, stage-1 recipe, training data, seed, tokenizer/base, export and ancestry receipts. No legal/MATH final, reset export or other replay sibling supplies a parent. Control uses the frozen stored verified FinQA solutions; OPR substitutes boundary-generated solutions below. No chain reset to base or new stage-1 training here.

Both arms retain D's full-weight recipe, 40 updates, LR 1e−5, ten-update warmup including zero LR, AdamW/clipping, fresh incoming-initialised EMA .99/.01 and verified 27B ContractNLI demonstrations; one student completion per legal prompt. Export 0/10/20/30/40. For I0 use its bound task/replay positions and denominator. For CR use stage-2 **32 legal +4 instruction +2 maths +2 knowledge +8 FinQA =48 rows/update**, task denominator 32, one optimizer/scheduler/EMA step per update: 1,280 legal and 320 FinQA exposures. Other nominated recipes retain their bound 32-row schedule, including the resulting task/replay exposure. Publish exact per-seed schedule before generation; no generic fallback changes these counts.

Replay loss is EOS-inclusive token-mean CE per row on stored versus OPR targets, prompt/padding excluded, no D importance correction or three-token skip on CE. Preserve D loss and all non-FinQA replay objectives; global loss weights do not depend on accepted response length. Any carried P2 coefficient or allocation is identical in both arms and must have compatible evidence under README. Use README’s admitted caps/draws rule, including I0’s inherited 2,048-token replay-draw filter and K/CR’s unfiltered schedules. No new filtering, hidden row removal or extra optimizer step; preserve bound batch-balancing behavior.

## On-policy pool construction and data

“Current checkpoint” means the **untouched incoming FinQA final for that seed**, before ContractNLI training. Freeze this generator for the whole stage; no refresh at step 10/20/30, no EMA generator, stored 27B response masquerading as a rollout, or evaluation-based quality filter. This keeps the task-boundary intervention reviewable and budgeted.

Generate for every distinct FinQA prompt in the frozen replay schedule. Use its normal memory-off FinQA prompt, thinking off, temperature .7, top-p 1, cap 2,048 including EOS, four predeclared independent attempts maximum per prompt, seeds derived from `opr-20261102|seed|prompt-id|attempt`. Take the first passing response in attempt order; no highest-score/shortest-response selection. Stop further attempts after success and retain attempted/rejected response hashes and reasons. Control sees the identical scheduled prompt, with its original stored solution. Sampling settings are a proposed fixed development choice, not a tuned claim about Chen's recipe.

Accept only complete EOS/stop responses passing the existing FinQA numeric/gold verifier, encoding, source and length checks; reject truncation or malformed final answer. Gold is available to the **verifier only**, never inserted into the generator prompt. Correct numeric output does not certify a faithful rationale; state that limitation. No response edits or teacher repair.

**Stored-target fallback fixed before generation (M5).** Let U_s be all distinct scheduled FinQA prompt IDs for seed s, and A_s those with a verified incoming-final response in the four attempts. Require **|A_s|/|U_s| ≥80% for every seed**, else PENDING/INCOMPLETE before either arm trains. For each uncovered ID, OPR uses the **same frozen stored verified response as control**, fixed by its original target hash; no repair, replenishment, prompt replacement, successful-subset selection or weakened verifier. Repeated IDs reuse their accepted response or fallback. All prompt IDs, positions and exposure remain identical. Report coverage per seed/attempt, generated-response substitution fraction |A_s|/|U_s|, stored-fallback fraction 1−coverage, and both fractions weighted by replay exposures; separately report accepted generated responses identical to stored text. This tests a predeclared ≥80%-covered source-refresh policy, not 100% on-policy replay.

Use only the nominated/CR **FinQA training** pool, reverified and frozen outside all 1,147 FinQA panel questions, ContractNLI test/diagnostic groups, general panels and monitoring/PD/SP candidates. Keep licences, dataset revisions, raw responses and exclusion receipts. Use README duplicate thresholds on candidate supersets; no panel-driven deletions. No January–June, July–September stream rows or October–November data are needed here; the stream split remains unchanged.

## Endpoint, panels and bars

**R2-M1 — manager decision fixed before any data.** Define **C_s = {FinQA-1,147 items incorrect at original base and correct at the common incoming FinQA final for seed s}**, using the **final decision lane’s** base and incoming correctness vectors. These references are independent of both trained arms. Compute and seal membership from the reference vectors before the reader accesses either arm’s final correctness; use that same C_s for both arms. This is a fixed arm-independent definition, not selection on outcomes.

The separate **pre-training** scoring job is a feasibility gate only: form C_s^pre by the same rule and require **|C_s^pre|≥100 for every seed before training**, else PENDING. Preserve its complete base/incoming item vectors and identities. In the final lane require **|C_s|≥100 per seed**, else PENDING for insufficient support; no alternate primary. Re-score base, incoming and all decision-bearing arm checkpoints together on one physical GPU/job. Compare each final reference vector with its pre-training vector on **all 1,147 IDs**, separately for base and incoming and for every seed: **≥99% item correctness agreement** (at least 1,136 matches, at most 11 disagreements) in each comparison, else **UNVERIFIED**. Do not average over references/seeds or demand exact equality. Report every disagreeing ID, both correctness values, checkpoint/scorer identities, agreement fractions, cohort sizes, intersection and symmetric difference; never delete disagreements or choose a more favourable cohort.

**Primary:** acquired-answer survival at ContractNLI step 40, OPR − stored replay, `100×number still correct on C_s/|C_s|`. Final C_s always comes from the same-lane references, including when a few pre-training items disagree within tolerance. Final-arm correctness never defines membership.

Admission/benchmark benefit: mean ≥2 points, 3/3 positive and paired two-sided 95% lower bound >0 using **t=4.302653**, df=2. Each guard lower bound ≥−1: whole FinQA final accuracy and retention from incoming, ContractNLI legal-test-100 tolerant final and gain, each general-v1 panel individually, HellaSwag-100, and each other actually acquired earlier task in the bound history (none beyond FinQA on this route). Also report absolute incoming/final scores, acquired-then-lost answers and both-cohort intersections. Do not inflate seed count with items.

Score base/incoming/final on FinQA, legal test, general instruction/maths and reference knowledge/HellaSwag; legal acquisition curve at 0/10/20/30/40 and step-20 retention diagnostics. Legal diagnostic panel is separate and never substitutes for test. Gain-AUC is descriptive, not a new gate. Report inherited FinQA acquisition lower bound >0, legal mean/lower bound ≥69.7 with positive gain, general ≥base−3 and earlier task ≥acquisition−1 with **one-sided t=2.919986** where registered; these absolute package summaries do not replace the new paired guards. Serving >1.5× base tokens/correct is “at cost”.

## Qualification, cost and outcomes

Require preflight proof of identical schedules, verified target provenance, correct CE masks/global reduction at world sizes 1/2/8, finite D/replay gradients and EMA, one-step invariants, worst-length rendering without truncation and export/resume equality. Generation uses **one batched one-GPU lane per seed**, with its own frozen generator and ready prompts; qualify batch/memory/throughput and ≥70% utilisation per lane as in README before generation/training spend. Three seed lanes may run concurrently; no single allocation serializes all seeds and no GPU-hour discount applies. Training uses eight GPUs; scorer lanes one GPU with three concurrent ready commands where qualified. Cohort feasibility precedes training on a separate ready frontier; the final decision uses a complete re-scored frontier, not mixed hardware scores. Apply README statuses and utilisation/eviction rules.

**Repriced M1, all three seeds:** core nomination/I0 **119.54–272.68 GPU-h** = six C0-like legal stages 92.94–153.36 + two qualifications 9.60–18.32 + scoring/IO 15–33 + generation 2–68. CR **169.82–333.52** = six CR legal stages 139.44–230.04 + qualification 13.38–24.48 + 15–33 + generation 2–46. Under I=K, use the CR price if CR admitted, else the nominated core price; without either, archive. **There is no K+FinQA-replay price or arm.** These are README’s long-cap rates and admitted-draw policy, not the rejected 6.572 receipt. Generation bounds charge every attempt; actual distinct prompt counts, lengths and one-GPU throughput replace allowances before send. Separate fresh controls are included; shared integration 20–40 is charged once in README. No paid run is authorized. Wall time UNMEASURED; flag >24 h per arm/seed or combined lane.

Missing pools, zero/insufficient cohort, failed generation coverage or incomplete controls: PENDING/INCOMPLETE. Wrong ancestry, overlap or identity: UNVERIFIED; mixed hardware: UNVERIFIED-HARDWARE. Valid evidence missing a bar: INCONCLUSIVE, incumbent unchanged. Never replenish after seeing retention. When this is an incumbent comparison, a pass nominates the component subject to v6 stack/confirmation rules; when it is a nomination-only/non-incumbent benchmark, a pass requires a later comparison against I before admission.

## What this does NOT claim

One boundary cannot establish sustained on-policy adaptation, optimal refresh frequency, truthful reasoning, TRACE replication or universal retention. A nominated pool is not validated by beating its own stored-target version. The result isolates response-source replacement on fixed prompts, not replay quantity, broad coverage, generic KL, or a combined K+CR system. It neither changes the product memory incumbent nor consumes reserved stream evidence.

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
| M5 | Applied | Stored-target fallback fixed before generation; ≥80% unique-prompt coverage per seed else PENDING; publish substitution/fallback fractions by IDs and exposures. |
| MINOR: cohort | Applied | C_s is final same-lane base-incorrect AND incoming-correct; ≥100 pre/final; pre-training vectors retained for ≥99% agreement diagnostics. |
| MINOR: citation | Checked | Chen primary abstract verified; source refresh is a local fixed-exposure adaptation, not a TRACE replication. |

### Round-3 disposition (manager-fixed before data)

Shared rules apply through the [README](README.md); allocation and cross-family re-review remain pending.

| Finding | Disposition | Fixed action |
|---|---|---|
| R2-M1 | Applied | Final same-lane arm-independent OPR cohort; ≥100 pre-training feasibility; ≥99% agreement per vector/seed, disagreements published. |
| r3-m1 | Applied | Pin 214dbe6f/7efbb6cb and timing-only precedence clarification; freeze v2 thresholds and all round-3 decisions before returns. |
| r3-m2 | Applied | Staggered reads allowed; later frozen specifications cannot change after earlier reads. |
| r3-m4 | Applied | One batched generation lane per seed; independent ≥70% utilisation qualification and per-lane wall-risk pricing. |
| r3-m5 | Applied; owner approval PENDING | Explicit per-branch allocation decision versus rough 290–420 GPU-h; scope approval is not allocation approval. |
| r3-m7 | Applied | Decimal half-up component rounding, summed row/branch totals and ceiled owner allowances; iSDFT and GRPO discrepancies corrected. |

## Builder deltas

Prospective handoff only; owner approval and cross-family review still precede any build.

| Area | Required delta |
|---|---|
| Binding | Seal nominated/incumbent recipe, caps/draws, replay schedule, parents and pre-training base/incoming feasibility vectors before generation; define final C_s solely from final same-lane references. |
| Pool/reader | Implement fixed four-attempt verification, stored-target fallback, per-seed 80% gate and unique/exposure substitution receipts; no prompt removal. |
| Qualification/cost | Test CE/fallback masks, exposure and ≥99% agreement statuses; qualify batched generation/utilisation per seed and inventory all attempts/scorer work against the core/CR price. |
