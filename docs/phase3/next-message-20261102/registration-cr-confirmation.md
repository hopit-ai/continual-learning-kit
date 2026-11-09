# CR−G confirmation — DRAFT, 6 November 2026

**DRAFT — seventh comparison in kit-message-v7; review and incremental allocation approval remain pending. No launch or spending authority.** Goal 4 / Paper 2. The [owner decision, “CR confirmation and arm-K memory limit (6 November, owner)”](../../programme-plan-v6-amendment-20261024.md#cr-confirmation-and-arm-k-memory-limit-6-november-owner) authorizes registering CR confirmation and adding it to v7, as an explicit exception to the six-comparison cap. It does not reopen the six approved registrations or extend their allocation to this row.

Scientific authority: the [original K/CR registration](../message3-tier-20261029/registration-k-cr.md), including its v2.1 qualification/input amendments, and the owner's subsequent same-hardware, likelihood-reference, near-tie and right-sizing rulings in the programme amendment. Copy the **CR and G definitions unchanged**, bound to kit **`059d6293`** and v6a's realised node-side manifests. Any difference in their recipe, pools, exposure, panels, references or bars is a defect, not a confirmation variant. This file changes the scientific seed set to **104, 105, 106 only**, fixes separate confirmation analysis and future incumbent timing, and applies the already approved operator rules. The [README](README.md#six-approved-comparisons-and-one-draft-confirmation) records the seventh-row exception and the v7-wide likelihood concurrency limit.

## Question and motivating exploratory result

Does cumulative coverage improve stage-3 retained accuracy over exposure-matched generic replay, while protecting every registered guard, on three fresh seed chains?

The [v6a reading](../return-v6a-reading-20261106.md) reports exploratory CR−G differences **+1.626, −0.357, +1.774 points**, mean **+1.014**, sample SD **1.190**, two-sided 95% interval **−1.94 to +3.97**, positive **2/3**: **INCONCLUSIVE**, not admitted. These are PARTNER-COMPUTED values, with the reading's transcription caveats; they are motivation only. K−C0 was also INCONCLUSIVE. **K is not carried forward.**

**IFEval is a pre-declared, decision-bearing guard, not a secondary caveat.** In v6a the post-FinQA boundary was mixed: **+3/+1/−8**. CR−G lost instruction accuracy in every seed at the later boundaries: **−5/−2/−2** after ContractNLI (mean **−3.0**) and **−2/−13/−2** after hard MATH (mean **−5.7**). Print confirmation IFEval per-seed contrasts, means, intervals and guard verdicts prominently beside the primary, at every stage. An average gain across the five primary domains cannot excuse a failed instruction guard. These observed costs do not authorize changing replay shares or relaxing the −1 bound.

Freeze this registration, implementation/reader version, input hashes, scorer authority and schedule manifests **before any new confirmation data**. Analyse **104–106 alone**: never pool with 101–103, never report an n=6 admitting interval, never borrow old controls or replace a fresh seed with an exploratory seed. Keep the v6a table separately labelled exploratory; its verdict stays INCONCLUSIVE regardless of this confirmation's outcome.

The builder must implement an isolated confirmation mode in the scheduler and reader: assert **`seeds == {104,105,106}`**, reject **any 101–103 cell as UNVERIFIED** before analysis, and emit **CR−G only**. Missing confirmation cells remain PENDING/INCOMPLETE. Do not extend the shared exploratory `SEEDS` constant to six seeds. Keep confirmation manifests, state, scores and reader outputs in separate trees; suppress K−C0, CR−C0, G−C0 and the exploratory reader's “confirmation opens” decision text. The Builder deltas below require a pre-data diff receipt against `059d6293`; the primary, guards, statistics and hardware identity rules are unchanged.

## Arms, starts and exact exposure match

Two arms only: **CR** (cumulative coverage CE replay) and **G** (matched generic CE replay). Fresh pinned Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218` chains, paired within each of seeds 104, 105 and 106: **FinQA → ContractNLI → qualified hard MATH**. Start from original base once; each stage inherits its own arm's preceding step-40 final. No reset, historical final/winner, K, C0, P2, adaptive allocation or new v7 component. Both arms train all three stages; stage-1 outcomes never gate continuation. Missing technical prerequisites leave affected work PENDING, not a shorter-chain substitute.

Both arms retain D's full-weight task recipe: LR **1e−5**, ten-update warmup including the initial zero LR, unchanged AdamW/clipping/weight decay, fresh optimizer/scheduler and incoming-initialised EMA **.99/.01** at each stage, one on-policy sample per task prompt, verified **27B demonstrations**, thinking off and **40 updates/stage**. D's task loss, full-vocabulary distillation, three-token skip and importance correction are unchanged. Bind common task targets/order per seed across arms. Export **0/10/20/30/40**. No extra updates, selected checkpoint or outcome-driven scientific restart.

| Stage | CR task | Instruction | GSM8K | ARC | Earlier FinQA | Earlier legal | CR total | G task + generic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FinQA | 32 | 4 | 2 | 2 | 0 | 0 | 40 | 32+8 |
| ContractNLI | 32 | 4 | 2 | 2 | 8 | 0 | 48 | 32+16 |
| Hard MATH | 32 | 4 | 2 | 2 | 8 | 8 | 56 | 32+24 |

G's generic total includes the same four shared instruction rows as CR, plus **4/12/20** generic extras at the three stages. Both retain all **32 task rows** per update; G/CR do not use C0/K's task-position replacement. Both have **1,280 task exposures/stage** plus **320/640/960 replay exposures**, identical total replay counts and loss weights. Only the named replay coverage differs.

Bind `data.train_batch_size = ppo_mini_batch_size = 40/48/56`, `trainer.balance_batch=False`, eight ranks, one row/rank/microbatch and **5/6/7 rounds**. Exactly one mini-batch, clipping operation, optimizer/scheduler/EMA step per update. For both arms,

`L = (Σ_32 d_i + Σ_replay CE_i)/32`.

Never divide by total rows. After rank averaging each local row contributes **loss/4** on eight ranks; the two-rank smoke uses **loss/16**, not /20, /24 or /28. Replay CE is EOS-inclusive token-mean per sequence, excludes prompt/padding, has no importance correction or initial-token skip, and uses the frozen stored responses. Replay rollouts and discarded EMA pad forwards remain charged. Independently verify global gradients at world sizes 1/2/8.

Use one shared instruction stream: reproduce `next_slots` seed-only generic cycling **without its length-filtering draw function**, so the same **160 IDs recur each stage**. For every seed/stage/update assert the four shared IDs match between CR and G. G extras use **`m3-cr-v2|seed|stage|generic-extra|id`** hash order, excluding that stage's shared IDs and previously used extras; publish all IDs. Other domains use that salt with their domain name and ID tie-breaks. Retain the original salt, substituting only the new seed values; no confirmation-specific reorder. Never call `next_slots.draw`; require **`skipped_draw_count == 0`** for every arm/domain. Overlength inputs fail preparation, never disappear from schedules.

Keep common original M3 length limits across both arms and all stages: student prompt **10,240**, response **2,048 including EOS**, teacher reprompt **14,336** plus response allowance (actor padded width **12,288**, EMA **16,384**). Context limits must cover each fully rendered admitted input, demonstration and response. Do not reintroduce MATH's old 2,048/6,144 override. CPU-tokenize the longest admitted legal and FinQA rows through the actual stage-3 rollout/teacher configurations and reject truncation. A missing fit is PENDING, not permission to change the recipe or filter rows.

## Frozen pools, targets and exclusions

Bind **every pool, target and schedule source to v6a's realised node-side manifest hashes under kit `059d6293`**, including candidate lists, raw/accepted responses, task demonstrations, exclusions, verifiers and the recorded hard-MATH mode. Record the manifest location, expected hash and observed hash for each source in the confirmation binding receipt. A current pointer or the original generation protocol alone is not identity evidence. **Missing artifacts/manifests or any hash mismatch ⇒ affected work PENDING plus owner escalation, never reconstruction or substitution.** Regeneration is eligible only if it reproduces the **identical recorded hashes**; it cannot clear PENDING without that proof. No expected hash may be inferred or replaced by a newly generated one.

Freeze licences and all three new seed schedules before training. Derive schedules for 104–106 only from those identical sources and the frozen scheduling algorithm/salt, substituting only the seed values; record the new schedule hashes separately from the v6a source hashes. New seed schedules are not new target selection. Require unique accepted rows sufficient for **no within-stage repeats**: GSM8K **80**, ARC **80**, earlier FinQA **320**, earlier legal **320**, shared instruction **160**, instruction pool for G **960**. Retain all inherited minima; report across-stage repeats separately. Insufficient pools leave CR/G PENDING with owner escalation, without weakened coverage or replenishment. The realised manifests and hard-MATH mode still require node-side binding; this document does not claim their hashes have been verified locally.

- **Instruction:** B's checked RLVR-IFeval pool, excluding refresher-64, unchanged between arms. G's extra generic rows come from this same eligible source.
- **GSM8K/ARC:** v6a's realised accepted `next_pools` GSM8K-train and ARC-Challenge-train base-success artifacts and hashes, from **1,024 candidates/domain**, with completion/encoding/gold checks. Bind the realised ARC `FROZEN.json` and passing **128-row acceptance**; never reuse bracket-format targets. Re-running the bounded construction does not establish identity; only identical recorded hashes do.
- **Earlier FinQA:** reverified B training demonstrations, excluding **all 1,147 panel questions**.
- **Earlier legal:** v6a's hash-bound `contract-replay.jsonl`, raw targets and 27B training journal through `c_targets.select_pool/verify`, including its original **≤1,024-candidate/≤4-attempt gold-blind schedule** receipts. Missing targets invoke the PENDING/escalation rule above, not a new generation schedule. Require document/group exclusions, terminal gold label, **≥8 reasoning words**, no think tags, complete stop/EOS, input **≤10,240**, response **≤2,048 including EOS**. Preserve attempt receipts; label correctness does not certify rationale truth. No dev/test/legal-panel documents.
- **Hard MATH:** bind the exact targets, schedule sources and **recorded `core-copy` or `m3-fallback` mode actually used by v6a**. Do not choose a mode from today's core M5 pointer or switch from v6a's fallback to a newly available core artifact. Missing mode/hash/artifact or mismatch leaves stage 3 PENDING with owner escalation; no easy-MATH or different task fallback. Regeneration must reproduce the same recorded mode and identical hashes.

Apply **normalized equality, five-shingle Jaccard ≥.8 and edit similarity ≥.9** across pools/panels. Freeze PD/SP and existing core holdouts against candidate-prompt supersets before preparation; preserve general/HellaSwag, legal document/group, FinQA and monitor/probe exclusions. Later checks verify identity only, never remove scored items. Overlap invalidates affected evidence. CR broad pools retain their own namespace regardless of core nomination/non-nomination: technical pool validity, not component admission, controls eligibility. No v6b nomination or new-row outcome changes these CR/G definitions.

Benchmark training splits and July–September development rules remain unchanged. January–June belongs to 4A; October–November stream confirmation stays untouched until its separately approved **15–31 December** build/audit. This seed confirmation consumes no reserved stream evidence.

## Primary, references, panels and guards

**Primary CR−G:** stage-3 step-40 **equal-weight retained accuracy over exactly instruction, general maths, knowledge, FinQA and legal test**. For arm a and seed s, define accuracy in percentage points and

`R(a,s) = [(instruction_3 − instruction_base) + (maths_3 − maths_base) + (knowledge_3 − knowledge_base) + (FinQA_3 − FinQA_acquisition,a) + (legal-test_3 − legal-test_acquisition,a)] / 5`.

`d_s = R(CR,s) − R(G,s)`.

General references are the **original base**; FinQA acquisition is **that arm's stage-1 final**, legal acquisition **that arm's stage-2 final**. Base cancels algebraically but its same-lane item scores remain mandatory. Do not use the other arm's acquisition score, a shared winning final or incoming MATH score for all five references. Guard absolute final scores so weak acquisition cannot manufacture retention. Each seed contributes one five-domain mean; domains/items are not extra seeds. HellaSwag is a guard, not a sixth primary domain.

Score **FinQA-1,147, legal-test-100, qualified hard-MATH-100, general-v1 and both likelihood panels** at base and every stage final, retaining required stage-start/acquisition references. Collect each stage's task curve at **0/10/20/30/40** and step-20 retention diagnostics. Legal test uses the inherited tolerant metric; the legal diagnostic panel stays separate and cannot replace test. General instruction keeps strict all-constraint IFEval accuracy. Publish absolute scores, retention, acquired-answer survival on each arm's acquisition cohort and their intersection, and answer transitions; cohorts remain descriptive and never enlarge n.

The original guard inventory is unchanged; each is a paired **CR−G** contrast:

| Boundary | Mandatory guards |
|---|---|
| After FinQA | Each general panel individually (**IFEval**, maths, knowledge); HellaSwag-100 likelihood accuracy; FinQA task final and gain from its stage start |
| After ContractNLI | Each general panel individually (**IFEval**, maths, knowledge); HellaSwag-100; FinQA absolute final and retention from each arm's FinQA acquisition; legal-test task final and gain from its stage start |
| After hard MATH | Each general panel individually (**IFEval**, maths, knowledge); HellaSwag-100; FinQA and legal-test absolute finals and retention from their own acquisition finals; hard-MATH task final and gain from its stage start |

**Every guard requires its paired two-sided 95% seed-t lower bound ≥−1 point.** In particular, report all three IFEval guard verdicts even if the primary already fails. There is no pooled-general guard replacing individual domains and no exception for the previously observed IFEval cost. **Gain-AUC stays diagnostic**, using normalized trapezoids at 0/10/20/30/40 relative to that stage's incoming score; sibling registrations that guard AUC do not add a new CR/G gate.

Report inherited absolute package bars separately, **not as gates for CR−G**: FinQA acquisition lower bound >0; legal tolerant mean/lower bound ≥**69.7** and positive gain; general ≥base−3 each boundary; prior tasks ≥acquisition−1 thereafter, with inherited **one-sided t=2.919986** where applicable. Standalone MATH checks remain PENDING without matching references; these chains do not replace C Goal-1. Report tokens/correct and the inherited **1.5× base serving ceiling**; exceedance is “at cost”.

## Reference-scored likelihood and hardware authority

Apply the [1 November reference-scorer ruling](../../programme-plan-v6-amendment-20261024.md#likelihood-guards-scored-by-the-reference-scorer-1-november-owner-fixed-before-any-kcr-outcome): **HellaSwag-100 and general-v1 knowledge use the independent HF reference implementation at every decision-bearing base, incoming and arm-final checkpoint**, labelled **REFERENCE-SCORED**. This includes knowledge's contribution to the primary, not only its guard. Preserve the registered reference precision and fp32 likelihood reductions; no downcast or alternative implementation to fit more workers. Freeze scorer/tokenizer hashes, exact items, prompts, conditional-token boundaries, length normalization, EOS convention and ties. Strict generated-letter knowledge failures remain diagnostic only; no outcome-dependent endpoint switching.

Retain the [near-tie ruling](../../programme-plan-v6-amendment-20261024.md#likelihood-scorer-validation-near-tie-ruling-1-november-owner-fixed-blind) unchanged: a production/reference validation passes only if **every disagreeing item** has a top-two length-normalized likelihood gap **strictly <0.001 nats/token in both scorers**. Any larger gap in either scorer leaves that validation PENDING; exact threshold equality does not pass. Publish checkpoint/item IDs, both selected options and both top-two gaps, keeping partner calculations PARTNER-COMPUTED. The frozen scorers store likelihood per option character; use pinned character/conditional-token counts only to express the validation gap in nats/token, as in [the existing validation implementation](../../../kit/likelihood_validation.py). Do not change option selection or normalize the scientific endpoint differently. The reference authority supersedes failed kit validation for these decision scores; it does not retroactively call the failed kit scorer valid, relax the near-tie threshold, or authorize a later switch back based on results.

Use **v6a's per-seed lanes: exactly one native job and one reference job per seed**, with a single physical GPU UUID/environment/allocation-job identity within each job. The native assignment remains `m3|{seed}`; every REFERENCE-SCORED cell binds that seed's single reference assignment. Each job collects all applicable panels for original base, required incoming/acquisition references and both arms, **including all 0/10/20/30/40 task-curve points and step-20 diagnostics**. Native and reference jobs may have distinct assignments, but panels within either job may not split across GPUs or allocations. Preserve the frozen reader's identity checks unchanged, including checks of diagnostic/curve cells; any off-assignment CR/G constituent makes the contrast **UNVERIFIED-HARDWARE**. Freeze and verify checkpoint/export hashes and assignment receipts. All panels retain their exact kit scoring configuration; no prompt/decoding/numerical change hidden as scheduling. Stage all inputs before either job starts; the serial per-seed queue is priced below.

## Statistics, outcome labels and incumbent precedence

For the three paired differences from **104, 105, 106 alone**, use sample SD and **df=2**, `CI95 = mean(d) ± 4.302653 × SD(d)/sqrt(3)`. Admission requires **mean ≥2 points**, **two-sided 95% lower bound >0**, **positive 3/3**, and **every guard lower bound ≥−1**. No one-sided substitution for the primary or paired guards. Print all three differences, mean, SD, bounds, positivity and each guard's bar result.

The radius is **2.48414×SD**: at mean +2 the primary needs SD <.805 to exclude zero; a zero-mean guard needs SD ≤.403 to clear −1. v6a's primary SD 1.190 is exploratory sensitivity context only, not a variance estimate guaranteed for fresh seeds. Publish available historical guard analogues with their provenance; unavailable analogues are UNMEASURED. Do not use old data as observations, prior pseudo-seeds or a pooled power rescue. An INCONCLUSIVE confirmation remains possible. Seed-t intervals are per contrast, not family-wide 95% coverage for seven comparisons; any bootstrap is labelled diagnostic.

| Evidence / result | Registered disposition |
|---|---|
| Complete valid evidence, all primary and guard bars pass | PASS / admitted for the exact CR recipe, subject to the future-message binding below |
| Complete valid evidence, any primary or guard bar fails | **INCONCLUSIVE**, incumbent unchanged; the failed bar/admission condition is **NOT MET**. Do not invent a different negative-efficacy threshold or relabel v6a. |
| Inherited package/goal criterion is NOT MET under its own reader | Preserve **NOT MET** for that package/goal report; it is not a substitute CR−G verdict or a new comparative gate |
| Missing seed, stage, pool, qualification, reference, score cell or required receipt | **PENDING/INCOMPLETE**, never a two-seed verdict or a scientific null |
| Wrong recipe/ancestry/base/data/scorer identity, leakage or invalid provenance | **UNVERIFIED**; hardware subtype **UNVERIFIED-HARDWARE**. Missing evidence needed to verify remains PENDING. |

Unresolved decision-bearing letter-format dependence elsewhere retains **FORMAT-CONFOUNDED → PENDING**. No n=2 fallback, fourth/replacement seed, optional stopping, dose extension, post-outcome reweighting, guard relaxation or scientific retry to seek a pass. Technical resume preserves the original RNG/optimizer/EMA state and all attempt costs.

**Incumbent binding, fixed now:** v6a admitted neither K nor CR, so **v7's other six rows keep their frozen I0 binding**, including their registered independent comparators, nomination-only exceptions and v6b M8/M9/X/P2/compatible-stacking prerequisites. This confirmation is independent of those outcomes. It cannot release, rebind, reparent or change an OPR trigger in those six rows, even if its results are available first. Their original seed set and specifications remain frozen.

**Only if this confirmation is admitted under all bars, CR takes precedence over I0 as the task-learning comparison incumbent starting with the message after v7 (v8), never during v7.** Future dispatch binds the exact CR chain, its own acquisition parents, domains, exposure and denominator. K is excluded; the old K-first rule cannot resurrect an INCONCLUSIVE K. If this confirmation is INCONCLUSIVE / not admitted, **retain the then-current incumbent**, including any valid v7 admission on I0. If confirmation evidence is PENDING or UNVERIFIED, retain the then-current incumbent without declaring non-admission; a future row whose comparator requires resolving CR admission stays PENDING rather than selecting a branch opportunistically.

A v7 component pass on I0 remains scoped to that ancestry: it neither outranks confirmed CR by a best-score comparison nor transfers automatically onto CR. Core P2/stacking is incompatible with CR unless separately tested; do not construct CR+P2, CR+OPR/Fisher/iSDFT/freeze or any combined winner here. Later combinations require their own registered one-change test and v6 connected-stack recheck. This is a prospective dispatch precedence, **not evidence that CR beats I0**, since G is the exposure-matched control. The product incumbent, frozen Qwen3-8B plus static k=3 memory, is unchanged.

This follows the [README's pinned dispatch precedence](README.md#common-incumbent-binding-rule) (line 27) and its [ancestry-scoped component rule](README.md#common-incumbent-binding-rule) (line 38). **Owner acknowledgement requested before future dispatch:** admitted CR takes precedence from v8 even if v7 admits a component on I0; that component is retained as ancestry-scoped evidence, not automatically carried onto CR. Non-admission retains the then-current incumbent. This acknowledgement remains pending; it does not reopen v7 bindings or permit post-outcome re-ordering.

## Qualification and v6 right-sized execution

Preparation, tokenization, hashing, binding, collection and reading run on **CPU with 0 GPUs**. Train with **8 GPUs** only while the eight-rank trainer is running; use **1 GPU per complete generation/scoring lane**, or **2 GPUs** for the already registered TP2 teacher preparation. No allocation holds GPUs waiting for input/export/checkpoints. Stage all lane inputs first. Stages remain serial within each arm/seed; independent ready chains may run on disjoint allowed nodes, with original core-node exclusion/submission priority and isolated confirmation manifests/state/output trees.

The [v6a partner finding](../return-v6a-reading-20261106.md#partner-operator-changes-how-not-what-assessed) supersedes the older blanket concurrency-3 scorer prescription: **likelihood scorer concurrency K≤2**, where K is worker concurrency, not scientific arm K. The partner's two-copy **.42** memory fraction is a qualification starting point, not a guaranteed fit or a license to alter precision. HF reference guards retain their own implementation and worker qualification; use fewer copies if necessary. Non-likelihood ready-command concurrency may use qualified v6 settings. Generation-only K=3 qualification does not qualify prompt-logprob or HF reference workers. Serialize engine starts per GPU, release the generator before its likelihood child and measure actual memory; preserve scorer numerics.

Use the [v6 allocation/watch rules](../../../kit/util70-integration.md): watcher/lease per allocation, per-GPU rolling and whole-life mean utilization, **≥70%** allocation utilization qualification, including holder-slice telemetry. v6a did not establish the operator's precise window or startup grace; obtain and report that operating definition before claiming qualification. Reference lanes' low utilization must be handled by qualified scheduling/co-location, never by dropping HF guards, changing precision or padding with meaningless work. Eviction/unqualified lanes are PENDING; re-score a failed complete lane together in a fresh job and retain its cost. Training resumes only from verified native state. Timing reprices work; it is not an efficacy gate.

Require current-code CR/G smoke and disposable **8B/eight-GPU node preflight** bound to admitted longest legal rows, materialised optimizer state, finite losses/gradients/EMA, correct masks/exposure/step counts, no truncation, reduction-gradient relative error **≤1e−4** at world sizes 1/2/8 and native checkpoint/resume/export equality. Measure **≥3 nonzero-LR updates**, then restart scientific chains untouched. Retain **GPU peak ≤90% and cgroup RAM ≤95%** qualification limits. The retrospective **0.95 arm-K memory deviation applies to K only**, not CR/G. A small-model smoke or historical v6a result alone cannot qualify changed code. Qualification missing/failing leaves PENDING/INCOMPLETE; identity failures are UNVERIFIED. Any numerical optimization needs same-input loss/gradient/update equivalence evidence and must apply consistently to both arms and all stages. Record every operator change in Part 3.

## Cost envelope and wall-time limits

Use [K/CR repricing, 29 October](../kcr-reprice-20261029.md), **not** the superseded 425–516 four-arm estimate, the old 6.572 legal anchor or the v6a aggregate **298.3 GPU-h** as a matched confirmation forecast. Charge **two arms × three seeds × three stages =18 scientific stages**. Both arms have the same cap-based rates; content-dependent decode/prefill differences require telemetry. No K base-forward work or C0 training is included.

v6a Part 3 labels 298.3 as **“recorded-process-GPU-hours-only”** and **“missing-allocation-or-idle-is-unknown”** ([raw return, Part 3](../return-v6a-text/parts-20261106.txt)). Its stage/arm coverage and inclusion of qualification, scoring and failed attempts are unresolved; it cannot establish an allocated-GPU-hour rate for these 18 stages. If it covered all 36 original K/CR stages, the implied rate would be about 8.29 process GPU-h/stage, materially below this cap-based forecast, but that scope is unverified. **Request in v7 Part 3:** a per-arm/seed/stage breakdown of process and allocated GPU-hours, GPU count and elapsed time, with training, preparation, qualification, ordinary/reference scoring, exports, idle time and failed/retried attempts separately reconciled to totals; explain the coverage of v6a's 298.3 where recoverable. Use measured current-code timings to reprice before send, not an unexplained aggregate discount. The 830 allowance remains a conservative request under the owner's wall-time priority, not evidence that this much compute will be consumed.

| Training component | GPU-h per arm/seed stage | Six stages across both arms and all three seeds |
|---|---:|---:|
| FinQA, 40 rows/update | 19.37–31.95 | 116.22–191.70 |
| ContractNLI, 48 rows/update | 23.24–38.34 | 139.44–230.04 |
| Hard MATH, 56 rows/update | 27.11–44.73 | 162.66–268.38 |
| **Training total** | | **418.32–690.12** |

This row deliberately calculates from the source's **published per-stage rates**, using decimal half-up to .01 after each component formula, then summing displayed components, as the v3 siblings do. Thus one arm's three-seed training is **209.16–345.06**; the source's **209.17** lower figure used underlying unrounded phase rates. The .01 difference is rounding basis, not a throughput saving. Do not add the superseded five-times-legal replay surcharge: the common padded caps already charge that work, including replay rollout/discarded EMA passes.

| Incremental confirmation component, all three seeds | H100 GPU-h |
|---|---:|
| Scientific training, 18 stages | 418.32–690.12 |
| Ordinary/reference scoring and validation, 330 call-equivalents × .05–.10 | 16.50–33.00 |
| Scientific setup/export/transfer reserve, 18 × .50–1.50 | 9.00–27.00 |
| One qualification per arm, 2 × 6.69–12.24 | 13.38–24.48 |
| **Reusable-input envelope** | **457.20–774.60** |
| Conditional identical-hash broad-pool regeneration, once | 2.00–25.00 |
| Conditional identical-hash regeneration of v6a's recorded hard-MATH fallback, once | 10.00–30.00 |
| **Envelope including both preparation contingencies** | **469.20–829.60** |
| **Incremental owner planning allowance requested, upper rounded up** | **830** |

Scoring scales the source inventory to **48 call-equivalents ×2 arms ×3 seeds =288**, plus **18 base-panel** and **24 production/reference validation passes**, total **330**. The arm allowance includes ordinary and HF reference finals, task curves and diagnostics; shared base/validation is charged once, not once per arm. These are provisional call-equivalents, not an assertion that every current lane command has been timed. Replace with an exact deduplicated inventory covering all required reference/arm/panel cells and measured HF throughput before send. Lower worker concurrency gives no GPU-hour discount. Qualifications retain eight rollout/scientific-like passes plus two diagnostic passes and 1–3 GPU-h launch/resume IO per arm, as the source prices them.

Existing verified pool/target artifacts cost **zero new generation** and must not be regenerated to spend a contingency. Use 457.20–774.60 if both preparation artifacts are reusable; add only an actually required prerequisite that reproduces the recorded v6a hashes. Missing/mismatched artifacts remain **PENDING plus owner escalation** under the binding rule; a budget contingency does not authorize reconstruction. The broad contingency retains the original 1,024 candidates/domain and is usable only for identical-hash regeneration. The MATH contingency applies **only if v6a recorded `m3-fallback`**, retaining **500 training targets**, ≤4 attempts/item, 2,560 output cap, verified first acceptance and the 2,048 student response limit on the registered TP2 path, with identical hashes required. It cannot fund a mode switch or replace a missing `core-copy` artifact with fallback targets. No new candidate search, changed accepted pool or screening panel is bought here. Missing upstream FinQA/legal targets require owner escalation and PENDING status; any identical-hash regeneration beyond these contingencies must be separately priced.

This **830-GPU-h request is additional to v7's approved six-row allocation**, not covered by it or by v6a's old tier allowance. The existing shared integration line in the six-row README is charged once; any additional confirmation-only integration/retry expense must be itemized and repriced. Current forecasts are assumptions, not confidence intervals, measured ceilings or new spending authority. Charge allocated GPUs × elapsed time, including idle ranks, failed attempts, startup, teardown and IO. CPU work does not retain GPU reservations. Replace allowances with current-code phase/reservation telemetry and record incremental allocation approval before send; never silently reduce scope/dose to fit an estimate.

Training alone is **8.72–14.38 h per arm/seed** on eight GPUs. Own scoring allowance is **2.40–4.80 h**, shared base/validation **.70–1.40 h per seed**, setup **.19–.56 h**, giving approximately **12.00–21.14 h warm-input chain**, or **12.84–22.67 h** with that arm's one-off qualification exposed. Conservatively serializing both arms' score work gives a **5.50–11.00 h per-seed scoring queue** and approximately **15.24–27.47 h** critical path including qualification: **WALL-TIME RISK >24 h at the upper end**. These serial examples take no concurrency speed credit. If cold broad/MATH preparation is exposed, add shared waits **2–25 h** and **5–15 h** respectively, without charging those GPU-hours again. Report actual per-arm/seed and combined-lane serial time, critical path, queue/load/transfer delays and every >24 h risk before send. Our-side paid execution under this documentation task is **$0**.

## What this does NOT claim

This is a fresh-seed confirmation on the same fixed benchmark panels, not new-task or untouched-stream replication. It does not turn v6a into a pass, establish family-wide 95% confidence, prove universal non-forgetting or restored plasticity, or show that coverage protects instruction following when its guard fails. CR−G isolates replay coverage at matched task/replay exposure; it does not establish a compute-optimal replay budget, superiority to I0/product memory, truth of stored rationales, or the benefit of K, P2 or a combined stack. A failed confirmation does not prove all cumulative replay ineffective. No result substitutes for C Goal-1, 4A, the required connected-stack recheck or October–November confirmation.

## Review disposition

Awaiting round-2 re-review after [Opus round 1 AMEND](review-cr-opus-r1.md); table intentionally empty per manager instruction. No sibling review approval is claimed for this seventh registration.

| Finding | Applied/rejected | One-line note |
|---|---|---|

## Builder deltas

Forward-looking requirements, to be completed after approval and before confirmation data. These are not claims of completed implementation or qualification.

| Area | Required delta |
|---|---|
| Realised input binding | Bind every pool, target and schedule source to v6a's node-side manifest hashes under `059d6293`, including the recorded hard-MATH mode; publish expected/observed hashes. Missing or mismatch means PENDING plus owner escalation; regeneration qualifies only with identical hashes. Freeze new seed schedules from those same sources and unchanged salt/algorithm. |
| Confirmation scheduler/reader | Add an isolated mode asserting `seeds == {104,105,106}`; reject any 101–103 cell as UNVERIFIED before analysis. Missing cells are PENDING/INCOMPLETE. Never extend the exploratory seed constant to six; emit CR−G only, with unchanged primary, guards, statistics and identity rules and no exploratory “confirmation opens” text. |
| Namespace and outputs | Separate confirmation manifests, state, scores and reader output trees from exploratory v6a and the other six v7 rows. Preserve the scientific `m3-cr-v2` salt and per-seed `m3\|{seed}` assignment lookup; output isolation must not alter schedule order or hardware identity semantics. |
| Per-seed scoring | Exactly one native job and one reference job per seed across both arms/base/references and all panels, including curves and diagnostics. Publish complete job/UUID/environment receipts and reject off-assignment cells as UNVERIFIED-HARDWARE; qualify complete-lane recovery without stitching attempts. |
| Code diff receipt | Before new data, publish and review the confirmation implementation/reader diff against kit `059d6293`, limited to the seed-set/confirmation-mode isolation and namespace changes above plus approved operator changes **3a/3b/3d/3g/3h**, **likelihood K≤2**, and the **holder-slice watcher**. Enumerate each changed file/hunk and its allowed category; any other delta leaves the build PENDING for escalation. No CR/G recipe, loss, pool, panel, reference, bar or hardware identity amendment is allowed. |
| Operator implementation | Pass the attention backend through `EngineArgs.attention_backend` (3a), serialize engine starts (3b), re-measure memory (3d), free the scoring engine before the likelihood child (3g), and wait for neighbour memory (3h). Enforce likelihood worker K≤2 and collect holder-slice telemetry without changing numerical precision or scoring authority. |
| Qualification and accounting | Exercise old-seed contamination, missing inputs, hash/mode mismatch and off-assignment curve rejection; verify isolated outputs, full three-seed CR−G and frozen formulas/guards. Complete the registered gradient/resume/longest-input preflight, exact scorer inventory and current-code timing; request the v7 Part-3 per-stage process/allocation reconciliation and report >24 h critical-path risks. |

## Owner approval (6 November 2026)
Owner: "Approve 830 + acknowledge". The allocation of up to 830 partner GPU-hours is approved, on top of the six v7 rows. The non-admission rule is acknowledged: retain the then-current incumbent, including any valid v7 admission. Build authority is granted. The tag needs separate owner approval.

## Registration delta: FinQA fresh-seed schedules (owner, 6 November 2026, before any data)
Owner: "Yes, approve regenerating FinQA orders with package B's procedure".

v6a's FinQA stage copies package B's native per-seed order, and package B has orders only for its own seeds. For seeds 104–106, the FinQA schedule is therefore **regenerated with package B's own ordering procedure** (same code, data, corpus and sampler; only the seed changes). Rules:
- **Gate:** before any 104–106 order is used, the procedure must reproduce byte-for-byte v6a's realised FinQA schedules for seeds 101–103. It must also reproduce package B's stored orders for any of seeds 104–105 that exist on the node.
- **Failure:** any mismatch leaves FinQA PENDING with owner escalation, never a fallback.
- **Unchanged:** legal and hard MATH keep their seed-shuffle-cycle rule.
- **Freeze:** this delta is fixed before any 104–106 output exists.
