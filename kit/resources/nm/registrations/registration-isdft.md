# iSDFT component 1 — new-task base-policy KL — DRAFT, 2 November 2026

**DRAFT — owner approval and an Opus cross-family review before any build. No execution or spending authority.** Goals 3–4 / P3; replan §3 row 7. The [shared registration](README.md) governs incumbent binding and execution. Exactly paired seeds **101–103**.

## Question, trigger and anchor order

Does adding a frozen-base policy anchor on new-task trajectories to D improve instruction retention without reducing legal acquisition or FinQA retention? Return-1 instruction loss motivates the row but does not establish persistence on the selected I. **Pre-fixed persistence gate:** use only I’s existing binding evidence at the ContractNLI step-40 boundary, strict general-v1 instruction versus its original base, seeds 101–103. Require mean retention <−3 points (the inherited instruction-loss margin); record all seed differences and the inherited interval, without selecting another checkpoint/panel. Complete valid evidence with mean ≥−3 archives this row **ARCHIVED UNREAD**; missing/wrong-recipe evidence leaves PENDING. Seal the gate before any new row is trained or read. In particular, an admitted K that has repaired the deficit does not trigger this experiment.

[iSDFT, 2609.24646](https://arxiv.org/abs/2609.24646) separates a frozen-base forward-KL anchor from its information-proximal teacher target. This registration tests **the anchor alone**, added to unchanged D; it does not introduce the paper's teacher-information budget, interpolation target, bisection or schedule.

**Correction M3:** K/CR verdicts arrive with v6a; under K or CR, P2/core stacking are incompatible and automatically omitted, so this row binds without waiting for v6b, subject to persistence. Under I0, wait for v6b M8/M9, eligible X, P2 and required compatible stacking. This corrects prerequisite timing, not the frozen K → CR → applicable P2/stacking → I precedence. K's replay KL, P2's parameter L2 and this new-task KL all use **the original base** `b968826d9c46dd6066d109eabc6255188de91218`. Neither D's moving EMA nor incoming FinQA weights replace it. Separate passes do not establish K+CR or K+P2; required missing evidence remains PENDING.

## Arms, loss and dose

From the bound recipe's three FinQA step-40 finals, fork fresh **I** and **I+task-KL** ContractNLI stages. Parent hashes, provenance and recipe match per seed. I is the exact selected D/replay recipe, with compatible admitted P2 retained in both arms if applicable. No new replay domains, no start substitution and no consumption of a new Fisher/OPR/freeze winner from this dispatch.

For each new-task row i and the student's own response prefix, define

`a_i = mean_{t in M_i} KL(p_base(.|x_i,y_i,<t) || p_student(.|x_i,y_i,<t))`.

Use full vocabulary, temperature 1, **forward KL base→student**, base eval/no-grad. M_i contains all response prediction tokens including EOS, excluding prompt and padding; no three-token skip or importance correction in this added term. Student and base see the same memory-off task prompt and student prefix, **no demonstration**. D retains its own distinct demonstration context, masks, three-token skip and importance correction unchanged. Replay rows receive **no new anchor term**. The anchor target is not refreshed from the incoming final.

`L_challenger = L_I + .10 × (Σ_new-task a_i)/32`.

**μ=.10 is fixed here**, a prospective component-test coefficient, not selected from task/evaluation scores or inherited from K's replay weight. There is no μ calibration/sweep or information-budget component. Sequence means and denominator 32 preserve the task share: e.g. 28 task rows in K versus 32 in CR remain their respective shares. Report realised task/anchor values, gradient norms, alignment and clipping; do not renormalise μ after seeing them. If μ=.10 is numerically infeasible, mark qualification incomplete and re-register rather than change it silently.

Both arms: 40 updates, full-weight BF16, LR 1e−5, ten-update warmup including first zero LR, same AdamW/clipping/weight decay, fresh optimizer/scheduler and incoming-initialised EMA .99/.01, verified 27B legal demonstrations, one on-policy sample/task prompt. Export 0/10/20/30/40. Preserve I's replay counts/positions/order, other losses and denominators. CR stage 2 retains 48 rows, denominator 32; K retains 28+4. Use README’s admitted caps/draws rule and a paired schedule/allocation policy: retain core’s inherited replay-draw filtering under I0; keep K/CR unfiltered. No new filtering or balancing deviation. D's teacher changes through its normal EMA only; never add the anchor to the teacher objective.

## Data and implementation qualification

Use the frozen ContractNLI training split, document/group exclusions, verified targets and replay candidate supersets already bound to I. No new teacher labels or held-out documents. All loss-bearing rows exclude legal test/diagnostic panels, FinQA-1,147, general/HellaSwag, Screen-S monitoring/probe data and PD/SP holdouts by README's exact/near-duplicate rules. No SEC stream data: January–June remains 4A's, July–September remains development and October–November untouched. Freeze all prompts, source revisions, masks and target hashes before training.

Apply README’s single admitted-caps/draws rule to both arms, including student, response and teacher limits. Current source legal widths are 12,288 actor/base and 16,384 EMA; verify the returned binding. Overlength failures block preparation, never disappear from the admitted schedule or induce a cap change.

Maintain a separate explicit BF16 frozen base, distinct from D's EMA, sharded over the eight training ranks. An already-present K frozen-base module can be shared **only as read-only identical weights**, with separate correct task/replay inputs; do not charge the same persistent weights twice or conflate its losses. Price all added reference forwards/collectives. Use bounded 256-token full-vocabulary projections and fp32 log-softmax/KL reductions, never whole batch×sequence×vocabulary buffers or a top-k/tail approximation.

Preflight: μ=0 reproduces I; analytic/small-reference forward-KL values and student gradients agree ≤1e−5 absolute on bounded tests; distributed loss reduction gradient relative error ≤1e−4 at world sizes 1/2/8. Verify no demonstration leakage to base/student, no anchor on replay, no reference gradients, stable base hashes, finite losses/gradients/EMA, one optimizer/scheduler/EMA step and longest-row memory with materialised optimizer state. Test zero/nonzero-LR updates and native export/resume equality. Unsupported memory/numerics leave PENDING; measured extra time only reprices.

## Endpoint, guards and decisions

**Primary:** final general-v1 instruction accuracy change from original base, (I+task-KL)−I, at ContractNLI step 40. Base cancels in the difference but its scores remain required. Do not switch to average general retention or knowledge after observing instruction results.

Require mean ≥2 points, 3/3 positive, paired two-sided 95% lower bound >0: `mean ± 4.302653 SD/sqrt(3)`. Each new guard uses that same lower bound ≥−1: each general panel individually, HellaSwag-100, FinQA absolute final and retention, legal-test-100 tolerant final, gain and normalised gain-AUC. Gain-AUC is trapezoidal accuracy at 0/10/20/30/40 divided by 40 minus the unchanged incoming score. It guards acquisition speed while the primary measures retention; it does not prove plasticity restoration. Legal diagnostic panel remains separate. Record acquired-answer survival and answer transitions without selecting a final-correct cohort.

Score base/incoming/final on every panel, legal curve at all five exports and step-20 retention diagnostics. Knowledge/HellaSwag use the full-precision **REFERENCE-SCORED** authority and every decision-bearing curve/reference/arm stays in a complete per-(seed,panel) one-GPU lane. Report inherited absolute package summaries with **one-sided t=2.919986** where applicable: legal ≥69.7 and positive gain, general ≥base−3, earlier task ≥acquisition−1 and positive FinQA acquisition. They are not new gates or substitutes for the comparative guards. Report token/correct and >1.5× base serving cost as “at cost”.

Complete valid evidence missing a bar: INCONCLUSIVE; missing cells or unresolved K/P2 binding: PENDING/INCOMPLETE; wrong base, data or checkpoint identity: UNVERIFIED; mixed scoring hardware: UNVERIFIED-HARDWARE. No fourth seed, chosen checkpoint, μ retry or post-outcome guard relaxation. A pass applies only to the bound recipe; adding the information-budget component or combining this with another new winner needs a separate comparison and immediate v6 objective/stack recheck.

## Cost and limits

**Repriced M1, all three seeds:** **K 123.07–229.34; CR 170.56–299.84; I0 119.37–212.90 GPU-h**. Start with README’s `6T+2Q+15–33` at the admitted legal caps. Add three challenger-stage full-width base passes plus .25 stage for challenger qualification: **3.25×(.562–2.528)** for 32-row I0/K or **3.25×(.843–3.792)** for 48-row CR. Charge conservative dummy/reference work as well as task rows; do not claim free K module reuse before exact-code measurement. Existing K base work is already in T_K; this addition prices the new task-anchor pass. Separate fresh controls are included; integration 20–40 is shared once. Archived persistence gate means no scientific row cost. CPU work zero GPUs, training eight, scorer lanes one with qualified concurrency 3. No paid execution authorized. Wall time UNMEASURED; flag >24 h per arm/seed or combined lane.

This does **NOT** test full iSDFT, show that new-task KL and replay KL are interchangeable, or establish the optimal amount of base anchoring. Anchoring may undo FinQA learning or suppress legal acquisition; the guards are essential. It cannot prove multi-stage or stream retention, general restored plasticity, compatibility with untested K+CR/P2 combinations, or October–November confirmation. A null at μ=.10 is a result at this dose, not a rejection of every base-policy anchor.

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
| MINOR: persistence | Applied | Test mean instruction retention <−3 on I’s existing legal-final binding evidence; otherwise archive; missing evidence PENDING. |
| MINOR: citation | Checked | iSDFT primary abstract verified; anchor alone remains the single component tested. |

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
| Binding | Run and seal the existing-evidence persistence gate; archive repaired instruction loss before training; omit P2 under K/CR. |
| Objective | Add only μ=.10 task-prefix forward KL; preserve admitted schedules/filtering, D/replay losses and original-base identity. |
| Qualification/cost | Test μ=0 equivalence, task-only masking, longest admitted caps and native resume; measure added reference passes and quote branch costs. |
