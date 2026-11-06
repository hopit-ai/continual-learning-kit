# Fisher-weighted versus isotropic base L2 — DRAFT, 2 November 2026

**DRAFT — owner approval and an Opus cross-family review before any build. No launch or spending authority.** Goal 4 / P2; replan §3 row 6. Apply the [shared registration](README.md); exactly paired seeds **101–103**. Read only **after P2's verdict** is resolved and its compatible recipe/anchor/coefficient status recorded.

## Question and one change

Does weighting a base-anchored quadratic penalty by a frozen diagonal importance estimate preserve general capabilities better than isotropic base L2 at matched initial regulariser-gradient norm? [Scholz et al., 2609.28395](https://arxiv.org/abs/2609.28395) motivates an LLM-scale EWC baseline; [Kumar et al., 2308.11958](https://arxiv.org/abs/2308.11958) motivates regenerative regularisation; [EWC, 1612.00796](https://arxiv.org/abs/1612.00796) motivates parameter-importance weighting. These sources do not establish benefit on our FinQA→ContractNLI chain.

Only the **diagonal weighting** changes. Same original base anchor, trainable tensors, coefficient rule, optimizer, D teacher, replay and starts. This is a base-anchored empirical-Fisher variant, not online EWC with an anchor at each task's final. Do not anchor Fisher to the FinQA final while isotropic L2 anchors to base.

## Arms, P2 binding and dose

**Correction M3:** bind K/CR from v6a and applicable core components from v6b, as README; Fisher itself always waits for v6b’s P2 verdict. P2 is omitted under K/CR, so ISO is necessarily non-incumbent and this row is benchmark-only there, with new recipe calibration. Let B be that exact recipe with its P2 penalty removed, if present. Train **ISO = B + λR_iso** and **FISH = B + λ_F R_F** from identical verified FinQA finals, fresh runs per seed. If compatible P2 is admitted in I, ISO is the fresh incumbent control and this is an incumbent comparison. If P2 is resolved not admitted, or passed only on incompatible ancestry, ISO is a **non-incumbent baseline**: FISH−ISO is readable as a benchmark but cannot admit FISH. Beating an unsuccessful regulariser does not prove improvement over I. Do not add a hidden third arm/admitting contrast; a later FISH−I registration would be required. This resolves the evidence notes' differing “after P2” versus “only after non-admission” suggestions in favour of the approved scope.

A P2 verdict is not its coefficient. Reuse the sealed P2 λ only when its recipe, starts, parameter set and calibration inputs match this comparison. Otherwise apply the same registered calibration rule on B's exact first scheduled batches before measurement, label it **new recipe calibration**, and do not claim it reproduces the old P2 experiment. PENDING P2 evidence always blocks this read; missing stacking blocks only where it is an applicable I0 prerequisite. Incompatible core stacking does not block the K/CR benchmark.

Both arms: full-weight Qwen3-8B, base revision `b968826d9c46dd6066d109eabc6255188de91218`; ContractNLI stage only, 40 updates, LR 1e−5, ten-update warmup with initial zero LR, fresh AdamW/scheduler and incoming-initialised EMA .99/.01, unchanged D demonstration/prefix/importance/masking conventions. Preserve the selected replay slots, task exposures and denominator: 32 rows for the corresponding I0/K recipe, or CR's 32+16 rows with denominator 32. Export 0/10/20/30/40; no later MATH stage. No extra dose, task loss reweighting, layer freeze or teacher regularisation.

## Penalty and frozen Fisher estimator

Let T contain all distinct trainable student parameters, counting ties once, including embeddings/head, attention, MLP and norms; P is its scalar count and Δ=θ−θ_base. Use

`R_iso = .5 Σ_j Δ_j²`; `R_F = .5 Σ_j w_j Δ_j²`.

Estimate a **diagonal empirical sequence-score Fisher proxy at the original base**, before any new run. For fixed response sequence i, define `ell_i = mean_response_plus_EOS log p_base(y_t | x,y_<t)`; accumulate **individual** `g_i = grad ell_i`, then `F_j = (1/N) Σ_i g_ij²` in fp32. Never square a batch-averaged gradient. Prompt and padding are excluded; EOS included, no D skip or importance weight. Equal response-sequence weighting is deliberate; this token-normalised empirical sequence-score proxy is not the exact model-expectation Fisher, token Fisher or Hessian. Record this approximation in the return and paper.

Use **256 fixed training/proxy rows**: 128 checked generic instruction, 64 GSM8K-train, 64 ARC-Challenge-train, all with frozen verified base responses. Hash order within domain by `fisher-20261102|domain|id`, take the specified counts. Sources/revisions, complete candidate supersets and raw targets must be frozen before estimating gradients. The three domains are fixed whether or not those replay components were admitted: they estimate importance only and are not new CE replay in either arm. Exclude all training-route evaluation panels, Screen-S monitoring/probe partitions and PD/SP holdouts, using README audit thresholds. Never estimate, normalise, select λ or inspect gradients on **any evaluation panel**. No stream data from any month is needed. Shortfall in any domain is PENDING; do not substitute validation/test rows or alter quotas.

Normalise globally: `m = (1/P) Σ_j F_j`, require finite m>0; `w_j = F_j/m`, so `(1/P)Σ_j w_j =1`. No layerwise renormalisation, top-k mask, damping, clipping or tunable exponent. Zero entries remain zero; publish quantiles, maximum, effective support and tensor-level sums. Compute one shared frozen F/w artifact for all seeds, hash it and the base before/after use. This normalises trace/average penalty scale to identity; it does **not** force equal realised penalties or gradient norms on the FinQA displacement. Keep trace-normalised weight summaries descriptive; match the initial regulariser-gradient scale below while retaining its direction.

Set ISO's common `λ=.10 median_s(||g_B,first,s||₂ / ||Δ₀,s||₂)`, `Δ₀,s=θ_in,s−θ_base`, for seeds 101–103, unless an exactly compatible sealed P2 λ exists. In the **same dedicated calibration pass**, set the single shared `λ_F = λ × median_s(||Δ₀,s||₂ / ||w⊙Δ₀,s||₂)`. Thus the median-seed ratio `median_s(||λ_F w⊙Δ₀,s||₂ / ||λΔ₀,s||₂)=1`; equality need not hold separately in all three seeds. Publish both coefficients, each seed’s raw/weighted displacement norms and realised ratio; median-ratio relative error must be ≤1e−4. Require every norm/ratio finite and positive, else PENDING; no epsilon/damping repair.

Calibration uses only the exact first **training** batch, incoming EMA, pinned rollouts/RNG/masks, deterministic settings and untouched FinQA parents. Accumulate/reduce/unscale before clipping; no optimizer/scheduler/EMA mutation and **no outcome/panel data**. Seal inputs/norms/coefficients then restart both arms untouched. Add each arm’s regulariser once per update before clipping; derivatives are `λΔ` and `λ_F w⊙Δ`. References receive no gradients; D’s EMA is not regularised. This matches initial strength by a fixed calibration rule, not a coefficient sweep. Subsequent norms, alignment and clipping may diverge as a consequence of anisotropy; report them without recalibration.

## Panels, endpoint and bars

**Primary:** FISH−ISO difference in final equal-weight general retention across **instruction, maths, knowledge**, each score minus that seed's original-base score. The base cancels algebraically, but same-lane base item scores remain required. Frozen three-panel set, never restricted to positive domains or admitted replay pools.

Require mean ≥2 points, positive 3/3 and two-sided paired lower bound >0 with **t=4.302653**, df=2. Each guard's two-sided lower bound must be ≥−1: individual general panels; HellaSwag-100; FinQA final and retention from the shared acquisition checkpoint; ContractNLI legal-test-100 tolerant final, gain and normalised gain-AUC. Gain-AUC uses trapezoids at 0/10/20/30/40 divided by 40 minus common incoming accuracy, matching P2's acquisition protection. No post-penalty “reset baseline”. Report acquired-correct FinQA survival and answer transitions descriptively.

Score base/incoming/final on every guard, legal curve at all five checkpoints and step-20 retention diagnostics. Knowledge and HellaSwag use full-precision **REFERENCE-SCORED** authority. Legal diagnostic panel stays separate. Every decision curve/reference/arm is in one complete per-(seed,panel) GPU lane. Absolute package summaries remain reported separately: legal ≥69.7 with positive gain, general ≥base−3, FinQA ≥acquisition−1, with **t=2.919986** for inherited one-sided bounds. No absolute package bar is silently weakened or used to replace the paired test. Report token/correct costs and >1.5× base serving as “at cost”.

## Qualification, budget and failures

Before execution independently check per-example squares against a small fp64 reference, tied-parameter counting, global trace normalisation, finite/nonnegative weights, exact base/Fisher immutability and no reference gradients. Compare penalty values/derivatives analytically and at world sizes 1/2/8 (relative gradient error ≤1e−4; bounded tiny-reference absolute error ≤1e−5). Confirm regulariser applied once, after D accumulation and before clipping, identical zero-λ path, longest-row production memory, complete optimizer state, export/resume equality and unchanged task exposure. Record Fisher and ISO gradient norms/cosines and clipping incidence; do not reject scientific runs merely because real anisotropy changes those ratios. Nonfinite computation or memory failure blocks qualification; timing only reprices.

**Repriced M1, all three seeds:** **K 125.24–235.12; CR 171.82–301.52; I0 121.54–218.68 GPU-h**. Each is README’s `6T+2Q+15–33`, plus **2–6** for penalty/shard work and a **2–8 provisional allowance** for one shared Fisher estimator and all three calibration passes. The old unsupported 24–50 estimate is withdrawn. These K/CR prices fund **benchmark-only** contrasts, not an incumbent admission. No old 6.572 anchor or frozen-parameter speed discount is used.

**MINOR — measured preflight repricing required, not yet performed.** After build/qualification authorization, preflight 32 fixed estimator rows (16 instruction, 8 maths, 8 knowledge, hash-first within the frozen 256) with production eight-rank sharding and padded lengths, including the longest row of each domain in a separate memory-only fixture. Record per-domain per-example wall seconds after warmup, startup/export GPU-hours and three exact first-batch calibration timings. Estimate `E = 8×(128 t_instruction +64 t_maths +64 t_knowledge +Σ_s t_cal,s)/3600 + H_startup/export`, using observed rates/length buckets, then add explicitly priced diagnostic/retry work and a declared 25–50% timing reserve on the estimated compute. Charge preflight work once; if the exact estimator gradients can be reused by hash, subtract only their already-computed rows from remaining work. Replace **2–8 with measured E** in this row and all README totals before send; absent measurement leaves **budget qualification PENDING**. No GPU work or fabricated receipt accompanies this documentation amendment. Stage rates can provision the rest of the envelope but cannot measure per-example-gradient overhead.

Store/stream fp32 Fisher/base shards; about 32 GB per 8B array is aggregate storage, not peak-memory qualification. CPU preparation zero GPUs; estimation/calibration/training eight where distributed; scorer lanes one with qualified concurrency 3. Shared integration is charged once in README; no duplicate estimator per seed. Wall time UNMEASURED; flag >24 h per arm/seed or combined lane.

Incomplete Fisher/proxy/calibration, unresolved P2 or missing controls: PENDING/INCOMPLETE. Wrong anchor, data overlap or identity: UNVERIFIED. Hardware mismatch: UNVERIFIED-HARDWARE. All valid bars not met: INCONCLUSIVE, incumbent unchanged. No extra seed, alternate Fisher source or λ sweep. In the incumbent branch only, passing supports replacing isotropic weighting subject to the v6 connected-stack/confirmation rule; otherwise it is a benchmark pass requiring a future FISH−I test.

## What this does NOT claim

**r3-m3:** matching the initial regulariser-gradient L2 norm does **not** match effective per-coordinate strength under AdamW. Adaptive first/second-moment preconditioning and clipping can make heavy-tailed Fisher weights concentrate or saturate the pull on a few coordinates while ISO distributes its pull evenly. The match is an initial raw-gradient norm convention, not equality of optimizer updates, realised penalties or later regularisation strength. Report weight tails and gradient/clipping diagnostics without recalibration.

This does not establish general plasticity restoration, Fisher optimality, faithful pretraining curvature, EWC replication or resistance across arbitrary future tasks. Base importance may protect old general behaviour at the expense of acquired FinQA; that is why FinQA is a mandatory guard. A favourable FISH−ISO contrast cannot repair an unadmitted P2 baseline by inference. Frozen proxy data and three seeds do not replace untouched stream confirmation or the required PD re-probe of an admitted P2 remedy.

## Review disposition

Response to [Opus round 1](review-opus-r1.md) and [round 2](review-opus-r2.md); round-3 manager decisions are fixed before any data, with cross-family re-review pending. Common findings apply through README; this table records row-specific changes.

| Finding | Applied/rejected | One-line note |
|---|---|---|
| M1 | Applied | Selected-ancestry cap-based pricing, qualification and per-branch budget in README; replace timing allowances before send. |
| M2 | Applied | README admitted caps/draws rule; source config verified, actual returned binding still required. |
| M3/M4 | Applied / preserved | Prerequisite timing corrected where applicable; dispatch/nomination precedence pinned at 214dbe6f and 7efbb6cb (timing-only clarification); v2 thresholds and round-3 decisions frozen in README before any return. |
| MINOR: controls | Applied decision | Separate fresh controls retained for independent release/recovery; costs and reason in README. |
| MINOR: house style | Applied | Review disposition and prospective Builder deltas tables added; no implementation or launch. |
| M6 | Applied | λ_F matches median initial regulariser-gradient ratio on Δ₀ to one; calibration training inputs only; no outcome data. |
| MINOR: estimator pricing | Applied rule; measurement PENDING | Withdraw 24–50; provisional 2–8 is replaced by measured eight-rank per-example preflight formula before send. No receipt fabricated. |
| MINOR: citation | Checked | Scholz primary abstract verified; EWC motivates the proxy, not its exact local calibration. |

### Round-3 disposition (manager-fixed before data)

Shared rules apply through the [README](README.md); allocation and cross-family re-review remain pending.

| Finding | Disposition | Fixed action |
|---|---|---|
| r3-m1 | Applied | Pin 214dbe6f/7efbb6cb and timing-only precedence clarification; freeze v2 thresholds and all round-3 decisions before returns. |
| r3-m2 | Applied | Staggered reads allowed; later frozen specifications cannot change after earlier reads. |
| r3-m3 | Applied | Fisher raw L2 gradient matching does not equalise AdamW per-coordinate update strength. |
| r3-m5 | Applied; owner approval PENDING | Explicit per-branch allocation decision versus rough 290–420 GPU-h; scope approval is not allocation approval. |
| r3-m7 | Applied | Decimal half-up component rounding, summed row/branch totals and ceiled owner allowances; iSDFT and GRPO discrepancies corrected. |

## Builder deltas

Prospective handoff only; owner approval and cross-family review still precede any build.

| Area | Required delta |
|---|---|
| Binding | Wait for v6b P2; mark K/CR benchmark-only, remove incompatible P2 and calibrate newly on the chosen B. |
| Estimator/calibration | Implement shared per-example squared-gradient artifact, tied-parameter audit and Δ₀ norm matching; seal both λ values without model mutation or outcome input. |
| Qualification/cost | Check analytic gradients and median ratio; measure 32-row domain preflight plus three calibrations and replace provisional E in row/package budgets. |
