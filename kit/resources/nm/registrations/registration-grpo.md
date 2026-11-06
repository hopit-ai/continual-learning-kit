# GRPO FinQA-stage-1 qualification — DRAFT, 2 November 2026

**DRAFT — owner approval and an Opus cross-family review before any build. No run or spending authority.** P3 RL column; replan §3 row 9. Inherit the [shared registration](README.md), evidence statuses and right-sizing. Paired seeds **101–103** only. **GRPO is never an incumbent candidate**, regardless of scores.

## Question and scope

Can a mechanically qualified verifier-reward GRPO baseline learn FinQA at matched dose and rollout budget well enough to serve as P3's RL column? [Retaining by Doing, 2510.18874](https://arxiv.org/abs/2510.18874) motivates on-policy data as a factor in forgetting; [RL's Razor, 2509.04259](https://arxiv.org/abs/2509.04259) motivates studying the RL/retention relation. Neither supplies a causal explanation for our eventual comparison without matching data, exposure and acquisition.

The single tested change is **the new-task learning objective**, D distillation versus verifier-reward GRPO. This is a qualification experiment, not a rung against the product/task incumbent. Its fresh matched D control is explicitly a **grouped-dose qualification control**, not historical standard-D performance. No replay, K, CR, P2, information budget, recovery or layer freeze in either arm. Those combinations require later P3 registration.

## Starts, grouped dose and controls

Both arms start from the original pinned Qwen3-8B `b968826d9c46dd6066d109eabc6255188de91218` once, independently per seed. FinQA stage 1 only; no ContractNLI, MATH or SEC continuation. Fresh D control and GRPO each run **40 optimizer updates**, same LR 1e−5, ten-update warmup including initial zero LR, AdamW parameters/clipping/weight decay and full-weight precision as D. Export 0/10/20/30/40. No outcome-dependent early stopping. The LR 1e−5 with β_KL=0 is a deliberately accepted qualification-dose risk; failure supports only “qualification not established at this dose”, not failure of GRPO generally.

Use **8 FinQA prompts ×4 independent completions =32 trajectories/update**, for **320 prompt slots and 1,280 training trajectories per arm/seed**, one optimizer update on each fresh group. This matches trajectory dose and rollout budget across D/GRPO while providing nondegenerate within-prompt groups; it differs from standard D's 32 prompts×1 completion. Do not claim matched unique-prompt exposure to historical standard D. Both arms see the same seed-specific ordered 320 unique training prompt IDs, one pass; response samples diverge naturally as policies diverge. Four completions are trained in **both** arms; D must not train only one after generating four.

Use prompt cap **2,048**, response 2,048, D reprompt 6,144 (actor width 4,096; EMA width 8,192), with a fixed eligible training-prompt superset and no runtime draw filtering. Overflow leaves qualification PENDING. Common sampler: thinking off, temperature .7, top-p 1, max 2,048 tokens including EOS, identical prompt/tokenizer and RNG derivation by seed/update/prompt/sample. Freeze the schedule before sampling. One generation attempt per completion: no rejection-resampling, best-of-four, extra all-wrong groups, dynamic question filtering or uncharged warmup rollouts. Charge the zero-LR update's 32 samples too. D's EMA follows incoming .99/.01 with the usual verified 27B demonstration; GRPO requires no distillation teacher. The teacher-forward difference is part of the objective cost, not hidden compute parity.

## Reward and exact objective

For every generated FinQA completion, frozen numeric verifier reward r is **1** for correct, complete, correctly encoded answers passing the existing FinQA checker and **0** otherwise. Gold and program results feed only the verifier, never the policy prompt. No length bonus, learned judge, partial credit or answer-format reward beyond the fixed validity check. Raw malformed/truncated outputs and their zero rewards remain in the group; do not drop them. Evaluate rewards on D's completions too for diagnostics, without changing D's loss or filtering its data.

For each four-sample group use `A_j=(r_j−mean(r))/(sqrt(mean((r−mean(r))²))+1e−6)`, detached. All-equal reward groups have A=0 and remain in the 32-trajectory denominator. Use a fresh-rollout frozen old policy equal to the current policy at generation; store its token log-probabilities. `rho_jt=exp(log p_theta−log p_old)` and minimise

`L_GRPO = −(1/32) Σ_j mean_{t in response_j} min(rho_jt A_j, clip(rho_jt,.8,1.2) A_j)`.

Mask prompt/padding only; include EOS when present, include generated tokens up to the cap for truncated responses. No D three-token skip, D importance term, separate entropy bonus, auxiliary CE, or base-KL penalty. **β_KL=0**, fixed, so this qualification does not bundle another base anchor. One pass/update, no PPO epochs or minibatch reuse; clipping is still implemented/tested even though fresh one-pass ratios initially equal one. Old policy is frozen per update, never the original base or EMA. Finite zero-gradient updates are allowed for homogeneous-reward groups; scheduler/update count is not extended to compensate.

D uses all four own-policy prefixes with unchanged D demonstration KL, masks, correction and sequence reduction; global denominator 32. This protocol is a proposed common P3 workload, not a claim that it exactly replicates either cited paper or the ungrouped incumbent. Freeze optimizer and sampler manifests, loss algebra and numerical conventions before build.

## Data and mechanical qualification

Use the frozen FinQA **training** split and verified 27B demonstrations already available to D. Hash eligible IDs by `grpo-qual-20261102|seed|id`, take first 320; fewer than 320 leaves PENDING. Demonstrations exist for every D prompt and are hidden from GRPO policy inputs; their generation expense belongs to D and is accounted if not already paid. Exclude FinQA-1,147 and every general/HellaSwag/legal/PD/SP/monitor/probe item and candidate superset under shared duplicate rules. Freeze licences, source revisions, gold/verifier, prompts, raw targets and schedule hashes. No panel-guided prompt selection or reward tuning. No SEC rows or reserved-month access.

Require unit/reference fixtures for correct/incorrect/malformed/truncated numeric outputs, reward isolation from prompts, exact grouped normalization including all-zero/all-one rewards, detached advantages/old log-probs, clipping gradient signs, EOS masks, no accidental KL/CE and reduction equivalence at world sizes 1/2/8 (relative error ≤1e−4). Test known mixed-reward groups so nonzero policy gradients are exercised independent of sampled success. Run both real production paths through zero and nonzero LR with finite optimizer states, unchanged batch/trajectory count, exact scheduler accounting, hashes, export/reload and native resume/RNG equality. Materialise optimizer state and longest-input rollouts. No extra trajectories may be used to make a smoke appear to learn; smoke fixtures establish mechanics, not efficacy. **Record non-gating telemetry in mechanical smoke and every scientific update:** full-vocabulary forward KL from frozen base to current policy on the same sampled response prefixes (prompt/padding excluded, EOS included; no gradient), mean verifier reward and mixed/zero-reward group frequencies versus update. Keep this reference pass out of the objective, charge its compute, and report the trend without using it to choose LR, add KL, stop early or increase dose.

## Qualification endpoints and bars

Decision-bearing panels: FinQA-1,147 at base and 0/10/20/30/40; general-v1 instruction/maths/knowledge and HellaSwag-100 at base/final, with step-20 retention diagnostic. No legal test qualification is inferred from a FinQA run. Knowledge/HellaSwag use full-precision **REFERENCE-SCORED** authority; generated panels keep frozen greedy thinking-off scoring and cap 2,048 (training sampling does not change evaluation). All curve/reference/arm inputs of each seed/panel are scored in one complete GPU lane.

**Primary qualification:** FinQA final acquisition is within D's band: paired GRPO−D **two-sided 95% lower bound ≥−1 point**, and GRPO's own FinQA gain from base has **one-sided 95% lower bound >0 and is positive 3/3**. D must also show positive FinQA gain in 3/3 with one-sided lower bound >0; otherwise the matched dose is insensitive and the qualification is PENDING, not a GRPO failure. Two-sided t=**4.302653**; one-sided t=**2.919986**, df=2. FinQA gain-AUC at 0/10/20/30/40 is diagnostic; do not select a better step or alter dose to match acquisition after the fact.

**R2-M3 — manager decision fixed before any data:** each retention guard (general instruction, maths, knowledge, HellaSwag) requires the paired GRPO−D **one-sided 95% lower bound ≥−3 points**, `mean(d) − 2.919986 SD(d)/sqrt(3)`. This adopts the inherited −3 package retention margin for qualification; report absolute changes versus base separately without replacing this paired comparison. FinQA non-inferiority remains **−1 with the two-sided t=4.302653**; GRPO’s own acquisition (and D sensitivity check) remains **one-sided >0 and positive 3/3**. This is a **non-inferiority qualification**, not an incumbent admission; all margins are fixed prospectively. Report equal-weight instruction/maths/knowledge retention descriptively, without a second benefit verdict. Any later affirmative improvement experiment must retain the common ≥2-point, positive paired interval, positive 3/3 and ≤1-point guard rule under its own registration. A qualification pass never opens incumbent admission. Report every panel even when one improves and another worsens.

Publish all-wrong/mixed/all-correct group frequencies, zero-gradient fraction, verifier acceptance, actual unique prompts/trajectories, reward curves, policy drift, generated token counts, acquired/lost answers, training/serving cost and >1.5× base serving as “at cost”. Training reward is not held-out accuracy. Too few mixed groups may explain a null, but never authorize more rollouts or a reward redesign within this registration.

### Historical D-versus-D SD audit (R2-M3)

A matched historical estimate would be `SD_s(score(D_repeat,s) − score(D_control,s))`, sample SD across seeds **101–103**, in percentage points, for the same task stage/dose/panel/scorer. The inspected repository evidence does **not** supply two independent matched D runs per seed; the requested paired SDs cannot be computed from it.

| Panel | Historical paired D-versus-D SD | Evidence / missing pairing |
|---|---|---|
| FinQA-1,147 | **UNMEASURED** | v4 has one D step-40 run per seed; Package B repeats its 877/878/859 scores, not independent D training. |
| General instruction / IFEval | **UNMEASURED** | Same single v4 D series (74/76/77); no independent matched D replicate. |
| General maths | **UNMEASURED** | Same single v4 D series (87/87/87); zero spread here is not a zero D-versus-D SD. |
| General knowledge | **UNMEASURED** | Same single v4 D series (78/75/75); no matched repeat under current reference authority. |
| HellaSwag-100 | **UNMEASURED** | v4 omits this panel; return-1 has no core-D final HellaSwag scores and format-confounded probe/base scores, not a reference-scored D repeat pair. |

Sources inspected: [v4 raw scores.csv.gz](../../phase2/evidence/v4-finqa-return/scores.csv.gz) (filter `arm=D`, `step=40`, `cap=2048`, seeds s101–s103; the 8192-cap rows score the same checkpoints and are not training replicates); [v4 result/provenance](../../phase2/v4-finqa-results-20261014.md); [Package B tier-1 reader, summaries](../evidence/p3b-tier1-return1/reading/kit-reading.json); and [return-1 evidence/provenance](../return1-reading-20261028.md), with [per-seed rows](../evidence/return1/per-seed.json) and [three-seed decisions](../return1-three-seed-decisions-20261029.json). D−base, D−F, D+replay−D, or D before/after another task changes the contrast; reusing a checkpoint twice would manufacture a zero. None is labelled a paired D-versus-D estimate. No historical grouped-dose D control is available either.

**Manager fallback, fixed before data:** keep the bars above despite unavailable matched SDs and explicitly plan for an outcome that is **likely INCONCLUSIVE**. This is a planning caution, not measured power or a claim that GRPO is ineffective. At true parity the observed SD must be ≤`sqrt(3)/4.302653=0.403` for FinQA and ≤`3 sqrt(3)/2.919986=1.780` for each retention guard. Four retention guards plus the strict FinQA guard may still fail jointly; no later margin relaxation, extra seeds or dose extension is permitted. A failed valid qualification means “qualification not established at this dose”, not evidence of inferiority.

## Budget, outcomes and limits

**Repriced M1, independent of K/CR/I0: 52.74–89.48 partner GPU-h**. At this row’s short caps, README re-evaluates K/CR’s calibrated phase model: D **7.7300–10.8220/stage**, GRPO **3.3130–4.7583/stage**, the latter dropping EMA work and adding the telemetry-only base forward. Three stages per arm give **33.13–46.74 training**, **4.61–9.74 qualification**, **12–24 scoring/reference** and **3–9 setup/export/transfer**, rounded by README’s single decimal half-up component rule. All **7,680 trajectories**, including zero-LR and zero-advantage samples, are charged; max output count 15,728,640. This is a cap-based forecast, not a measured teacher-cost saving. Missing D demonstrations require a separate generation price and PENDING status, never uncharged work. Shared integration 20–40 appears once in README. CPU zero GPUs; training eight; separate generation/scoring one; no concurrency discount. No paid execution authorized. Wall time UNMEASURED; flag >24 h per arm/seed or combined lane.

Mechanical failure, missing qualified D control or missing evidence: PENDING/INCOMPLETE; invalid identities/provenance: UNVERIFIED; mixed GPU scores: UNVERIFIED-HARDWARE. Complete valid evidence missing qualification bars: INCONCLUSIVE, qualification not established at this dose; this is not a demonstration of GRPO inferiority, especially with the unmeasured historical SDs. No fourth seed, extra reward attempt, silent algorithm switch or incumbent change. A qualified GRPO becomes **eligible only as P3's RL column**; subsequent replay/teacher-source comparisons must bind a common workload and preserve all relevant controls.

This does **NOT** establish RL as a product method, a superior continual-learning incumbent, a causal proof that on-policy data alone prevents forgetting, matched compute across objectives, or retention after ContractNLI/MATH. Both D and GRPO are on-policy here; the objectives differ. It does not validate GRPO with replay, base KL, arbitrary rewards or a longer horizon. The product/task incumbent stays unchanged even if every GRPO number is better.

## Review disposition

Response to [Opus round 1](review-opus-r1.md) and [round 2](review-opus-r2.md); round-3 manager decisions are fixed before any data, with cross-family re-review pending. Common findings apply through README; this table records row-specific changes.

| Finding | Applied/rejected | One-line note |
|---|---|---|
| M1 | Applied | Selected-ancestry cap-based pricing, qualification and per-branch budget in README; replace timing allowances before send. |
| M2 | Applied | README admitted caps/draws rule; source config verified, actual returned binding still required. |
| M3/M4 | Applied / preserved | Prerequisite timing corrected where applicable; dispatch/nomination precedence pinned at 214dbe6f and 7efbb6cb (timing-only clarification); v2 thresholds and round-3 decisions frozen in README before any return. |
| MINOR: controls | Applied decision | Separate fresh controls retained for independent release/recovery; costs and reason in README. |
| MINOR: house style | Applied | Review disposition and prospective Builder deltas tables added; no implementation or launch. |
| MINOR: GRPO telemetry | Applied | Accepted LR/no-KL qualification-dose risk; per-update base KL and reward trend recorded, priced and non-gating. |

### Round-3 disposition (manager-fixed before data)

Shared rules apply through the [README](README.md); allocation and cross-family re-review remain pending.

| Finding | Disposition | Fixed action |
|---|---|---|
| R2-M3 | Applied; SDs UNMEASURED | Repo audit cannot identify independent matched D repeats; per-panel evidence cited, likely INCONCLUSIVE explicit; retention −3 one-sided, FinQA −1 two-sided, acquisition >0 and 3/3. |
| r3-m1 | Applied | Pin 214dbe6f/7efbb6cb and timing-only precedence clarification; freeze v2 thresholds and all round-3 decisions before returns. |
| r3-m2 | Applied | Staggered reads allowed; later frozen specifications cannot change after earlier reads. |
| r3-m5 | Applied; owner approval PENDING | Explicit per-branch allocation decision versus rough 290–420 GPU-h; scope approval is not allocation approval. |
| r3-m7 | Applied | Decimal half-up component rounding, summed row/branch totals and ceiled owner allowances; iSDFT and GRPO discrepancies corrected. |

## Builder deltas

Prospective handoff only; owner approval and cross-family review still precede any build.

| Area | Required delta |
|---|---|
| Binding/objective | Freeze 320 prompts, four samples each, short caps and one-pass verifier reward objective; no KL penalty or extra samples. |
| Telemetry | Add frozen-base KL on actual sampled prefixes and per-update reward trend in smoke/scientific runs; no gradient or outcome gate. |
| Qualification/cost | Verify grouped masks/reductions/resume; apply fixed −3 one-sided retention and −1 two-sided FinQA bars, retain >0/3-of-3 acquisition and likely-INCONCLUSIVE interpretation; charge telemetry/reference passes. |
