# v2b — context-distillation consolidation — DRAFT, 2 November 2026

**DRAFT — owner approval and an Opus cross-family review before any build. No launch or spending authority.** Goal 2 / P4; replan §3 row 4. The [shared registration](README.md) supplies statistical, provenance, execution and reporting rules. Exactly paired seeds **101–103**, never more than three.

## Question and one change

Does replacing D's single-trace teacher context with retrieved k=3 verified memory cases improve the student's **memory-off** performance on later development filings? Compare against a fresh implementation of **4A arm-5 W=D on the cumulative ledger**, not January–June checkpoints or a newly selected K/CR recipe. W remains the bound provisional D. Only the teacher's privileged context changes; the student never sees retrieved cases during rollouts, training inputs or primary evaluation.

[OEL, 2603.16856](https://arxiv.org/abs/2603.16856) motivates learning from experience; [PMD, 2607.01480](https://arxiv.org/abs/2607.01480) motivates procedural-memory distillation; [Cortex, 2610.20399](https://arxiv.org/abs/2610.20399) motivates testing consolidation after context removal. This is our component experiment, not a replication of their complete systems or an extraction/rewrite ablation.

## Arms, chronology and dose

Both training arms start once from pinned Qwen3-8B `b968826d9c46dd6066d109eabc6255188de91218`. **Control W:** D's EMA teacher receives the accepted target trace as its demonstration. **Challenger CD:** the same teacher receives the retrieved cases **instead of** that demonstration; never append the target's answer. Both teachers follow the same incoming-initialised EMA .99/.01. This is memory-conditioned D, not an extra 27B teacher.

Use July, August and September 2026 **development** batches in calendar order. Seal all arms' month-t arrival predictions before releasing that month's feedback. Build one common frozen-base-success ledger. After August (development boundary 2), train **one 40-update stage** over the released cumulative July–August ledger, then score September prospectively. No July stage and no terminal September stage: this is 4A arm 5's every-second-month boundary rule applied to a three-month pilot, not its six-month trial. July/August arrival scores are no-update checks. Six training stages total: two arms × three seeds. There is no downstream October prediction in this experiment.

Dose per arm/seed: effective batch 32, 1,280 ledger-row exposures, LR 1e−5, ten-update warmup including initial zero LR, fresh AdamW/scheduler, clipping and weight decay exactly W's recipe. D task forward-KL, importance correction, three-token skip and sequence reduction stay unchanged. One on-policy student completion per scheduled row; same ordered prompt IDs per seed in both arms. Export at 0/10/20/30/40. Use seed-shuffled cycling if the ledger is smaller than 1,280; publish unique traces and repetitions. No replay addition, information budget, parameter freeze or changed collector.

Frozen base without memory and frozen base + static k=3 memory are **scoring-only references**, re-scored in the same assigned lanes as the paired arms. They do not supply a second admitting comparison. Score both trained arms with memory on as secondary deployment readouts, using identical retrieved cases in both arms. Primary inference is memory off for both.

## Data and retrieval

Bind **data/p4_sec/slice-v4-scale**, the 193-question July–September development slice in the SEC notes’ scale readout, its complete manifest, filing/issuer components, gold audit and exclusion hashes before generation. Use the two registered SEC families, `fcf_margin_change` and `cash_conversion_change`; no quota expansion or new task family. Reuse the v4 development selection/holdout rules as frozen; do not import 4A's rule excluding July–September itself. **January–June belongs to 4A; no rows, ledger entries or exports from it. October–November remains untouched.** Do not browse reserved outcomes for exclusion checks; use custodian fingerprints.

Permanent issuer-component holdouts never enter the collector training ledger, memory, teacher context or any selection. Admit only frozen-base responses correct under both historical and explicit-unit scorers, complete within 2,048 tokens, ≤2,500 trimmed characters, passing source/unit checks, with training permission. Keep raw unedited responses, dates, source/checkpoint/verifier hashes and rejection reasons. No gold rationale or teacher solution substitutes for a missing success. Audit candidate supersets against task/general/PD/SP panels using the README duplicate rules before inference.

**r3-m6 — projected supply, published before send.** The [scale manifest](../../../data/p4_sec/slice-v4-scale/manifest.json) has **40 July +56 August =96 trainer candidates** after permanent holdout exclusions, before the success/length/source filter. Using the existing [scale readout’s explicit-unit base success rate](../../phase4/4a0-sec-slice-v4-notes.md) of **83/193 ≈43%** gives **96×0.43=41.28, about 41 unique eligible traces**, and **1,280/41≈31.2 passes per arm/seed** over that common ledger. This is a planning projection, not a measured accepted count: dual-scorer, EOS/length and provenance filters may reduce it further, and the candidate subset may have a different success rate. Publish the actual unique count, per-ID exposure multiplicities and retrieval support before training; do not add data, replace rejected rows or expand the dose after seeing a thin ledger. Zero/no-context support retains the existing PENDING rule.

For each ledger training query, retrieve from the **same cumulative released July–August ledger**, excluding the query itself, its issuer component, accession and required filing groups. Rank by `2×same-family + Jaccard(operand-label words) + .25×same-form`, tie by lower ID, as 4A static memory. Supply source rows, question and original response for at most k=3 cases. Fewer eligible cases means fewer cases, including zero; do not refill from forbidden months or another issuer in the same component. Freeze IDs, ranking and rendered contexts before training. Training retrieval at the August boundary can use all released August feedback; arrival retrieval can use only earlier-month feedback. This distinction is recorded per query.

Use common student prompt cap 10,240, response cap 2,048 including EOS, and common teacher reprompt cap **14,336**, plus 2,048 response/prefix allowance (16,384 padded EMA width), in both arms. This is the long-cap C0-like price, including retrieved cases; if the current filing or control demonstration cannot fit, preparation is PENDING rather than silently increasing the cap. CPU-render/tokenize the actual worst cases before build binding. On retrieval overflow drop the lowest-ranked **whole case**, never truncate the current filing, response prefix or demonstration. Log effective k and drop counts. A zero-sized ledger leaves the experiment PENDING; absence of retrieved context on every row leaves the intervention uninformative/PENDING rather than a negative distillation result. Do not silently skip overlength training rows or change schedules.

## Panels, endpoint and bars

**Primary:** per-seed September permanent unseen-issuer holdout explicit-unit accuracy, CD(memory off) − W(memory off), in points. Fixed whole-month denominator; missing September holdouts means INCOMPLETE, no pooling with training issuers or replacement by retrospective September-after-training scores. Explicit-unit scoring requires ±0.05 in requested percent/percentage points, correct suffix and no ×100 rescue; historical accuracy is descriptive.

Require mean ≥2, positive 3/3 and paired two-sided 95% lower bound >0, `mean ± 4.302653 SD/sqrt(3)`. **All guards** require paired lower bound ≥−1: **September deployed memory-on explicit-unit accuracy on all 52 September questions**, CD − W with identical July–August-only retrieval; general-v1 instruction, maths and reference-scored knowledge; HellaSwag-100. General panels and holdouts are scored at incoming and step 40, plus retention diagnostics at step 20. **July/August retention and absolute-final checks are descriptive for this pilot**, separately per month/mode and pooled item-weighted over 45 holdouts; they cannot pass/fail the row. Retention uses each mode’s sealed arrival reference with arrival memory context fixed, and absolute final scores are also reported so weak acquisition cannot manufacture retention. References and both arms are re-scored together per seed/panel, including the frozen and memory references.

**M7 denominator and sensitivity freeze.** The [SEC slice notes, scale readout](../../phase4/4a0-sec-slice-v4-notes.md) give totals 61/80/52; the linked [scale manifest](../../../data/p4_sec/slice-v4-scale/manifest.json) and `sec.issuer_holdout` in its three batches give the actual permanent unseen-issuer denominators below. **R2-M2 — manager option (a), fixed before any data:** the memory-on guard uses **all 52 September questions**; none enter the July–August training/memory ledger. Freeze September arrival contexts before releasing September feedback; no September case enters retrieval for this guard. The **13-item September memory-on permanent holdout is a descriptive stratum**, while memory-off on those 13 remains the primary. Within each endpoint CD and W use the **same IDs**; missing responses count as missing evidence, not a smaller denominator. These counts are per seed and are not multiplied by three or by two modes.

| Month / endpoint | Month total | Memory-off N | Memory-on N | One item, points | Role |
|---|---:|---:|---:|---:|---|
| July | 61 | 21 | 21 | 4.762 | Descriptive prior month |
| August | 80 | 24 | 24 | 4.167 | Descriptive prior month |
| September holdout | 52 | 13 | 13 | 7.692 | Memory-off primary; memory-on descriptive stratum |
| September all questions | 52 | — | 52 (all-question N) | 1.923 | Memory-on guard only |
| July+August | 141 | 45 | 45 | 2.222 | Descriptive pooled prior months |

At paired sample SD s, the observed mean must be ≥2 and **>2.48414s**, with all three seed differences positive. This is a detection threshold, not an 80%-power calculation. For the September **13-item memory-off primary**, the smallest attainable all-positive benefit is **one net item in every seed =7.692 points**; one item in only one seed fails 3/3 and the interval. If one-item variation gives differences (2,2,1) items, s=4.441 points and the mean 12.821 has lower bound 1.788; the same SD requires mean >11.032. An 80%-power paired two-sided t-test planning MDE is **3.26404σ** under independent normal seed differences (noncentral t, df=2); at σ=4.441 it is **14.496 points**, before the extra 3/3 and ≥2 requirements. This normal approximation is illustrative for these discrete counts, not guaranteed power. No matched three-seed historical σ is available: actual power/MDE is **UNMEASURED**, not inferred from the single-seed SEC readout. The all-52 September memory-on guard has 1.923-point item granularity and keeps its strict two-sided −1 bar: at zero mean it still needs SD≤0.403 and may remain INCONCLUSIVE. The larger denominator reduces granularity; it does not guarantee power. Its 13-item holdout stratum is descriptive only. No later demotion, denominator change or memory-off primary pooling is permitted.

Report original 4A +5-point/3-point package bars only as context: **they do not become v2b bars and this pilot cannot pass 4A**. Where inherited absolute package summaries use one-sided bounds, use t=**2.919986**, df=2. Report general changes from original base and incoming, answer transitions, per-family results, arrival accuracy, effective-k strata and unique-trace supply. Strata and issuer bootstrap are descriptive, never extra seeds or alternative primaries.

Report share of memory gain retained `(trained memory-off − frozen)/(frozen+memory − frozen)` separately for each arm; zero denominator stays undefined and negative denominators are flagged. The distinct 4A recovery ratio is N/A here because this pilot has no monthly direct-W arm; do not relabel cumulative W as that arm. Record input/output tokens per correct and all-in collector/retrieval/training costs; output tokens/correct >1.5× frozen is “at cost”, not an exclusion. A serving advantage needs measured total cost, not memory-off accuracy alone.

## Execution, costs and failure modes

Three-seed controls are fresh, never reusing a 4A winner. Score the complete per-(seed,panel) frontier on one physical GPU/job, three concurrent ready commands where qualified, reference authority for knowledge/HellaSwag. CPU work has zero GPUs, generation/scoring one, training eight; no held GPU while waiting for feedback/export. Follow README's resume, utilisation, hardware and missing-evidence rules.

**Repriced M1, independent of K/CR/I0: 119.54–212.68 partner GPU-h** = six long-cap C0-like stages **92.94–153.36** + two qualifications **9.60–18.32** + ordinary/reference scoring **12–24** + scientific setup/export/transfer **3–9** + collector **2–8**. Caps are 12,288 actor and 16,384 EMA padded tokens, with retrieved cases charged inside the teacher width; all repeated ledger exposures and modes are included. Rendering is CPU/zero GPU; any larger teacher cap requires prospective repricing, not hidden context. Shared integration 20–40 is in README once. This pilot has its own fresh controls and does not spend 4A funding. Replace unmeasured collector/scorer allowances with qualified timings before send. Wall time UNMEASURED; flag >24 h per arm/seed or combined lane.

Before scientific execution test context separation (no case tokens in student inputs), retrieval exclusions, chronological feedback seal, unchanged D masks/reductions, finite gradients/EMA, common schedule hashes, zero/nonzero-LR updates and export/resume equality with production longest inputs. Timing is priced, not an efficacy gate. Thin ledger, wrong source gold, leakage, no retrieval support, missing mode references or hardware mismatch remain explicit PENDING/INCOMPLETE/UNVERIFIED; complete valid bars not met are INCONCLUSIVE, incumbent unchanged. Never add traces or stages after inspection.

## What this does NOT claim

A development win identifies a teacher-context benefit over W at this boundary. If 4A fails it nominates a repaired consolidation challenger; if 4A passes it nominates a next rung. It **does not** establish superiority to the product memory incumbent, 4A success, held-out confirmation, sustained self-improvement, a better collector, or universal non-forgetting. Those require their own comparisons, connected recheck and untouched October–November confirmation. Two families, one later month and three seeds cannot supply broad independent stream replication.

## Review disposition

Response to [Opus round 1](review-opus-r1.md) and [round 2](review-opus-r2.md); round-3 manager decisions are fixed before any data, with cross-family re-review pending. Common findings apply through README; this table records row-specific changes.

| Finding | Applied/rejected | One-line note |
|---|---|---|
| M1 | Applied | Selected-ancestry cap-based pricing, qualification and per-branch budget in README; replace timing allowances before send. |
| M2 | Applied | README admitted caps/draws rule; source config verified, actual returned binding still required. |
| M3/M4 | Applied / preserved | Prerequisite timing corrected where applicable; dispatch/nomination precedence pinned at 214dbe6f and 7efbb6cb (timing-only clarification); v2 thresholds and round-3 decisions frozen in README before any return. |
| MINOR: controls | Applied decision | Separate fresh controls retained for independent release/recovery; costs and reason in README. |
| MINOR: house style | Applied | Review disposition and prospective Builder deltas tables added; no implementation or launch. |
| M7 | Applied | 21/24/13 holdouts retained; July/August descriptive; September memory-off primary unchanged, memory-on guard all 52 and 13-item stratum descriptive. |
| MINOR: citations | Checked | OEL, PMD and Cortex primary abstracts verified; each motivates consolidation, not this exact recipe. |

### Round-3 disposition (manager-fixed before data)

Shared rules apply through the [README](README.md); allocation and cross-family re-review remain pending.

| Finding | Disposition | Fixed action |
|---|---|---|
| R2-M2 | Applied, option (a) | September memory-on guard uses all 52; 13-item memory-on holdout descriptive; memory-off primary unchanged. |
| r3-m1 | Applied | Pin 214dbe6f/7efbb6cb and timing-only precedence clarification; freeze v2 thresholds and all round-3 decisions before returns. |
| r3-m2 | Applied | Staggered reads allowed; later frozen specifications cannot change after earlier reads. |
| r3-m5 | Applied; owner approval PENDING | Explicit per-branch allocation decision versus rough 290–420 GPU-h; scope approval is not allocation approval. |
| r3-m6 | Applied | 96 candidate traces ×43% ≈41 unique; about 31.2 passes over 1,280 exposures, published prospectively. |
| r3-m7 | Applied | Decimal half-up component rounding, summed row/branch totals and ceiled owner allowances; iSDFT and GRPO discrepancies corrected. |

## Builder deltas

Prospective handoff only; owner approval and cross-family review still precede any build.

| Area | Required delta |
|---|---|
| Binding/data | Bind scaled v4 manifest/batch hashes, 21/24/13 holdouts and all-52 September memory-on guard IDs; publish projected ~41 and actual unique ledger supply; seal chronology/retrieval limits. |
| Runner | Swap teacher context only; enforce 14,336 reprompt cap, identical student inputs/schedules and exclusions. |
| Reader/cost | Make July/August checks descriptive including pooled 45-item diagnostics; keep 13-item September memory-off primary and all-52 memory-on guard (13-item memory-on stratum descriptive); print item granularity/MDE and long-cap costs. |
