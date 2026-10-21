# 4A registration: consolidation versus SEC memory, January–June 2026

20 October 2026 · Registered under plan v6 decision 2 (owner: "I am good with all 4"): "Register 4A before tier 2 returns, binding W at send." Replaces the [18 October draft](4a-registration-draft-20261018.md), with [plan v6](../programme-plan-v6-20261020.md)'s amendments and its [reviews](../plan-v6-evidence/). It authorises no build, run, dispatch or spend. **No January–June 2026 model output exists.**

## 1. Status: what is frozen and what is bound later

| Label | Items |
|---|---|
| **FROZEN NOW** | Question, stream rules, ledger and admission, arms and the v1 inclusion rule, dose, seeds, scoring protocol, primary outcome, bars, reporting additions, cost formula, what changes the plan. |
| **FROZEN BEFORE ANY JANUARY–JUNE OUTPUT** (values recorded under rules frozen now) | Candidate count N; stream manifest and batch hashes; gold audit; common input cap; v1's July–September admission decision, which fixes the memory incumbent. |
| **BOUND AT SEND** | **W** (recipe and B export identities); **timing** (message 1 if ready, else message 2; never holding message 1); **budget** (ceiling from measured W timing, §11); kit and software hashes; node and scoring-GPU identity. |

Order: v1 reads out on July–September, the recorded values above are completed, then v0 runs. After v0 reads out, arms, dose and bars are not amended. v0 on Modal is a preview only and never enters a 4A comparison.

## 2. Question

Does consolidating verified successes into Qwen3-8B's weights, on top of memory, beat memory alone on later months, while keeping earlier-month and general skills? This is plan v6's rung **v2**, tested against the memory incumbent, never the bare base.

## 3. Stream — FROZEN NOW

- **Months:** January–June 2026, six consecutive complete months, all after Qwen3-8B's release (April 2025). July–September stays development data; October–November 2026 confirms later.
- **Build rule:** v4 builder; families `fcf_margin_change` and `cash_conversion_change`; **quota 40 per family per month**; at most two questions per latest filing; seed `4a0-slice-v1`; **`--amendment-rule restating`**: only later 10-K/A and 10-Q/A count, and one disqualifies a question only if it restates the operand concepts for the report or comparative period, or says "restate/restatement/restated" in a financial-statement or note section. Unreadable amendments reject conservatively; Part III-only amendments stay eligible.
- **Candidate count:** **up to 2,400 hash-ordered candidate filings per month, or the largest uniform count the cache allows, never below 600**, fixed and recorded in the manifest before any model output. At 600 the 20 October re-scan found 37, 53, 38, 59, 51 and 43 questions (281); cash conversion was near quota, FCF margin 2–19 a month, and January and June exhausted their candidates. A build at 600 must reproduce those counts, or the build stops for repair.
- **Shortfalls are declared, never back-filled** or replaced.
- **Exclusions:** all July–September probe filings, issuer groups, required join filings and co-registrants (`slice-v4-cash`, `slice-v4-scale`) and the v3 slice; no filing group reused across months; B's exact and near-duplicate rules against task panels and general-v1.
- **Permanent holdouts:** `4a0-sec-issuer-holdout-v3:` CIK hash modulo 4 = 0, over whole co-registrant components; new links reviewed before freezing. Holdouts never enter memory, notes, training, teacher requests or selection.
- **Arrival and order:** a question arrives on its latest required public filing date. Months in calendar order; within a month by acceptance time, accession, item ID.
- **Feedback timing:** every arm's whole-month predictions are sealed (hashes recorded) before that month's feedback (ledger admission, memory, note revision, training) is released. An update after month t first helps month t+1.
- **Gold audit (before any output):** every gold checked against the original filing: displayed-cell or XBRL binding, periods, signs, independent arithmetic, accounting basis and the recorded non-restating amendment IDs. Failures are removed and declared; the audit hash goes into the manifest.
- **Input cap:** fixed after a CPU renderer check, within W's training limits.

**Command.** The builder writes January–June only under `--feasibility`, which labels output feasibility-only. One bounded extension adds a primary label for these months, changing no selection, rejection, holdout or serialization rule; it gets a cross-family review and its hash is recorded. Then:

```sh
python scripts/p4_sec_slice.py build --version 4 --feasibility \
  --months 2026-01,2026-02,2026-03,2026-04,2026-05,2026-06 \
  --families fcf_margin_change,cash_conversion_change --per-family 40 \
  --max-candidates <N> --allow-shortfall --amendment-rule restating \
  --out data/p4_sec/slice-4a-jan-jun --cache '<SEC_CACHE_PATH>'
```

## 4. Trace ledger and collector — FROZEN NOW

One shared ledger; the collector is the **frozen base 8B (arm 1)**, whose sealed predictions are reused. Updated arms never add successes. No teacher (27B) solutions; a teacher condition needs its own registration. A response is admitted only if:
- it is correct under both the historical and the explicit-unit scorers;
- it is complete and unedited (finished within 2,048 output tokens) and passes source and unit checks;
- its trimmed response is at most 2,500 characters;
- it is not a holdout, and its item allows training.

Records carry source, date, hashes, collector checkpoint, verifier version and provenance. Gold rationales never replace responses.

## 5. Arms — FROZEN NOW

All start from base `b968826d9c46dd6066d109eabc6255188de91218`, with identical documents, prompts, decoding and scoring.

| Arm | What happens | Updates per seed |
|---|---|---:|
| 1. Frozen | No memory, no updates; supplies the collector | 0 |
| 2. Frozen + static k = 3 memory | Fixed retrieval over earlier-month ledger traces. **Always kept as a control.** | 0 |
| 3. Frozen + v1 revisable memory | **Only if v1 is admitted on July–September** under plan v6's common rule against static k = 3 (≥ 2-point gain with a positive paired 95% interval; ≤ 1 point lost on the other endpoint, confirmed by its bound). Otherwise absent. | 0 |
| 4. Direct W | One 40-update stage after each month 1–6 on that month's new traces; no memory at inference | 240 |
| 5. **Memory + consolidation (PRIMARY)** | Incumbent memory at inference; after months 2, 4 and 6, **one 40-update stage over the cumulative ledger**; memory kept | 120 |

- **Memory incumbent:** v1 if admitted, otherwise static k = 3; fixed before any January–June output, never switched on its results.
- **Static k = 3:** rank by `2 × same family + Jaccard(operand-label words) + 0.25 × same form`, ties by lower ID; exclude the target's issuer component, accession and required filing groups; fewer cases if fewer qualify; show source rows, question and original response; on overflow drop whole lowest-ranked cases, never the current document.
- **Shared memory:** arms 2, 3 and 5 use identical case IDs and, for v1, identical note versions, all produced from the frozen collector's released feedback. Arm 5 never revises notes from its own outcomes.
- **Declared amendment** (plan v6, before any output): consolidation is 120 updates per seed against direct W's 240, replacing the design's two stages per boundary.
- **Thin months:** an empty month contributes no stage and none is made up. Stages cycle a seed-shuffled schedule; unique-trace and repetition counts are recorded.

## 6. W's recipe — selection FROZEN NOW, identity BOUND AT SEND

**Selection:** [package C §3 with its F+replay amendment](../phase3/package-c-registration-20261018.md), over D, F, D+replay and F+replay on B's completed, unreset step-40 results, ranked by: (1) confirmed joint acquisition and sequence retention; (2) number of confirmed retention margins, including both stages' general panels; (3) mean sequence MATH final; (4) lower maximum task serving-token ratio. Exact ties: F, D, D+replay, F+replay. If all fail, W is still selected and labelled provisional. 4A outcomes and reset exports never select W.

**Recipe from package B:** full weights; batch 32; 40 updates per stage; LR 1e-5 with ten-step warmup including a zero-LR first step; fresh optimiser and schedule per stage (and fresh incoming-student EMA for D); 2,048-token response cap; exact losses and masks. F learns the accepted responses; D conditions its teaching copy on them (a transfer of B's recipe to self-traces). Replay recipes keep positions 8/16/24/32 (28 task + 4 checked generic base responses), B's mixed loss and the amended D over-cap skip rule, with overlap rechecked against 4A holdouts. Dose: **1,280 task exposures per stage, 1,120 with replay**. No LoRA substitution.

## 7. Seeds — FROZEN NOW

**101–105**, matched across arms 4 and 5. Deterministic arms 1–3 run once.

## 8. Scoring — FROZEN NOW

**Machine:** every 4A comparison is scored on **one physical GPU on the partner's node**: pinned scorer, greedy, thinking off, one attempt, 2,048 output tokens, the common input cap. Arms 1–3 are re-run there; cross-machine noise (about ±3 per panel) is the size of the margins.

**Primary measure:** **explicit-unit accuracy on permanent unseen-issuer holdouts** (±0.05 in the requested percent or percentage points; conflicting suffix fails; no ×100 rescaling). The historical scorer is reported separately.

**When:**
- **On arrival:** each arm scores the whole month-t batch in its current state (weights and memory through month t−1); predictions are then sealed. Month 1 is the no-history check.
- **After every training stage:** all arrived holdouts, and each general-v1 panel (IFEval, knowledge, maths) separately; for arm 5, holdouts with memory on (deployed) and off. Earlier holdouts keep their arrival-time memory context, so retention isolates the weight change.
- **After month 6:** every saved checkpoint of arms 4 and 5, memory off, on all January–June holdouts (descriptive; no checkpoint selected).

**Commands** (per month, `--batch 2026-0m`): `memory --slice <slice> --runs <runs> --batch <m> --out <runs>/memory-<m> --k 3` (`<runs>` holds `frozen-<batch>/`); `eval generate --slice <slice> --batch <m> --model <checkpoint> --out <dir>`; `eval score --slice <slice> --batch <m> --responses <file> --out <dir>`; `readout --slice <slice> --runs <runs>`.

**Kit extensions before send:** the v1 memory arm, fixed-context and memory-off re-scoring, the self-trace stage driver, recurrence tagging and §10's readouts. Every row runs through the real runner with stand-ins, failure paths included, and passes review before send.

## 9. Primary outcome and bars — FROZEN NOW

**Primary outcome:** per seed, the unweighted mean across months 2–6 of (arm 5 − memory incumbent) explicit-unit holdout accuracy, in points.

**Improvement established** only if: **mean ≥ +5 points**; positive in ≥ 4/5 seeds; positive in ≥ 3/5 monthly seed-means; and a positive two-sided 95% paired seed-t interval (4 degrees of freedom).

**Retention**, after every stage: each earlier-month holdout loses ≤ 3 points against its first-scored reference; each general panel ≤ 3 points against both the incoming checkpoint and the original base; each confirmed by a paired one-sided 95% upper bound below 3. **Joint success requires every margin.** A crossing bound is inconclusive, and **inconclusive means not established**, never equivalence.

**Reported, not bars:** issuer-component bootstrap across dates; historical scorer; all-arrival and recurring-issuer scores; per family; direct W against memory and frozen (secondary, never substituted for the primary). Format-only gains do not count.

## 10. Reporting additions — FROZEN NOW (no new bars)

- **Recovery ratio**, per month: (direct W − frozen) ÷ (memory − frozen).
- **Share of memory's gain kept:** (arm 5 memory-off − frozen) ÷ (memory − frozen). A zero denominator stays undefined.
- **Recurrence split:** a holdout's recurrence count is the number of earlier months whose ledger holds a same-family trace with operand-label Jaccard ≥ 0.5; strata high (≥ 2) and low (0–1). The primary difference is reported per stratum (descriptive if under 10 questions). It decides only whether v6 runs: yes if the high stratum's gain exceeds the low's, both with ≥ 10 questions.
- **Paired answer transitions** (correct → wrong, wrong → correct) for every retention and arm comparison, plus answers acquired then lost.
- **Every checkpoint scored;** arms 4 and 5 also compared at equal cumulative updates (40, 80, 120) and equal training GPU-hours.
- **Shift logging:** student negative log-likelihood on each stage's schedule, before and after it. **Readout logging** at each saved checkpoint, as tested in the kit; diagnostic only.
- **Cost per arm:** tokens per correct (input + output; output only) and all-in cost (collector, failures, training, replay, retrieval, checkpoints). Over 1.5× frozen output tokens per correct is "at cost", never an exclusion. A cost advantage needs joint success and total cost no higher than memory's; otherwise "gain at extra cost" with a projected break-even volume.

## 11. Cost — formula FROZEN NOW, ceiling BOUND AT SEND

**Partner training:** 45 stages over five seeds (30 direct W, 15 consolidation). Measured 8×H100 timings:

| W | Per stage | 45 stages |
|---|---:|---:|
| F (about 20 min) | 2.7–3.3 GPU-h | 122–149 GPU-h |
| D+replay (about 50 min) | 6.3–6.7 GPU-h | 284–302 GPU-h |

D and F+replay are priced from their own measured timings at send; until then D uses D+replay's range and F+replay uses F's.

**Scoring:** at most about 31,000 predictions (up to 480 questions and 120 holdouts; 300 general-v1 items per scoring; arms 1–3 once). At the approximate 1.22 s per question on one H100, about 11 GPU-h; **reserve 60** for longer memory prompts, startup and reruns. **Startup, exports, storage and failures: 80** (about 0.7 TB for nine full-weight checkpoints per seed across five seeds).

**Ceiling** = 1.5 × (45 × measured W stage GPU-h) + 140, rounded up to the next 50: about **400 GPU-h if W is F**, **600 if D+replay**, at or below plan v6's 600–850. Spending stops at the approved ceiling.

**Our side:** EDGAR $0 plus gold-audit labour; local stand-in dry run $0; up to $10 Modal for a kit smoke test within the plan's remaining cap. v0's $5–10 is v0's.

## 12. What changes the plan — FROZEN NOW

- **Joint success:** consolidation becomes the incumbent (v2 admitted); v3's stream arm and cost validation follow; the recurrence split decides v6.
- **Not established, or memory wins:** incumbent unchanged, product memory-first, result recorded as learning. SEAL stays closed; any repaired challenger must beat memory.
- **Retention failure:** a separately registered prevention change (v3 or v4), not an amendment here.
- **Leakage, wrong identities, invalid gold or provenance, broken scoring:** stop, repair, file a dated prospective amendment.
- **Weak scores never stop completion.** No best-seed or best-checkpoint selection, silent amendment, or added dose, months or families after outputs.

## 13. Limits

- Two families, one chronology and thin January–March cannot show broad accounting competence or universal non-forgetting; five seeds cannot create independent issuers; pretraining exposure is unknown.
- A frozen collector does not test a closed loop (v7); selecting W is not evidence of its superiority; k = 3 or v1 is not every memory system.
- Six months is the registered workload, not a guarantee of power. A headline claim beyond this stream needs October–November confirmation.

## Amendments after cross-family review (Astra, 20 October; binding)

1. **Status.** This is a conditional registration, finalised after the July–September v1 decision and before v0. Frozen before any January–June output: the candidate count N; the audited manifests; v1's implementation and settings; admission endpoints and the interval method; the common input cap. Until W is bound, the cap must accommodate every candidate recipe.
2. **Arm 5 (memory + consolidation)** exactly matches the incumbent's retrieval and note versions. Static k = 3 stays a control. No separate k = 3 + consolidation arm.
3. **Retention references.**
   - Archive each holdout's rendered arrival context.
   - At arrival, score arm 5 both memory-on and memory-off; each later mode is compared with its own arrival reference.
   - Use paired one-sided 95% seed-t bounds across five seeds, keeping the registered ≤ 3-point margin (not "bound below 3").
4. **Recurrence tagging.**
   - Tag each question at arrival, using only previously released ledger months and the pinned label tokenizer.
   - Before v0, freeze the high-minus-low gain aggregation, month weighting, unique-question denominators and handling of missing strata.
   - v6's own routing rule and thresholds are frozen separately, before 4A reads out; the split only triggers v6.
   - Reused January–June evidence is exploratory.
5. **Executable handoff.**
   - Publish the reviewed primary-label build invocation.
   - Generate and score memory arms from the rendered memory directory, keeping prompts where the readout expects them.
   - The memory builder must enforce the input cap with whole-case dropping (the current code does not); this is a kit item before send.
6. **Stages and cost.** Skip a direct stage only for an empty monthly ledger, and consolidation only for an empty cumulative ledger. The cost formula gives about 350–400 GPU-h if W is F and about 600 if W is D+replay. Holdout and prediction counts come from the manifest, since hash partitioning does not guarantee exactly 25%.
7. **Accounting.**
   - Charge the collector once per deployable system.
   - Separate shared study spend from retention-scoring overhead.
   - Use six-month arrival-correct denominators; report all-in tokens and GPU-hours per correct, including note writing and revision.
   - Expose replay background information equally across arms.
8. **Amendment-index horizon (manager, 20 October).** The restating-amendment screen uses one fixed horizon, recorded in the manifest with the hashes of its index files, independent of the months requested. Default horizon: 2026-Q3, the last quarterly index complete at build time. The 20 October scan found a January-only build saw fewer amendments (39 questions) than January inside the full build (37). The builder change is in progress (`--amendment-horizon`); N and the horizon are frozen together before any January–June output.

## Build values frozen (manager, 20 October; before any January–June model output)

- **Candidate count N = 2,400** hash-ordered candidate filings per month, the registered maximum. The gzip-compressed cache (33.1 → 2.1 GB, verified; an offline rebuild of the 600-candidate scan was byte-identical) makes it feasible on disk. January and June will exhaust their candidates below N.
- **Amendment horizon = 2026-Q3**, the last complete quarterly index at build time, recorded in the manifest with the hashes of its index files.
- **Build:** `p4_sec_slice.py build --version 4 --purpose 4a-primary --months 2026-01..2026-06 --families fcf_margin_change,cash_conversion_change --per-family 40 --max-candidates 2400 --seed 4a0-slice-v1 --allow-shortfall --amendment-rule restating --amendment-horizon 2026-Q3`, with `SEC_CACHE_GZIP=1`.
- **v1 decided:** not admitted (see [result](v1-revisable-memory-result-20261020.md)). The memory arms are static k = 3 only.
- **Still to freeze before v0:** the gold audit and the common input cap.

## Slice built and gold-audited (manager, 21 October; before any January–June model output)

- **Built** with the frozen command (N = 2,400, horizon 2026-Q3, `--purpose 4a-primary`), finished 20 October at 23:02; exit 2 = declared shortfall.
- **Questions per month:** January 37, February 80, March 80, April 75, May 80, June 43, total 395. Both families reach their 40 quota in February, March and May. January and June exhaust their candidates. Manifest sha256 begins `ed5a8b17357e3e96`.
- **Gold audit:** all 395 answers independently recomputed from the displayed cells and checked against XBRL, 0 issues (`manager-gold-recompute.txt` in the slice). The 15 questions with less common capital-spending tags (capital improvements, other PP&E, oil and gas, real-estate development, productive assets) were checked by hand; each line is labelled "Capital expenditures" or "Purchase(s) of property and equipment", matching the line the question names. All kept.
- **Still to freeze before v0:** the common input cap. Memory prompts are limited to 10,240 input tokens (with 2,048 output, 12,288 in all), dropping the lowest-ranked whole retrieved cases on overflow and never the current documents. The static memory builder must enforce this before v0 runs.

## Registration final (manager, 21 October, before any January–June model output)

All items the registration required before v0 are now frozen:
- **N** = 2,400 and the **amendment horizon** = 2026-Q3;
- the **audited manifest** (395 questions, sha256 begins `ed5a8b17357e3e96`; gold recompute 0 issues);
- **v1's decision** (not admitted; memory arms = static k = 3 only);
- the **admission endpoints** and **interval method** (as registered, including Astra's amendments);
- the **common input cap**: 10,240 input tokens with whole-case dropping, enforced by `p4_sec_slice.py memory --input-cap` (fix/memory-input-cap). The stand-in maximum is 5,568 tokens, so nothing is dropped.

**Bound at send:** W, timing, budget, kit and software hashes, node and GPU identity.

**From here:**
- v0 (frozen + static memory on January–June, on Modal) runs as a preview. Only the arms re-run on the partner's node bear 4A's comparison.
- After v0 reads out, 4A's arms, dose and bars are not amended.

## Deviation recorded (manager, 21 October, after v0)

- **Astra's amendment 4 was not met before v0.** It required freezing the recurrence split's aggregation (high-minus-low gain, month weighting, unique-question denominators and missing-strata handling) before v0. The manager ran v0 without writing that receipt; the kit builder found the gap.
- **Consequence:** the recurrence split in 4A is **exploratory**, and its result cannot trigger rung v6 (routing) confirmatively. Any v6 decision needs the split frozen first and then confirmed on untouched, source-disjoint months (October–November 2026), which plan v6 already requires for confirmation.
- **Unchanged:** every other 4A rule, arm, dose and bar. No v0 output was used to choose the split's definition, but the order requirement was missed, so it is disclosed.

## Approved budget (owner, 21 October: "Approved with 600 partner GPU-hours. If it is tight, make it more")

- **Approved ceiling, counted as 8 GPUs × time while a job holds the node:**
  - **600 partner GPU-hours** if W is F or F+replay (forecast about 314–494);
  - **800** if W is D or D+replay (forecast up to about 700 at the slower measured D+replay stage time). The owner authorised raising the ceiling where 600 is tight.
- **Queue time between chained Slurm jobs is not charged.** It is reconstructed from the partner's job start and end records.
- **Spending stops at the approved ceiling, not the formula cap.** Any further increase is the owner's decision.

