# Next partner message: draft registration

20 October 2026. Implements approved [plan v6 §§1–6](../programme-plan-v6-20261020.md), decision 4. Design only; no runs. [B and its amendments](package-b-registration-20261014.md), [tier-1 results](p3b-tier1-results-20261019.md), [runbook](../../kit/README-p3b.md), [SEC notes](../phase4/4a0-sec-slice-v4-notes.md) and [literature refresh](lit-refresh-20261020.md) govern inherited details.

**Shared registration.** Seeds 101–105, paired throughout. Bind W only after completed unreset tier 2, using [C §3 and amendments](package-c-registration-20261018.md): rank confirmed joint B success, confirmed retention-margin count, sequence MATH final, then lower maximum task serving-token ratio; exact ties F, D, D+replay, F+replay. Weak complete evidence selects provisional W; incomplete evidence stays pending. Recovery/reset exports never select W.

Use one physical partner scoring GPU/environment throughout; re-score shared base/incoming controls there. Greedy, thinking off, decoding seed 0, 2,048 output tokens; unchanged panel prompts/scorers. Report five-seed means, paired seed intervals and question-bootstrap intervals separately, counts, finish rates, tokens/correct including failures, and correct→incorrect/incorrect→correct IDs. Undefined denominators remain undefined. Common admission: benefit ≥2 percentage points, positive paired 95% interval, other endpoint loss ≤1 point with its bound confirming the margin. Crossings are inconclusive. B/C/4A bars remain unchanged.

**Input identities.** Variables below mean the runbook's durable trees, resolved at send:

| Input | Registered path and receipt |
|---|---|
| Base / teacher | Qwen3-8B `b968826d9c46dd6066d109eabc6255188de91218` / Qwen3.6-27B `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`; snapshot/file hashes |
| D/F FinQA final | Unique `$V4_EXPORTS/**/{D,F}-finqa-alone-sSEED/hf-step40`; `$V4_WORK/runs/ARM-finqa-alone-sSEED/env/export-identity-step40.json` |
| Replay FinQA final | `$OUT/runs/finqa-RECIPE-sSEED/hf-step40`; sibling `export-40.json` |
| Tier-2 final | `$OUT/runs/{standalone,sequence}-RECIPE-sSEED/hf-step40`; sibling `export-40.json`; identity records incoming receipt |
| FinQA / MATH targets | `$V4_WORK/v4/report-main/scheduled/finqa-sSEED/{train.parquet,manifest.json}`; `$OUT/targets/schedule-SEED.jsonl` and `manifest.json` |

Receipts must match weight files, including retained F+replay 101–103 and corrected D+replay; exclude `failed-v3`. FinQA evaluation is `$FINQA/test.parquet` (1,147); MATH is A's frozen 100-item `math_screen`, with its 500-item training pool. General-v1 is `kit/panels/general-v1.jsonl`, SHA-256 `ce5743ef03f59f57b18231ccd88aee66f0350c8fe3fce8a4abaeb816dd00a1e6` (IFEval, knowledge, maths separately).

**v0+: coexistence.** Question: can a task-trained checkpoint retain SEC competence and benefit from its own successes? Choose D+replay's FinQA finals if W∈{D,D+replay}, otherwise F+replay's, five seeds. Arms: each checkpoint memory-off and static k=3 own-success memory; frozen base with the same two arms, re-scored on-node once.

Freeze the audited **193-question** July–September `data/p4_sec/slice-v4-scale`, including declared shortages, batch/source/audit hashes and holdout components; manifest SHA-256 `3e98419636a327c004464cb31b921db54acf78a06cc2e92f873c6a5e83b13782`. No slice rebuild. Collector equals the evaluated checkpoint: admit only its earlier memory-off responses passing both historical and explicit-unit scorers. Preserve SEC exclusions for permanent holdouts, target issuer components, accessions/filing groups and current-month answers; no base traces substituted.

Primary: explicit-unit unseen-issuer accuracy and tokens/correct, by month and pooled; historical scores secondary. Plan-v6 §2 exception to common admission: July memory-off ≥ frozen; August–September own-memory ≥ frozen+memory. Show paired issuer-clustered 95% intervals and seed uncertainty; crossing zero leaves coexistence unconfirmed. Report own-memory minus own-memory-off and trace yield. B general margins are reported, not re-judged: maths already misses by about one point. This development demonstration changes confidence in coexistence, never admits v2 or changes 4A.

Builder interface: add `--checkpoint-path`, `--checkpoint-receipt`, `--collector-checkpoint-path` and collector receipt to the SEC entry point; require collector/evaluator identity equality, separate per-seed journals, and checkpoint/batch/response receipts in memory manifests. Support plain node execution and base controls; preserve SEC's 12,288 context limit. Paths/hashes and GPU identity bind at send; scientific choices freeze now.

**v3: diagnostic interpolation.** Question: does scaling an update improve acquisition–retention trade-offs? For every D/F/D+replay/F+replay FinQA final (20), and available tier-2 standalone/sequence finals (up to 40), form `θλ = θstart + λ(θfinal−θstart)`, λ={.25,.5,.75}. Stage start is base for FinQA/standalone MATH, exact matched FinQA final for sequence MATH. λ=0 and 1 are controls, never substitute base for a sequence start.

Score FinQA/general-v1 throughout, plus MATH on tier-2 blends; report acquisition–retention curves with paired uncertainty against λ=1. Interpolate every floating weight in FP32, cast to the original export dtype; require matching tensor keys/shapes and identical nonfloating buffers/tokenizer. Record both parent receipts and λ in each export receipt. Bind available finals at send, listing missing slots. No recovery/reset blends.

This is inspired by grafting (2610.05872), not donor-grafting replication. A common-rule-sized retention benefit without acquisition loss beyond its margin warrants registering v3 proper **only after v2 admission**; otherwise defer. No λ is selected from these evaluation panels, no blended checkpoint becomes incumbent, and multiplicity makes apparent favourable points exploratory. Any stream λ must be fixed independently before stream evaluation.

**v4: replay breadth, FinQA→MATH.** Question: do broad base skills and earlier acquired skills need distinct replay? Start all four arms from W's exact tier-1 FinQA finals. Generic-only uses W's family objective plus B replay even if W is plain D/F. Reuse B's five sequence controls only when starts, recipes, schedules and receipts match exactly; otherwise run five controls.

Freeze 40 updates, batch 32, B optimiser/caps/schedules. Generic occupies positions 8/16/24/32. Reference is 28 MATH+4 generic; challengers replace task positions 4/12/20/28: broad-only or earlier-only =24+4+4; both =24+4+2 broad (4/20)+2 earlier (12/28). Reduced MATH exposure is part of the intervention. D uses per-item token-normalised task KL plus EOS-inclusive replay CE weighted by slot fractions; F keeps B's SFT reduction. Preserve D's over-cap skip-in-place audit.

Generic reuses B's checked base pool: RLVR-IFeval `47c03c73621c4aab2b824b7818681117d662770e`, open-instruct checker `11826255077617a46919ce75cadf1f3d53f30dac`, `IF_FUNCTIONS_MAP`; no refresher rows.

Broad uses frozen base's own answers to [GSM8K train](https://github.com/openai/grade-school-math) (MIT; final-number exact rational check against gold) and [ARC-Challenge train](https://huggingface.co/datasets/allenai/ai2_arc/blob/main/README.md) (CC-BY-SA-4.0; strict final choice label equals `answerKey`). One greedy, thinking-off, seed-0, 2,048-token response per candidate; complete checked answers only, no source solutions or teacher substitution. Keep licences/attribution; answer checks do not verify rationales.

Hash-order each source by `sha256("next-v4-broad-v1|"+source+"|"+id)`, ID tie-break; retain at most 1,024 deduplicated candidates/source before generation, no replenishment. Cycle accepted source pools equally, seed-shuffled. Earlier-task uses unique B verified 27B FinQA training targets, never evaluation items.

Apply B's shared-source-ID, NFKC/casefold/whitespace equality, 5-shingle Jaccard ≥.8 and edit similarity ≥.9 exclusions, including short prompts, within/across pools and against all FinQA/general-v1/MATH evaluations. Freeze untouched **HellaSwag-100**: [Rowan/hellaswag validation](https://huggingface.co/datasets/Rowan/hellaswag/blob/90242dd8c0c481249d6e24cbe6c6f2cefaf7444d/README.md) at `90242dd8c0c481249d6e24cbe6c6f2cefaf7444d`, first 100 by `sha256("next-v4-holdout-v1|"+ind)`, tie by ind; original context/endings labelled A–D, prompt suffix `End with Answer: <letter>`, strict final-label accuracy. No HellaSwag split supplies replay; exclude pool overlaps. Pin broad-source revisions/checkers before generation; freeze accepted IDs/outputs and exclusions before training, binding hashes at send. Empty pools leave inputs incomplete.

Score MATH 0/10/20/30/40; FinQA/general-v1/HellaSwag 0/20/40; step 40 primary. Broad's primary is general-v1 maths retention; earlier-task's is survival fraction of FinQA answers wrong at base/right at stage start. Broad first against generic; earlier-only against generic if broad fails, otherwise both against broad. Require ≥2-point primary gain with positive paired 95% interval; other retention endpoint and untouched panel lose ≤1 point with confirming bounds. **Stricter acquisition exception:** MATH gain-AUC and final-score paired lower bounds versus generic-only must be ≥0; incremental admissions also protect the surviving incumbent. Report denominators and paired transitions. Probe success nominates components; admission awaits connected finals, broad first. Failure preserves incumbent.

**Order, cost, delivery.** Scoring jobs (v0+, v3), then probe, C, then registered/audited 4A; full-node training serial. Return 1 after C, return 2 after 4A. Never delay message 1 for 4A; continuation belongs to message 2.

Measured F ≈20–25 min×8 GPUs=2.7–3.3 GPU-h/stage; D+replay ≈50 min×8=6.7. Probe: 15 new stages cost 40–50 or 100 GPU-h; unmatched controls add 13–17 or 33. Allow preparation/scoring/startup within **100–200**. v0+ <5+≈1 base GPU-h; v3 10–20, up to 180 blends. C retains 423 if W=F, otherwise 600 pending legal repricing. Total **539–826 without 4A; 1,139–1,676 with 600–850 for 4A**, matching v6's rounded envelope. Do not add C twice. Reprice at send from tier-2 timings; allowances are estimates, not measurements.

Builder checklist:

- Receipt-bound input manifest, W decision, frozen pools/panels/licences and exclusions.
- Plain preparation, interpolation, training, SEC/scoring and readout commands; partner supplies sbatch.
- Collector-identity and blend-endpoint checks, loss/slot audits, local stand-in rehearsal before send.
- Paired curves/transitions, admission tables, tokens/costs, durable raw evidence and compact return archive.
- Fold B logging teardown/reader fixes into the kit; record kit/environment hashes, send-time budget and missing inputs.

VERDICT: DONE

1. v0+ uses the replay sibling and its own verified SEC successes.
2. v3 screens interpolation without selecting λ on evaluation panels.
3. v4 separates broad and earlier-task replay with an untouched panel.
4. Scoring→probe→C→optional 4A; first return follows C.
5. Draft only: no runs; identities and measured repricing bind at send.

## Amendments after cross-family review (Opus, 20 October; binding, applied before any build)

1. **Contamination disclosure.**
   - **Broad replay is in-distribution for the general-v1 maths panel** (GSM8K test, ids `gsm8k-<test idx>`) and near-distribution for knowledge (ARC is MMLU-like). A maths benefit therefore shows the skill is protected by matching replay, not by breadth.
   - **HellaSwag-100 is the only out-of-pool panel.** Its change is reported as a secondary transfer endpoint, not only as a guard. ARC shares HellaSwag's letter-answer format; format transfer is disclosed.
   - **Pool IDs are namespaced** `gsm8k-train-<idx>` and `arc-c-train-<id>`, so the shared-ID rule cannot collide with panel IDs.
2. **"Both" is factorial.**
   - Batch slots: 20 MATH + 4 generic (positions 8/16/24/32) + 4 broad (4/12/20/28) + 4 earlier-task (2/10/18/26).
   - The single-addition arms keep 24 MATH + 4 generic + 4 added slots.
   - The lower task exposure of "both" (800 vs 960 MATH prompts) is disclosed and is part of that arm's recipe.
3. **Acquisition rule = plan v6's common rule.** MATH gain-AUC and the final score may lose at most 1 point against generic-only, with paired 95% lower bounds ≥ −1. No stricter superiority test.
4. **v3 proper uses λ = 0.5, fixed now.** Screen results decide only whether v3 proper runs, never its λ.
5. **v3 storage.** Blend, score, then delete. Keep tensor-hash receipts and a re-creation command, not the weights. Write blends to node-local NVMe only; about 180 blends × 16 GB would otherwise need about 3 TB.
6. **Inputs frozen now:**
   - the GSM8K prompt (general-v1's maths instruction) and the ARC prompt (`End with Answer: <letter>`);
   - the GSM8K checker (eval_bed's maths number extractor);
   - a new HellaSwag-100 scorer, built and frozen before any output;
   - licences: HellaSwag MIT; ARC CC-BY-SA, with its notice carried on the pool file.
   - Under D, earlier-task items use EOS-inclusive cross-entropy on the 27B targets (a teacher-text anchor, not SDFT KL).

   Kit checklist addition: the slot-map extension to `p3b_replay_train.py`, which today hard-codes positions 8/16/24/32.
7. **Controls re-run fresh.** W is selected on general-v1 and MATH, so its reused sequence runs carry selection noise. The generic-only reference arm is therefore re-run fresh, five seeds (about 13–33 GPU-h), instead of reusing B's sequence finals.
8. **v0+ bar.** "Met" means the seed-mean is at or above the reference. "Confirmed" means a non-inferiority lower bound ≥ −2 points. A mean below the reference means coexistence is not shown.
9. **Wording and capacity.**
   - Probe total, including stages, preparation, scoring and startup: about 100–200 GPU-h, plus 13–33 for the fresh controls.
   - With one physical scoring GPU, about 300 scorings for v3, the probe and v0+ run serially, roughly 1–2 days of wall-clock.
   - If the scoring GPU is not B's, the λ = 0 and λ = 1 controls are re-scored (about 6 GPU-h).

## Amendment, 20 October (Owner, 20 Oct: "Go ahead with 1,2,3")

- **v4, the replay-breadth probe, moves from FinQA → MATH to FinQA → ContractNLI,** inside package C's after-FinQA route, where the second task has real headroom (8B 42, teacher 78).
  - Arms: a reference with generic replay (the replay sibling of W's family; shared with C's W arm when W already is that recipe), + broad, + earlier-task, + both, at the registered slot maps (24 + 4 + 4 for single additions, 20 + 4 + 4 + 4 for both), five seeds.
  - Pools, checkers, the untouched HellaSwag-100 panel, the contamination disclosure and the admission rules are unchanged, except that MATH is replaced by ContractNLI as the acquisition endpoint (C's test-reserve tolerant score and gain).
- The FinQA → MATH probe is dropped from the next message.
- v0+ and the v3 interpolation screen are unchanged.

## Amendment, 22 October (owner: "Yes, ActivityNet-only fix is OK."; before any HellaSwag output)

The `90242dd8c0c481249d6e24cbe6c6f2cefaf7444d` pin could not be staged:
it is a loader-only revision, and its GitHub source has been DMCA-blocked since
2026-09-14. `ind` is not unique across sources: 433 values are shared by
ActivityNet and WikiHow rows. The panel is re-pinned to
`218ec52e09a7e7462a5400043bb9a69a41d06b76`, with validation parquet
`data/validation-00000-of-00001.parquet` SHA-256
`899813071e1e95efafec90f856e1987d2150fa4d020fc005df6962c259f660cd`.
The population is ActivityNet rows only (`source_id` starts with `activitynet~`):
3,243 rows with unique `ind`. Selection, prompt, labels and strict parser are
unchanged. The frozen panel SHA-256 is
`2b0fc85c6e80854bb47982d35a18d15bc83a5655de6202634547071eccbe487d`.
Found by the partner's agent at row M2; the owner approved the ActivityNet-only
fix on 22 October, before any HellaSwag output.

Before a tag for this amendment, the manager must run
`python3 scripts/check_pinned_inputs.py --contract-source "$CONTRACT_SOURCE" --receipt pinned-inputs.json`
on a network-enabled CPU host and retain an all-inputs-passing receipt alongside
the export review. Stand-in/export tests alone do not establish stageable pins.
