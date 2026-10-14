# 3C second-task screen — pre-registration, 11 October 2026

Question: is there a second task with fresh Qwen3-8B learning room and enough
verified teacher/student targets to justify standalone acquisition and then
FinQA → B? This is package 3C's screen only (roadmap §4, programme ledger §7).
No training, full-pool qualification, sequence, recovery experiment or frontier
comparison is authorised here. Written before any new model output.

**Freeze status:** design registered; the machine-readable data annex is not
complete until both candidates have been materialised and hashed. MATH source
data is unavailable in this checkout. No real inference may precede completion
of that annex. CPU stand-ins are engineering evidence only. Do not change these
rules after outputs; do not call absent data hashes a completed freeze.

## Candidates and immutable selection

1. **LiveCodeBench**, using `kit/beds/code.py` and `code-split-v1.json`.
   The training population is the 223 pre-cutoff problems in the archived
   `test5.jsonl` + `test6.jsonl` copies, excluding panel IDs and normalised panel
   text exactly as the existing bed does. This is an explicit two-file subset,
   not the entire historical release. The held-out panel is the existing 51 in
   its original order. Source bytes, every pool/panel prompt and test payload,
   IDs and ordered lists are hashed in the annex. No training/test selection
   depends on earlier model success. Screen strata are easy/medium/hard, with
   34/33/33 training questions (minimum successes 24/24/24).
2. **Hendrycks MATH**, original train/test partitions, all seven subjects, levels
   1–5. Predefined subset: text-only questions (no `[asy]`), with the final boxed
   reference answer parseable as a single exact rational under
   `kit/beds/math_screen.py`. Exclude duplicate normalised questions within a
   split; exclude train questions matching any test question. Order original
   paths lexically to choose a representative duplicate, without model output.
   Select 100 train and 20 test questions per level: a 500-question training
   pool and 100-question held-out panel. Screen 20 training questions per level
   (minimum 14 successes in each). Subject results are descriptive. All source
   files, exclusions, selected answers and prompts are hashed. This is a
   rational-answer competition-maths subset, not a claim about all of MATH.

For every selection, sort by `(sha256(UTF8(salt + '|' + id)), id)` and take the
registered count within each stratum. Salts: `3c-v1-code-screen`,
`3c-v1-math-pool`, `3c-v1-math-panel`, `3c-v1-math-screen`.
Pool and panel do not overlap. Too few eligible source questions is an
incomplete preparation, not permission to replace levels or change quotas.
The CPU `freeze` command writes both candidates and the ordered/hash annex
together, before either can run. The LCB-only audit can be prepared separately.

**Contamination disclosure (pre-freeze review amendment):** the LCB pool and
panel predate Qwen3's 29 April 2025 release and the Qwen3.6-27B teacher; MATH's
test split is a standard benchmark. Prior training exposure is possible.
Headroom only catches memorisation that appears as saturation; it cannot
establish uncontaminated evaluation. This is a disclosure, not an admission gate.

The [MATH authors' repository](https://github.com/hendrycks/math) and its
[licence](https://github.com/hendrycks/math/blob/main/LICENSE) state MIT; the
[EleutherAI dataset card](https://huggingface.co/datasets/EleutherAI/hendrycks_math)
also lists MIT and preserves train/test splits. Keep the licence with the local
download. Do not use the combined 12,500-row mirror as a train split. LCB's
existing split records a generic CC declaration with unresolved variant; this
package does not redistribute its problem statements or tests.

GSM8K, ToolAlpaca, Chemistry and Spider-v2 are excluded for the settled reasons
in the proposal. Candidate order cannot be changed by outcomes.

## Measurements and decisions

**Step 1:** verify source identities and scoring on CPU, then score the fresh
untrained `Qwen/Qwen3-8B` revision
`b968826d9c46dd6066d109eabc6255188de91218`, without adapters, on the entire
candidate panel. Reuse `kit/eval_bed.py`, one GPU, deterministic greedy decoding,
thinking disabled, seed 0, 2,048 output tokens, 12,288 context. A baseline
strictly above 90% rules out that candidate: LCB ≥46/51; MATH ≥91/100.
Exactly 90% proceeds. Missing/duplicate/foreign outcomes are incomplete.
Length finishes are reported and scored as returned, never retried.

Coding uses all released private tests at both steps (the existing training
eight-test cap is not used for screen verification). CPU adequacy reporting
covers test counts, distinct inputs/outputs, duplicates, stdin/functional
shapes, and whether constant-output mutants survive on every screen/panel
problem. Handwritten correct and faulty programs validate both scorer modes,
including timeout. These checks expose weak tests; no new numeric admission
gate is added. Passing finite tests is not proof of algorithmic correctness.

MATH extracts the last balanced `\\boxed{...}` answer and compares exact
rational values, including equivalent fractions (also `\frac12`), correctly
grouped thousands separators (`1,000` or `1{,}000`), finite decimals and bounded
arithmetic. These normalisations also apply to gold eligibility before the joint
freeze. No floating tolerance, substring matching, solution-text fallback
or model judge. Unsupported prediction syntax scores wrong and is archived.
CPU equivalence checks include scaled/sign-flipped fractions, arithmetic,
near-miss decimals, malformed boxes, units, sets, intervals and symbolic
expressions. Validate every selected gold answer and a deliberately shifted
answer. The subset restriction and parser limitations stay visible.

**Step 2:** only after a complete Step-1 pass, run pinned
`Qwen/Qwen3.6-27B`, revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`,
on the frozen 100 training questions. Reuse v4's schedule: seeds
27101/27102/27103/27104, temperatures 0/.3/.6/.8, top-p 1, at most four
attempts, 2,560 teacher tokens, thinking disabled. Choose the first verified
demonstration. Frozen initial 8B rewrites that demonstration, at most four
attempts, temperature .7, top-p .95, v4 per-ID/attempt seeds, 2,048 tokens,
6,144 input tokens. The teacher input cap is 4,096. Verify completed targets
using the same scorer and v4's 2,048-token EOS-inclusive student target limit.
An executable code solution is a worked answer; no prose word-count gate.
For MATH require working before the final box; no word-count gate.
Archive every raw response, verifier result and rejection, including truncation
and input-cap failures. No test inputs, expected test outputs, reference MATH
solution or gold answer enter generation prompts. Rewrites see only the
question, selected demonstration and v4 rewrite instruction.

Admission requires **≥85 distinct demonstration ∩ rewrite successes /100**,
and **≥70% in every frozen eligible difficulty stratum (`n_s ≥10`)**, rounding
required counts upward. Four unsuccessful attempts are completed scientific
failures for that question, not missing observations. Budget/wall-time/resource
exhaustion before completing the registered attempts means **INCOMPLETE**, even
if successes already exceed 85. No adaptive extra attempts, seeds, demonstrations,
question substitutions or cap extensions. Operational repairs may resume the
same archived requests within the approved cumulative budget; they may not
resample completed requests or erase earlier evidence.

The first candidate passing both steps is selected for 3C qualification; do
not run MATH once LCB qualifies. A completed scientific failure advances to
the next candidate. An incomplete candidate leaves selection unresolved; do
not treat it as failure to advance the order. If both complete and fail, record
no qualified second task. Full-pool ≥80% coverage and demonstrated standalone
acquisition remain later requirements on the partner. A screen pass proves
target availability and headroom, not learnability, retention or goals 3–4.
Pool-weighted coverage, platform/subject breakdowns, test adequacy, tokens,
rejection reasons and timing are descriptive; none rescues a failed decision.

## Cost and paid boundary

**The original $20 cumulative Modal ceiling is insufficient for the revised
pre-freeze timing plan. Proposed reservation: $126.9841 for both candidates
($127 rounded up for owner approval)**, including failed starts, CPU, GPU,
host memory and collection. This is a request, not an authorised increase.
No paid run in this build. The manager must first obtain and record owner
approval for the larger amount in `docs/phase2/ledger.md`; do not trim the
science to fit $20. Per candidate reserve $2.7911 for Step 1 and $60.7009 for
Step 2 (totals calculated before rounding). Step 1 uses one H100;
Step 2 uses two H100s, teacher TP2 then frozen-8B TP1 in a fresh subprocess
(both allocated GPUs remain charged). Use the v4 27B check environment:
vLLM 0.18.0, torch 2.10.0+cu129, CUDA 12.9.1, Flash Attention, Triton GDN,
batch-invariant mode, no precision fallback.

[Modal published rates](https://modal.com/pricing), checked 11 October 2026:
H100 $0.001097/s, CPU $0.0000131/core/s, memory $0.00000222/GiB/s.
Reserve using 8 cores + 64 GiB, 25% compute margin, 600 seconds startup,
2 seconds idle, $0.10 ancillary allowance per invocation. Function limits:
Step 1 1,000 seconds; Step 2 **19,260 seconds**, of which 60 seconds are reserved
for finalisation. Its 19,200-second work deadline is twice the conservative
9,600-second projection derived from attempt 4's measured 27B load/round times
and the ledger's cold-download times, plus explicit verification/IO allowances.
The projection assumes all four attempts for all 100 questions in each phase,
and uses the 27B timings as a conservative proxy for the unmeasured 8B phase.
There is **no persistent model cache**: `/tmp` is inside a single-use container.
The increased startup allowance accommodates the larger image containing the
1.3 GiB test file; both model downloads and loads remain inside the function
budget. These are planning estimates, not guarantees. Preserve partial evidence
and report incomplete on cap exhaustion. Detailed arithmetic and unchanged paid
commands live in `3c-second-task-screen-runbook.md`. This review amendment lands
before the joint annex binds the registration and implementation hashes.

These are predeclared scientific task-admission rules. Plan v4 §15.21's
direction against disqualifying operational gates does not remove them. No
extra review, parity, judge, quality percentage or campaign gate is introduced.

## Data source for candidate 2 (declared 12 October 2026, before any model output)

The authors' MATH release is no longer distributed (the Berkeley tarball returns 403; Hugging Face
`hendrycks/competition_math` is disabled after a takedown). Owner, 12 October: use the EleutherAI mirror
("A. since it is a research norm"). Source: Hugging Face `EleutherAI/hendrycks_math` at revision
`21a5633873b6a120296cce3e2df9d5550074f4a3` (labelled MIT; train 7,500 / test 5,000, fields problem, level, type,
solution, as in the original). The mirror does not keep the original file numbers, so item IDs are
`<split>/<subject>/<row index>` at that revision, materialised by `scripts/materialise_math_mirror.py`. Selection
rules, strata and decision rules are unchanged. The rights-holder's takedown is disclosed; any publication will say
which copy was used.

