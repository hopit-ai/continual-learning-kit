# Hard-MATH headroom screen — registration, 20 October 2026

Question: does a hard MATH subset give frozen Qwen3-8B real learning room (30–70%), so that it can become the
connected continuation's third task? Authority: [package C amendment of 20 October](package-c-registration-20261018.md)
(owner, 20 Oct: "Go ahead with 1,2,3"). The registered 3C/tier-2 MATH subset (500 training / 100 evaluation,
[3C registration](3c-second-task-screen-preregistration.md)) is too easy for this purpose: frozen 8B scores about
89–93/100 on it. This screen measures **student headroom only**. Teacher coverage (pinned `Qwen/Qwen3.6-27B`,
revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`) is screened separately on the partner's node later. Written
before any model output on these items. Do not change these rules after outputs.

## Data, frozen before inference

- **Source:** the 3C source unchanged: `EleutherAI/hendrycks_math` at revision
  `21a5633873b6a120296cce3e2df9d5550074f4a3`, original train/test splits, materialised by
  `scripts/materialise_math_mirror.py`; IDs `<split>/<subject>/<row index>`. The build re-runs 3C's own
  `math_inputs` and refuses unless it reproduces the 3C frozen annex exactly (source digest, old pool and panel IDs).
- **Eligibility:** 3C's, unchanged: text-only (no `[asy]`), last-box gold parseable as one exact rational by
  `kit/beds/math_screen.py`, exact normalised duplicates and train/test overlap removed.
- **Hard levels:** MATH **Level 5**. Levels 4–5 are used only if Level 5 cannot fill every set below after all
  exclusions (recorded in the manifest; not an outcome-dependent choice).
- **Disjointness:** exclude every item of the old 500 pool, 100 panel and 100 screen sample by ID; then apply the
  registered near-duplicate rule (`kit/p3b_replay_data.py`: normalised equality, 5-shingle Jaccard ≥0.8 with
  comparable size, or normalised Levenshtein similarity ≥0.9) on problem text, greedily in `hmath-v1-dedup` hash
  order. Held-out candidates are checked against the old 600 and the general-v1 panel; training candidates against
  those plus every surviving held-out candidate. Each exclusion and its matched item are recorded.
- **Sets** (stratified by MATH type, proportional quotas by largest remainder on surviving counts; within a type,
  sort by `(sha256(salt + '|' + id), id)` and take the first n):
  - **screen panel, 200**, from the test split, salt `hmath-v1-screen`; the only set scored here;
  - **evaluation set, 100**, from the remaining test-split items, salt `hmath-v1-eval`; reserved, never scored by
    this screen;
  - **training pool, 500**, from the train split, salt `hmath-v1-pool`; reserved for the later registration.
- Too few items is an incomplete preparation, not permission to change quotas or levels beyond the rule above.
  Every selected gold and a shifted answer are validated by 3C's equivalence audit. Files, hashes, level/type counts
  and exclusions are in `data/math_hard/manifest.json`.

## Measurement

Frozen untrained `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`, no adapters, scores the
200-item screen panel once through `kit/eval_bed.py generate --bed math_screen` (the 3C step-1 scorer): the 3C
prompt, chat template with thinking disabled, deterministic eager batch-invariant greedy decoding, seed 0, 2,048
new tokens, 12,288 context, one H100. Length finishes are reported and scored as returned, never retried. The
verifier is 3C's: last balanced `\boxed{}` compared as an exact rational; unsupported syntax scores wrong.

## Decision rule (frozen)

- **QUALIFIES (student headroom)** if the complete panel's frozen accuracy is in **[30%, 70%]**, i.e. **60–140
  correct of 200, inclusive**, decided on the integer count. Above 140: FAIL_TOO_EASY; below 60: FAIL_TOO_HARD.
- Missing, duplicate, foreign or non-boolean outcomes: **INCOMPLETE**, not a failure.
- Report the 95% Wilson interval for the panel and a per-type table (n, correct, accuracy, Wilson interval) and
  finish reasons. The interval and per-type rows are descriptive; none rescues or vetoes the decision.
- A pass establishes student headroom only. Registration of the third task additionally needs teacher coverage
  and its own approved registration and budget. If no hard subset qualifies, the third stage is dropped and recorded,
  as the amendment says. Learnability, retention and acquisition are not tested here.

Contamination disclosure (as 3C): MATH's test split is a standard benchmark; prior exposure is possible. Headroom
only detects memorisation that shows as saturation.

## Cost and paid boundary

No paid run in this build. Launcher `scripts/math_hard_screen_modal.py`: default is a CPU dry run (no Modal
import or network); `--rehearse` runs the same scorer path on CPU stand-ins; `--execute` is the manager's paid
action after a ledger estimate row. One H100 GPU function, timeout 1,200 s, startup 300 s, no retries, one
container, after a CPU cache step (2 cores, 1,200 s) on the volume that already holds this 8B revision.
Worst case at $3.95/H100-hour with 8 cores, 64 GiB, 25% margin and $0.10 ancillary: **about $2.69**; expected
well under $1. Within the owner's standing authority for small GPU runs on our side ($5–10, 12 October).
