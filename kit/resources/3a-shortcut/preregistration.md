# 3A: Chemistry numeric-rank shortcut attribution

Registered specification, 11 October 2026; round-2 amendment requested by the
owner after review of `cebfccd9`, after CPU fixtures and before any checkpoint
scoring. The initial specification is retained in that commit. Panel bytes are
unchanged; only diagnostics and analysis/interpretation are amended. Paid
execution awaits the owner's separate allocation in
`docs/phase2/ledger.md`. Roadmap 3A only; no training or published-run attribution.

Question: how much of the two kept standalone SDPO Chemistry checkpoints' gain
over base depends on the third-smallest numeric-option cue? The historical
32/56 and 32/55 concentration of net gains is descriptive, not attribution.

Models: Qwen/Qwen3-8B at programme revision `b968826d` (full revision from
`kit/v4_contract.py`); `sema-chem-r1-a1/hf-step40` and
`sema-chem-r2-a1/hf-step40` (partner-log row 24). No other checkpoints or selection.
Use the authors' pinned 210-question test split (51 affected: 32 logD and 19
logarithmic aqueous solubility; 159 unaffected). The repository's older 400-item
`chem_L3_heldout.jsonl` and mixed-level `chem_probe_by_level.jsonl` are historical
panels, not these checkpoints' test population. Frozen taxonomy and source hashes
are inherited from the v4 data tooling.

Panels, generated without reading checkpoint responses:

- Original: all 210 questions, unchanged system/user messages and gold letters.
- Letter permutations: three fixed cyclic option rotations (1, 2, 3 positions),
  all 210 questions each. Texts and gold option identity stay fixed. Original
  plus these rotations put each original option at every letter once.
- Balanced: four independently drawn distractor sets for each affected question,
  one per gold numerical rank 1–4 (204 instances). Molecule, stem, system,
  gold value and its exact spelling stay fixed. New distractors are sampled
  from the same property's training-option value distribution, without
  replacement within a set, separated by at least 0.10 log units from gold
  and one another. Four label rotations ensure each question's gold occurs at
  each letter once; a hash-fixed per-question rank order avoids a common
  rank-to-letter mapping. Values retain source spelling; no synthetic fallback.
- Rank-preserved redraw control: four independent new distractor sets per
  affected question, always gold rank 3, with the same four gold-letter
  placements as Balanced (204 instances). This tests distractor replacement
  sensitivity while retaining the cue. Unaffected questions receive only
  Original and Letter permutations; they have no invented numeric intervention.

Before scoring, a separate mechanical validator reparses the written prompts,
checks preservation of stem/system/gold text, disjoint train/test identities,
unique numeric values and separation, training-distribution membership, gold
rank/letter counts per question and property, and accuracy of every fixed
rank rule (smallest, second, third, largest). Each rule must score exactly 25%
on Balanced overall, within each property and within each question. Also
report fixed-letter rules, nearest/farthest-from-option-mean heuristics, and
three training-only gold-value heuristics per panel and property: pick inside
the exact closed training-gold range, pick nearest its frequency-weighted
median, and pick an exact training-gold option string. Report expected accuracy
under uniform ties and a uniform four-option fallback when no option matches.
Also report training-gold string membership separately for golds/distractors.
These descriptive diagnostics do not license outcome-driven regeneration.
Freeze every panel, validation report, source, scorer, and this document by
SHA-256 in the shipped manifest. Resolve construction defects before scoring;
retain any subsequent departures and never silently substitute panel bytes.
This is mechanical validation and empirical plausibility, not an independent
experimental measurement of molecular properties or chemical distractor quality.

Scoring: unchanged `kit/eval_bed.py` Chemistry bed and authors' MCQ verifier,
thinking off, deterministic eager/batch-invariant, greedy n=1, B=2,048,
BF16, one physical GPU per job, context 12,288 (programme scoring convention).
Run each model's four panel groups on its assigned GPU; parallel jobs may use
other physical GPUs of the same type/software. Record identities, fingerprints,
raw responses, output token counts, truncation and actual job/allocated GPU time.
There are 12 scoring rows, 1,248 instances/model, 3,744 completions total.
No accuracy threshold, qualification pause, or extra approval between rows.
Failures can be fixed and retried locally with original evidence retained.

Reader and uncertainty (predefined): average repeated instances within each
underlying question before averaging questions. Report base and checkpoint
accuracy, gain = checkpoint minus base, and tokens per correct answer = all
output tokens / correct instances (undefined when zero correct), separately
for affected, each affected property, unaffected and all where applicable.
Report format failures and truncations descriptively. Never treat variants as
independent questions. Use 10,000 paired bootstrap resamples of question IDs,
seed 31011, percentile 95% intervals, stratified by the two affected properties
(or unaffected family), preserving all variants and all models together.
Intervals describe question uncertainty conditional on these two trained runs,
not uncertainty over a population of training seeds; no seed significance test.

Headline rank effect, per checkpoint: affected gain on Balanced minus gain on
Rank-preserved redraws, with a paired question interval. Both panels redraw
options and balance gold letters; their training-gold-string familiarity and
value-band dilution are roughly matched. Report unaffected Letter-minus-Original
gain and headline rank effect minus that unaffected change as a selective rank
gain change. Unaffected questions receive a letter intervention only, so this
is a control rather than a fully matched numeric-intervention comparison.

Also report Balanced-minus-Original, affected Letter-minus-Original,
Balanced-minus-matched-Letter and Rank-preserved-minus-matched-Letter gains;
retain the original selective difference as descriptive. The matched letter
reference averages Original and the three rotations, covering all four letters.
Balanced-minus-Original mixes rank removal with training-gold familiarity and
band dilution; it is not the headline rank contrast. Report the untrained
model's perturbation effects alongside the gain effects.

“Gain depends on the rank cue” has strong support if BOTH checkpoints have
positive affected gain on Rank-preserved redraws, the 95% interval for
Balanced-minus-Rank-preserved gain is wholly below zero, and the selective rank
change interval is wholly below zero. For “unaffected gain does not materially
shrink”, require its Letter-minus-Original interval wholly above −5 percentage
points. These are interpretation criteria, never run gates. Failure means
inconclusive or mixed support, not evidence of no effect. Report seed disagreement.

The narrow training-gold value band is a second identified shortcut, recorded
as a descriptive, option-only finding. Nearness to the training-gold median
remains highly predictive in both redrawn panels. Exact gold-string familiarity
also changes when distractors are redrawn; report it without attributing model
behavior to memorization. Exact training ranges are logD [2.36, 3.08] and
solubility [−2.843, −1.62]; one held-out solubility gold (−2.85) lies just outside
the latter. Specify boundary/tie conventions rather than silently rounding.

A positive surviving gain on Balanced bounds rank-cue dependence only; a
value-band cue remains untested. It does not prove chemical reasoning or
validate another paper's gains. A loss on rank-preserved redraws relative to
Original may reflect familiarity, band dilution or general distractor
sensitivity. A loss under letter permutations suggests label/format sensitivity.
A replicated headline loss supports rank-cue dependence under these interventions,
with residual distractor-difficulty and imperfect matching stated. No in-band
panel is added; a separate band-removal intervention remains an owner option.

Headline descriptive lost-gain fraction: (rank-preserved affected gain − balanced
affected gain) / rank-preserved affected gain, only for a positive denominator.
Also retain the Original-referenced fraction as a secondary descriptive measure:
(original affected gain − balanced affected gain) / original affected gain.
Never clip either ratio to [0,1]; show raw point changes and both denominators.
For each reference, report its contribution to whole-panel lost gain with weight
51/210, and its fraction of original whole-panel gain only when that denominator
is positive. The Rank-preserved reference changes only affected questions;
these weighted contributions are not a measured full-population intervention.
Ratios are descriptive, not the fraction causally mediated by a learned rank
algorithm. No adaptive prompting, checkpoint selection, serving-cap changes or
panel regeneration based on outputs. Later changes are recorded as amendments.

Budget: planning estimate 24 partner GPU-hours core, ceiling 50 including
reserve/failed/idle allocation time; optional base-only Modal technical smoke
within the roadmap's $5 allowance. No paid work is executed by this build.
