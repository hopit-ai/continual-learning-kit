# TRACE operator kit

The owner selected eight-task SeqFT and two descriptive freeze3 contrasts,
within D-gold and D-27B. Use seeds 101–103. The runtime registration ships at
[resources/trace/registration.md](resources/trace/registration.md). Preparation
and binding record the SHA256 of those exact bytes.

Qwen2.5-7B-Instruct uses Apache-2.0. TRACE data terms are task-specific. FOMC
uses CC BY-NC 4.0, MeetingBank and ScienceQA use CC BY-NC-SA 4.0, NumGLUE uses
ODC-By plus underlying content grants, and Py150 requires per-file source
notices. C-STANCE and 20Minuten need applicable grants. Section 8 records the
evaluation and gated original-tokenizer conditions. Preserve staged notices.

## CPU commands

```bash
python -I kit/trace_jobs.py inventory
python -I kit/trace_jobs.py price
python -I kit/trace_rehearsal.py --out /tmp/trace-cpu-new
python -m unittest discover -s tests -p 'test_trace_registration.py' -v
```

The rehearsal needs only standard-library Python. It uses synthetic stand-ins
through every normal row. Its evidence is TECHNICAL-ONLY. All model imports
are lazy. Network acquisition is disabled by offline environment flags and
`local_files_only=True` on Hugging Face loads.

## Bind local node inputs

Create a JSON config on the node. Paths are local inputs and remain local.
A file reference is an object with `path` and SHA256 `sha256`. JSONL data files
contain one unique globally namespaced `id` per record. A model record has
`model`, `revision`, `files` and `sha256`. `files` maps every relative staged
file to its content hash. For the 8B base, `sha256` is the canonical-map digest
of safetensors, safetensors index and config.json, matching `nm_prepare`.

The top-level config contains the following fields.

| Field | Required input |
|---|---|
| out | Absolute durable output directory |
| technical_synthetic | false for production |
| estimated_ceiling_gpu_h | 3500, warn-only |
| base | Qwen3-8B model record, registered revision |
| teacher | Qwen3.6-27B model record, registered revision |
| judge | Qwen2.5-7B-Instruct record, registered commit |
| original_tokenizer | Original gated Llama-2 tokenizer record and revision |
| sdpo | Existing pinned SDPO installation directory |
| trainer_python, score_python | Installed-stack interpreters |
| studies | Optional array of s8, gold, 27b. Default all three |
| datasets | Required cleared tasks, each with train/eval/test JSONL references |
| audit_review | Optional hash-bound JSON manual adjudication list |
| licences | JSON rights inventory reference |
| sec_clearance | Existing hash-only clearance JSON reference |
| exclusions | Existing panels/replay/few-shot exclusion JSONL reference |
| panels | Frozen panel inventory JSON reference |
| calibration | Ninety checkpoint-independent pair JSONL reference |
| admission | CPU prompt/scorer/length/demo fidelity receipt reference |
| metrics | Task/panel names mapped to callable/fixture references |
| qualification | s8/gold/27b mapped to installed-stack command manifest refs |

Each task row has `id`, `task`, `group`, `prompt` and `answer`, preserving the
released source membership. Namespace groups across source datasets. The full
released train dose is 5,000 rows/task. Parent groups are split-disjoint.
Preparation audits the entire training candidate superset before subset ranking.
Normalised prompt/answer pairs are deduplicated before choosing the fixed
500 distinct contents/task. Full S8 retains its released dose and records
intra-split duplicates. Indexed overlap screening prepares each unique text
once. It records candidates in `audit/manual-review.json`. Preparation returns
PENDING while any candidate lacks adjudication. Supply `audit_review` as a
hash-bound JSON array, each entry with candidate `id`, `disposition: "cleared"`
or `"confirmed_overlap"`, and `evidence_sha256`. Rerun preparation after binding
the review. Confirmed overlaps and held-out source groups require data repair.
The exclusion ledger includes IFEval, knowledge, maths, FinQA, ContractNLI,
MATH, replay, calibration and all few-shot populations.

The rights inventory keys cover the requested tasks plus `original_tokenizer`,
`base`, `teacher`, `judge`, `panels` and `calibration`. Each entry includes
`permitted: true`, dated `evidence_sha256`, scope and source-grant records.
Missing task grants mark affected studies PENDING. Gold and 27b require only
FOMC, MeetingBank and NumGLUE-cm. Set `studies: ["gold", "27b"]` for that scope.
Preparation reports blocked studies while binding cleared contrasts.
SEC clearance has `reserved_outcomes_opened: false`, `intersections: 0` and
`near_match_clearance: true` backed by existing custodian/hash evidence.

`panels` contains `general-v1`, `trace_general` and `open`. General-v1 must
match the shipped kit population byte-for-value, including original `math`,
`knowledge` and `ifeval` panel keys. TRACE general keys are MMLU, BBH, GSM8K,
TyDiQA, PIQA and BoolQ. Each uses the pinned builder schema described below, including
`rows`, inference settings, subset counts and dataset provenance. Open rows have `id`,
`panel` and `prompt`, with 175 Self-Instruct, 300 LIMA and 178 CoNa rows.

Calibration rows have opaque `id`, `panel`, `prompt`, `a`, `b` and
`checkpoint_independent: true`. Seal 30 per open panel. Keep source/model
identifiers outside answer text. Build them with the manager-side workflow in
[resources/trace/operator/README.md](resources/trace/operator/README.md).
It selects from the three pinned public files after approved local staging;
this build downloads nothing. Author and review the requested answers manually
or with an authorised agent; no API authoring is part of this build.
The builder emits sealed JSONL, its SHA-256 and the matching JSON array.
Keep its LOCAL quality/order key on our side. Use `trace_calibration.py --pairs pairs.json
--out packets` on our side to create independent Astra/Claude Opus high packets.
That command expects a JSON array converted from the sealed calibration JSONL.
The packet API accepts exactly two ordered declared rater names, defaulting to
`gpt-6-astra` and `Claude Opus`; the CLI accepts `--raters FIRST SECOND`.
Follow the frozen rater declaration for opaque tokens and blinding. Keep the
reversal mapping locally. GPT-6.1-Sol adjudicates disagreements after the
original independent labels are retained.
The local judge labels both orders. The numeric return includes its labels.
The paper states calibration against frontier-model raters. Human-rater count
is zero. Resolved rater backends and paid-token authority bind before calls.

### Metric and admission authorities

A metric callable reference has `path`, `sha256`, `function` and a `fixtures`
JSON file reference. The callable accepts `(row, prediction)` and returns a
0–100 per-item float. Each fixture has `row`, `prediction` and `score`. At least
20 fixtures per metric are checked to 1e-6. Use `trace_metrics.general` for the
frozen instruction/maths scorers, and `trace_metrics.score` for staged TRACE
metric implementations. The latter rows carry `metric_authority` with the
upstream file reference, `function`, result `key`, `scale_to_100`, dependency
file references and `uses_source` for SARI. Staged source modules keep their
licences/notices. TRACE's source `metrics.py` and dependencies require binding.
Knowledge uses the inherited independent reference implementation directly.

Fill only local paths and hashes in
`resources/trace/operator/task-authorities-template.json` for the eight tasks.
Stage TRACE unmodified with root `metrics.py` and sibling `evaluations/`.
The SARI result key is the JSON path `["sari", 0, "sari"]`.
String result keys retain their existing meaning.

Metric environment pins are rouge 1.0.1, fuzzywuzzy 0.18.0, sacrebleu 2.4.3,
sacremoses 0.1.1 and packaging 26.3 (the vendored SARI script imports it).
Do not install python-Levenshtein: Py150 edit
similarity is pinned to fuzzywuzzy's difflib matcher and the bridge refuses
the other backend. Use nltk 3.5 on Python <=3.11 or nltk 3.9.1 on
Python >=3.12. BLEU is computed upstream and unused by registered keys.
`datasets` need not be installed for metrics. The bridge supplies a scoped
compatibility module and loads the hash-bound vendored datasets 2.21.0 SARI
script locally, with no network fetch. Its licence and provenance ship beside it.
The task fixtures test perfect-answer invariants. The operator README documents
the manager-only task verification command and metric venv; the six general
scorers have a separate independent upstream parity check described below.

`admission` includes `original_tokenizer_census`, `supervised_masks`,
`prompt_fixtures`, `scorer_fixtures`, `demo_fidelity_gold` and
`demo_fidelity_27b`. Each has `valid` and an `evidence_sha256` for the detailed
probe result. Bind the registration's numeric thresholds in those receipts.
The original-tokenizer census uses the same rows with Qwen's port, including
L-EOS. Before any GPU row, preparation checks every S8 train/eval/test row
on CPU for BPE suffix alignment and records the conditional-EOS correction
share. Require zero wrong supervised tokens and at most one EOS-position shift
per affected row. Report original and Qwen affected shares on the same rows.
Qwen minus original above 1 percentage point warns and continues. Missing
original rates warn as UNMEASURED. The affected share never blocks S8 at prepare.
Held-out suffix census results are diagnostic. Training reads the sealed
pre-tokenised rows. Gradient clipping matches upstream TRACE at exactly 1.0,
manager-verified on 2026-11-11 in `utils/ds_utils.py` at commit
`3c2f2322cc93c109adfeda96c2c906fcb0d1192b`.
The summary fidelity checks cover the fixed 50 training-only rows per
source. Gated original tokenizer access and licence acceptance are explicit.

A qualification command manifest contains `argv` and `files`, a list of file
references covering its executable scripts and configs. It exercises the
installed stack's actual numerical/native/scorer probes. The child receives
`TRACE_CONFIG`, `TRACE_STUDY` and `TRACE_QUALIFICATION_OUT`. It writes a JSON
receipt at that final path containing `identity`, `leakage`,
`freeze_bit_equality`, `scorer_fixtures`, `licences`, and `evidence_sha256`.
The five dispositions describe valid correctness evidence. Include noise-floor,
world-size/reduction, EMA recurrence, native resume and export identity in the
hashed detailed evidence. Timing and utilisation appear as diagnostics.

Operator assets now ship in `resources/trace/operator/`: callable declarations,
twenty authored cases per metric, qualification command templates and admission
receipt templates. Complete local paths/hashes with one command:

```bash
python -I kit/trace_operator.py bind --config node-inputs.json --out bindings
```

Use `bindings/config.json` as TRACE_CONFIG. The binder checks the actual staged
metric callable against every expected fixture score to 1e-6 before publishing
the config. Source-dependent fixtures are contract cases, not a claim that an
absent upstream scorer has been tested. Preserve source licences and supply the
exact upstream task function, result key and units in `operator.metric_authorities`.
General panel authorities are supplied by the kit and refuse overrides.
The frozen instruction/maths callable and fixtures need no upstream code.

Native numerical probes and CPU semantic/fidelity observations require the
installed stack, licensed data and tokenizers. Bind their measured evidence and
command assets using the documented operator schema. Templates alone cannot
qualify a node. The qualification wrapper executes every bound native probe
and validates its measurements, including the fixed D/D noise tolerance.
It never substitutes CPU rehearsal evidence for production qualification.
The generated 27B summary audit is deferred until after targets; CPU bind seals
its receipt without changing the prepared config. All other CPU authorities
are checked before targets. See the operator README for exact schemas and the
remaining native-probe integration dependency.

## Run with operator-owned sbatch

The inventory prints allocation rows, GPU counts and dependency row IDs.
Use `inventory --config local-config.json` to print the selected study scope.
Submit those plain rows with your own sbatch and afterok dependencies. The
CPU prepare precedes targets; CPU bind follows targets and precedes GPU
qualification and training. Qualification flows directly into
training under the estimated ceiling approved with this send. The kit contains
no scheduler submission or monitoring service.

Within each allocation use the same command shape.

```bash
export KIT=/local/staged/kit
export TRACE_CONFIG=/local/trace-config.json
bash "$KIT/jobs/trace/run-task.sh" train-gold-101
```

Replace the example row ID with the inventory row. `prepare`, `bind` and
`collect` are CPU rows. `targets` uses two GPUs. Training and qualification
use eight. Each scoring row uses one physical GPU for the complete comparison,
all general-v1 states and local judging. CPU prepare precedes targets, which
must finish before CPU bind can validate and seal the 27B schedules.

Use dependency edges, rather than printed inventory order, to schedule work.
All three seeds and all study sources can train concurrently after their own
qualification. A completed seed can score and judge on its separate GPU while
other seeds/sources train. CPU prepare, bind and collect need no GPU allocation.
Scoring and judging within one complete comparison retain the same physical
GPU; stage ancestry and the persistent S8 optimizer remain sequential.
These independent edges already ship, so no scheduling or registered settings
are changed here. `price` uses the study key `27b`, matching `inventory`.

GPU rows are wrapped by `util_watch.py` with a 70% target. Missing/low telemetry,
cap-hit rates and estimated ceiling overruns warn and continue. Numeric/scorer,
identity, licence, leakage and frozen-state correctness need repair before the
affected row continues. Keep all partial attempts and allocation costs.

Rerunning a completed row verifies its receipt and returns it. An interrupted
D stage uses its native model/optimizer/EMA/RNG/mask checkpoint. S8 resumes at
its most recent ten-update native boundary. Endpoint exports use fresh attempt
directories. Replay verifies and reuses sealed exports after a crash. A score retry repeats the complete
same-GPU lane. Follow the installed stack's qualified recovery semantics.

Accounting covers reserved GPU count times elapsed row lifetime, including
loads, exports and idle tails. Reconcile scheduler allocation lifetimes and
abruptly killed rows with local attempt receipts before returning totals.
Unexpected kills can leave a RUNNING receipt requiring measured reconciliation.
Timing estimates and ceiling overruns never remove scientific cells.

## Return

`collect` writes ESSENTIAL as numbered `return-*.txt` parts, at most three
pages to paste. It writes uncapped FULL parts and numeric JSON under `full/`
for retention on the node. Overflow from ESSENTIAL spills into FULL. The
`missing.full_spill` count identifies omitted summary records. `full.records`
and `full.sha256` bind the complete local packet.

Each page has at most 75 lines and every ASCII line is shorter than 90
characters. One keyed count/summary record occupies each line. The importer
retains good records when a line is mistyped. SHA256 is ADVISORY and never
rejects a return. Every import is TRANSCRIPTION-UNVERIFIED.

ESSENTIAL contains identity, departures, licence/audit status, per-seed OP/BWT
and source effects, GPU-h by phase, failures, L-EOS census, cap-hit rates,
utilisation summary and explicit missing-data records. FULL includes every
ESSENTIAL candidate plus all-state sums, judge counts, calibration labels,
per-attempt ledger, token counts and allocation telemetry. Complete source
receipts and telemetry remain in the local evidence tree, hash-bound to FULL.

`trace_read.parse_pages` imports either record dictionary. `returned_scores`
reconstructs every score mean from FULL, and `descriptive` recomputes three-seed
estimates and guard intervals. Each seed remains one paired observation.

- `OP.study.seed.arm.offset` carries OP, TRACE BWT and T-minus-1 BWT.
- `effect.source.seed.offset` carries mean retained-task F-minus-D, then the
  individual FOMC and MeetingBank effects. `effect.27b_minus_gold.seed` carries
  the paired descriptive difference between source effects.
- `SA.study.seed.metric.offset` carries a common denominator followed by sums
  for every state in `trace_read.score_states(..., all_states=True)`. Chunks
  repeat the denominator. Legacy `S` records retain diagonal/final ordering.
  Every record identifies its own ordering independently.
- `J.study.seed.state.offset` carries Self-Instruct, LIMA, CoNa arrays with
  W/T/L/malformed counts for forward then reverse order.
- `C.offset` carries both local calibration labels in the sealed pair order.
  `calibration_ids` hashes the inventory retained on our side.
- `departures.offset`, `N.task` carry departure IDs and train/eval/test counts.
  `EOS.task.split` carries rows, affected rows, suffix errors and position errors.
  `EOSr.task.split.offset` carries Qwen share, original share and their difference.
- `A.opaque` carries row/GPU-h/status/warning count. Associated clock/error
  records retain attempt timing and error type. `gpu_h.phase` carries totals.
  `Q.study` gives identity/leakage/freeze/scorer/licence qualification fields.
- `U.opaque` and its fields carry allocation telemetry with source hashes.
  `utilisation.mean` is reserved-GPU-seconds weighted across complete allocations.
  Allocation counts, incomplete counts, below-70 counts and minimum window
  mean disclose coverage. Missing evidence has `missing.*` records.
- `tokens_caps.role` carries calls/input tokens/output tokens/prompt cap hits/
  response cap hits. `cap_rates.role` carries both hit rates. Null counts denote
  unavailable legacy evidence. Training counts refer to joint/target caps.
- Missing telemetry and serving costs remain UNMEASURED. Hash records bind
  cohorts, recipes, models, prompts, environments, vectors and budget ledgers.

The owner includes tagged TRACE in the v6b-reading message. If its tag is still
pending when the reading arrives, TRACE follows in the next send. Inventory
TRACE as three descriptive comparisons.

### Frozen S8 general panels (kit-message-v7i)

The kit supplies the six general metric authorities from OpenCompass **0.1.5**,
commit `9b21613d171ccac154530a3c7bbeba6743b469f4` (22 Sep 2023). Byte-identical
source, Apache-2.0 LICENSE and per-file hashes are in
`resources/trace/vendor/opencompass-0.1.5/`. When `s8` is selected, bind refuses MMLU, BBH, GSM8K,
TyDiQA, PIQA or BoolQ in `operator.metric_authorities`. Other study scopes
ignore these unused keys with a warning naming them. The eight TRACE task
authorities remain operator-supplied. The bridge executes the pinned evaluators
and postprocessors unchanged, with scoped package glue and local HF accuracy.
It restores import state on success and failure. It downloads nothing. General accuracy requires the real NumPy and
scikit-learn libraries in the staged metric environment; model scoring uses
the node's existing torch/transformers stack. No package installation occurs
during build or scoring.

Build `trace_general` on CPU from a JSON manifest using:

```sh
python3 -I kit/trace_general_build.py --manifest general-inputs.json --out trace-general.json
```

Set manifest `schema` to `trace-general-inputs.v1`. Its `datasets` object must
have exactly MMLU, BBH, GSM8K, TyDiQA, PIQA and BoolQ. Each panel has an explicit
nonempty dataset `revision` and `files`, a dictionary from each required logical
filename below to `{"path": "/staged/file", "sha256": "<64 hex digits>"}`.
Paths may differ from logical filenames; hashes must already be supplied.
Merge the emitted object into `panels['trace_general']` alongside the unchanged
`general-v1` and `open`, then hash-bind the complete panels JSON as usual.
All raw data must already be staged under applicable grants. No dataset bytes
are included in the kit. File hashes, revisions, per-file and subset counts,
config hashes and fixed shot IDs are recorded in the built artifact.

| Panel | Exact logical files and raw schema |
|---|---|
| MMLU | `dev/<subject>_dev.csv` and `test/<subject>_test.csv` for all 57 `mmlu_all_sets` in `configs/datasets/mmlu/mmlu_ppl_ac766d.py`; headerless UTF-8 CSV: question, A, B, C, D, target label. Dev rows 0–4, in order, are the fixed five shots. |
| BBH | `<task>.json` for all 27 tasks in `bbh_gen_5b92b0.py`'s two task lists; object `{"examples": [{"input": "...", "target": "..."}]}`. The three CoT examples come verbatim from the corresponding vendored `bbh/lib_prompt/<task>.txt`. |
| GSM8K | `test.jsonl`: HF `gsm8k/main` test rows with `question` and original `answer` containing `#### `. The pinned dataset postprocessor removes the prefix and commas; the pinned config supplies its four fixed demonstration rounds. |
| TyDiQA | `<language>/validation.jsonl`, one per all 11 `langs` in `tydiqa_gen_978d2a.py`: HF gold-passage validation rows with `passage_text`, `question_text`, `answers: {"text": ["gold", "alternative"]}`. Golds remain nonempty lists; 0-shot. |
| PIQA | `validation.jsonl`: HF validation rows with `goal`, `sol1`, `sol2`, integer `label` 0 or 1; 0-shot. |
| BoolQ | `val.jsonl`: the pinned SuperGLUE BoolQ val export, with `passage`, `question`, and string `label` (`"true"` or `"false"`). The builder requires exactly these two strings, maps them to 1/0, and refuses JSON booleans, integers and other values with a clear error. 0-shot. |

`trace_general_build.expected_files(panel)` returns the complete exact logical
file inventory. Configs are read as AST data, never imported by the builder;
prompt words and exemplars come from the vendored files. Preparation rejects
legacy general shapes, missing subsets, wrong rendering modes and changed caps.

Rendering is frozen: plain-string templates are **raw text**, without a chat
template. HUMAN/BOT rounds become user/assistant messages using Qwen3's native
chat template with thinking disabled. For PPL the label is the assistant turn,
and the entire rendered sequence is scored. MMLU and PIQA use raw text; BoolQ
uses rounds. For each candidate the sequential bf16 model supplies float32
logits. The pinned `_get_ppl` reduction sums shifted token cross-entropies,
ignores pad targets and divides by the full non-pad input length (including the
initial token), with no answer-only mask or `normalizing_str`. Tokenization
matches upstream `get_logits` (default special tokens, max length 2048 and the
tokenizer's truncation side). In label order, MMLU drops the LAST dev example
while `len(tokenizer.encode(rendered_prompt)) > 2048` and shots remain. The
reduced shot list carries forward across labels. Each row records the shot IDs
used per label and residual truncation after all shots are exhausted. First
minimum wins ties. Full-precision repeat tolerance is at most 1e-6 (registration H).

Generation is greedy, temperature 0: GSM8K 512, BBH 512 and TyDiQA 50 new tokens.
The prompt cap is 2048 with existing left truncation. No extra stop strings are
introduced. Prompt-cap hits, truncated token counts and response-cap hits are
recorded; truncation warns and continues. PPL records per-label losses and
its cap/truncation counts too. Existing TRACE-task and general-v1 rendering
and decoding retain their prior behaviour.

MMLU/PIQA/BoolQ use the pinned AccEvaluator on the selected label; GSM8K uses
its pinned prediction extractor and AccEvaluator on build-time-processed gold;
BBH uses MCQ postprocessing on prediction and gold for its 17 MCQ tasks and
BBHEvaluator for its 10 free-form tasks; TyDiQA uses TydiQAEvaluator F1 over a
list of golds. The primary panel score is the naive mean of subset scores:
57 MMLU subjects, 27 BBH tasks, 11 TyDiQA languages; the other panels each have
one subset. Per-subset scores and pooled item mean are retained, plus the
pinned `mmlu-weighted` group descriptively. The primary `sum/n` represents
subset means so the existing TRACE return reconstructs the same panel score.

The six general fixture sets exercise their real kit scorers (at least twenty
each), including label-selection ties and both BBH evaluator branches. On the
manager side, run the independent upstream check offline, after copying out
of the read-only references tree (the script does the copying):

```sh
python3 -I scripts/trace_general_upstream_verify.py references/opencompass-0.1.5-9b21613d general-verification.json
```

It runs every fixture and additional extraction/tie cases, upstream template /
retriever prompt comparisons, and CPU fake-logit loss parity in a separate
`python -I` worker. Native bf16 GPU behaviour and repeatability still require
the node's qualification; CPU parity is not a GPU qualification receipt.


### Calibration rater JSON and configuration

The node config's `raters` file reference points to a JSON object keyed exactly
`"gpt-6-astra"` and `"Claude Opus"`. Each value maps every sealed pair ID to a
label in -1, 0, 1. To use another declared pair, set the top-level config key
`"calibration_raters": ["FIRST", "SECOND"]` and use those exact keys in the
raters JSON. Exactly two distinct names are required and their order is kept.
The default is `["gpt-6-astra", "Claude Opus"]`. Packet filenames are
`gpt-6-astra-high.json` and `claude-opus-high.json`, with the original rater
names preserved inside each packet. `J_status` includes the pending reason,
including mismatched rater keys or an absent raters JSON.

The manager can count MMLU shot-dropping on staged rows with a local Qwen3
tokenizer directory. This CPU command downloads nothing.

```sh
python3 -I scripts/trace_general_mmlu_census.py --panels staged-general.json --tokenizer /local/Qwen3-tokenizer
```

`--panels` accepts the general builder output or `prepared.json`. The report
counts items needing shot-dropping and items remaining over 2048, both overall
and per subject. Rebuild older staged MMLU rows to include the shot-rendering
metadata used by the dropping loop.
