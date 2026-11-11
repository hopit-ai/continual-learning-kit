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
TyDiQA, PIQA and BoolQ. Each has `rows`, `prompt_cap`, `response_cap` and
`temperature`, with frozen shot messages in each row. Open rows have `id`,
`panel` and `prompt`, with 175 Self-Instruct, 300 LIMA and 178 CoNa rows.

Calibration rows have opaque `id`, `panel`, `prompt`, `a`, `b` and
`checkpoint_independent: true`. Seal 30 per open panel. Keep source/model
identifiers outside answer text. Use `trace_calibration.py --pairs pairs.json
--out packets` on our side to create independent Astra/Fable high packets.
That command expects a JSON array converted from the sealed calibration JSONL.
Only ID/messages go to the raters. Keep reversal mapping locally. Opus
adjudicates disagreements after the original independent labels are retained.
The local judge labels both orders. The numeric return includes its labels.
The paper states calibration against frontier-model raters. Human-rater count
is zero. Exact Fable/Opus backend and paid-token authority bind before calls.

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

These local qualification command manifests, metric fixture bindings and node
assets remain to be supplied. They are visible readiness dependencies. The
CPU rehearsal validates their interface using stand-ins. Production numerical
qualification runs on the node before its scientific descendants.

## Run with operator-owned sbatch

The inventory prints allocation rows, GPU counts and dependency row IDs.
Use `inventory --config local-config.json` to print the selected study scope.
Submit those plain rows with your own sbatch and afterok dependencies. The
CPU bind precedes all GPU qualifications. Qualification flows directly into
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
all general-v1 states and local judging. CPU binding precedes GPU rows.

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
