# A2: J(b), P and R

Build approval: owner, 8 November 2026, “3 Yes”. Operational departure and
selected-arm spending authority: owner, 10 November 2026, “Decisions 1, 3 and 7:
approve as recommended”. Bind that dated reference and a cap at most 601 GPU-h
in the selected arm’s configuration before its first row. It never submits jobs, downloads inputs or reads a 4A
outcome. Only the branch selected later by A2's registered 4A reading is sent.
J(a) routing is separately governed and is not implemented here.

The scientific specification is [the frozen registration](resources/a2/registrations.md),
SHA256 `276f4f6e04c3d0c228a47e04f80fa1d8de22335a5af5aee1ee68dabab98bf789`.
Paths, hashes, verification receipts and pricing are operational bindings only.
The existing a4 and nm files are imported without modification.

`resources/a2/J-rows.json`, `P-rows.json`, and `R-rows.json` are independent
inventories. `RP-rows.json` implements the prospectively required shared fresh
control when R and P are dispatched together. It contains three trajectories,
never a combined R+P intervention. Its shared control is charged once to study
expenditure and fully to each standalone comparison. Each row is one allocation.

The manager supplies a durable JSON configuration and binds its exact file
hash in the send record. Use fixed absolute paths across jobs:

```json
{
  "arm": "R",
  "disposition": "active",
  "out": "/durable/a2/R",
  "inputs": {"path": "/durable/bindings/a2-inputs.json", "sha256": "FULL_SHA256"},
  "archive": "/durable/a2/R-return.tar.gz",
  "text_out": "/durable/a2/R-text",
  "sdpo": "/durable/software/sdpo",
  "tokenizer": "/durable/models/Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218",
  "trainer_python": "/durable/envs/training/bin/python",
  "inference_python": "/durable/envs/inference/bin/python",
  "job_python": "/durable/envs/training/bin/python",
  "stage_max_seconds": 14400,
  "score_max_seconds": 600,
  "connected": {"path": "/durable/bindings/connected.json", "sha256": "FULL_SHA256"},
  "i0": {"path": "/durable/bindings/i0.json", "sha256": "FULL_SHA256"},
  "budget": {
    "approved_gpu_hours": 601,
    "approval_reference": "Owner, 10 November 2026: Decisions 1, 3 and 7: approve as recommended"
  },
  "allow_confirmation": false
}
```

This is a complete configuration shape for initial binding, with placeholders to
replace with real paths, hashes and approval. The 601 GPU-h maximum applies only to the single arm selected by the v6b
reading; it is not an allowance for each arm or a public release authorization. `connected` is required for J; `i0` for P and RP; R ignores both.
`budget` is required before real GPU work. `inference_environment` belongs in the
inputs document shown below, rather than this configuration. Optional `kit_commit`
is the full 40-character kit Git identity; otherwise local HEAD is read. Real runs
set `technical_synthetic` to false (the default); `fail_task` is a CPU rehearsal
failure-injection setting and must be absent on real runs. `arm` is J, P, R or RP;
`disposition` is active or archived. `out`, `archive`, `text_out`, `sdpo`,
`tokenizer`, `trainer_python`, `inference_python`, `job_python`,
`stage_max_seconds` and `score_max_seconds` are configuration keys, with the
locations/units shown above. `inputs` is a hash-bound file reference.

The limits are scheduling allowances, not measured runtimes or approved budgets.
Generate the actual rows with the configured limits before send. Supply the row's
exact `nodes`, `gpus`, `cpus`, `time`, and `depends_on` to your own sbatch.
Dependencies include `--kill-on-invalid-dep=yes`; resubmit dependants against a
new upstream job ID after a failed/PENDING attempt. Never await inputs on a GPU.

```bash
export KIT=/durable/kit A2_CONFIG=/durable/bindings/R-config.json
export JOB_PY=/durable/envs/training/bin/python
"$JOB_PY" -I "$KIT/a2_jobs.py" rows --config "$A2_CONFIG" --out /durable/R-rows.json
bash "$KIT/jobs/a2/run-task.sh" A2-R-bind
# Execute each remaining selected row in its own allocation with its resources.
```

CPU rows use zero GPUs and 8 CPUs. Native stages use 8 GPUs and 32 CPUs, with
fresh optimizer/scheduler and incoming-student EMA. Generation uses one GPU and
8 CPUs. Full comparison scoring is one allocation on one physical GPU, with
one allocation nonce and environment identity for all seeds, both systems and
references. The frozen inherited rule requires a complete comparison in one allocation;
per-seed or per-batch comparison splitting would change that rule. At 600 s
per command the J/P/R comparison is 52:17 and RP is 77:02; RP confirmation is
28:05. Inventories list these under `long_lanes`. **Submit them**: they fit the
captured site limit `MaxTime=7-00:00:00` (`SITE_MINUTES` 10080). They are long
single-GPU serial lanes, so watch the 70% utilisation rule (warn-only on our
side) and tell us in Part 3 if one is evicted. Only a row whose ceiling exceeds
the site MaxTime is listed under `scheduling_conflicts`; at the captured limit
there are none. Set `SITE_MINUTES` if your limit differs.
Standalone confirmation is 18:43. Rewrite rows use the actual hash-bound monthly
admitted trace count when emitted with `--config`, and split into command batches
as necessary so every rewrite ceiling is at most 24 h. The checked-in inventories
use the frozen supply ceiling of 80 questions/month (14:42 for rewrite at 600 s).
Do not use their ceiling counts as measured admissions. Serial model loads are
included in the unqualified per-command allowance; throughput/utilisation still
needs node qualification. Context building,
data conversion, Hydra resolution and exports are CPU rows. Each allocation has
one utilisation watcher and nonblocking GPU leases; low utilisation warns only.
PENDING exits 75, invalid evidence exits 2. Completed artifact hashes must verify
before a row is skipped. Failed native states remain in `partials/`; restart the
same scientific seed from update zero. A native step-40 completion is exported
without retraining. No best checkpoint or additional scientific attempt exists.

Input schemas are executable in `a2_bind.py`; `a2_sim.py` provides a complete
synthetic example. All references are `{path: absolute_path, sha256: full_hash}`.

- `inputs` points to `a2-node-input.v1`, with `technical_synthetic: false`,
  `base`, `audit`, `general`, `inference_environment`, and six-entry `slice`, `ledger`, `contexts` maps.
  Slice files must match 4A's frozen audited batch hashes and total 395/92.
  The audit binds those hashes, both families and January–June only. `general`
  is the unchanged general-v1 JSONL. Ledger and context references name each
  directory's `manifest.json`; every member is verified before use. Ledger
  admission is rechecked, without opening any score or 4A reader. The contexts
  are the exact archived memory-off AND memory-on arrival records from 4A,
  including recorded `recurrence`, `stratum` and `recurring_issuer`. Both modes
  are copied unchanged for every arm, never rebuilt. Missing arrival contexts
  are PENDING for every arm, including non-holdouts. No recollection of the
  frozen raw ledger is permitted. Shared k=3 memory always uses that ledger.
- J additionally supplies `connected`, pointing to a VERIFIED continuous
  seed-101/102/103 lineage receipt with core recipe/stacking dispositions,
  every stage's parent/final, acquisition, effective configuration, data and
  schedule manifests and student export. It binds the original base and common
  terminal SEC start, exclusions against January–June and sealed confirmation
  manifests, and any July–September exposure. Missing/unresolved lineage is
  PENDING; reset or stitched ancestry is invalid. Current collection uses each
  seed's latest completed student, never EMA, fixed for the whole month.
- P additionally supplies `i0`, pointing to verified I0 terminal ContractNLI
  source receipts, pool, stored targets, draw rules, overlap receipt and seed
  maps. Each map contains exactly 40 `slots` lists (1-based positions), its
  verbatim realised `accepted_order` including cycling, and `skipped_draws`.
  Only passed-over exclusions are permitted. An insufficient copied ordering
  is PENDING, with no new shuffle, draw, pool or target generation. The same
  seed map repeats at months 2, 4 and 6. The ordinary map gives 1120 SEC and
  160 generic positions per stage, inside batch 32. CR never replaces I0.
- R passes only archived prompt text, the admitted raw response and admission
  receipt across the rewrite boundary. Hidden item fields never reach its
  parser, model or verifier. Its exact prompt lives in `rewrite-system.txt`.
  Every parse, length, generation or verification failure retains the raw
  response at the same position. Failures are counted and never retried.

Qualify then proceed in this one partner message. Bind `budget`,
`approved_gpu_hours` (positive and <=601), and `approval_reference` up front.
Leave timing and price refs out of the config: no post-seal operational edits
are permitted. Only future `allow_confirmation` and `confirmation` may change.
The generated inventory identifies the exact `qualification` set. Run its CPU
bind/context/preparation/feedback rows and January–February arrival scoring
(including J collection), plus R/RP rewriting, then each system's first real
seed-101 month-2 training stage. These are registered study work, retained for
normal export and later months; no seed, order, schedule or setting changes.
Do not allocate GPUs while awaiting upstream rows. Submit each row with its
emitted dependencies; an allocation checks readiness immediately and exits.

The next row is CPU-only `A2-ARM-price`, immediately after those first stages
and before any other row. It reconciles actual allocation ends, writes measured
8-GPU stage timing receipts under `out/qualification/`, and computes the full
selected-arm price. It prices nine stages per system (RP's shared fresh control
once), all chronological and complete-comparison calls, J collection, actual
admitted rewrite counts, serial loading, failures/retries, and dormant future
confirmation. Preparation, export and transfer rows use zero GPUs; their GPU
idle/failure time is covered by the registered 80 GPU-h overhead. The measured
per-command allocation rate includes model loading; the inherited scoring floor
and loading/failure reserves remain. Training uses the registered 1.5 multiplier
and rounding to 50 before separately priced confirmation. Qualification is
already inside those workloads. The total cannot be below actual qualification
spend plus the measured baseline for remaining work. These are conservative
measured forecasts, not guarantees that later stages have identical runtimes.

`price.json` and `price-ref.json` are hash-bound row artifacts. Timing refs,
source allocation receipts, effective setups, preparation manifests, config,
protocol and approval are bound and rechecked by the gate. Never edit them or
add `challenger_timing` after sealing. An optional initial `D_timing` may reuse
existing **4A D** 8-GPU evidence: a hash-bound timing JSON with `origin: 4A`,
`qualified_8b: true`, `gpus: 8`, positive allocation `seconds`, matching
`technical_synthetic` (false in production), exact `setup`, and file refs
`source`, `effective_setup`, `prepared`, `effective_config`, `schedule`.
`source` is a COMPLETE allocation receipt with actual start/end and GPU-hours;
`effective_setup` and `effective_config` are JSON descriptors matching the
A2 stage's `timing-setup.json` exactly: D recipe, full 40 updates, batch 32,
LR 1e-5, warmup 10, fresh optimizer/incoming EMA, model revision, trainer
software, eight GPU hardware descriptions, normalized full effective command
and inherited recipe hashes. Preserve the original native configuration beside
this verified descriptor. The original `schedule` is 1280 JSONL positions with
no generic replay. Missing/mismatching evidence refuses; it never silently
falls back to a different receipt. The first A2 D stage is still registered
work and supplies an exact setup comparison, even when 4A timing is reused.
Pricing uses the larger of the A2 and matching 4A per-stage timings; the price
artifact records both timings, their references and the selected source.
Without `D_timing`, its allocations measure D just like the challenger.
First-stage timing includes all failed/recovery attempts; an export-only
recovery cannot supply a misleading near-zero training rate. RP
measures both P and R challengers independently. If every I0 map has zero
generic positions, P stays NOT APPLICABLE; configured P rows request no GPUs,
and RP qualifies/prices the shared control and R only, with no P training cost.

Every GPU allocation, including qualification, failures, retries and idle time,
uses the existing `a2_budget.reserve` ledger and the **same** owner cap. With
14400-second stages, 600-second command bounds and supply-ceiling rewrite
counts, the successful prefix reservations (including 30-second handoff per
GPU row, excluding already elapsed allocation startup) are J **69.183333**,
P **68.183333**, R **94.900000**, RP **128.966667 GPU-h**. Configured R/RP rows
use actual admitted counts and may reserve less. These are scheduling bounds;
there is no extra qualification allocation or separate budget. Retries debit
the same cap; each reserve also counts elapsed startup and concurrent holds.

If measured price <= cap, dependants proceed automatically with no manager or
partner approval step. If it exceeds the cap, every later scientific row
returns **NotReady / PENDING (75): owner escalation**, giving full price, cap
and qualification spend. Do not change the sealed cap. Run the CPU collector
`A2-ARM-collect` even after that stop (its dependencies use `afterany`). It
returns the computed price, qualification spend and allocation accounting in
both the standalone archive and readable/pasteable text. If qualification has
not completed, it reports NOT PRICED and the spend so far, then exits PENDING
(75) so collect stays unsealed. Only a completed OWNER ESCALATION price seals
the qualification-only return. After recovery, rerun collect to return the real
reading. Preserve all durable attempts, including failed/idle allocations;
resubmit failed rows/dependants
with fresh scheduler IDs. Actual scheduler ends, never scheduled EndTime, are
required for measured pricing and accounting; retry only the CPU price row if
ends are still pending. Queue gaps cost zero GPU-h. Build approval and 4A's
funding remain separate. The frozen registration's before-send pricing language
is superseded only for this operational sequence by the dated owner departure.

The January–June reading is exploratory. Bars are mean >= +2, positive 3/3,
two-sided paired df=2 t=4.302653 lower bound >0; every individual holdout/memory
mode and IFEval/knowledge/maths guard needs lower bound >= -1. No pooled general
panel substitutes. Granularity and paired answer transitions are printed from
integer outcomes. 4A's +5/3-point package rules remain contextual and unchanged.
All complete comparisons and references are remeasured together; mixed physical
UUID/allocation/environment is UNVERIFIED-HARDWARE. No missing score is zero.
Invalid provenance is UNVERIFIED; missing bindings/references are PENDING;
unfinished score matrices are INCOMPLETE; complete evidence missing a bar is
INCONCLUSIVE. Utilisation, rewrite yield and cost create no efficacy gates.

The confirmation rows are dormant. They return PENDING without opening
October–November. After the separately governed 15–31 December build and audit,
an operational configuration may add `allow_confirmation: true` and a hash-bound
`confirmation` receipt. Those are the only mutable configuration additions.
It must bind the already frozen June systems, the 14 December cutoff, disjoint
source audit/exclusions/release controls and October/November batch files.
Each month needs 40 questions, 10 holdouts and five holdout components. Score
fixed June weights and raw June memory, with no feedback or training. Terminal
guards rescore all January–June archived contexts and all three general panels.
Exploratory failure never gates confirmation. Confirmation is analysed alone.
At the holdout floor, item granularity is 10 points monthly and 5 in the mean;
the protocol warns that “a +2 bar is effectively unreachable.” No endpoint change
is allowed. Actual confirmation data is never read during the CPU rehearsal.

Each collector produces a standalone archive plus `RETURN-TEXT-*-of-*.md` parts.
Paste every part from the selected arm/phase verbatim. Lines are ASCII and under
90 characters. Import with:

```bash
python -I kit/a2_return.py --arm R --text /path/RETURN-TEXT-*.md --out tmp/R-reading.json
python -I kit/a2_return.py --arm R --archive /path/R-return.tar.gz --out tmp/R-archive.json
```

Readers recompute all bars from exact item counts. Returns remain PARTNER-COMPUTED;
independent reverification is UNVERIFIED. Text checksums are advisory and never
turn a transcription into an independently verified measurement. The archive
contains CPU reader evidence; bulk checkpoints, native states, raw answers,
schedules, rewrite receipts and contexts stay durable with a full hash manifest.
Preserve them. An unselected arm uses `disposition: archived`; every row returns
ARCHIVED UNREAD without opening its node paths or allocating a GPU.

CPU-only rehearsals use the real shell, watcher, allocation and child dispatch:

```bash
python -I kit/a2_dryrun.py --arm J --out tmp/a2-J
python -I kit/a2_dryrun.py --arm P --out tmp/a2-P
python -I kit/a2_dryrun.py --arm R --out tmp/a2-R
python -I -m pytest -o pythonpath=. tests/test_a2_arms.py tests/test_a2_rewrite.py \
  tests/test_a2_review.py tests/test_a2_sec_prompts.py \
  tests/test_a4_kit.py tests/test_a4_v6b.py -q --basetemp tmp/a2-tests
```

Fixtures label all evidence TECHNICAL-ONLY, including simulated future release.
They exercise every row, missing input -> PENDING, failed training/resubmit,
unreleased confirmation -> PENDING, immutable completion, standalone archive,
text import, and every archived row -> ARCHIVED UNREAD. They claim no scientific
result, GPU qualification or measured SEC throughput.

Named standalone CPU importers are `a2_j_read.py`, `a2_p_read.py` and
`a2_r_read.py`; they accept the same `--archive`/`--text` and `--out` arguments.
`data/*/teaching-contexts.jsonl` preserves the actual unique D demonstrations,
including raw fallbacks, beside each immutable position schedule. The rehearsal
also covers RP's 141 rows and verifies its one shared control and two contrasts.

Pasted parts include readable D/E decisions and intervals. Their repeated exact
item counts and membership receipts are packed as deterministic gzip/base64
T records, so the importer can recompute the bars without hundreds of paste
parts. All parts are still ASCII with lines under 90 characters. Full native
metric journals remain in the archive/durable evidence; text reports their
numeric means. Transport checksums remain advisory; malformed or missing
structural records are incomplete evidence.

The exported kit contains the registration verbatim. Its relative links to the
plan and review files refer to repository documents absent from a kit-only
export. Read this bundled registration as the authority; no link target is
needed to execute the binder. The cost forecast likewise identifies historical
repository sources that are absent from that export.

Owner input inventory (all file references use `path`, an absolute file path,
and `sha256`, the full SHA256; model directories use their pinned identity files):

| Document | Every consumed field / file |
|---|---|
| Main input | `schema` = a2-node-input.v1; `technical_synthetic`; `base` = local frozen revision; `slice`, `ledger`, `contexts` keyed by each month 2026-01 through 2026-06; `audit`; `general`; `inference_environment`. The latter is the inference interpreter's exact `runtime_versions()` dictionary (including an empty dictionary if that is what the environment reports), bound as a file. |
| Audit JSON | `audited` true; `months` January–June; `families` [fcf_margin_change,cash_conversion_change]; `slice_hashes` for every monthly batch. |
| Each slice JSON | Immutable audited SEC items, unique `id`; `sec` includes `template`, `accepted`, `accession`, `issuer_holdout`, source/exclusion/issuer component and required-filing identities; the unchanged 4A source/unit admission schema, `qa`, `pre_text`, `table`, `post_text`, `filename`. Full bundled source items pass through unchanged to inherited admission/scorers; never supply reconstructed minimal items. |
| Each ledger manifest | `manifest.json` with `files` mapping member names to hashes. Must include `traces.jsonl`, `admission.jsonl`, `sealed.json`; all other manifest members are hash-checked too. Each trace reads `id`, `item` (exact slice item), `month`, `response`, `response_sha256`, `collector_checkpoint` with `model_file_hashes`, `provenance` with `unedited` true. Responses are admitted again by the unchanged 4A checker. The whole-month seal and verifier provenance must be preserved. |
| Each contexts manifest | `manifest.json` plus every listed member, including `prompts.jsonl`. Each record contains `id`, `month`, `off`, `on`, `recurrence`, `stratum`, `recurring_issuer`. Each mode retains `prompt`, `case_ids`, `cases_dropped`, `input_tokens`. No alternative prompt generation is allowed for January–June. Binding records each verified manifest and JSONL hash plus independent per-item memory-off prompt hashes. |
| General JSONL | Unchanged `panels/general-v1.jsonl`; `id`, `panel`, `prompt` and all inherited scorer-specific membership/answer fields. Separate ifeval, knowledge and math panels. |
| `connected` JSON, J | `status` VERIFIED, `seeds` [101,102,103], `continuous` true; hash-bound `recipe_dispositions`, `exclusions`; `lineages` by seed, each with `base`, `stages`, `terminal`, `complete_registered_sequence` true. Every stage has `parent`, `final`, `seed`, `reset` false and file references `acquisition`, `effective_config`, `data_manifest`, `schedule_manifest`, `export`. Each final's local model identity must verify against its export receipt. |
| Connected exclusions JSON | `jan_jun_overlap` [], `oct_nov_overlap` [], `sealed_confirmation_manifest_sha256` nonempty, `jul_sep_exposure` present (disclosed). Recipe dispositions and stage evidence files are read/hashed, never a core outcome reader. |
| `i0` JSON, P/RP | `verified` true; `source` I0; `stage` terminal-contractnli; `seeds` [101,102,103]; file references `pool`, `stored_targets`, `draw_rules`, `overlap_receipt`, `maps` by seed. |
| Replay map JSON | `seed`, 40 `slots` lists of distinct integer positions 1–32, `accepted_order` (including inherited cycling), `skipped_draws`. Pool JSONL reads `id`, `prompt`, `response`; stored-target JSON maps IDs to exact response strings. Overlap receipt reads `excluded_ids`, matched against actual SEC holdout/general-v1 overlaps; no new draws are invented. |
| `budget` config object | `approval_reference`; positive `approved_gpu_hours` <=601; optional initial `D_timing` for matching 4A reuse. `challenger_timing` and price refs are generated artifacts, not configuration changes. See bootstrap instructions above. |
| Optional December config | `allow_confirmation` true plus `confirmation` file reference only after release. Receipt `schema` a2-confirmation.v1, `cutoff` 2026-12-14, `build_date` 2026-12-15 through 2026-12-31, `frozen_june_sha256`, `audit`, `slice` October/November references. Audit JSON: `source_disjoint`, `fixed_exclusions`, `audit_verified`, `release_controls`, all true. Confirmation items retain the full SEC schema including `issuer_group`. |

A complete main-input shape (replace all placeholder hashes and paths):

```json
{
  "schema": "a2-node-input.v1",
  "technical_synthetic": false,
  "base": "/durable/models/Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218",
  "audit": {"path": "/durable/4a/audit.json", "sha256": "FULL_SHA256"},
  "general": {"path": "/durable/kit/panels/general-v1.jsonl", "sha256": "FULL_SHA256"},
  "inference_environment": {"path": "/durable/bindings/inference-environment.json", "sha256": "FULL_SHA256"},
  "slice": {
    "2026-01": {"path": "/durable/4a/slice/batch-2026-01.json", "sha256": "FULL_SHA256"},
    "2026-02": {"path": "/durable/4a/slice/batch-2026-02.json", "sha256": "FULL_SHA256"},
    "2026-03": {"path": "/durable/4a/slice/batch-2026-03.json", "sha256": "FULL_SHA256"},
    "2026-04": {"path": "/durable/4a/slice/batch-2026-04.json", "sha256": "FULL_SHA256"},
    "2026-05": {"path": "/durable/4a/slice/batch-2026-05.json", "sha256": "FULL_SHA256"},
    "2026-06": {"path": "/durable/4a/slice/batch-2026-06.json", "sha256": "FULL_SHA256"}
  },
  "ledger": {
    "2026-01": {"path": "/durable/4a/ledger/2026-01/manifest.json", "sha256": "FULL_SHA256"},
    "2026-02": {"path": "/durable/4a/ledger/2026-02/manifest.json", "sha256": "FULL_SHA256"},
    "2026-03": {"path": "/durable/4a/ledger/2026-03/manifest.json", "sha256": "FULL_SHA256"},
    "2026-04": {"path": "/durable/4a/ledger/2026-04/manifest.json", "sha256": "FULL_SHA256"},
    "2026-05": {"path": "/durable/4a/ledger/2026-05/manifest.json", "sha256": "FULL_SHA256"},
    "2026-06": {"path": "/durable/4a/ledger/2026-06/manifest.json", "sha256": "FULL_SHA256"}
  },
  "contexts": {
    "2026-01": {"path": "/durable/4a/contexts/2026-01/manifest.json", "sha256": "FULL_SHA256"},
    "2026-02": {"path": "/durable/4a/contexts/2026-02/manifest.json", "sha256": "FULL_SHA256"},
    "2026-03": {"path": "/durable/4a/contexts/2026-03/manifest.json", "sha256": "FULL_SHA256"},
    "2026-04": {"path": "/durable/4a/contexts/2026-04/manifest.json", "sha256": "FULL_SHA256"},
    "2026-05": {"path": "/durable/4a/contexts/2026-05/manifest.json", "sha256": "FULL_SHA256"},
    "2026-06": {"path": "/durable/4a/contexts/2026-06/manifest.json", "sha256": "FULL_SHA256"}
  }
}
```

The binder's own outputs `protocol` and `software` bind implementation and runtime
identities; they are not extra owner configuration. Model directories are read
locally by the inherited identity verifier, tokenizer, trainer, export verifier
and scorers; configuration/tokenizer/weight files must already be installed.
SDPO's checkout is read at its pinned Git revision. No remote fallback exists.
The wrapper reads scheduler allocation start/end receipts (`scontrol`/`sacct`),
`SLURM_JOB_ID`, `SLURM_JOB_START_TIME`, measured GPU UUIDs (`nvidia-smi`),
`CUDA_VISIBLE_DEVICES`, inherited lease descriptors and watcher state.

Generated files subsequently read are: `config.json`, `binding.json`, per-row
`state/*.json`, `attempts/*/*/receipt.json` and hashed artifacts; context
`prompts.jsonl`/`manifest.json`; score `prepared.json`, `cells.json`, `raw/*.jsonl`;
whole-month `seals/*.json`; current-ledger `traces.jsonl`, `admission.jsonl`,
`manifest.json`; rewrite `rewrite-inputs.json`, cache/index JSON and `.started.json`
markers; training `data/*/{schedule.jsonl,teaching-contexts.jsonl,prepared.json,
train.parquet,val.parquet,test.parquet,manifest.json}`, `runs/*/{trained.json,
export-40.json,summary.json,manifest.json,started.json,resolved-config.yaml,
resolved-config.err}`, native `sdft/global_step_40/actor`, HF `hf-step40`, metric
and token journals; `frozen-june.json`, `confirmation-binding.json`, readings,
standalone return tarballs, and every `RETURN-TEXT-*-of-*.md` part. Partials stay
for audit. Native model/scorer files and every member of a supplied manifest are
also read under the inherited implementation hash inventory. No 4A outcome file
is an input.

Rewrite diagnostics distinguish generation acceptance from actual stage use.
A stage length failure retains the raw demonstration at its scheduled position,
counts unique affected IDs and repeated fallback positions, and reduces the
reported effective accepted count. A missing cache is PENDING. Engine setup
failure creates no one-attempt marker; the marker is written just before model
generation, so an interrupted actual generation cannot be retried. Corruption of
an entire context archive fails provenance validation before scoring; a missing
or mismatched prompt at the rewrite boundary is a counted raw fallback against
its independently recorded manifest binding.

Inherited implementation/resource files read or hash-checked (relative to kit):

`a4_allocation.py`, `a4_bind.py`, `a4_collect.py`, `a4_common.py`, `a4_dryrun.py`,
`a4_handoff.py`, `a4_jobs.py`, `a4_lane.py`, `a4_ledger.py`, `a4_memory.py`,
`a4_nll.py`, `a4_read.py`, `a4_reward.py`, `a4_run.py`, `a4_score.py`,
`a4_sdft_entry.py`, `a4_sdft_runner.py`, `a4_sec.py`, `a4_sim.py`, `a4_train.py`,
`a4_payload.sh`, `README-4a.md`, `resources/a4/slice.json`,
`resources/a4/registration.md`, `resources/a4/SHA256SUMS`,
`resources/a4/amendment-20261102.md`, `resources/a4/v6b-rows.json`, `README-4a-v6b.md`,
`jobs/a4/run-task.sh`, `resources/p3b/v4-recipes.json`, `beds/finqa.py`, `eval_bed.py`,
`score_forgetting.py`, `scorers.py`, `canonical.py`, `c_train.py`, `c_data.py`,
`c_sdft_entry.py`, `c_sdft_runner.py`, `p3b_loss.py`, `sdft_objective.py`,
`sdft_actor.py`, `sdft_verl.py`, `sft_entry.py`, `p3b_train.py`, `p3b_common.py`,
`p3a_common.py`, `lane_pool.py`, `util_watch.py`, `m3_manifest.py`, `m2_gpu.py`,
`inference_lifecycle.py`, `v6_infra.py`, `v4_teacher.py`, `v4_sft.py`, `v4_contract.py`,
`v4_state.py`, `v4_timing.py`, `v4_phase0_timing.py`, `p3b_reward.py`, `p3b_read.py`,
`next_w.py`, `c_bind.py`, `resources/p3b/model-hashes.json`, `panels/general-v1.jsonl`

A2 additionally reads or hash-checks all `a2_*.py`, `README-a2.md`, every file
in `jobs/a2/` and every file in `resources/a2/`, including each emitted inventory,
the bundled registration, forecast and exact `rewrite-system.txt`.
