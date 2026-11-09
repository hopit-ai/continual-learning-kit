# Message after return 2

The six v3 registrations are preserved byte for byte in
[resources/nm/registrations](resources/nm/registrations/README.md), including the
2 November owner approval, “Approve all six + allocation”. Their precedence and
thresholds are frozen. This implementation grants no tag or send approval.

Use the standalone NM wrapper on the integrated v6 tree. It does not submit jobs,
contact a service or open challenger outcomes to select a recipe. All new code
uses `nm_` names for the six rows and isolated `cr_` modules for confirmation;
the shared text reporter assembles all seven. Exploratory reader behavior and
scientific objectives remain unchanged. This unit does not regenerate the
infrastructure authority.
The registration copies retain their original relative
source citations as archival text; the implementation does not resolve those
links or fetch data.

## Rows and allocation

[rows.json](resources/nm/rows.json) contains 444 explicit, topologically ordered
rows: 260 CPU, 131 one-GPU and 53 eight-GPU rows. All request one node. CPU and
inference may share a node; training owns its eight allocated GPUs. Verified 27B
demonstrations are prerequisites, so this inventory requests no additional TP2
teacher. Missing demonstrations stay PENDING and require their separately priced
prerequisite rather than adding an unregistered generation job.

| Family | Rows | Work before fresh paired training |
|---|---:|---|
| v2b | 103 | July contexts/arrival/seal, August contexts/arrival/seal, common ledger/schedule |
| OPR | 77 | Three separate batched generation qualifications and lanes, three feasibility lanes, coverage/fallback |
| Fisher | 75 | Eight-rank 32-row preflight, shared 256-row estimator, CPU publication, three untouched first-batch calibrations |
| iSDFT | 65 | Existing-evidence instruction persistence gate |
| Freeze | 65 | Exact bound decoder-block mask |
| GRPO | 59 | Hash-first 320 training prompts; eight prompts × four completions/update |

Each family has two mechanical qualification paths, six fresh training stages,
CPU exports, CPU lane preparation, complete per-(seed,panel) scoring, accounting,
a CPU reader and a CPU collector. The 841 score-call slots include the OPR
feasibility references and v2b arrivals; generation and qualification work are
listed separately. Complete decision lanes contain six cells for ordinary
retention panels, twelve for legal/GRPO acquisition curves and seven for SEC
panels with their static-memory reference. July/August diagnostics and September
memory-on holdout strata do not become efficacy gates.

Use the row's resources and dependency list when writing sbatch commands:

```bash
# CPU row
sbatch --nodes=1 --gpus=0 --cpus-per-task=8 ...
# One complete generation/scoring lane
sbatch --nodes=1 --gpus=1 --cpus-per-task=8 ...
# One eight-rank qualification/estimator/calibration/training row
sbatch --nodes=1 --gpus=8 --cpus-per-task=32 ...
```

Every dependency needs `--kill-on-invalid-dep=yes`; readers/accounting use
afterany, other rows afterok. A PENDING or failed prerequisite needs a new
submission and dependants resubmitted against its new scheduler ID. Do not
allocate a GPU to wait for files. An archived binding means do not submit its
scientific rows. If an archived row is invoked, it records ARCHIVED UNREAD and
opens no outcome.

```bash
export KIT=/durable/kit
export NM_CONFIG=/durable/nm/launch.json
export JOB_PY=/durable/envs/job/bin/python
bash "$KIT/jobs/nm/run-task.sh" NM-grpo-bind
# Repeat for the chosen explicit row, in its scheduler allocation.
```

The wrapper owns one nonblocking per-GPU lease/idle check and one `util_watch`
with the row's exact `--expect-gpus` count. Children inherit `M3_GPU_LOCK_FDS` and
`UTIL_WATCH_ACTIVE`. CPU rows expose no CUDA devices. Every submission creates
`attempts/<row>/<nonce>`; a failed scoring lane is fully remeasured in a new
allocation. There is no historical GPU assignment for admission. The reader
labels a within-lane nonce/UUID/environment mixture UNVERIFIED-HARDWARE.

The wrapper passes each row's `lane_concurrency` to `lane_pool`. Scoring defaults
to K=1. A bundle's `lane_qualifications[panel]` may bind a K=3 receipt for that
exact family/panel workload, never a general-panel receipt reused for SEC,
FinQA or GRPO. The CPU plan records `qualification_binding`; its digest must
match the receipt's `binding_sha256`. That binding includes the frozen panel,
scorer, engine profiles, base revision and implementation seal. The receipt has
`schema=nm-lane-qualification.v1`, `concurrency=3`, `telemetry_complete=true`,
`numeric_match=true`, `memory_safe=true`, and measured `mean_util` and
`min_window_mean` both in [70,100]. Missing/unqualified receipts select K=1;
changed hashes or a receipt for another panel refuse. Each attempt records
requested/actual K in `lane-concurrency.json`. Qualified K=3 uses the pool's .31
memory fraction. Full-precision HF references and cohort/arrival lanes remain
serial. Generation has its own batch/memory settings and per-seed utilisation
qualification; general-panel measurements cannot qualify it. The scientific
generation lane requires complete telemetry, mean and rolling-window utilisation
at least 70%. A missing full rolling window leaves qualification PENDING.

## Bound inputs

The operator's JSON launch document has this shape (paths are examples):

```json
{
  "out": "/durable/nm/results",
  "job_python": "/durable/envs/job/bin/python",
  "infrastructure_update": "<SHA256 from the bound v6 launch>",
  "send_approval": "<separate approved send reference>",
  "hosts": [],
  "inputs": {
    "grpo": {"path": "/durable/nm/grpo-inputs.json", "sha256": "<SHA256>"}
  },
  "budget": {
    "branch": "CR",
    "approved_gpu_hours": 1565,
    "approval_reference": "2 November: Approve all six + allocation",
    "row_max_seconds": {"NM-grpo-C-s101-train": 14400}
  }
}
```

List a hard duration bound for **every** GPU row you intend to submit; the single
entry above illustrates the format only. Branches/approved allowances are K_CR
1362, K_core 1307, K_archived 1028, CR 1565 and I0 1270 GPU-h. Reservations include
loading, elapsed allocation time and teardown. All families in a dispatch use the
same durable output/budget ledger. CPU accounting reconciles actual Slurm starts
and terminal ends, including failed/rejected allocations. A process duration is
only a lower bound. Missing scheduler evidence leaves accounting/collection
PENDING; elapsed time above 24 hours is explicitly flagged.

A reference always has `path` and `sha256`. Files may be absent while upstream
work is pending; their absence is never a negative verdict. The source bundle
contains the family-specific data and these common fields:

- `binding_spec`: a reference to the prerequisite/recipe document below, for the
  four incumbent-dependent families.
- `base`: `model`, original `revision`, and `sha256` of the canonical mapping of
  `config.json` plus all safetensors and safetensor-index filenames to their SHA256 values.
- `tokenizer`, `sdpo`, `trainer_python`, `inference_python`: staged offline paths.
  The native entry checks the pinned verl bytes. No model download is performed.
- `source_revisions`, `licenses`, `holdout_authority`, `candidate_superset` and
  `excluded_superset`: frozen source/exclusion evidence. Superset rows have `id`
  and `text`; audit uses normalized equality, five-shingle Jaccard ≥.8 and edit
  similarity ≥.9. Reserved outcomes must not be opened to build these packets.
- `panels`: complete frozen item arrays by panel; `panel_authorities`: references
  to those same arrays. General/knowledge/HellaSwag/legal panels retain their
  exact 100 IDs; FinQA retains all 1,147. Instruction is strict, legal tolerant,
  and knowledge/HellaSwag use the inherited REFERENCE-SCORED implementation.
- `training_rows`: exact admitted per-seed schedules for legal comparisons,
  preserving core draw filtering and balancing, or K/CR unfiltered geometry.
  Rows have `id`, `prompt`, `response`, `gold`, `domain` and `m3_replay`.
- `qualification`: per-arm references to current-code production qualification
  evidence. A successful process or a CPU stand-in does not create this evidence.

Starts name seed, `stage=finqa`, `step=40`, `model` and the same model digest
convention. Each recipe binds seeds 101–103, exact pool IDs, schedule hashes,
40 updates, denominator 32, task/replay counts, FinQA slots, caps, draw policy,
balancing and any compatible P2 coefficient. K keeps 28+4; CR keeps 32+16 and
8 FinQA slots. Legal prompt/response/teacher caps are 10240/2048/14336; a different
returned cap leaves the row PENDING for prospective repricing.

The prerequisite document contains `evidence` references for K, CR, M8, M9, X,
P2, nomination and applicable stacking, plus `recipes` references for K, CR, I0
and, when relevant, nominated replay. Each verdict reference has a `selector`
array of JSON keys into an **existing unchanged reader's** output. For example,
M3 uses `['decisions','K']` and `['decisions','CR']`. A Boolean X admission also
needs `complete_selector`, e.g. `['status','p']`, requiring COMPLETE; a comparison
object can explicitly use `adapter: 'x-admitted'` to invoke the existing pure
reader unchanged. Missing, invalid and unresolved evidence never becomes “no”.

The resolver requires the K/CR disposition before freezing I, selects K before
CR before I0, and records inapplicable P2/core stacking under K/CR. Only I0 waits
for M8/M9, eligible X, P2 and required compatible stacking. Fisher always waits
for P2 and is benchmark-only unless its ISO control is the exact admitted
incumbent. OPR chooses I with FinQA slots, otherwise admitted CR, otherwise the
nominated core benchmark; it never constructs K+FinQA replay. iSDFT's persistence
reference binds recipe digest, instruction panel, step 40 and all three existing
retention differences; mean ≥−3 archives it.

The first incumbent-dependent bind seals `out/incumbent.json` before its extra
family gates. All subsequent binds must match its selection, selected recipe,
recipe digest, K/CR evidence and trace. Under I0 its required v6b evidence is
also fixed. Under K/CR, Fisher may add P2 evidence and OPR may add nomination
evidence; previously bound evidence cannot change. A lock serializes competing
binds. Per-family launch projections permit new returns for future families
without reopening this dispatch-wide selection. No challenger result is an
input to incumbent resolution. Missing or wrong-recipe iSDFT persistence stays
PENDING; it cannot choose a different I.

## Family-specific preparation

v2b supplies `development_batches` for July/August/September (61/80/52 items,
21/24/13 permanent holdouts). Contexts are archived before arrivals. Whole-month
frozen/static-memory predictions are sealed before feedback release; no-update
C/T references are explicit identical-weight aliases. The unchanged dual-scorer
collector admits unedited successes from July/August only. Retrieval excludes
self, issuer component, accessions and required filing groups. One cumulative
40-update stage follows August. Teacher cases replace the target demonstration;
student inputs stay memory off. Whole lowest-ranked cases are dropped on overflow.
The ledger's actual unique supply, repetitions and effective retrieval are retained.

OPR supplies `stored_verified` and `target_sha256` for every scheduled FinQA
response. `generation` binds batch size, memory fraction and a fixed ready
qualification batch ID list per seed. Four independent attempts at most, first
passing response, frozen FinQA-final generator, .7/top-p 1 and 2048 including EOS.
Coverage below .8 in **any** seed blocks both arms. Uncovered IDs retain the exact
stored targets at every exposure. Separate feasibility lanes require ≥100 acquired
items. Final cohorts use final-lane base/incoming vectors before either outcome
vector is accessed. Every base/incoming vector must agree with its pre-training
vector on at least 1136/1147 IDs; disagreements and cohort differences are reported.

Fisher supplies frozen `fisher_domains`: 128 instruction, 64 maths and 64 knowledge
training/proxy rows, with verified original-base responses. The estimator squares
individual sequence-score gradients, accumulates fp32 on each parameter's
device and normalises one global trace on device; it never squares a batch-mean
gradient or copies gradients to CPU per example. Each normalized shard is copied
once for final publication. CPU publication computes exact
streaming quantiles, support and tensor sums. Model/Fisher bulk hashing and prompt
rendering occur on CPU; allocations check their sealed metadata. The inherited
reference loader and sharding numerics are reused unchanged, with a scoped NM
admission adapter consuming the CPU hash receipt. Three dedicated first-batch passes
stop before optimizer/scheduler/EMA mutation. Calibration seals λ and λ_F using
the registered median rules; nonpositive/nonfinite norms leave PENDING. Raw norm
matching does not equalise AdamW coordinate updates. `fisher_pricing` supplies
measured startup/export work and a declared .25–.50 timing reserve; domain/example
and all three calibration timings replace the provisional estimator allowance.

GRPO supplies `training_candidates` with existing verified D demonstrations.
Both arms train all four trajectories for each of 320 unique prompts. The clipped
GRPO objective has detached within-group advantages, denominator 32, no D skip,
no CE/entropy/base-KL term, and no extra attempts for homogeneous groups. The
base forward-KL pass is telemetry only. Every update records reward, mixed/zero/
all-correct groups and policy drift. A qualification pass opens only P3's RL
column, never incumbent admission.

The iSDFT term is .10 × new-task full-vocabulary forward KL(base || student),
response/EOS mean, sum/32, with no new term on replay. Freeze masks decoder blocks
0–2 only, including all their projections/norms; embeddings/head/final norm stay
trainable. Ties spanning frozen/unfrozen roles refuse qualification. Frozen
coordinates are excluded from optimizer/decay, remain in scalar P2 logging and
are checked bit-identical at checkpoints. The frozen EMA blocks must match the
unchanged student or the exact .99/.01 `mul_`/`add_` recurrence in the native
dtype, including resumed updates. This permits only the pinned EMA's known
rounding, rather than demanding bit identity from floating-point multiplication.

## Readers, returns and verification

New benefit bars are mean ≥2, positive 3/3, two-sided df=2 lower bound >0 with
t=4.302653; guards use lower bound ≥−1. GRPO alone uses its registered FinQA −1
two-sided, retention −3 one-sided t=2.919986 and positive acquisition/3-of-3 bars.
An insensitive grouped-D control is PENDING. Absolute package summaries remain
separate. Valid unmet bars are INCONCLUSIVE; missing evidence is PENDING/INCOMPLETE;
identity errors are UNVERIFIED. No seed/checkpoint/threshold selection is added.

The collector's default return is **one cumulative dispatch cycle** covering all
families collected so far. Read `<out>/return-cycle.json`: its `parts` list names
the latest `attempts/NM-<family>-collect/<nonce>/text/RETURN-TEXT-*-of-*.md` files.
Paste that complete set once per return cycle. **At most four ESSENTIAL parts
TOTAL across all six families**, each at most 11,400 characters with every
physical line under 90 characters. Do not concatenate the historical cycles
from individual collectors. Each successful collection replaces its family's
entry, re-packs the complete dispatch, and atomically publishes the pointer.
Concurrent collectors serialize publication. Overflow refuses publication,
leaves the previous pointer intact, and retains FULL detail on the node.
The reader's short `summary.txt` is a local progress summary.

The existing `report_text.py`/`text_return.py` entry points use the NM adapter
with the same KIT framing and T/D/E rows. T rows carry the externally bound
`T|AUTH|<sha>`, seed order, binding/spec hashes, hashed lane identities, total
attempt counts (including failed attempts), and allocated GPU-hours. Each
`T|NM|count|<seed>.<panel>|[...]` carries exact integer correct-counts: denominator,
base, incoming, C, T. Legal and GRPO FinQA curves carry five counts per arm in
0/10/20/30/40 order; other panels carry final counts only. OPR adds
`<seed>.finqa-cohort` with denominator, C, T counts on the arm-independent cohort.
The importer repeats the native reader's percentage, subtraction and trapezoid
arithmetic without rounding, then applies the frozen per-seed bars. Reported
bounds/decisions are descriptive and never determine imported verdicts.

Family evidence includes GRPO D sensitivity (reconstructed from counts); OPR
cohort sizes, agreements, up to 11 disagreeing IDs per reference/seed, and
coverage/fallback by IDs and exposures; Fisher's norms, median ratio and both
coefficients; and iSDFT persistence. No nonce/UUID/environment is pasted raw.
Descriptive package summaries, prior-month diagnostics and per-phase attempt
detail stay in `text/full/RETURN-TEXT-*.md` (complete dispatch) and
`family-detail/full/RETURN-TEXT-*.md` (this family). Paste FULL only on request.

The CPU importer recomputes registered bars, retaining PARTNER-COMPUTED /
independent-reverification=UNVERIFIED provenance. It cannot reinspect raw item
vectors from text alone. Checksums are advisory: missing, damaged or mismatching
checksums label TRANSCRIPTION-UNVERIFIED and never reject a structurally complete
return or decide its scientific verdict. Missing parts remain incomplete;
duplicated or mixed identities are structural errors. A GRPO insensitive D
control with complete endpoints completes both read and collect rows while
retaining its registered PENDING/INCOMPLETE scientific outcome. Missing primary
evidence still exits PENDING immediately and requires resubmission.

Re-derive every returned family's verdict on CPU from the pasted files:

```bash
python -I kit/nm_return.py --text pasted-part1.txt pasted-part2.txt --out local.json
```

The dispatch result contains a `families` map keyed by the six registered names.
Each entry contains its own reconstructed status, primary, guards and provenance.
For single-family debugging, `report_text.py --inputs <json> --out <fresh>` takes
an input map with `_mode: "nm"`, `family`, `reading` (reading.json path),
`accounting` (accounting.json path), and `infrastructure_update` from the bound
launch. Production reporting verifies that external authority; `--standin` is
for explicit CPU fixtures only. Use the collector's dispatch cycle for returns.

Raw vectors, reader inputs, allocation proofs and telemetry stay on the node.
An optional extra archive is produced only when the source bundle requests
`return_archive: true`. Reproduce that optional archive on CPU:

```bash
python -I kit/nm_return.py --archive /durable/nm/return.tar.gz
```

The builder's CPU suite includes independent loss/gradient checks, exact D-path
fixture equivalence, Fisher calibration/quantiles, cohort agreement boundaries,
all six readers and every row through the real shell wrapper/watcher with explicit
stand-ins. Run only these files, with basetemp inside the worktree, then remove it:

```bash
python -m pytest tests/test_nm_*.py -q --basetemp tmp/nm-pytest
rm -rf tmp/nm-pytest
```

CPU stand-ins always return TECHNICAL-ONLY and cannot be relabelled scientific.
Production H100 memory and native distributed/resume/export equivalence remain
measured preflight requirements. CR utilisation and current-code timings are
reported monitoring evidence.
The manager regenerates the v6 authority last from the committed integrated tree;
this unit does not edit it or relax its release gate.

## Seventh v7 row: CR−G confirmation (owner approved 6 November)

The [confirmation registration](resources/nm/registrations/registration-cr-confirmation.md)
adds CR−G with its own **830 GPU-hour** allowance. The other six rows retain their
frozen I0 ancestry, definitions and seeds. Confirmation admission can affect the
v8 task-learning comparator only. It does not change v7's other six rows or the
product incumbent. No K, C0, combined winner or historical n=6 analysis is run.
This build does not grant tag/send authority.

For v7c, submit only [CR rows.json](resources/cr-confirmation/rows.json) from
the v7c folder. Keep the original six-row inventory and v7b folder for the frozen
six registrations. [rows-v7.json](resources/nm/rows-v7.json) is a combined
reference inventory; do not deploy all seven from one folder or parent root.
The additional [78 explicit rows](resources/cr-confirmation/rows.json) comprise
52 CPU rows, 20 eight-GPU rows (18 scientific stages and two qualifications), and
six one-GPU scoring rows. The merged v7 inventory has 522 rows: 312 CPU, 137
one-GPU and 73 eight-GPU. Every row requests one node; use its exact CPUs, GPUs
and dependencies. There are no target-generation/reconstruction rows.

| Confirmation phase | Rows | GPUs per row | Dependencies |
|---|---:|---:|---|
| Bind / prepare all source and fresh seed schedules | 2 | 0 | bind then prepare |
| Qualification inputs / qualification / qualification reader, each G and CR | 6 | 0 / 8 / 0 | prepare, then serial within arm |
| Stage preparation / training / CPU export, G and CR × 104–106 × three stages | 54 | 0 / 8 / 0 | both qualifications; own preceding stage export |
| Native/reference assignment | 1 | 0 | prepare |
| Complete native/reference CPU lane preparation | 6 | 0 | assignment and all six stage exports for that seed |
| Complete native/reference scoring | 6 | 1 | own CPU lane preparation |
| Accounting / reader / collector | 3 | 0 | afterany on the listed rows |

Use the **same real wrapper** as the six rows:

```bash
export KIT=/durable/kit-message-v7c/kit NM_CONFIG=/durable/v7c-cr-launch.json
export JOB_PY=/durable/envs/job/bin/python
# Submit explicitly, adding partition/account/node/time settings at your site.
sbatch --parsable --nodes=1 --cpus-per-task=8 \
  "$KIT/jobs/nm/run-task.sh" NM-cr-bind
sbatch --parsable --nodes=1 --cpus-per-task=8 \
  --kill-on-invalid-dep=yes --dependency=afterok:$CR_BIND_JOB \
  "$KIT/jobs/nm/run-task.sh" NM-cr-prepare
# Continue using every row's depends_on and right-sized resources in resources/cr-confirmation/rows.json.
# Example final CPU reader (the six scoring IDs are scheduler job IDs):
sbatch --parsable --nodes=1 --cpus-per-task=8 \
  --kill-on-invalid-dep=yes --dependency=afterany:$ACCOUNT:$N104:$R104:$N105:$R105:$N106:$R106 \
  "$KIT/jobs/nm/run-task.sh" NM-cr-read
```

Every dependency uses `--kill-on-invalid-dep=yes`. A missing/failed prerequisite
requires resubmitting the affected row and reconnecting descendants to its new
scheduler ID. No GPU allocation waits for inputs, tokenization, hashes, exports
or collection. CPU rows expose no GPUs; training owns eight GPUs only while its
native trainer runs. All attempts retain their nonces, leases, watcher data and
costs. Scoring recovery re-runs the **entire seed/lane in one fresh allocation**.
No cross-attempt stitching is permitted.

For the revised v7c deployment, create a **CR-only launch JSON** with its own
parent `out`. Run only `NM-cr-*` from the v7c code folder. All six other families
(v2b, GRPO, OPR, Fisher, iSDFT, Freeze) use the unchanged v7b code folder and
existing six-row parent root. The CR parent, `cr_confirmation.out`, six-row root
and v6a/v6b roots must be separate, non-nested trees. CR has its own dispatch
and return-cycle pointer. See [v7c status and deployment](README-v7c.md).

```json
{
  "out": "/durable/v7c-cr-parent",
  "job_python": "/durable/envs/kit/bin/python",
  "infrastructure_update": "4daf1f7f5624e3f9190831a1832d1183cb8b4032d310a9d0d027562ff058f400",
  "send_approval": "OWNER_BOUND_CR_SEND_APPROVAL",
  "cr_confirmation": {
    "out": "/durable/cr-confirmation",
    "v6a_root": "/durable/restored-v6a",
    "v6b_root": "/durable/restored-v6b",
    "six_row_root": "/durable/v7b-six-row-parent",
    "other_roots": [],
    "hosts": ["approved-non-core-node"],
    "realised_inputs": {"path": "/durable/node-v6a-inventory.json", "sha256": "OWNER_BOUND_SHA256"},
    "diff_receipt": {"path": "/durable/kit-message-v7c/kit/resources/cr-confirmation/diff-receipt-v7c.json", "sha256": "04861c64970d3fb8e04936ec2042852120cc94cd04fb063a5084bb71066df380"},
    "operator_receipt": {"path": "/durable/operator-qualification.json", "sha256": "QUALIFIED_RECEIPT_SHA256"},
    "trainer_python": "/durable/envs/trainer/bin/python",
    "sdpo": "/durable/pinned-SDPO",
    "tokenizer": "/durable/pinned-Qwen3-8B",
    "finqa_panel": "/durable/FinQA",
    "panels": ["NAME=/durable/frozen-panel"],
    "holdouts": ["NAME=/durable/frozen-holdout"],
    "host_reservation_bytes": 1099511627776,
    "budget": {
      "branch": "CR_confirmation", "approved_gpu_hours": 830,
      "approval_reference": "owner: Approve 830 + acknowledge, 6 November"
    }
  }
}
```

Before using this launch, load the complete [row duration map](resources/cr-confirmation/row-max-seconds.json)
into `cr_confirmation.budget.row_max_seconds` (all **78** rows):

```python
import json
from pathlib import Path
launch = json.loads(Path("/durable/v7c-cr-launch.json").read_text())
launch["cr_confirmation"]["budget"]["row_max_seconds"] = json.loads(
    Path("/durable/kit-message-v7c/kit/resources/cr-confirmation/row-max-seconds.json").read_text())
Path("/durable/v7c-cr-launch.json").write_text(json.dumps(launch, indent=2) + "\n")
```

GPU bounds use the registration's repricing **upper** GPU-hours divided by
allocated GPUs, plus **20% margin**, rounded up to the next minute. Training
uses 31.95 / 38.34 / 44.73 GPU-h for FinQA / legal / hard MATH; qualification
uses 12.24 GPU-h on eight GPUs. Native/reference lanes use 76 / 26 panel calls
per seed at .10 GPU-h/call, with no concurrency discount. The resulting seconds
are 17,280 / 20,760 / 24,180 for training, 6,660 for qualification, and
32,880 / 11,280 for native/reference scoring. CPU rows have operational bounds
of 14,400 seconds for preparation/tokenization and 3,600 for other CPU work;
they reserve zero GPU-hours. These are per-attempt ceilings, not a larger
allocation approval; actual allocations, failures and retries still debit 830.
Replace with measured current-code bounds and reprice before send.
The parent launch still binds `infrastructure_update`, `job_python` and the
separate send approval. Never infer an infrastructure seal from the local file.
CR rows read **`cr_confirmation.budget`**, not a top-level `budget`. Reservations
and allocation reconciliation live in `cr_confirmation.out`, separately from
the six-row ledger. A CR row cannot debit I0/K/CR or any six-row allowance.

`realised_inputs` pins an existing **node-side** inventory under kit `059d6293`.
Its `kit_commit` is `059d6293`, `hard_math_mode` is the **recorded** `core-copy` or
`m3-fallback`, and `sources` enumerates each existing file as
`{path, sha256, role, manifest?}`. Mark manifest entries `manifest:true`; every
member of their `files` map must also appear with the **recorded expected hash**.
The binder publishes expected and observed hashes, including failures. Required
roles are listed in `cr_confirmation.ROLES`: candidate lists, raw/accepted
responses, demonstrations, exclusions, verifiers, licences, pools, panels,
references, scorers, ARC FROZEN/acceptance, legal raw targets/journal/replay,
hard-MATH mode/targets, and schedule sources. The inventory names
`pools_manifest`, `base_binding`, `hellaswag_manifest`, `native_inventory` and
`math_source` as exact entries in `sources`. `math_source` must be the recorded
mode receipt. No current M5 pointer is consulted.

Each `sources` row is an object with a nonempty `path` and its lowercase
64-character `sha256`; every hash must match. `role` is optional for auxiliary
inputs such as tokenizer members and FinQA native inputs. When present it must
be a string, and the inventory must still cover every role in `ROLES`.
The optional `manifest` flag must be boolean. With `manifest:true`, the bound
file must be a JSON object; an omitted `files` key means an empty manifest
(the original behavior). A present `files` key must be an object mapping member
names to hashes, and each member must also have its own verified source row.
The bound native inventory has a nonempty `nodes` array of host strings or
objects with a `node` string, and an optional `message2_nodes` string array.
Malformed launch settings, diff receipts and native inventories produce an
owner-escalated PENDING input-binding record with expected/observed diagnostics.
If `cr_confirmation.out` is absent or invalid, the wrapper stores this attempt
under `out/cr-input-errors` (or beside the launch config if `out` is unusable).
The binding's final status is written once, after all bind-row input checks.

`task_sources` has exactly `finqa`, `contract_nli`, `math_hard`; each binds
`corpus` and `v6a_schedules` mapping `101`, `102`, `103` to realised schedules,
with every path in the inventory. Legal and hard MATH use the original verified
rows **in original pre-shuffle order** and `algorithm:"seed-shuffle-cycle-v1"`;
the unchanged algorithm must reproduce all three old orders before deriving
**104–106**. FinQA follows B's native per-seed order in v6a and does **not** follow
that cycle rule. The owner-approved 6 November delta uses
`algorithm:"b-native-finqa-regeneration.v1"`, `b_work` (B's original work root),
and `corpus` pointing to `b_work/v4/report-main/common/finqa.parquet`. Bind that
original COMMON parquet **in its original order**, its `manifest.json`, the
original `v4/datasets/datasets/v4_finqa/test.parquet`, all pinned tokenizer files,
and the three realised M3 JSONL schedules in the same hash inventory. Never
recover the COMMON corpus by pooling, sorting or deduplicating old exposures.

`cr_finqa.py` runs B's **unchanged `v4_campaign_ops.schedule`** in a disposable
CPU process and separate workspace. The adapter substitutes only the seed in
the original FinQA job; phase remains `main`, corpus bytes and sampler stay the
same. B shuffles IDs using `random.Random(seed)`, takes the first 1,280 when
available (otherwise its native repeated exposure rule), and calls
`v4_sft.main` / `schedule_common_rows` to write the scheduled COMMON parquet.
The M3 projection uses the existing `finqa_rows` verifier and JSONL writer.
Legal and hard MATH continue to call `cycle()` only under their own protocol.

**Gate before use:** the same native call must reproduce the bound realised
M3 schedules for **101–103 byte-for-byte**, including target bytes, repeated
exposures and JSONL encoding. The adapter also discovers B's stored
`finqa-s104/105-schedule.json` and `scheduled/finqa-s104/105` folders under the
original `b_work/v4/report-main`; any present files at those paths must be
hash-bound and reproduced. Stored 104/105 order discovery is limited to these
B report-main paths; copies anywhere else on the node are out of scope.
Stored order JSON is compared byte-for-byte, and stored parquet IDs/targets are
compared through the same M3 JSONL projection. An omitted, incomplete, changed
or mismatching stored order is **PENDING with owner escalation**. There is no
FinQA fallback and no `cycle()` substitution. Fresh **104–106** files are
released only after all checks pass; preparation repeats the gate before
freezing them. Resumed task binding checks the reproduction receipt and every
fresh schedule hash.

`input-binding.json` and `targets-finqa/receipt.json` retain
`finqa_reproduction`: expected/observed hashes and byte-equality for each
reproduction check, B code hashes, and each fresh order/schedule hash. The CR
reader carries the proof into the pasted `T|NM|finqa|reproduction` row and emits
`W|NM-cr|...owner_escalation|1` rows on a mismatch. This is partner-computed
provenance, not independent scientific reverification. Missing source/order
evidence stays PENDING; targets are never regenerated. The CPU stand-in uses a
fixture tokenizer and real B ordering/writing/verification; its qualification
rows are explicitly TECHNICAL-ONLY and do not establish an eight-GPU preflight
or baseline diff approval. `technical_synthetic: true` requires the host list
to be exactly `["CPU-STANDIN"]`; a non-stand-in host is refused before any
synthetic bypass runs.

Confirmation hard-MATH schedule binding additionally
re-runs `check_math_root` on the staged, frozen `targets-math` source tree,
including its pool/evaluation disjointness checks, before binding fresh schedules.
The replay scheduling salt remains **`m3-cr-v2`** and the assignment key remains
**`m3|SEED`**. Both arms retain the exact CR/G layouts, task/replay denominator,
loss, lengths, exclusions, references, acquisition parents and bars from v6a.
All sources and three fresh seed schedules freeze before training.

Missing artifacts/manifests/mode, insufficient pools, or any expected/observed
hash mismatch means **PENDING plus owner escalation**. A new artifact's hash is
never a replacement expectation. No fallback, replenishment, mode switch or
reduced chain can clear that stop. Old/foreign seed score evidence anywhere in
the confirmation trees is **UNVERIFIED**, even when cells are otherwise missing.
Shared exploratory `SEEDS=(101,102,103)` constants remain unchanged. Only the
wrapper's explicit `KIT_M3_MODE=cr-confirmation` enables 104–106 and G/CR.

The [scoring inventory](resources/cr-confirmation/scoring-inventory.json) lists
**306 unique logical arm/seed/stage/step/panel cells**: 228 native calls and 78 HF
panel cells in 39 processes loading both reference panels. This is six jobs:
one complete native and one complete reference job per seed, covering both arms,
base and incoming/acquisition references. Native jobs retain the task curves at
**0/10/20/30/40** and step-20 retention diagnostics. Reference jobs retain v6a's
**0/40 only** layout: **13 checkpoints / 26 panel cells per seed**. Intermediate
likelihood cells at 10/20/30 are not required. Base/incoming model aliases may
cause repeated loads; every actual call is charged. The registration's 330
call-equivalent forecast also reserves validation passes; the logical cell
inventory does not remove that allowance. Measure throughput and reprice before
send, retaining the 830 allowance. A missing curve/export keeps the
confirmation incomplete. Scorer UUID/environment/allocation identities include
curves; off-assignment or cross-job constituents are UNVERIFIED-HARDWARE.

Native scoring is serial. HF reference concurrency is **at most two**, with
unchanged bf16 model precision and fp32 likelihood reductions. Optional
`operator_receipt` evidence can describe **3a/3b/3d/3g/3h**,
`likelihood-K2` and `holder-slice`, pin the shared operator file bytes, and record
scorer numerics, memory, rolling window and startup grace. It is not required
before data. Missing, unreadable, mismatched or unqualified operator evidence is
monitoring only under the owner's “No unmeasured tight gates” rule.

The CR reader records all utilisation, telemetry completeness, holder-slice and
window/startup-grace concerns in `reading.json`'s `utilisation_warnings` list and
in `W|NM-cr|<warning-code>|1` rows in both ESSENTIAL and FULL pasted returns.
Missing telemetry, utilisation below 70%, missing holder slices and legacy or
mismatched operating definitions never change the scientific verdict or make a
lane PENDING. `util_watch --startup-grace SECONDS` records the configured grace
(default: `--window`); `holder_slices` lists measured UUIDs only in holder mode,
and `holder_slice_telemetry_complete` is true only with complete holder telemetry.
The grace field is recorded operating context; the watcher retains its existing
full-window rolling statistics and separate startup measurements.
Evicted/incomplete scientific lanes remain PENDING and retain their cost.
Input hashes bound to v6a, the code seal after the pre-data freeze, fresh seed
isolation, one allocation per lane, UUID/environment identity and missing-cell
checks remain strict.
Current-code longest-input, gradient (world sizes 1/2/8), finite state, eight-GPU
memory and native resume/export checks remain in the inherited qualification.
Historical v6a or a small-model smoke cannot stand in for that qualification.

Generate the pre-data code-diff receipt:

```bash
"$JOB_PY" -I "$KIT/cr_confirmation.py" \
  --diff-receipt /durable/cr-diff-to-review.json
```

The v7c receipt restores the conservative round-3 dependency closure: 479
sealed files (the original 478 plus the evening approval resource), 143 changed
files and 348 baseline hunks. The owner approved the 169 additional inherited
pairs after all three review groups. The former dependency gap is closed;
[the operator note](README-v7c.md) describes cache isolation and production paths
that CPU rehearsal did not exercise.
Use a full clone with baseline **`059d6293`** present; check
`git rev-parse --verify '059d6293^{commit}'` before generating or verifying review.
Every hunk requires a substantive allowed category: `seed-mode`, `namespace`,
`finqa-native-order`, `technical-standin`, `3a`, `3b`, `3d`, `3g`, `3h`,
`likelihood-K2`, `holder-slice`, `owner-v7c-fix`, or `owner-v6-howchange`.
`owner-v7c-fix` is restricted to `kit/cr_confirmation.py`, `kit/cr_jobs.py` and
`scripts/publish_kit.py`, with approval recorded in the receipt.
`owner-v6-howchange` accepts only
exact (file, hunk SHA256) pairs in the byte-pinned
[26-pair approval](resources/cr-confirmation/owner-approved-hunks.json) and
[169-pair approval](resources/cr-confirmation/owner-approved-169.json).
`technical-standin` covers only CPU fixture bypasses; their output is TECHNICAL-ONLY.

<!-- v7c-receipt-status -->
The current [diff receipt](resources/cr-confirmation/diff-receipt-v7c.json) is **COMPLETE**.
The receipt records the owner approval quote, date and reviewed pairs for
`owner-v7c-fix`, bound by its code seal and the launch JSON `diff_receipt.sha256`.
Use the receipt SHA256 and tag commit stated in the owner's send message; stop if
`sha256sum` of the receipt or `git rev-parse HEAD` differs from those values.
<!-- /v7c-receipt-status -->
Bind a COMPLETE receipt's shipped path and SHA256 in the CR-only JSON as shown
in [the operator note](README-v7c.md). `verify_diff` checks the full code seal and
independently regenerates every baseline hunk digest. Raw patch text stays in
the private review inventory, since removed baseline lines can contain private
literals; no diff bytes are omitted from the independently checked digests.
A COMPLETE diff receipt grants no tag, send or realised-input approval.
Node-side binding and optional operator evidence
are also not claimed by this CPU-only build. Run the CPU fixture through all
78 real wrapper/watcher rows with:

```bash
python -I scripts/cr_confirmation_dryrun.py --runnable --out /tmp/cr-native-finqa-dryrun
```

The stand-in calls B's real scheduler, verifies the three historical schedules
and any stored 104/105 orders, and exercises every G/CR preparation, training,
export and score row. It returns TECHNICAL-ONLY evidence. The default dry-run
mode separately exercises all 78 rows with missing inputs and verifies the
PENDING behaviour. Neither mode regenerates the infrastructure authority.

The reader emits **CR−G only**. It preserves the five-domain equal-weight retained
primary, each arm's own acquisition references, paired df=2 seed-t formulas and
all guards. It prints IFEval contrasts, means, intervals and bar results at **all
three boundaries**, even if another bar fails. Package bars, survival/cohorts,
answer transitions, strict-letter diagnostics and serving cost remain separate
local reports, never additional CR−G gates. The Part-3 accounting receipt requests
per-arm/seed/stage process and allocated GPU-hours, GPU counts, phase/idle/failed
attempt reconciliation, recoverable coverage of v6a's 298.3, and actual
critical-path/load/queue/transfer timings. The provisional **15.24–27.47 h** serial
path carries a **>24 h risk**; no measured utilisation or timing is asserted.

Paste CR's newest return cycle from the **CR-only parent**
`out/return-cycle.json`. It contains at most **four ESSENTIAL parts for CR**,
with FULL detail retained locally. The six-row return stays in its v7b parent. CR `T|NM|count` rows hold integer correct/total counts in
seed order 104–106; the CPU importer recomputes the primary and every guard rather
than trusting printed summaries. No old seed contributes. Prominent primary and
three IFEval endpoint lines accompany the counts. A PENDING/UNVERIFIED provenance
status cannot be repaired by counts. Use `nm_return.py --text PART... --out PATH`
to reproduce the pasted result. Checksums remain advisory transcription aids;
local independent provenance verification stays UNVERIFIED/PARTNER-COMPUTED.
