# Combined message 2+3: operator entry point

One partner message. Message-2 core has Opus round-3 SEND. The message-3 tier
joins only if it reaches **Opus SEND by 2 November 2026**; otherwise send and run
core alone using [README-message2](README-message2.md). Tier failures, missing
inputs and pending reviews never delay core. This package grants no launch authority.
The partner supplies all `sbatch` commands, allocations and dependencies. No
script submits, dispatches, polls or retries jobs. Seeds remain 101–103.

Read the scientific instructions unchanged: [core M/S/X](README-message2.md),
[K/CR](README-message3.md), [PD](resources/pd/README.md),
[SP](jobs/message3-sp/README.md), and [SP node verification](resources/sp/NODE-VERIFY.md).
Use the combined wrapper below for tier jobs so PD/SP get the same node exclusion
as K/CR. Whole-node exclusive allocations are mandatory; all tier nodes must be
outside the complete core-held node set, including train-only nodes. The combined
node lock serializes K/CR, PD and SP on a shared tier node. Scheduler exclusivity
must span the child process lifetime. No training/scoring overlap on a node.

## Freeze the local launch inventory

1. Follow core's send binding, hardware assignment, manifest freezing and `m2_bundle`
   input handoff exactly. Use its **durable frozen launch.json** (with node_slot on
   every task), not the earlier unbound source. INPUTS is the CPU bundle handoff;
   it is not a new GPU experiment. Every source task appears in the combined inventory.
   If `build` fails because of a tier input, run core with README-message2's wrappers
   directly; they are the same scripts.
2. Build K/CR with `jobs/message3/build.sh`, setting `M2_MANIFEST` to that frozen
   core launch and including every core node/root in K/CR settings. Build the SP
   inventory/sidecar with `jobs/message3-sp/plan.sh`, including both launches and
   the pinned return-1 decisions. Inspect the generated files; plan.sh always
   exits zero, so its exit code alone does not prove successful preparation.
3. Prepare three PD input plans using the PD runbook; each binds the qualified
   unchanged core, exclusions, starts and registered seed. Use the shipped sealed
   `resources/pd/scientific/send-binding.json`: copy it into each plan's
   `send_binding` and set `scientific.send_manifest` to that file as shown in
   `plan-inputs.template.json`. Its nine starts bind the evidenced original v4
   plain-D FinQA parents and C sequence-D ContractNLI step-40 finals, plus the
   pinned base registration. Follow the scientific README's exact receipt and
   checkpoint construction rules, then run `verify-node` before launch.
   The node entry is
   **pd_node.py**, which starts one whole seed in a clean PD tree. Scientific
   preparation remains required. The revised PD request is **222–264 H100 GPU-h**
   for all three seeds, within the approved tier component allowance. A late whole-seed
   restart adds 73.86–87.87 GPU-h; record it against the tier allowance. No automatic retries.
4. Create operator-local settings matching the example below; replace all
   placeholder locations/nodes. Keep settings, hardware and location maps local.
   All six state roots must be separate, non-nested trees. Use the native family
   handoff/storage rules across tier nodes; a node-local directory is not shared
   merely because another node uses the same path string. Freeze complete hardware
   for tier nodes as `{node: {complete_inventory: true, gpus: [{node, uuid,
   environment}, ...]}}`, using the inference Python's `m2_assignment.py` inventory
   on **every allocated GPU**, including training GPUs. SP still requires its own
   separate hardware authority and current complete parent-occupancy inventories.

```json
{
  "core_manifest": "/staged/core-durable/launch.json",
  "kcr_manifest": "/staged/kcr-launch.json",
  "sp_manifest": "/staged/sp/sidecar.json",
  "sp_inventory": "/staged/sp/inventory.json",
  "core_local_root": "/local/nvme/core",
  "core_durable": "/durable/core",
  "kcr_root": "/local/nvme/kcr",
  "pd_root": "/local/nvme/pd",
  "sp_root": "/local/nvme/sp",
  "control_root": "/durable/combined-control",
  "pd_nodes": ["tier-node"], "sp_nodes": ["tier-node"],
  "tier_hardware": "/staged/tier-hardware.json",
  "trainer_python": "/venv-trainer/bin/python",
  "inference_python": "/venv-inference/bin/python",
  "sp_node_pools": "/staged/node-pools.json",
  "pd_plans": {"101": "/staged/pd-101.json", "102": "/staged/pd-102.json", "103": "/staged/pd-103.json"},
  "tier_send": {"status": "values from the send message", "date": "values from the send message", "review_sha256": "values from the send message"},
  "pd_allocation_approved": "values from the send message"
}
```

Copy the values given in the owner's send message before `build`. The JSON
snippet above is a placeholder, **values from the send message**: the owner supplies
`status: "SEND"`, the actual review date (no later than 2026-11-02), the SHA256 of
shipped `docs/send/final-review-redacted.md`, and `pd_allocation_approved: true`
under the approved 1,800 GPU-h tier allowance. The wrapper compares the supplied
SHA256 to those exact review bytes. The shipped historical FIX review does not
itself grant SEND; the owner's send message must supply the actual decision.
These gates affect tiers only. Set them before building and before PF-tier;
changing them afterwards requires rebuilding and repeating tier preflight.
Paths in this example are portable placeholders, never partner locations.

```bash
python3 "$KIT/combined_send.py" build --settings "$SETTINGS" \
  --out "$SEND/combined.json" --inventory "$SEND/combined-inventory.md"
export COMBINED_MANIFEST="$SEND/combined.json"
export JOB_PY="$INFERENCE_PY"
```

`combined.json` is a static inventory, not a submitter. Every M/R/RA/S/X, K/CR,
PD and SP task is listed with prerequisites and an estimate or an explicit
unmeasured allocation. Its portable Markdown inventory contains no paths or
hardware identifiers. [Reference inventory](combined-inventory.md) is generated
on CPU from the real producers; it illustrates all rows, not scientific inputs.
Regenerate against the actual frozen S spec and actual hardware before send.
The approved tier allowance is **1,800 GPU-h**, with component allowances
**K/CR ≤1,400, PD ≤264, SP ≤216**. Current estimates are K/CR 786.30–1,387.16,
PD 221.59–263.62 and SP 54.75–216 GPU-h. Their upper case totals 1,866.78,
about **67 GPU-h above 1,800**, plus any PD whole-seed restart (up to about
**88 GPU-h**). A compatible committed core target copy removes up to 30 GPU-h
of new K/CR generation. Monitor these warnings through the partner's actual
allocation receipts; they are **warnings, not gates**, and no cells are dropped.
The 1,800 allowance is not a promise that all component upper cases fit together.
Core proxy counts and allowances are separate. Per-job estimates overlap component
envelopes: do not add both. SP's reference job rows sum to 53.75–215 GPU-h;
the separate **1 GPU-h startup reserve** gives the 54.75–216 envelope. Unknown
coverage and failed/restarted allocations remain additional measured charges.

## Submission order: first preflights, then core, then isolated tiers

1. Submit `PF-node-*` as the first exclusive jobs on their frozen **core** slots,
   with core's prepare wrapper through the combined entry below. INPUTS must
   already be bundled. Then submit `PF-ARC` on its frozen slot after its node
   preflight. No core job starts before its own node preflight; ARC failure leaves
   broad-dependent work pending while pool-free/plain-D, S and core collection
   proceed on healthy nodes, exactly as core's existing graph specifies.
2. Submit every `PF-tier-*` as the first exclusive job on its listed **tier** node.
   These jobs check rapidfuzz in both runtimes, NumPy/PyArrow/datasets/verl and
   tier trainer/inference imports, readable cgroup current/peak/OOM counters, and
   every allocated physical GPU UUID plus the inference environment and child
   torch assertion. Missing counters/imports/UUIDs leave only that tier node
   PENDING. The receipt binds settings and hardware bytes. These checks do not
   replace the four K/CR arm qualifications, PD smoke or SP scientific exclusions.
3. Submit all **core M/R/RA/S/X jobs**, with the core manifest's dependencies and
   explicit `node_slot`. Keep independent collectors/partial reducers independent
   as documented by core. Never add a tier dependency to any core job. Use core's
   partial-input/collector rules; a global afterok chain would suppress useful returns.
4. Once every core job is submitted, record that fact (this command submits nothing):

```bash
python3 "$KIT/combined_send.py" core-submitted --manifest "$COMBINED_MANIFEST"
```

5. Submit K/CR `M3-init` and the rest of its static graph on separate available
   tier nodes. Stages 1–2 do not wait for hard-MATH targets. A broad PENDING leaves
   only G/CR and their coverage preparation pending; C0/K continue. Stage 3 copies
   a committed core target snapshot read-only if available, otherwise runs its
   own registered two-GPU fallback without waiting on core. `M3-read` retains
   partial evidence. Per-arm preflights remain mandatory before chain training.
6. Submit PD-101, PD-102, PD-103 on allowed tier nodes via the combined wrapper,
   after core-submitted and PF-tier on that node, one whole seed per eight-GPU allocation. Supply current already retained
   source exports; no core retention extension or restore is requested for PD.
7. Submit SP sidecars with **afterany** parent dependencies from `sidecar.json`.
   Set `SP_INPUTS`, `SP_PARENT_HARDWARE`, `SP_OWNER_EXPORTS`/`SP_MODELS` and
   `SP_HARDWARE` per its runbook. The combined wrapper checks node inventories
   with the author's `node_verification.py run` **before every SP panel job**.
   Missing inventories prevent scoring; complete overlap checks preserve every
   pair and flag only affected pairs. SP-read also writes the canonical pair-aware
   `verified-tables/` through the author's report adapter; preserve that report
   with the raw results. The older `tables/` is an intermediate reader output.
   Missing/deleted/expired exports remain missing. SP never restores, pins or
   extends retention. Never make parent jobs depend on SP receipts or success.

One selected row per operator-submitted allocation:

```bash
bash "$KIT/jobs/combined/run-task.sh" PF-node-0
bash "$KIT/jobs/combined/run-task.sh" PF-ARC
bash "$KIT/jobs/combined/run-task.sh" PF-tier-0
# Then core jobs on their frozen slots; after all core submissions, tier examples:
bash "$KIT/jobs/combined/run-task.sh" M3-K-101-finqa-train
bash "$KIT/jobs/combined/run-task.sh" PD-101
# Use actual SP-* task IDs from the generated inventory.
```

Once dispatched, core returns its native exit codes; combined wrapper `ValueError`
gates return 3 (PENDING), while unexpected errors, including missing files, return 1.
No exception changes another family's state. K/CR returns 0/75/1; PD failures stay with
that seed; SP's underlying wrapper always exits zero and writes its state record.
The combined SP pre-scoring inventory gate can return nonzero and still blocks
no parent. Requeue manually with identical frozen inputs and each family's
restart rules. For PD requeue, first move `seed-N` and `bound-N.json*`
together to `pd_root/failed-<k>/`; then requeue into clean paths with the identical
frozen input plan. Preserve the failed tree and supervisor PENDING receipts.
Record failed allocation charges, queue, restore and transfer time.
Return core evidence independently even when every tier is PENDING.

## Returning results as text (current)

First create `report-inputs.local.json`, listing only the roots available on this node.
Omitted core/tier families report PENDING; omitted 4A always reports DEFERRED / NOT_CONFIGURED. Multiple
roots per family are supported and numbered separately. A `source` can be a
native result tree or its existing native archive for M, X, S and 4A. KCR and SP
use their native trees. Readers run afresh; saved reading JSON is never trusted.
X/S trees use their existing collectors to make temporary **local** archives;
no archive needs to be sent. No models, GPU execution or network calls are used.
Use the same frozen PD anchors/original export resolver and SP inventory as the
native readers; PD combines all three seed receipts, without filling missing
cells. Include every PD launcher timing, including failed attempts, exactly once.
Do not count nested training or streamed scoring again.

```json
{
  "M": [{"source": "/results/core"}],
  "X": [{"source": "/results/x", "screen": "/results/s"}],
  "S": [{"source": "/results/s"}],
  "KCR": [{"source": "/results/kcr"}],
  "PD": [{"evidence": "/results/pd/all-seeds.json",
          "exports": "/results/pd/original-resolvers.local.json",
          "trusted": "/results/pd/frozen-anchors.json",
          "launcher_timings": ["/results/pd/seed101/launcher-timing.json",
                               "/results/pd/seed102/launcher-timing.json",
                               "/results/pd/seed103/launcher-timing.json"]}],
  "SP": [{"source": "/results/sp", "inventory": "/results/sp/inventory.json",
          "input": ["/results/core", "/results/kcr"]}],
  "4A": [{"source": "/results/4a"}],
  "jobs": [
    {"family": "core", "manifest": "/results/core-launch.json", "durable": "/results/core-durable"},
    {"family": "KCR", "manifest": "/results/kcr/m3-launch.json", "state": "/results/kcr/state"},
    {"family": "PD", "manifest": "/results/pd-launch.json", "state": "/results/pd-state"},
    {"family": "SP", "manifest": "/results/sp-launch.json", "state": "/results/sp/state"}
  ]
}
```


For this message, paste WhatsApp chat text only. After all runs **or any partial
set**, run the CPU-only reporter with the node's existing Python environment:

```bash
python3 "$KIT/report_text.py" --inputs report-inputs.local.json --out return-text
```

Use a fresh output directory. **ESSENTIAL is the default: paste Parts 1-3,
then Part 4 only if present.** The set contains at most four numbered parts,
each ASCII, without tabs and at most 12,000 characters. Files are
`return-text/RETURN-TEXT-k-of-N.md`: **paste each whole file, KIT line to SHA256
line, one part per message; never paste terminal output or `*.local.json`**.

The parts contain:

1. Every registered decision verdict, primary estimate, interval, seed count
   in the registered direction (negative for PD, positive otherwise) and key guard failures. Part 1 alone supports every decision.
2. Per-seed primary endpoints and failed/unavailable guards or guards within
   0.5 points of their bar, plus mean, sample SD, native bound and seed count
   in that direction. Seeds are 101-103; 4A keeps its registered five seeds 101-105 and
   monthly inputs. Screen S retains its registered bootstrap bound.
3. Job completion, recorded GPU-hours per family and total against allowances,
   wall times and warnings, including coded PENDING reasons.
4. Overflow from Parts 2/3, only when needed. Evidence is never silently dropped;
   oversized input scopes refuse the essential return instead of emitting a fifth part.
   Split the local input map into smaller family/root groups and run separate
   fresh reports into separate output directories. Paste and import each report
   separately, including its Part 1 and its own FULL-set reference.

On `Text report refused`, inspect `<out>/refusal.local.txt` locally, fix the
input/configuration issue, and rerun into a fresh output directory. Never paste
that diagnostic: it can contain private paths and traceback details. If refusal
is due to capacity, split the input maps as above; retained FULL files and local
results stay on the node.

All text floats use four significant figures. Selection and scientific verdicts
use the unrounded native results. `null` means unavailable, never zero.
Stand-ins are marked TECHNICAL-STANDIN and provide no scientific evidence.
The full-precision `full-results.local.json` stays on the node, bound by
RESULTS-SHA256. Every essential part gives the FULL set's count and digest.

The reporter **always also writes** the complete numbered detail set to
`return-text/full/RETURN-TEXT-k-of-N.md`. Keep it on the node; send specific
FULL part numbers only when requested. `--full` selects that complete set in
the command's output (use a fresh output directory); it still writes ESSENTIAL.
Do not paste the entire FULL set by default, and do not mix outputs from
separate report runs. Retain the result trees, local settings, hardware maps,
and all `*.local.json` on the node; settings and hardware maps are never printed.


The jobs array reads each frozen manifest's `tasks` and includes every task,
including tasks without a state file. `J` rows group DONE, FAILED and PENDING
jobs using compact indexes into the **shipped public inventory**, whose digest
is in ESSENTIAL Part 3 (`T|JOB-INVENTORY`). The importer resolves these to registered job
names only when its own public inventory digest matches. Indexes come from
`combined-inventory.md` in table order, followed by additional PF-node/PF-tier
slots 0-127; they do not depend on private manifest settings or node counts.
`u`-prefixed indexes are local manifest ordinals for unrecognized tasks: their
names remain private and are explicitly unresolved. The private
`job-map.local.json` retains that mapping for later verification. Missing
completion evidence is PENDING; a failed core execution is reported only from
its manifest-bound execution receipt. Without job manifests the summary says
unknown/PENDING. Keep the manifests on the node. The PD supervisor can supply a
small manifest with its three registered seed jobs and per-job supervisor
receipts in its state directory.

`D` rows are family, comparison, verdict, primary estimate, lower/upper interval,
seeds in the registered direction and key guard failures. M
`admission-evaluated` means the admission procedure finished, not a bar passed. A missing upper bound is intentional for
one-sided registered intervals. `R` rows preserve numeric reader leaves with a
metric pointer and seed. Repeated records may use compact JSON column tables
(`@columns`, `@rows`, `@offset`); the importer expands the already rounded values.
`E` rows carry essential endpoints: family, endpoint, P/G/I kind, ordered seed
values, mean, sample SD, lower/upper bound, count in the registered direction, n,
bar and method. The importer names the count `seeds_in_direction` and records
`seed_direction` (negative for PD, positive otherwise).
`W` rows group warning/PENDING codes and counts. Registered static reasons are
coded using their public wording; other free-text reasons get opaque codes and
stay on the node. Unknown/free-text labels get per-run HMAC aliases. The random
HMAC key is kept only in private `metric-map.local.json`, with the local alias
map. Numeric values under unknown keys are dropped and counted in
`W|family|UNKNOWN_NUMERIC_DROPPED|count`; numeric metadata is never treated as a
scientific metric merely because its key was aliased. For K/CR's
two-sided interval, the displayed upper endpoint is the symmetric counterpart
of the native reader's lower endpoint; one-sided intervals keep upper=null. SP remains
descriptive; DEFERRED never means a scientific admission. Only registered
incumbent-bound X comparisons can yield admission. Cost totals are recorded
process GPU-hours, include recorded failures, and leave unmeasured allocation,
idle and missing timing costs PENDING. The 1,800 GPU-h tier allowance is monitored
as a warning, never a new scientific gate.

Each part starts `KIT <tag> <commit> PART k/n` and ends `SHA256 <hex>`, the SHA256
of its exact ASCII body including the final LF, excluding header/footer.
The body includes `T|KIT|tag|commit` and `T|PART|k|n`; the importer requires
exact equality with the outer header, including in Part-1-only reads and FULL
supplements. Use `--expect-commit` with the full published commit from the send
message to additionally pin the expected kit. The
reader restores line wraps, smart quotes, non-breaking spaces, CRLF and leading
`> ` plus invisible U+200B-U+200F, U+2060 and U+FEFF before checking the canonical body. Part 1 also includes `RESULTS-SHA256`
over canonical `full-results.local.json` (sorted keys, compact JSON, ASCII
escapes, no NaN; excluding its final LF). **These detect transcription errors
and truncation only. They do not authenticate results or independently verify
science.** All imported decisions and metrics are PARTNER-COMPUTED with
independent re-verification UNVERIFIED. The FULL-set digest hashes the ordered list of SHA256 hashes of each part's
canonical content-row JSON (excluding KIT/PART/MODE/FULL-SET rows and outer framing).
Requested supplements must match that digest/count and kit identity; when all
FULL parts arrive, the importer verifies the aggregate digest too. These
references provide no authentication. The full-results hash can be checked
against the retained JSON when archives become possible again; the text alone
cannot reconstruct private provenance or prove that digest's contents.

Our side, copy **single messages**, saving each whole part as one text file.
A multi-message copy adds timestamps and sender prefixes; these are refused as
unframed text. Set `EXPECTED_COMMIT` to the full published commit from the send
message, then run:

```bash
python3 scripts/read_text_return.py essential-*.txt --expect-commit "$EXPECTED_COMMIT" --out text-reading.json
# Later, any requested FULL parts, without repasting the essential set:
python3 scripts/read_text_return.py full-part-17.txt --base text-reading.json --expect-commit "$EXPECTED_COMMIT" --out supplemented.json
# Part 1 alone, explicitly accepting missing essential Parts 2/3/4:
python3 scripts/read_text_return.py pasted-part1.txt --part1-only --expect-commit "$EXPECTED_COMMIT" --out decisions.json
```

The importer rejects bad checksums (naming the part), duplicates, mixed kit
identities and missing parts. It never guesses a damaged value. Keep **all result
trees and local JSON files on the node**. Paste only the numbered markdown parts.
Local settings, model/export maps, hardware maps, paths, hostnames, usernames,
scheduler IDs and GPU UUIDs are never printed in those parts or reporter output.
The local input map, `*.local.json` and `refusal.local.txt` are excluded from the text return.

## Return (when archives are possible again)

Return core evidence independently of tier status. Include:

- Core archives: `message2-return.tar.gz`, `reprobe-return.tar.gz`,
  `return1-rescoring.tar.gz`, `screen-s-return.tar.gz`, `addon-return.tar.gz` and
  `addon-extended-return.tar.gz` (when present), with their hash manifests, summaries,
  partial/missing/PENDING records, logs and failure receipts; the frozen launch and
  scoring-assignment identities, PF-node/PF-ARC receipts and input/send binding evidence
  specified in README-message2. Include each independent collector even if another fails.
- `control_root/core-submitted.json` and every `control_root/preflight/*.json`.
- K/CR state, frozen launch/input/schedule/assignment manifests, qualifications,
  per-task receipts and logs, generation-source/target evidence, exposure/score/runtime
  evidence and `m3-read.json` / `m3-read.md`; retain partial/PENDING arm evidence.
- For each PD seed, its complete seed tree (including `seed-evidence.json`),
  `bound-<seed>.json` receipt and any supervisor PENDING/failure receipts; include
  archived failed attempts as well as the final attempt. Follow PD's native checkpoint
  exclusions and return contract.
- SP sidecar/inventory identities, receipts, raw results, pool-verification evidence,
  intermediate `tables/` and canonical `verified-tables/`, including missing-entry
  and pairwise UNVERIFIED reports.

Exclude **every `*.local.json`**, `refusal.local.txt`, the operator settings file and `tier_hardware`,
plus local model/export maps and node-pool inventories containing locations,
hostnames or GPU UUIDs. Apply the native family redaction/collection contracts to
returned evidence; retain private hardware/location authorities locally. Do not
return live model weights unless the native runbook explicitly requests them.

## Our-side offline verification and export

```bash
PYTHONDONTWRITEBYTECODE=1 python3 "$KIT/combined_dryrun.py" --out "$FRESH_CPU_DIR" --standin
python3 scripts/publish_kit.py "$EMPTY_EXPORT_DIR"
```

The CPU run records `summary.json` and the portable per-row `wrapper-walk.jsonl`.
The CPU run walks every generated task through its real CLI/shell wrapper with
scientific child stand-ins. It injects broad PENDING, K/PD/SP failures, checks
both core trees byte-for-byte, and requeues core collection. It does not qualify
GPU science or measure GPU-h. Keep its summary and portable row inventory.
Publisher copies all kit tier files, the frozen SP panel bytes, runtime helper
and selected standalone tests; EXPORT-INVENTORY.json hashes every copied file.
It scans the entire export. No operational settings, local maps, private smoke
inputs, names, usernames, hostnames or scheduler IDs belong in the shipped tree.
Do not rewrite pinned scientific inputs to redact a private build log.
