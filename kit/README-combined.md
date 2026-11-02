# Combined message 2+3: operator entry point

One partner message. Message-2 core has Opus round-3 SEND. The message-3 tier
joins only if it reaches **Opus SEND by 2 November 2026**; otherwise send and run
core alone using [README-message2](README-message2.md). Tier failures, missing
inputs and pending reviews never delay core. This package grants no launch authority.
The partner supplies all `sbatch` commands, allocations and dependencies. No
script submits, dispatches, polls or retries jobs. Seeds remain 101–103.

Read the scientific instructions: [core M/S/X](README-message2.md),
[K/CR](README-message3.md), [PD](resources/pd/README.md),
[SP](jobs/message3-sp/README.md), and [SP node verification](resources/sp/NODE-VERIFY.md).
Use the combined wrapper for tier jobs. Request exactly the GPUs used by each
job: partial-node allocations are allowed. CPU work requests zero GPUs and may
share the same host to reach node-local storage. Keep the frozen hostname/input checks and verify measured UUIDs within each
allocation. Historical UUID placement never prevents relocation. Inference uses per-allocated-GPU leases; eight-rank training owns
all eight GPUs of its allocation. Independent inference and CPU jobs may share
hosts with core. There is no combined tier node lock or blanket node exclusion.

Every GPU job runs through `util_watch.py`, writing `util.jsonl` under the local
control root (standalone tier runners write under their own root). The site
requires average activity above 70% over a rolling window across allocated GPUs.
Right-sizing and eight participating ranks do not prove compliance. Measure each
unchanged scorer, load/save gap and training shape before production admission.
Do not change batches, doses, seeds, panels, losses or scorer settings to raise
utilisation. Never poll for missing checkpoints while holding GPUs: exit PENDING.

| Family | gpus | nodes | cpus | depends-on | sbatch resource line |
|---|---|---|---|---|---|
| CPU prep/export/read/report | 0 | 1 | 8 | frozen inputs | `--nodes=1 --gpus=0 --cpus-per-task=8` |
| K/CR native score/validation | 1 | 1 | 8 | CPU ready proof, all checkpoints | `--nodes=1 --gpus=1 --cpus-per-task=8` |
| Broad generation | 1 | 1 | 8 | CPU candidates | `--nodes=1 --gpus=1 --cpus-per-task=8` |
| SP panel lane | 1 | 1 | 8 | CPU adapters/binding | `--nodes=1 --gpus=1 --cpus-per-task=8` |
| Reference seed/panel lane | 1 | 1 | 8 | CPU ready proof | `--nodes=1 --gpus=1 --cpus-per-task=8` |
| TP2 teacher fallback | 2 | 1 | 8 | CPU reuse decision = generate | `--nodes=1 --gpus=2 --cpus-per-task=8` |
| Eight-rank train/qualification | 8 | 1 | 32 | CPU prepared data | `--nodes=1 --gpus=8 --cpus-per-task=32` |

All requests use one node. CPU/inference rows may share a node; eight-rank rows
may not share their GPUs. Add the frozen hostname constraint separately. Wait
using scheduler dependencies outside allocations. Submit a scoring lane only
when every checkpoint it will score exists. Score every checkpoint of a native family/seed lane in one allocation on its
measured UUID; reference comparisons also stay within one complete allocation. If relocating a lane, rescore the entire lane in one job;
partial relocation remains UNVERIFIED-HARDWARE.

The 1 November [near-tie validation amendment](resources/m2/likelihood-near-tie.md)
applies to core and K/CR likelihood/reference validation. It preserves scores
and choices and uses pinned CPU-derived lengths solely for nats/token gap
diagnostics. Both disagreeing-item gaps must be strictly below 0.001; otherwise
validation stays PENDING, with no automatic retry.

## Freeze the local launch inventory

1. Create the full **local** core binding on CPU from your original archives.
   The shipped `resources/m2/send-binding-expected.json` contains public content
   expectations, including all archive/score/receipt hashes and seeds 101-103;
   checkpoint locations stay local. Set these operator-local shell variables:
   `B_ARCHIVE` (the original inner tier-2 B return, not its outer bundle),
   `C_ARCHIVE`, `TEACHER_ARCHIVE`, `RETURN1_ARCHIVES` (directory containing
   exactly the three byte-identical original archives listed below), `RESCORE_PATHS` (your local
   Tier A export resolver), and `SEND` (a fresh local output directory).

   `RETURN1_ARCHIVES` must contain exactly these original files:
   `package-c-return.tar.gz`, `next-return-1.tar.gz`, and
   `math-hard-teacher-return.tar.gz`. Set `C_ARCHIVE` and `TEACHER_ARCHIVE`
   to the corresponding files in that directory. Pre-check their bytes:

   ```bash
   printf '%s  %s\n' \
     'dc79e6583fba475cbc8ba357bce99634a60122191212a8c710226bc8abdd7b3d' "$B_ARCHIVE" \
     'cad18fee21b94b609c88be8ce74eb41134b6c3a1e4602b79a6b3b96d822770c9' "$RETURN1_ARCHIVES/package-c-return.tar.gz" \
     '262263e7a6ceb8afd74c7e3fb3a450195f2efbe4bca6ee1f4204e4dfecac1827' "$RETURN1_ARCHIVES/next-return-1.tar.gz" \
     'c408eea08fe6fc567d94b35a46d5cfad986e555b4aa58cbd6fbb4d72d0123db7' "$RETURN1_ARCHIVES/math-hard-teacher-return.tar.gz" | sha256sum --check
   ```

   Create `RESCORE_PATHS` using the [17-key JSON template](resources/m2/rescore-paths.template.json)
   and the [resolver format](README-message2.md#three-seed-send-re-probe-and-independent-return-1-rescoring).
   The keys are the decisions file's exact redacted `checkpoint` strings for
   the 17 distinct Tier A checkpoints, **not your own paths**. Keep those keys;
   replace each value with `{"model":"/local/model","receipt":"/local/export.json"}`.
   Only the pinned frozen base (the template's first entry) may omit `receipt`.
   The template's value locations are placeholders. The public check confirms
   Tier A checkpoint coverage and recorded hashes; live tensor hashes are checked
   when rescoring. No checkpoint is regenerated.

   ```bash
   python3 -I "$KIT/p3b_read.py" --source "$B_ARCHIVE" --out "$SEND/b-reading"
   python3 -I "$KIT/m2_send.py" --out "$SEND" \
     --b-source "$B_ARCHIVE" --b-reading "$SEND/b-reading/reading.json" \
     --c-source "$C_ARCHIVE" --teacher-source "$TEACHER_ARCHIVE" \
     --return1-archives "$RETURN1_ARCHIVES" --rescore-paths "$RESCORE_PATHS" \
     --decisions "$KIT/resources/m2/return1-decisions.json" \
     --format-check "$KIT/resources/m2/receipts/hellaswag-format-check.attempt2.json" \
     --arc-prompt-check "$KIT/resources/m2/arc-prompt-FROZEN.json" \
     --public-expect "$KIT/resources/m2/send-binding-expected.json"
   ```

   This refuses before writing the binding if **any non-path field** differs.
   Paths are neither compared by this public check nor printed. Set core settings
   `bind.expect` to `$SEND/send-binding.json`, `bind.rescore_paths` to
   `$RESCORE_PATHS`, `bind.format_check` and `bind.arc_prompt_check` to the shipped
   paths above. Also set `bind.b_archive` to `$B_ARCHIVE` when `bind.b_source`
   is the live B export tree: the binder hashes that original archive itself
   and enforces the fixed shipped public expectation before the local binding.
   The ordinary binder checks connected parent checkpoint files and receipts;
   rescore-reference live tensor hashes are checked when rescoring. Keep the full binding and resolver
   local; neither needs a file handoff or a pasted return.
   For X4 set core settings `addon.smoke` to
   `$KIT/resources/m2/receipts/p2-smoke.attempt5.json` (the real passing attempt 5).
   For each PD plan, load `smoke_receipt` from the shipped sealed JSON object
   `$KIT/resources/m2/receipts/pd-smoke.attempt3.json`. These are the original
   qualifying receipts, not new smoke runs. S's registration and SP's frozen
   panel/verification resources already ship; their normal on-node checks remain.
   Do not substitute PD's separate sealed scientific binding for the core binding.
   Use core's **durable frozen launch.json** (with node_slot on
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
   The PD entry is `python -I kit/pd_node.py --phase PHASE --plan PLAN --out PD_STATE`.
   Submit separate prep/pre/train/finalize/post/collect jobs with 0/1/8/0/1/0
   GPUs; every job requests one node. Resource rows and exact sbatch arguments
   are in [the PD runbook](resources/pd/README.md) and
   `resources/pd/job-resources.json`. The estimate is now **63–98 H100 GPU-h**
   for all three seeds, within the existing PD <=264 component allowance;
   a whole-seed restart adds about 20.74–32.46 GPU-h plus CPU costs. No automatic retries.
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
For v6b, use its own `v6b_send` record (SEND date, review_file and review_sha256)
as described in README-v6b; the historical tier date cutoff applies only to older
combined sends. These gates affect tiers only. Set them before building and before PF-tier;
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
PD 62.21–97.38 (rounded request **63–98**) and SP 54.75–216 GPU-h.
Their upper case totals about **1,700.54 GPU-h before unpriced rescoring**.
A PD whole-seed restart adds about **20.74–32.46 GPU-h**, plus CPU/queue costs. A compatible committed core target copy removes up to 30 GPU-h
of new K/CR generation. Monitor these warnings through the partner's actual
allocation receipts; they are **warnings, not gates**, and no cells are dropped.
The 1,800 allowance is not a promise that all component upper cases fit together.
Core proxy counts and allowances are separate. Per-job estimates overlap component
envelopes: do not add both. SP's reference job rows sum to 53.75–215 GPU-h;
the separate **1 GPU-h startup reserve** gives the 54.75–216 envelope. Unknown
coverage and failed/restarted allocations remain additional measured charges.

## Submission order: first preflights, then core, then isolated tiers

1. Follow the core manifest's resource fields and CPU preparation gates on its
   frozen slots. INPUTS must already be bundled. ARC failure leaves only its
   dependent work PENDING; independent branches and collection proceed.
2. Submit `PF-tier-*` as **0-GPU CPU jobs** on their listed hosts. They check
   runtime imports and readable cgroup counters and bind settings/hardware bytes.
   Productive GPU jobs check only their allocated UUIDs before model use. These
   checks do not replace K/CR qualifications, PD smoke or SP exclusions.
3. Submit all **core M/R/RA/S/X jobs**, with the core manifest's dependencies and
   explicit `node_slot`. Keep independent collectors/partial reducers independent
   as documented by core. Never add a tier dependency to any core job. Use core's
   partial-input/collector rules; a global afterok chain would suppress useful returns.
4. Once every core job is submitted, record that fact (this command submits nothing):

```bash
python3 "$KIT/combined_send.py" core-submitted --manifest "$COMBINED_MANIFEST"
```

5. Submit K/CR CPU preparation first. For each stage the sequence is
   `*-prepare` (0 GPUs), `*-train` (8), `*-export` (0), `*-score` (1).
   Every scoring/validation row also has a CPU `*-ready` checkpoint proof row.
   Exports merge/hash/publish on CPU after the GPU allocation ends. Each arm's
   `M3-preflight-ARM-inputs` is CPU-only; `M3-preflight-ARM` uses eight GPUs.
   Broad preparation/generation/finalisation requests 0/1/0 GPUs. A broad PENDING
   leaves only G/CR dependent work pending; C0/K continue independently.
   `M3-math-resolve` copies committed core targets on CPU if present. If its local
   `math-resolution.json` says `copy`, omit `M3-math-targets`. Otherwise submit
   that two-GPU teacher job, then `M3-math-ready` on CPU. The ready row never
   waits; schedule it after the teacher for the fallback path. Stages 1–2 never
   depend on this branch. Collectors retain partial evidence.
6. For each PD seed, submit the six right-sized phases in the PD runbook after
   core-submitted and the applicable node preparation. CPU prep requires at least
   352 GiB reserved spool/headroom capacity per seed and binds the hostname; keep all phases on
   that storage host. The persistent `.kit-pd-storage.json` ledger and lock live
   in the PD seed parent on the PD filesystem; keep all seeds under that parent,
   never copy/delete its ledger while reservations remain. Pre scores freeze E_s;
   train saves native checkpoints;
   CPU finalize builds complete exports; post re-scores update 0 plus every curve
   point on one GPU; CPU collect retains closure and reload agreement. Supply
   current already retained sources. PD HF cells run at **K=2 maximum**, sharing one
   pre-created job nonce; their rolling utilisation is **UNMEASURED**. Finalize
   deletes each native snapshot after its HF export verifies. The combined manifest contains all 19 PD
   phase rows, including CPU `PD-read`; retired `PD-SEED` holders are absent.
7. Submit SP sidecars with **afterany** dependencies from `sidecar.json`.
   `SP-verify-pools` and each `SP-prepare-*` request zero GPUs. The combined entry
   runs node verification before these CPU rows. Preparation runs adapters,
   binding, complete model hashes and provenance checks before GPU admission.
   `SP-panel-*` requests one GPU and checks the prepared files' freshness, then
   scores ready entries through the profile-capped lane pool (qualified vLLM K≤3). Each missing/deleted/expired
   export marks only its own entry MISSING-INPUT; the remaining entries still score.
   `SP-length`, `SP-read` and canonical pair-aware `verified-tables/` publication
   are CPU jobs. Preserve the raw evidence and never extend parent retention.
8. Submit `REF-FAMILY-assign` and `REF-FAMILY-SEED-prepare` on CPU after
   restoring/binding the native family root. Submit each `REF-FAMILY-SEED`
   only after all its checkpoints exist and CPU verification succeeds. Default prepare
   dependencies name the family/seed final export or score rows; an earlier decision
   frontier must supply explicit `reference_jobs[].requires` and a complete list. It uses
   one GPU for both panels of base/incoming/all finals, with K=2 maximum concurrency.
   All non-m3 REF phases are pinned to the first configured core host (`node` in
   each inventory row), where their node-local inputs and reference trees live.
   `REPORT-text` is a separate
   zero-GPU afterany job; reference guards execute inside CPU readers/reporters.

One selected row per operator-submitted allocation, with its resource request:

```bash
bash "$KIT/jobs/combined/run-task.sh" PF-node-0
bash "$KIT/jobs/combined/run-task.sh" PF-ARC
bash "$KIT/jobs/combined/run-task.sh" PF-tier-0
# Then core jobs on their frozen slots; after all core submissions, tier examples:
bash "$KIT/jobs/combined/run-task.sh" M3-K-101-finqa-train
# PD uses separate right-sized submissions from resources/pd/README.md.
# Use actual SP-* task IDs from the generated inventory.
```

Once dispatched, core returns its native exit codes; combined wrapper `ValueError`
gates return 3 (PENDING), while unexpected errors, including missing files, return 1.
No exception changes another family's state. K/CR returns 0/75/1; PD failures stay with
that seed; SP's underlying wrapper always exits zero and writes its state record.
The combined SP pre-scoring inventory gate can return nonzero and still blocks
no parent. Requeue manually with identical frozen inputs and each family's
restart rules. For PD, archive the entire failed PD_STATE tree, then restart
all phases in a fresh tree with the identical frozen input plan. Preserve failed
watcher logs, timing, agreement and PENDING receipts.
Record failed allocation charges, queue, restore and transfer time.
Return core evidence independently even when every tier is PENDING.

### Utilisation watcher

The combined and message-3 task wrappers run under `util_watch.py`, including
CPU tasks with zero visible GPUs. Every combined attempt writes a fresh JSONL and
adjacent `.summary.json` under shared `control_root/utilisation/<group>/`.
This root must be visible on every execution and report host. PD watcher files
stay outside the seed trees so CPU prep can enforce its empty-tree rule.
GPU rows also retain independent `.started.json` markers, including retries and
attempts whose watcher never finishes. Combined launches use one outer watcher.
UUIDs and scheduler identities stay on the node.
For other GPU entry points use
`python -I "$KIT/util_watch.py" --out <fresh-path.jsonl> -- <command...>`.
Defaults are `--interval 5 --window 300 --warn 75`. The watcher forwards signals,
preserves child exit status, and warns without stopping or changing the work.
It samples only the allocated CUDA/Slurm GPU set; unavailable telemetry is unknown.
`UTIL_PHASE_FILE` is set for optional `kit.util_phase.mark(name)` markers.

The combined reporter automatically adds every shared group's utilisation root
and a per-report started-GPU-row count file. For standalone reports add
`"utilisation": [{"family": "core", "roots": ["<control_root>/utilisation/core"]}]`
to the local report input map (repeat for available families; `summaries` accepts
explicit summary paths). ESSENTIAL Part 3 and FULL include compact
`T|UTIL|family|jobs|gpus|mean_util|worst_window|seconds_below_70` rows. Jobs counts
observed attempts, GPUs sums allocated slots across attempts, and mean activity
is weighted by GPU-seconds. Below-70 seconds sums rolling-window job-seconds.
Missing/incomplete telemetry is `null`. CPU-only attempts have zero GPUs and
null activity. See `docs/phase3/text-return-utilisation-20261102.md` for details.

## Returning results as text (current)

First create `report-inputs.local.json`, listing only the roots available on this node.
Omitted core/tier families report PENDING. v6b always includes its mandatory 4A
source; missing staged inputs remain PENDING. Multiple
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

The combined REPORT-text dispatcher creates `text_out/<nonce>/` per attempt;
paste from the latest completed attempt. For standalone reports use a fresh
output directory. **ESSENTIAL is the default: paste Parts 1-3,
then Part 4 only if present.** The set contains at most four numbered parts,
each ASCII, without tabs and at most 12,000 characters. Files are
`return-text/RETURN-TEXT-k-of-N.md`: **paste each whole file, KIT line to SHA256
line, one part per message; never paste terminal output or `*.local.json`**.

The parts contain:

1. Every registered decision verdict, primary estimate, interval, seed count
   in the registered direction (negative for PD, positive otherwise) and key guard failures. Part 1 alone supports every decision.
2. Per-seed primary endpoints and failed/unavailable guards or guards within
   0.5 points of their bar, plus mean, sample SD, native bound and seed count
   in that direction. Seeds are 101-103; 4A uses its amended three seeds,
   paired t df 2, 3/3 positive seed means and ≥3/5 positive monthly seed-means. Screen S retains its registered bootstrap bound.
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
metric pointer and seed, with one scalar `pointer=type:value` per logical row (`n` number, `b` boolean,
`s` string, `z` null). Numeric rows must contain finite numbers.
There are no JSON object or table blobs. Every physical report line is at most
90 ASCII characters; long logical rows wrap and the importer rejoins them.
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
of its canonical 90-column ASCII body including the final LF, excluding
header/footer. **Checksums are advisory for copy-pasted chat text.** A missing/truncated checksum or mismatch
never refuses an otherwise structurally valid part: all its rows are labelled
`TRANSCRIPTION-UNVERIFIED`, and the importer names that part. Numeric values
are never guessed or repaired.
The label comes from the actual exact-tag Git checkout when available, otherwise
from the copied kit's `KIT-IDENTITY.json`. Publishers must pass the intended
release with `publish_kit.py --tag`; an untagged build is labelled `untagged`,
never assigned an older release label automatically.
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
FULL parts arrive, the importer reports the aggregate digest as advisory too. These
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

The importer reports checksum mismatches without refusing the text. It requires
matching header/body identities, the expected commit when supplied, each
registered decision exactly once, section row counts, and valid numeric fields.
Duplicates, missing parts, mixed identities and structural damage are reported
and refused. These structural checks do not independently verify science. Keep **all result
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

## Re-scoring decision checkpoints (v5)

The six failed partner likelihood validations remain failed. The opposite picks
have knowledge gaps 0.010/0.023 and HellaSwag gaps 0.005/0.003 (kit/reference),
outside the near-tie rule. Score HellaSwag-100 and general-v1 knowledge guards
with the unchanged independent validation references; these results are labelled
REFERENCE-SCORED. Keep the failed validation files. Never combine kit and reference
values for one panel in a contrast. At decision checkpoints, HellaSwag and
knowledge guards require reference scores: kit-only guard rows stay PENDING
even if kit validation passed. Each (seed, panel) must use one physical GPU
and environment across base, incoming, and every arm's stage finals; mismatches
are UNVERIFIED-HARDWARE.

Create `checkpoints.local.json` as a JSON array containing every decision
checkpoint: `{"arm":"K","seed":101,"stage":"finqa","step":40}` for M3,
or `{"arm":"legal-generic","seed":101,"step":40}` for M. Use native
`x-` arm names for X, including each tier-2 comparison's own pre/control/challenger.
Include seeds 101–103, base (`arm: base`, step 0; M3 stage `finqa`), each incoming
(step 0), and each stage final (step 40). Paths/hashes come from native receipts.
Use separate lists/roots per family (`m3`, `core`, `x-tier1`, `x-tier2.NAME`).

For M3, run `M3-rescore-SEED-ready` on CPU, then `M3-rescore-SEED` in its
one-GPU allocation. The manual generated-panel block below is for core/X.
First re-run generated panels with the native scorer's exact configuration.
Set `OUT`, `FAMILY`, `CHECKPOINTS`, `SEED`, `GPU`, `FINQA_PANEL_ROOT`,
`KIT` (the kit directory), `REPORT_INPUTS` (your `report-inputs.local.json`),
and `TEXT_OUT` (your `return-text` output directory) locally.
Run this block for each seed (101–103) on that seed's assigned host. Keep
`GPU` assigned to the same physical UUID for each seed/panel throughout the list.
Score X's inherited generated denominators in `references/` without requiring
the failed kit likelihood validation; HellaSwag/knowledge guards follow below.

```bash
export OUT FAMILY CHECKPOINTS SEED GPU FINQA_PANEL_ROOT KIT
if [[ "$FAMILY" == x-* ]]; then
    M2_ASSIGNMENT="$OUT/scoring-assignment.json" M2_FAMILY="$FAMILY" M2_FAMILY_SEED="$SEED" \
      python3 -I "$KIT/util_watch.py" --out "$OUT/util/x-native-$SEED.jsonl" -- \
      python3 -I "$KIT/x_references.py" --out "$OUT" --seed "$SEED" --gpu "$GPU" \
        --panel-root "$FINQA_PANEL_ROOT" --generated-only
fi
python3 -I "$KIT/util_watch.py" --out "$OUT/util/native-$SEED.jsonl" -- \
  python3 - <<'PY'
import json, os, subprocess, sys
from pathlib import Path
out, family = os.environ['OUT'], os.environ['FAMILY']
assignment = Path(out) / ('m3-scoring-assignment.json' if family == 'm3' else 'scoring-assignment.json')
for c in json.load(open(os.environ['CHECKPOINTS'])):
    if c['seed'] != int(os.environ['SEED']): continue
    if family.startswith('x-') and c['arm'] == 'base': continue
    for panel in ('general-v1', 'finqa', 'test', 'math_hard'):
        script = 'm3_score.py' if family == 'm3' else 'm2_score.py' if family == 'core' else 'x_score.py'
        arm = c['arm'] if family in ('core', 'm3') else c['arm'].removeprefix('x-')
        if arm.startswith('t2-') and arm.endswith('-pre'): arm = 'pre'
        cmd = [sys.executable, str(Path(os.environ['KIT']) / script), '--out', out,
               '--arm', arm, '--seed', str(c['seed']), '--step', str(c['step']),
               '--panel-set', panel, '--gpu', os.environ['GPU'], '--panel-root', os.environ['FINQA_PANEL_ROOT']]
        if family == 'm3': cmd += ['--action', 'score', '--stage', c['stage']]
        subprocess.run(cmd, check=True, env=dict(os.environ, M2_ASSIGNMENT=str(assignment),
                       M2_FAMILY=family, M2_FAMILY_SEED=str(c['seed'])))
PY

```

Finish the native one-GPU allocation before the following CPU preparation.
Prefer the explicit combined `REF-*` jobs. Set `reference_jobs` in local combined
settings to a list of `{family, out, checkpoints, lane_assignment, requires}`:
`out` is the restored native family root, `checkpoints` is its complete local
list, and `requires` names existing combined checkpoint/export rows. Supply one
entry per core, M3, X tier-1 and each X tier-2 contrast. Defaults are local
`reference-FAMILY` roots requiring operator restoration; no restoration happens
inside a GPU job. `lane_assignment` is optional reviewed v2 JSON with exactly
`FAMILY.hellaswag|101` and `FAMILY.knowledge-ll|101` keys for seeds 101–103.
Each value contains node, UUID and environment. The two panels may use distinct
GPUs. Omitting this map preserves native UUIDs for both panels.

CPU assignment/preparation (0 GPUs):

```bash
python3 -I "$KIT/ref_score_job.py" --action assign --out "$OUT" --family "$FAMILY" \
  --lane-assignment "$REFERENCE_LANES"
python3 -I "$KIT/ref_score_job.py" --action prepare --out "$OUT" --family "$FAMILY" \
  --checkpoints "$CHECKPOINTS" --seed "$SEED" --panel "$PANEL"
```

Then one separate 1-GPU job for that complete `(family, seed, panel)` lane:

```bash
python3 -I "$KIT/ref_score_job.py" --action score --out "$OUT" --family "$FAMILY" \
  --checkpoints "$CHECKPOINTS" --seed "$SEED" --panel "$PANEL" --gpu "$GPU"
```

The score entry automatically uses the utilisation watcher and allocation lease.
Run the reporter only in its CPU allocation after scoring jobs finish:

```bash
python3 -I "$KIT/report_text.py" --inputs "$REPORT_INPUTS" --out "$TEXT_OUT"
```

The assignment's UUID is verified at execution, independent of ordinal numbering;
run each seed's commands on its assigned host/GPU. The reference assignment freezes
one entry per (family, seed, panel), and can also be supplied from a reviewed native
assignment with `--from-assignment`. Results use the reader's unit names under
`reference-scores/`, with manifests, model identity, method, UUID/environment,
likelihoods and cost. Native readers and M/X collectors retain these units.

The [v6 allocation-lane integration](util70-integration.md) defines complete
comparison jobs, allocation identities, command counts and added scoring costs.
Every one-GPU row uses the shared lane pool inside exactly one watcher.

## 4A in v6b (required)

4A's 197 rows from `resources/a4/v6b-rows.json` are always in group `4a`.
Set `a4_config` to the absolute config from [README-4a-v6b](README-4a-v6b.md),
and `a4_nodes` to its storage hosts. Keep its output separate from other roots.
The combined launch binds the config bytes when present; rebuild after changes.
Missing inputs return PENDING and must be staged before manual resubmission.
4A has an independent approval and does not wait for the K/CR send gate.

Run every row through `jobs/combined/run-task.sh ROW`. CPU bind/context/ledger/
prepare/export/finish/read/collect rows request `--nodes=1 --cpus-per-task=8`
with zero GPUs. Native training requests
`--nodes=1 --gpus=8 --cpus-per-task=32`; complete scoring lanes request
`--nodes=1 --gpus=1 --cpus-per-task=8`. Training ends at native checkpoints;
CPU jobs merge/export. Qualified vLLM lanes use K=3; HF NLL uses K=1.
Each allocation has one watcher, measured identity and a fresh attempt nonce.

Use every row's explicit `depends_on` edges: afterok for productive stages,
afterany for `A4-read`, always with `--kill-on-invalid-dep=yes`. Submit
`REPORT-text` afterany its complete listed frontier, including `A4-collect`.
The reporter automatically includes the configured 4A source, even when a
supplied report-input file omits it. Paste its 4A decision and utilisation lines;
4A cannot be omitted or labelled DEFERRED when inputs exist. Three-seed
amendment and 800 GPU-h ceiling are in the 4A runbook. The 400–450 GPU-h proxy
is a planning figure within that separate ceiling, not a measured SEC forecast.
