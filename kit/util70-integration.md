# v6 allocation lanes

One-GPU scoring/generation rows use `lane_pool.py --concurrency 3` (PD HF
pre/post use **K=2 maximum**, rolling utilisation **UNMEASURED**) inside
one `util_watch.py` lifetime. The partner supplies `--nodes=1 --gpus=1
--cpus-per-task=8`; PD pre/post also use eight CPUs. Resource fields in the
actual generated launch manifest are authoritative. CPU rows request zero GPUs;
the teacher requests two and distributed training requests eight. All families
use the same host-local `/tmp/kit-gpu-leases/GPU-UUID.lock` namespace. Training
leases all eight of its allocated GPUs; a CPU allocation may share its host.

The recorded H100 general-panel measurement was 88% mean, with an 89% worst
300-second window and byte-identical outputs for three concurrent processes.
The K=3 memory fraction is .31. This measurement does not qualify other panels,
HF reference commands, PD, generation, or training. Their rolling utilisation
must be read from their own watcher receipts. Scorers, batches, seeds, panels,
doses, and numerical decision thresholds are unchanged.

## Jobs, readiness and command counts

The inventory has a command-count column. A core CPU staging job writes the
exact expanded `command_count` and ordered `commands` into
`LOCAL_ROOT/TASK/lane-commands.json`. Counts depend on frozen nominations,
availability and dispositions; a symbolic inventory entry is resolved there
before GPU admission. No GPU command polls for another job's checkpoint.
Missing inputs exit PENDING immediately. The runner records all launches,
completion statuses, GPU UUID, shared allocation job identity and memory fraction.

| GPU row | Expected commands on its one GPU |
|---|---|
| M4 boundaries | 28 checkpoint/panel commands per seed |
| M incoming/final frontiers | Legal incoming 5, final 12; MATH incoming 4, final 16; skipped frozen branches do no scoring |
| R incoming | 25: base plus four incoming arms, five panels each |
| R individual final | 14 per arm/seed |
| R-compare-broad / R-compare-full | 43 / 81 per seed |
| RA101 / RA102 / RA103 | 12 / 12 / 17, including fresh retained/reused contrast inputs |
| Optional RB101 / RB102 / RB103 | 106 / 106 / 111 |
| M8-compare / M9-compare / M-final-score | Up to 62 / 79 / 119 per seed; the frozen nomination/admission determines executed arms |
| M3b broad generation | Three disjoint candidate shards; the CPU reducer retains original candidate order |
| S7 own-response generation | Six commands for a root incoming, eight for a non-root incoming: one domain and partition per command |
| X3 own-response generation | Eight domain/partition commands; the generic-only disposition skips generation |
| S4 inventory | Two panel commands per slot; slots retain separate registered identities |
| S8-S10 | Two NLL commands per domain/checkpoint comparison, two per eligible probe, and two generated-panel commands per distinct checkpoint; exact count in CPU plan |
| X incoming | Pre checkpoint panels and inherited references; aliases run on CPU |
| X-complete | 63 with all four arms READY: five pre panels, ten inherited reference panels, 48 curve panels; CPU aliases are additional reducers |
| X15-X20 complete | 39 per comparison/seed with both arms READY: five pre, ten inherited reference and 24 curve panels |
| X adaptive/drift measurement | Two NLL commands per domain at every frontier, including a fresh stage-start measurement |
| FV core/reprobe/X | One unchanged production/reference validation command per registered start available for that seed; CPU plan gives exact count |
| K/CR base | Six panels per seed |
| K/CR stage score | 15, 16, 17 by stage: six fresh incoming panels plus the original 9, 10, 11 curve panels |
| K/CR rescore | 198 per seed: base and every registered curve/guard cell across all four arms and three stages |
| K/CR validation | One unchanged validation command per panel/seed; each internally compares base and the registered incoming checkpoint |
| K/CR broad generation | Three disjoint candidate shards |
| REF-FAMILY-SEED | Two times the number of decision checkpoints in its CPU-frozen local list, both panels in one job |
| PD-SEED-pre / PD-SEED-post | 3 / 18 unchanged HF scoring calls |
| SP-panel-ASSIGNMENT | One unchanged panel call per inventory entry; the row records the number of entries |

Short validation, S4 and PD pre jobs retain their independent readiness,
registration and restart boundaries. Validation and S4 still use K=3 but cannot
fill three slots when only one or two commands are ready; PD uses K≤2. Long generation commands can
fill three slots even with only three shards. These short lanes and the
20-second launch stagger remain explicit utilisation risks; their resource
request is one GPU, never an eight-GPU holder.

Reference jobs are `REF-FAMILY-assign` (CPU), `REF-FAMILY-SEED-prepare` (CPU,
verifies both panels), and `REF-FAMILY-SEED` (one GPU). Supply a complete
checkpoint list and native restored root for the required decision frontier.
Families include core, reprobe, K/CR and the registered X comparisons. Core and
reprobe reference control jobs and the combined report can run before tier
files exist; K/CR reference jobs still require their tier binding.
Reference guards needed by admission must be available before the CPU admission
reader is requeued. A later complete frontier must re-score every input of its
contrast in one job; do not append old reference cells to a new job. Never
submit a job whose list includes a checkpoint that requires its own decision.
CPU preparation archives an earlier same-family/seed reference frontier under
`reference-history` before a retry or later frontier. Each new GPU attempt writes a fresh nonce tree; its current-family/seed pointer
selects only that job in readers. Every checkpoint is rescored on each attempt. Readers validate its reference authority
separately from native scores and exclude the archived attempts from tables.

## Allocation identity and retained evidence

The scheduler chooses the physical GPU. Historical assignment documents are
kept byte-for-byte. A new proof binds their SHA256 and historical family/seed
record, plus the actual UUID/environment and a fresh allocation job identity.
The runner checks the physical UUID at every launch and completion. A later
job never requests a historical UUID as its admission condition.

Core complete comparison jobs re-score their own baselines, incoming checkpoints
and controls. `R-compare-*` feeds nomination; `M8-compare` and `M9-compare` feed
admission; `M-final-score`, `X-complete` and `X15-X20-complete` feed final readers.
CPU publication keeps earlier jobs in immutable `lane-evidence/TASK` score sets.
CPU readers choose a complete superset for each family/seed and refuse competing
equal-coverage jobs. Hash checks cover each set. They never combine incomparable
frontiers to manufacture a complete contrast. CPU materialised reader views are
removed before snapshot publication; original sets remain intact.

Every adaptive NLL contrast remeasures its stage-start checkpoint on the current
job's GPU, including at steps 20/30/40. The registered scalar EMA still uses its
previous signal; the within-frontier before/after difference is local to one job.
A failed GPU attempt requires CPU staging before retry. The new attempt rescans
all prerequisites and re-scores the whole lane; failed attempts remain costs.

RA retained/reused inputs are fresh calls in the same package job. Original
rulings and historical scores stay as provenance. No new score is relabelled as
historical. K/CR uses separate scoring workspaces per job; its final per-seed job
re-scores all arms, incoming checkpoints, panels and registered curve points.
Actual final UUIDs must remain distinct across its seeds. Reused seed GPUs or
mixed allocation jobs produce UNVERIFIED-HARDWARE, without a GPU admission wait.
SP scores the full connected assignment in one job. Its CPU reader freezes
incoming cohorts after complete panel scoring. Actual SP/parent overlap,
repeated SP assignment UUIDs, or mixed jobs are reported UNVERIFIED-HARDWARE;
scoring does not refuse a scheduler-selected GPU because of an old desired UUID.
PD retains its approved pre-training cohort, same-job 18-cell post curve and
strict pre/post update-0 reload agreement checks.

## Additional scoring cost

All numbers below are additional calls relative to the merged pre-integration
graph, not claims of measured GPU-hours. Let `h(C)` be the measured sum of the
calls in set C, divided by 3600 for seconds; watcher allocation wall time is the
billable one-GPU job time. Concurrency means process-time sums are not allocation
GPU-hours. Charge model loading, incomplete attempts, tails and queue/CPU work
separately.

| Changed lane | Additional calls | Additional GPU-h |
|---|---:|---|
| R broad/full comparison jobs | 124 per seed, 372 total | Unmeasured; three jobs each of h(43) and h(81), with concurrency |
| M8/M9/final complete jobs | Up to 260 per seed, 780 total | Unmeasured; nomination/admission branches determine exact call set |
| X tier-1 complete | Up to 63 per seed, 189 total | Unmeasured; disabled/deferred arms remain skipped |
| X tier-2 complete | Up to 39 per comparison/seed, 702 total for six comparisons | Unmeasured; READY comparisons only |
| X adaptive measurements after step 10 | One fresh baseline per domain/frontier | Unmeasured; exact domain/frontier counts are in CPU plans; training dose unchanged |
| RA A101 / A102 / A103 | 2 / 4 / 4, ten total | Historical serial call costs sum to 0.060993 / 0.117036 / 0.116565 (0.294594 total); these are planning proxies, not new allocation measurements |
| Optional RB101 / RB102 / RB103 | 10 / 11 / 26 | Historical serial proxies 0.297272 / 0.308210 / 0.601411; current allocation cost unmeasured |
| K/CR intermediate stage jobs | 216 total fresh incoming panel calls | Unmeasured; replaces CPU aliases with same-job scientific scores |
| K/CR final seed jobs | 98 extra per seed versus the former 100-command rescore, 294 total | Unmeasured; 198 commands per complete job |
| Reference lane merger | Zero calls | Same panels/checkpoints, two panels in one job; current HF utilisation unqualified |
| Reprobe reference frontier | Two calls per decision checkpoint in the supplied list | Unmeasured; makes the required independent reference guards explicit in the inventory; each new frontier repeats its complete contrast |
| PD, SP and generation sharding | Zero scientific calls | PD remains 63–98 GPU-h planning range; CPU costs and current allocation rates unmeasured |

Do not add historical per-process proxies to the revised allocation budget as
if they were measured concurrently. The inventory leaves unmeasured components
unknown instead of claiming the 1,800 GPU-h allowance is proven sufficient.

## Infrastructure authority

`resources/v6/infrastructure-update.json` is generated against `main` with
`scripts/v6_infra_authority.py`, covering exact existing Python and shell-code
pin transitions. Existing registrations, SHA256SUMS, runtime maps, historical
receipts and pinned send expectations are unchanged. The authority's exact
SHA256 must be bound by the newly generated send/protocol/freeze identity before
any checker accepts an amended byte pin. Arbitrary byte changes still fail.

New identities include the authority SHA256; the core protocol implementation
map (including `lane_pool.py`, `lane_evidence.py`, `util_watch.py`); launch task,
resource, command and dependency hashes; S/X bindings that incorporate that
protocol; new scoring/validation/reference proofs and manifests; M3 scoring
workspaces and SP inventory/sidecar hashes; and PD's prospective
`split_job_sha256`. The qualified PD numerical core and its historical code and
smoke identities do not change. New bindings are produced from the node's own
archives; pinned historical send bindings are never regenerated in this build.
The launch's `lane_runtime` identity seals the exact authority and all thirteen
shared lane, allocation, reference, M3 and SP implementation files listed in
`v6_infra.LANE_CODE`. Child processes receive that frozen seal. PD includes the
same runtime identity in its split-job implementation hash.

PD storage now reserves a peak 16 checkpoint bounds plus 64 GiB per host under
flock (normally 352 GiB/seed). Finalize verifies each HF export and then deletes
its native snapshot; post keeps all HF exports until all 18 cells finish. This
prospective deviation from the old 256-GiB/one-pending-export storage rule must
be recorded beside the 2 November ruling. Generation configs are byte-preserved
from each source. PD GPU launchers discover CUDA identity in a short subprocess
so the launcher retains no CUDA contexts while its eight training ranks run.
