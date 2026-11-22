# V7c operator note: CR-only, separate parent root

> **CR safety for newer tags:** CR runs only from a `kit-message-v7i1` clone,
> checked against the owner's commit and receipt SHA256 from the approved send.
> Clone this tag into a **NEW directory**, never into an existing experiment clone.
> CR refuses unless diff-receipt-v7i1.json records review COMPLETE and owner
> approval APPROVED. Never edit it; if it reads PENDING, ask the owner.
> A code-seal mismatch requires a new reviewed receipt.

<!-- v7i1-receipt-status -->
CR refuses unless diff-receipt-v7i1.json records review COMPLETE and owner
approval APPROVED. Never edit it; if it reads PENDING, ask the owner.
The original v7c receipt stays unchanged as historical review evidence.

## Folder and root selection

Use a **fresh kit-message-v7i2 clone, regenerated six-row launch JSON and fresh
parent root A** for v2b, GRPO, OPR, Fisher, iSDFT and Freeze. Their registered
settings are unchanged; v7i2 fixes their scheduler checkpoint save. Use v7i1 for **only `NM-cr-*`**.
Give CR its own parent dispatch root B, launch JSON and confirmation data tree C.
A, B, C, restored v6a and restored v6b must be pairwise separate, non-nested trees.
CR writes B's own `return-cycle.json` with at most four ESSENTIAL parts; the
six-family return stays in A. Adding CR to A's bound launch changes its source
seal and is unsupported. Both `six_row_root` and `v6b_root` are required
before CR launch settings are accepted.
`six_row_root` is an isolation label only: any empty, separate folder passes.

Deploy a **full clone of the approved public tag**, retaining all panels,
resources and scripts. A selected Python-file copy is insufficient:
`kit/panels/general-v1.jsonl` is required by scoring. The new delta is portable: `v7i-baseline.json` stores before text only for
newly edited files, from v7i plus the test-redaction cherry-pick. Every inherited
v7c → v7i transition binds hashes and the first public tag/commit where the v7i
content shipped (or private source if unavailable). No historical v7c text or
patch is shipped. Public clones need no private Git objects to verify it.

For example, after the release tag is approved:

```bash
git clone "$KIT_REMOTE" /durable/kit-message-v7i1
git -C /durable/kit-message-v7i1 checkout kit-message-v7i1
git -C /durable/kit-message-v7i1 rev-parse HEAD
```

Before GPU rows, run on the Linux node's CPU: `python -m pytest -q tests/test_cr_round2.py -p no:cacheprovider` (includes the TERM-resistant detached-descendant check); Linux execution here is **UNVERIFIED**. The CPU bind row also runs a CUDA-free detached-worker teardown preflight with the same supervisor; missing `/proc` children inventory or unconfirmed teardown refuses binding. Later inventory failure prevents confirmed teardown.

## CR-only root-B launch

Write `/durable/v7i1-cr-launch.json` using the following shape. Replace the node,
input, tokenizer, trainer and approval values with the actual bound evidence.
The receipt path is fixed; replace its PENDING hash only after owner approval.
Before launch, check them against the owner's send message, which states the
receipt SHA256 and the tag commit: stop if `sha256sum` of the receipt or
`git -C /durable/kit-message-v7i1 rev-parse HEAD` differs from the owner's values.
The [existing operator instructions](README-next-message.md) explain the inputs.

```json
{
  "out": "/durable/v7i1-cr-parent-B",
  "job_python": "/durable/envs/kit/bin/python",
  "infrastructure_update": "89f7b8dd80dc111f65978cf887469044c54feda070150303194390ee5057a32e",
  "send_approval": "OWNER_BOUND_CR_SEND_APPROVAL",
  "cr_confirmation": {
    "out": "/durable/v7i1-cr-data-C",
    "six_row_root": "/durable/v7i2-six-row-parent-A",
    "v6a_root": "/durable/restored-v6a",
    "v6b_root": "/durable/restored-v6b",
    "other_roots": [],
    "hosts": ["APPROVED_NON_CORE_HOST"],
    "realised_inputs": {"path": "/durable/node-v6a-inventory.json", "sha256": "OWNER_BOUND_SHA256"},
    "diff_receipt": {
      "path": "/durable/kit-message-v7i1/kit/resources/cr-confirmation/diff-receipt-v7i1.json",
      "sha256": "PENDING_OWNER_APPROVED_RECEIPT_SHA256"
    },
    "operator_receipt": {"path": "/durable/operator-qualification.json", "sha256": "QUALIFIED_RECEIPT_SHA256"},
    "tokenizer": "/durable/pinned-Qwen3-8B",
    "sdpo": "/durable/pinned-SDPO",
    "trainer_python": "/durable/envs/trainer/bin/python",
    "bind_python": "/durable/envs/trainer/bin/python",
    "finqa_panel": "/durable/FinQA",
    "panels": ["NAME=/durable/frozen-panel"],
    "holdouts": ["NAME=/durable/frozen-holdout"],
    "host_reservation_bytes": 1099511627776,
    "budget": {
      "branch": "CR_confirmation",
      "approved_gpu_hours": 830,
      "approval_reference": "OWNER_BOUND_830_GPU_H_APPROVAL"
    }
  }
}
```

Before binding, load the complete shipped duration map into the **nested** CR
budget. Run this snippet with the selected environment's `python -I`:

```python
import json
from pathlib import Path
path = Path('/durable/v7i1-cr-launch.json')
launch = json.loads(path.read_text())
launch['cr_confirmation']['budget']['row_max_seconds'] = json.loads(
    Path('/durable/kit-message-v7i1/kit/resources/cr-confirmation/row-max-seconds.json').read_text())
path.write_text(json.dumps(launch, indent=2) + '\n')
```

CR selects `cr_confirmation.budget`; failed attempts and retries remain charged
to the separate `CR_confirmation` ledger. The 830 GPU-hour allowance is
warn-only: crossings record WARN in the row receipt and account output and never
refuse the row. Its reservations and reconciled allocation costs live under C. A top-level six-family budget does not supply the CR allowance.

`bind_python` is required inside `cr_confirmation` and is included in the launch
digest. It runs only the bind-time native FinQA worker in the training environment;
`job_python` still runs scoring in the inference environment.

CR may run several rows in one holder with the registered resources and
dependencies. Each row reserves its GPU count times its own prospective duration
plus wrapper age and the existing 30-second slack. Charge is row GPUs ×
wrapper-measured MONOTONIC elapsed time through confirmed teardown of the whole
process tree, including the watcher's child in its own session. UTC start/end
remain audit fields; wall-clock adjustments do not change exact charges.
Linux child-subreaper adoption keeps orphaned descendants under the wrapper;
TERM-resistant descendants receive KILL and cleanup is checked before completion.
Only an attempt-bound, hash-consistent terminal receipt supplies exact cost.
Existing receipt/pointer hash mismatches are rejected, never repaired.

If teardown is unconfirmed, the wrapper dies, or completion evidence is missing,
its reservation stays while the scheduler holder runs. After the scheduler
confirms the holder ended, charge row GPUs × (holder end − row UTC start),
labelled `COMPLETE-UPPER-BOUND`. This deliberately includes possible holder idle
time as a conservative bound. Reconciliation reports scheduler holder GPUs ×
life, row sum and `holder_unattributed_gpu_hours`; rows exceeding the holder
produce a warning, never a refusal. Missing scheduler GPU/life evidence is
reported explicitly. The no-idle-holding rule remains mandatory. Receipt and
completion hashes retain the existing local operator trust boundary; an
operator who replaces all evidence can still forge local records.

The 26 GPU-row ceilings total exactly 896 GPU-hours, 66 above 830, before the
30-second reservation slack (1.383333 GPU-hours across all rows). These are
prospective bounds, not measured runtime or a spend estimate.

After actual approvals and realised evidence are bound, invoke CR rows:

```bash
export KIT=/durable/kit-message-v7i1/kit
export NM_CONFIG=/durable/v7i1-cr-launch.json
export JOB_PY=/durable/envs/kit/bin/python
bash "$KIT/jobs/nm/run-task.sh" NM-cr-bind
```

## Coverage review and cache mitigation

The dependency coverage is closed under the owner's 8 November evening approval
of all three review groups, including the seal-rule hunk 112 and conditional
cache hunk 96. The explicit frozen list is `cr_dependencies.FROZEN_DEPENDENCIES`;
the receipt seals it and the approval resources. It includes optional imports,
computed worker/plugin paths, literal child entry points and conservative kit
Python/shell coverage. A regression test rejects an uncovered confirmation
import or runtime plugin. Repository bytes and baseline diffs are covered;
installed external packages remain outside this repository receipt.

The original **26 exact (file, hunk SHA256) pairs** remain in
[owner-approved-hunks.json](resources/cr-confirmation/owner-approved-hunks.json).
The additional **169 exact pairs** and the evening approval text are in
[owner-approved-169.json](resources/cr-confirmation/owner-approved-169.json).
Both resource hashes are pinned in `cr_confirmation.py`.
`owner-v6-howchange` accepts only exact pairs in the pinned resources. The 15
coverage hunks assigned existing categories keep those review assignments by
exact pair. A category carries forward only for an unchanged (file, SHA256)
pair. The three changed CPU stand-in files are `technical-standin` by exact
pair after a named manager review, and the three v7c fix files
(`kit/cr_confirmation.py`, `kit/cr_jobs.py`, `scripts/publish_kit.py`) are
`owner-v7c-fix`, which is valid only for those three paths and only with the
owner's recorded approval of their reviewed pairs.

Before CR settings are evaluated, and again before the allocation operation
imports runtime scorers, CR clears all four cache-related environment variables:
`M2_MODEL_HASH_RECEIPTS_FILE`, `M2_MODEL_HASH_RECEIPTS_FILE_SHA256`,
`M2_MODEL_HASH_RECEIPTS_SHA256` (the spelling actually read by the cache), and
`M2_EXECUTION_PHASE`. Every child inherits the cleaned environment. Model
identity therefore uses fresh hashing. Tests launch CR with all variables set
and an invalid cache receipt; the real installed hook takes its uncached path
and child processes receive none of these variables. No cache/scorer file is
modified by this mitigation.

## Runtime paths beyond CPU rehearsal

The CPU stand-in never starts the production `m2_cached_score.py` HellaSwag
child or a real vLLM engine and `inference_worker.MemoryProfileWorker`. These
paths are included in the reviewed, sealed closure. Their execution on a real
GPU node, node-side realised-input acceptance, and scientific qualification
remain requirements beyond CPU rehearsal. CR uses its own serial scoring
operation and bypasses the `v6a.py` manifest-executor rescore helpers.

## Resealing after merges

Run `python -I scripts/reseal_cr_v7i1.py` from the full checkout after merging any
sealed-file change. It binds inherited hashes against approved v7c commit
`921e8e4613a542f3108a26fd7e7c1bc5938e3a61`, and regenerates only post-v7i/redaction hunks, the owner packet, and the CR
resource SHA256SUMS. Later TRACE hunks in shared files are labelled
`owner-v7i1-fix (TRACE)`; review them after the merge. It retains approvals only for identical code and hunk bytes;
changed bytes reset them to PENDING. The manager fills `review` and
`owner_approval` after review and owner approval, sets `status` to COMPLETE and
`escalation` to null, reruns resealing and verifies the final receipt. The new
receipt is not itself in the code seal, avoiding a self-hash cycle; the launch
JSON binds its SHA256 supplied in the owner's send message.
