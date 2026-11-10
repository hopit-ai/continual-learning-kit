# V7c operator note: CR-only, separate parent root

> **CR safety for newer tags:** CR runs only from a `kit-message-v7c` clone,
> checked against the owner's commit and receipt SHA256
> `04861c64970d3fb8e04936ec2042852120cc94cd04fb063a5084bb71066df380`.
> Clone any newer tag into a **NEW directory**, never into the v7c clone.
> On a newer tag, the CR refusal "confirmation code changed since diff review"
> is expected, not a defect. A newer tag's launch JSON carries no `cr_confirmation`.

<!-- v7c-receipt-status -->
The current [diff receipt](resources/cr-confirmation/diff-receipt-v7c.json) is **COMPLETE**.
The receipt records the owner approval quote, date and reviewed pairs for
`owner-v7c-fix`, bound by its code seal and the launch JSON `diff_receipt.sha256`.
Use the receipt SHA256 and tag commit stated in the owner's send message; stop if
`sha256sum` of the receipt or `git rev-parse HEAD` differs from those values.
<!-- /v7c-receipt-status -->
Against `059d6293d0dc8359df418abe216ce438e6bf147e`, the coverage closure and
both pinned owner approval lists cover **479 sealed files**, **143 changed files**
and **348 hunks**. Its SHA256 is
`04861c64970d3fb8e04936ec2042852120cc94cd04fb063a5084bb71066df380`.
This is code-review evidence; send approval, public-tag approval and node-side
realised-input evidence remain separate requirements. No scientific definition,
`nm_*.py` or infrastructure-authority bytes changed.

## Folder and root selection

Keep v2b, GRPO, OPR, Fisher, iSDFT and Freeze on the **unchanged kit-message-v7b
folder, six-row launch JSON and parent root A**. Use v7c for **only `NM-cr-*`**.
Give CR its own parent dispatch root B, launch JSON and confirmation data tree C.
A, B, C, restored v6a and restored v6b must be pairwise separate, non-nested trees.
CR writes B's own `return-cycle.json` with at most four ESSENTIAL parts; the
six-family return stays in A. Adding CR to A's bound launch changes its source
seal and is unsupported. Both `six_row_root` and `v6b_root` are required
before CR launch settings are accepted.

Deploy a **full clone of the approved public tag**, retaining all panels,
resources, scripts and history. A selected Python-file copy is insufficient:
`kit/panels/general-v1.jsonl` is required by scoring. Do not use a shallow clone
or a Git-free archive: receipt verification regenerates the baseline inventory.
For example, after the release tag is approved:

```bash
git clone "$KIT_REMOTE" /durable/kit-message-v7c
git -C /durable/kit-message-v7c checkout kit-message-v7c
git -C /durable/kit-message-v7c rev-parse --verify '059d6293^{commit}'
```

## CR-only root-B launch

Write `/durable/v7c-cr-launch.json` using the following shape. Replace the node,
input, tokenizer, trainer and approval values with the actual bound evidence.
The shipped receipt's path and SHA256 below are concrete; preserve its bytes.
Before launch, check them against the owner's send message, which states the
receipt SHA256 and the tag commit: stop if `sha256sum` of the receipt or
`git -C /durable/kit-message-v7c rev-parse HEAD` differs from the owner's values.
The [existing operator instructions](README-next-message.md) explain the inputs.

```json
{
  "out": "/durable/v7c-cr-parent-B",
  "job_python": "/durable/envs/kit/bin/python",
  "infrastructure_update": "4daf1f7f5624e3f9190831a1832d1183cb8b4032d310a9d0d027562ff058f400",
  "send_approval": "OWNER_BOUND_CR_SEND_APPROVAL",
  "cr_confirmation": {
    "out": "/durable/v7c-cr-data-C",
    "six_row_root": "/durable/v7b-six-row-parent-A",
    "v6a_root": "/durable/restored-v6a",
    "v6b_root": "/durable/restored-v6b",
    "other_roots": [],
    "hosts": ["APPROVED_NON_CORE_HOST"],
    "realised_inputs": {"path": "/durable/node-v6a-inventory.json", "sha256": "OWNER_BOUND_SHA256"},
    "diff_receipt": {
      "path": "/durable/kit-message-v7c/kit/resources/cr-confirmation/diff-receipt-v7c.json",
      "sha256": "04861c64970d3fb8e04936ec2042852120cc94cd04fb063a5084bb71066df380"
    },
    "operator_receipt": {"path": "/durable/operator-qualification.json", "sha256": "QUALIFIED_RECEIPT_SHA256"},
    "tokenizer": "/durable/pinned-Qwen3-8B",
    "sdpo": "/durable/pinned-SDPO",
    "trainer_python": "/durable/envs/trainer/bin/python",
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
path = Path('/durable/v7c-cr-launch.json')
launch = json.loads(path.read_text())
launch['cr_confirmation']['budget']['row_max_seconds'] = json.loads(
    Path('/durable/kit-message-v7c/kit/resources/cr-confirmation/row-max-seconds.json').read_text())
path.write_text(json.dumps(launch, indent=2) + '\n')
```

CR selects `cr_confirmation.budget`; failed attempts and retries remain charged
to `CR_confirmation=830`. Its reservations and reconciled allocation costs live
under C. A top-level six-family budget does not supply the CR allowance.

After actual approvals and realised evidence are bound, invoke each CR row in
its own scheduler allocation with its registered resources and dependencies:

```bash
export KIT=/durable/kit-message-v7c/kit
export NM_CONFIG=/durable/v7c-cr-launch.json
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
