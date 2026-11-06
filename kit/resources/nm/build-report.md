# NM fix round 2 handoff

N1 and N2 are FIXED. Frozen registrations, precedence, seeds, doses, thresholds,
scorer arithmetic, GPU/resource requests and utilisation rules are unchanged.

| Finding | Resolution |
|---|---|
| N1 | Production reader/CLI and CPU stand-ins distinguish missing primary evidence from registered PENDING outcomes with computed endpoints. A computed GRPO D-insensitive reading completes its read row, passes afterok to collection, and reaches dispatch text. A real-wrapper GRPO-only rehearsal imports that text through `nm_return --text` and exactly re-derives PENDING/INCOMPLETE, primary, guards and zero D sensitivity. Missing-primary pipeline/CLI regressions still exit 75. |
| N2 | Every collector publishes one locked, cumulative dispatch cycle, replacing its own family rather than adding another family-specific paste. `out/return-cycle.json` points to the latest immutable part set. ESSENTIAL is capped at four parts TOTAL, each at most 11,400 characters, lines at most 89. Exact per-seed/panel/arm integer counts reproduce native percentages, paired differences, gains, AUCs and all registered bars, with the original arithmetic order. Package summaries, prior-month diagnostics and per-phase attempts are retained in FULL on the node and omitted from ESSENTIAL. Overflow refuses a new cycle without changing the previous pointer; FULL remains locally available. |

## Measured capacity

The capacity fixture exercises all six families together, all **66 allowed OPR
disagreements** (11 for each seed/reference), 64-character item IDs, nonconstant
seed/curve values, prime-sized general/legal panels, the registered FinQA/SEC
denominators, every accounting phase, six-digit attempt totals and long GPU-hour
precision. It is explicit CPU transport evidence, not a scientific run.
The before formatter is the exact `45fc01ae` implementation, run on the same
readings/accounting and kit identity as the after formatter.

| Format | Total pasted characters | Total ESSENTIAL parts |
|---|---:|---:|
| Before, six separate family sets | 59,935 | 7 |
| After, one complete dispatch | 29,983 | 3 |

After part sizes are **11,083 / 11,073 / 7,827** characters; longest line **89**.
The hard budget also refuses larger/unexpected evidence rather than truncating
it. FULL has no four-part limit and stays on the node until requested.

The real-wrapper 444-row stress rehearsal retained **447 attempts**: 444
COMPLETE, two PENDING, one injected FAILED, followed by new-nonce resubmissions.
On the same real-wrapper evidence the former format was **55,171 characters
in 7 parts**. Its final dispatch is **29,858 characters in 3 parts** (11,063 / 11,123 / 7,672),
with 89-character maximum lines and all six verdicts exactly re-derived. It also
includes all 66 OPR disagreements. The separate GRPO D-insensitive rehearsal
completes read and collect while retaining its registered scientific status.
Archive, hardware and replay-shortfall paths remain covered.

## Integration and verification

Changed NM files: `kit/nm_read.py`, `nm_pipeline.py`, `nm_text.py`, `nm_sim.py`,
`README-next-message.md`, this report and NM jobs/pipeline/read/text tests.
**Only shared-file edit this round: `kit/text_return.py`**, one dispatch-routing
condition. `kit/report_text.py`, inventory rows and all seven frozen registration
files are unchanged. No dependency, GPU run, service call, tag or dispatch.

Final verification:

- `python -m pytest tests/test_nm_*.py -q --basetemp tmp/nm-fix2-final`:
  **154 passed** in 96.77 seconds, including every-row and GRPO-only dry runs.
- `python -m pytest tests/test_text_return.py -q --basetemp tmp/nm-fix2-shared-text`:
  **72 passed** in 195.03 seconds.
- Both suites' basetemps and all intermediate test/measurement directories were
  deleted after reducing their results into `tmp/util70/nm-fix2-validation.json`.

The final commit is recorded in the external fix-round handoff. Modified modules compile and whitespace
checks pass. The production H100 qualification and manager infrastructure-authority
regeneration requirements remain as documented in the earlier handoff.

---

# NM fix round 1 handoff

All review findings against `fd97c64e` are fixed. The following initial-build
record is retained as history; this section describes the current tree.
The frozen registrations, precedence, thresholds, seeds, doses, scorer numerics
and resource requests are unchanged. No GPU, Modal, external service, tag or
scientific dispatch was used.

| Finding | Status | Change and regression evidence |
|---|---|---|
| B1 | FIXED | Default pasted-text return per family through report_text/text_return; shared KIT framing and T/D/E records, unrounded paired values for primary/every guard, package summaries, GRPO D/general diagnostics, OPR cohorts/agreements/disagreeing IDs/coverage/fallback, Fisher norms/median/coefficients, iSDFT persistence, lane hashes and all-attempt allocation accounting. CPU import recomputes bars; checksum damage is advisory. Archives are opt-in. Six-family round trips, altered guard/decision, threshold precision, missing/damaged checksums, wrapping, missing/duplicate/mixed parts and CPU CLI tested. |
| M1 | FIXED | Pass the inventory lane_concurrency, bind K=3 qualification to each exact family/panel workload, otherwise record K=1. Missing receipts at preparation or execution select K=1. Cross-panel/changed evidence refuses. CPU planning and actual execute-call arguments tested. |
| M2 | FIXED | fp32 accumulators and trace normalization stay on device; addcmul_ per example, one final host publication per shard. Analytic individual-square and two-coordinate normalization fixtures rerun; host copies forbidden during accumulation/normalization by regression. |
| M3 | FIXED | Lock/seal out/incumbent.json at first applicable resolution. Later selection, recipe and K/CR evidence must match; only resolved applicable P2/nomination evidence can be added. Changed-evidence and changed-recipe second binds refuse. Pending later evidence can resolve without prematurely freezing its temporary hash. Real pipeline bind tested. |
| m1 | FIXED | Missing or wrong-recipe iSDFT persistence raises Pending; tested, including an already-frozen I. |
| m2 | FIXED | Missing top-level/nested spec bindings use ready/get; all six readers tested with missing panels/checkpoints/scorers. |
| m3 | FIXED | Exact NON-INCUMBENT BENCHMARK label in OPR/Fisher status or scope/interpretation; no incumbent admission. Native and pasted readers tested. |
| m4 | FIXED | GRPO descriptive equal-weight general-retention stats added and transported; direct diagnostic test. |
| m5 | FIXED | Each score-prepare depends on its own seed's two exports. September contexts depend on the frozen v2b schedule, with only SEC lanes depending on them. DAG/resource fields and every seed checked. |
| m6 | FIXED | Restore task demonstration mask check on distillation task rows. Both D and iSDFT tested; verifier-only GRPO remains independent of demonstrations. |
| m7 | FIXED | Confirmed pinned EMA uses .99/.01 mul_/add_. Keep student bit identity; compare frozen EMA to exact native-dtype recurrence, including resume count. fp32/bf16 fixtures at 0/4/8/20/40 updates accept rounding and refuse movement. |

## Files and shared integration surface

NM-owned edits: `nm_common`, `nm_bind`, `nm_loss`, `nm_fisher`, `nm_actor`,
`nm_native`, `nm_jobs`, `nm_pipeline`, `nm_read`, `nm_return`, `nm_sim`; new
`kit/nm_text.py`; regenerated `resources/nm/rows.json`; runbook and this report.
Only 121 inventory dependency lists changed; all other row fields are identical.

**Shared-file edits: `kit/report_text.py`, `kit/text_return.py`.** The reporter
accepts `_mode=nm` with reading/accounting/externally bound authority inputs;
production authority verification remains required. The transport exposes the
common framing helper, dispatches NM writing/importing to its evidence adapter,
and preserves existing family behavior. NM source seals include both shared
files. No infrastructure authority or registration was edited.

Two new test files (`test_nm_pipeline.py`, `test_nm_text.py`) and regressions in
NM binding/read/math/actor/jobs tests cover every finding.

## Final verification and dry run

- `python -m pytest tests/test_nm_*.py -q --basetemp tmp/nm-fix1-release`:
  **146 passed** in 79.69 seconds.
- `python -m pytest tests/test_text_return.py -q --basetemp tmp/nm-fix1-text-existing`:
  **72 passed** in 196.38 seconds.
- Modified Python modules compile; `git diff --check` passes. All registration
  copies compare byte-for-byte with the frozen handoff.
- Main real-wrapper/watcher rehearsal: **444 rows, 447 unique attempts**:
  444 COMPLETE, two PENDING, one injected FAILED. Pending dependency and failed
  trainer resubmissions use new nonces. All six text returns rederive their
  technical verdicts without an archive. Each fixture return fits one part,
  roughly 5–8 KB, max line length 89; multipart transport is tested separately.
- **142 archived paths**, replay coverage/cohort shortfalls, mixed hardware and
  optional archive readback also pass. CPU stand-ins allocate zero paid GPU-h.
- All test basetemps are under this worktree's tmp/ and removed after reduction.

Real H100 longest-input memory, distributed/native resume/export qualification,
per-panel K=3 receipts, rolling utilisation and measured pricing remain production
requirements. Serial fallback preserves scorer numerics but carries no utilisation
claim. The manager still regenerates the infrastructure authority from the clean
integrated commit. Private metadata and the refreshed portable bundle remain
`tmp/util70/nm-git` and `tmp/util70/nm.bundle`; the final commit ID is in the
external handoff because a commit cannot contain its own ID.

---

# Initial build record

Builder NM, `build/next-message`, starting from integrated v6b snapshot
`d57b5146`. The six v3 registrations and the owner's 2 November approval were
read before implementation. Precedence, thresholds, doses and seed count were
not amended. No v6a/v6b verdict or new scientific outcome was assumed or opened.
No GPU, Modal, external service, tag or dispatch was used.

## Files

- `kit/nm_common.py`, `nm_bind.py`: constants, immutable receipts, externally bound
  authority handling, prospective K/CR/I0 resolution and per-family launch seals.
- `kit/nm_data.py`, `nm_development.py`: exclusion/schedule helpers, development
  chronology/retrieval, replay fallback/coverage and arm-independent cohorts.
- `kit/nm_loss.py`, `nm_actor.py`, `nm_native.py`, `nm_entry.py`: isolated native
  objectives/worker hooks, analytic Fisher/isotropic penalty boundary, block mask,
  checkpoint metadata and GRPO telemetry.
- `kit/nm_model.py`, `nm_prepare.py`: CPU model/index hashing, prompt rendering,
  exact schedules, native command composition, bounded allocation admission and
  CPU HF exports. The inherited reference loader/sharding numerics are reused;
  its isolated NM admission adapter consumes the sealed CPU hash receipt.
- `kit/nm_fisher.py`, `nm_fisher_stats.py`, `nm_generate.py`, `nm_score.py`:
  per-example estimator/preflight, exact streaming weight summaries, per-seed
  batched generation and unchanged scoring authorities.
- `kit/nm_jobs.py`, `nm_execute.py`, `nm_pipeline.py`, `nm_budget.py`: explicit DAG,
  real allocation wrapper, one watcher/lease, attempt trees and allocation costs.
- `kit/nm_read.py`, `nm_return.py`, `nm_sim.py`: frozen readers, portable CPU
  readback and explicit technical stand-ins.
- `kit/jobs/nm/run-task.sh`, `kit/jobs/nm/sim/nvidia-smi`;
  `kit/README-next-message.md`; `kit/resources/nm/rows.json`,
  `cpu-rehearsal.json`, this report and the seven unchanged registration snapshots.
- Nine new `tests/test_nm_*.py` files.

**Initial-build shared-file edits: none.** No pre-existing kit implementation, wrapper,
registration, resource pin, inventory, runbook or test was changed. The standalone
NM inventory/wrapper is the integration surface; it does not silently add an
unsupported group to the shared combined dispatcher. Manager integration can
use its explicit commands or retain the standalone wrapper.

## Settled-rule compliance

| Rule | Implementation/evidence |
|---|---|
| D1/D7 | 444 one-node rows: 260×0 GPU, 131×1 GPU, 53×8 GPUs; explicit CPU/sharing requests. No new teacher request because verified demonstrations are prerequisites. |
| D2 | Preparation, tokenisation, large hashes, calibration sealing, estimator publication, exports, readers and collection are CPU rows. GPU reference admission uses CPU-bound file metadata. |
| D3 | All required checkpoints/plans are checked before a scorer starts. Missing dependencies exit 75 immediately; no external-output polling. |
| D4/S4 | Contrasts use one complete job per seed/panel; observed UUID/environment/nonce, no historical GPU admission pin. Mixtures return UNVERIFIED-HARDWARE. |
| D5/S5 | Copied frozen registrations, original base, seeds 101–103, 40 updates and exact denominators. Fresh controls; no combination of new winners. |
| D6/S1/S2 | One watcher with exact expected GPU count and fresh path, one parent lease/idle check, inherited `M3_GPU_LOCK_FDS`. Ordinary qualified lanes cap at K=3/.31; reference lanes remain serial. |
| S3 | Launch-supplied infrastructure SHA only; missing/different authority refuses. Text header reports that externally bound SHA. Authority file unchanged. |
| D8 | New NM files, dense existing-kit style, `python -I`, no added dependencies or service calls. |
| D9 | CPU tests and all-row real-wrapper rehearsal; no GPU/Modal. Co-author trailer on every commit. Scratch basetemps removed after evidence reduction. |

The iSDFT KL shares D's student forward. Task and replay ranks therefore retain
one actor forward/backward per row, while their read-only base calls remain
aligned. Frozen P2 coordinates remain in scalar logging but receive no gradient
or decay. Fisher calibration restarts untouched parents and preserves original
measured calibration timings on resubmission. Published cost components use
Decimal half-up rounding and totals sum the published component amounts.

## Verification

**81 new tests passed:** 73 component tests and 8 real-wrapper tests.

Commands used:

```bash
python -m pytest tests/test_nm_binding.py tests/test_nm_budget.py \
  tests/test_nm_prepare.py tests/test_nm_actor.py tests/test_nm_data.py \
  tests/test_nm_math.py tests/test_nm_read.py tests/test_nm_return.py \
  -q --basetemp tmp/nm-pytest-final-component-ledger
python -m pytest tests/test_nm_jobs.py -q \
  --basetemp tmp/nm-pytest-shipping-wrapper
```

The final archived-reader CLI regression was run separately after the component
run. It proves an ARCHIVED UNREAD CLI invocation does not open either outcome or
specification files. Compilation, shell syntax, whitespace and static undefined-
name checks passed. Star-import diagnostics follow the surrounding kit style.

Independent CPU checks include individual gradient squares, trace/quantiles,
median Fisher calibration, analytic gradients, 1/2/8 reduction algebra, clipping
signs, reference isolation, exact inherited-D fixture equivalence, frozen masks,
GRPO zero/mixed groups, native configuration validation, issuer components,
source/plan tampering, all endpoints/guards and return reproduction.

Existing regressions:

```bash
python -m pytest tests/test_lane_pool.py tests/test_v6_infra_authority.py \
  tests/test_rung_p2.py tests/test_kit_run_grpo.py -q --basetemp tmp/nm-pytest-existing
```

**145 passed, 15 failed.** The identical 15 failure IDs were reproduced against an
untouched `d57b5146` snapshot inside this worktree, also 145 passed/15 failed.
They comprise the manager-owned current-pin authority release gate, 13 older P2
fixture/pin/authority/attempt assertions and one existing GRPO checkout-refusal
fixture. They were not changed to weaken authority or restore obsolete five-seed
expectations. The internal failure-ID comparison is in
`tmp/util70/nm-baseline-results.json`; the baseline snapshot/logs were removed.

## Dry run

All **444 rows** completed through `kit/jobs/nm/run-task.sh` and the real
`util_watch.py`, using explicit CPU stand-ins. The main rehearsal retained **447
unique attempts**: 444 completed, two PENDING and one failed. It demonstrated a
missing dependency, blocked dependant, interrupted trainer and fresh-nonce
resubmission. Every scorer proof agreed with its watcher's mock measured UUID.

Another **142 archived row paths** were exercised without challenger reads.
Separate wrapper tests blocked on <80% replay coverage and <100 cohort support,
and propagated mixed hardware as UNVERIFIED-HARDWARE through the real reader and
archive readback. All six archives reproduced their readers; text lines stayed
at most 90 characters. CPU results remain TECHNICAL-ONLY, never scientific
admissions. Allocated/paid GPU work was zero.

## Open release requirements

There are no new threshold/precedence design choices for the owner to make.
Production requirements remain explicit rather than fabricated:

- Actual returned K/CR/M8/M9/X/P2/stacking and parent/data bindings must be supplied
  at each applicable frontier; absent evidence stays PENDING.
- Real eight-H100 loss/gradient, longest-input memory, optimizer, native resume
  and export equivalence qualifications remain required. CPU fixtures do not
  establish these. Production qualification receipts are mandatory gates.
- Per-seed generation rolling utilisation, reference/SEC/GRPO utilisation,
  estimator/calibration timing E, exact reservations and >24-hour risks need
  measured preflight repricing. Serial HF references and loading/saves remain
  utilisation risks; numerics/batches were not changed to hide them.
- The manager must regenerate the infrastructure authority last from the clean
  integrated tree and obtain the separate tag/send approval.

The normal worktree index accepted the first commit, then refused the final
writes. Subsequent commits use private metadata at `tmp/util70/nm-git`, as the NM
instructions permit. A portable `tmp/util70/nm.bundle` and final commit details
are supplied with the final handoff. The ordinary worktree ref may therefore
still show the first commit; use the private metadata/bundle for the complete
series.
