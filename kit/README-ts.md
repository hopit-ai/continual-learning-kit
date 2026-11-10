# Teacher-signal comparison operator draft

**DRAFT - NOT FOR SEND**

This build implements the owner-approved four-arm SDPO-source adaptation. Registration, realised replay binding, mechanical GPU qualification, measured pricing and separate run approval remain pending. Seeds are 101–103. Each arm starts from the original base for its seed, then inherits its own preceding stage final along FinQA → ContractNLI → hard MATH.

## Arms and objective

| Arm | Teacher context | Task mask | Attempts per task |
| --- | --- | --- | --- |
| D | Verified 27B worked solution | One | One |
| M | Verified 27B worked solution | Any of eight passes | Eight |
| S | First successful distinct sibling, with the edge cases below | Any of eight passes | Eight |
| H | First successful distinct sibling, with the edge cases below, then verified 27B fallback | One, subject to named model-output error masking | Eight |

Attempt zero supplies every loss-bearing response token. Auxiliary request seeds derive from seed, stage, update, scheduled position, question ID and attempt index. The complete incumbent batch finishes first. The seven siblings for each task run concurrently through `asyncio.gather`, each with its own seed and sampling-parameter copy. Replay receives its stored-target CE.

A distinct sibling differs from attempt zero in token IDs and whitespace-normalised text. Normalisation joins whitespace-separated words with a single space. Selection uses the first successful distinct sibling by fixed attempt index. When attempt zero succeeds and every successful sibling is a duplicate, S uses the fixed positive message `The attempted answer passed verification.` H uses the verified 27B fallback in that case.

The manager-declared failed-target edge case applies to both S and H. If attempt zero failed and all successful siblings are non-distinct from it, use the first successful sibling anyway. The distinctness rule protects a student's own correct answer. A failed target does not qualify for that protection. Telemetry counts these rows under `nondistinct_success_context`.

All-failure S uses its plain teacher prompt and task mask zero. Model-origin decoding, response-cap, finish-telemetry, checker and unexpected selection cases produce task mask zero with a named reason for the scheduled row. An oversized sibling teacher context also produces a named zero-mask row and a plain prompt. Request identity, incomplete infrastructure output, input schedule and model-state corruption remain technical failures. The ledger and return text include the reason counts. Missing demonstrations and malformed auxiliary record schemas raise kit/data errors before model-output handling. Each stage with masked task rows emits a warn-only masked-rate line in the return text, including its numerator and denominator. The warning never gates a chain.

The objective is `(sum(mask * D_task_loss) + sum(replay_CE)) / 32`. It uses full-vocabulary forward KL, the first-three-token skip, the incumbent mean importance correction capped at two, and EMA .99/.01. Every scheduled batch takes the optimizer, scheduler and EMA steps, including replay-only M/S batches. Nonfinite task loss on a masked row remains a technical failure under the incumbent strictness.

Each stage has 40 updates and exports at 0/10/20/30/40. Full-weight AdamW uses LR 1e−5, weight decay .01, clipping one and ten-step warmup including zero LR. Optimizer, scheduler and EMA initialize afresh from the incoming student each stage.

## Prompt construction and finish semantics

Sibling and feedback routes render the literal incumbent `TEACHER_TEMPLATE` into a teacher-only copy. The adapter snapshots the raw student prompt before generation. Construction checks that snapshot against the student prompt and verifies that teacher construction preserved it. This structural check accepts short labels or table numbers occurring naturally in the question. The verified 27B routes retain the incumbent `check_student_prompt_clean` guard.

The wrapper presents every context as an example response, including the fixed feedback message. Its exact wording is retained below.

```text
<original last user content>

This is an example for a response to the question:
<context>

Now answer with a response of your own, including the thinking process.
```

The pinned naive reward manager derives its finish-gate flag by testing for `tokenizer.eos_token_id` anywhere in the valid response token IDs. TS calls the incumbent FinQA reward function with exactly that flag for attempt zero and every sibling. The raw vLLM `finish_reason` is also carried for every attempt. The pinned async server discards the distinction between `stop` and `length`, so TS uses a passive, request-context-local engine observer to carry that field through its own `TokenOutput` subclass and agent `extra_fields`. D and M use identical observers. Original valid token IDs are preserved in full.

Thus an alternate stop token follows the pinned EOS-membership gate. FinQA requires its inherited `Answer: <number>` format. The real FinQA fixture exercises both a correct formatted short answer and a bare table-number overlap during structural prompt construction. Legal and rational-MATH retain their frozen text parser semantics. All checker inputs stay isolated from feedback construction.

Near-duplicate telemetry compares each of the seven siblings with its attempt zero, independently of success. It reports a numerator, denominator and rate using character-trigram-set Jaccard similarity ≥0.9 after whitespace normalisation. Exact normalised duplicates are included. Named technical-error rows are excluded from this rate and counted separately under `near_duplicate_excluded_pairs`. Selection follows the declared distinctness rule.

## Replay binding, rows and scoring

The [row inventory](resources/ts/rows-ts.json) contains both DAGs. Its `registration` and `replay_binding` fields are null. Select the admitted binding after the v7b return and freeze its realised recipe receipts.

| Binding | Task rows | Replay rows by stage | Prompt/response/teacher caps |
| --- | --- | --- | --- |
| I0 | 28 | 4 / 4 / 4 generic | 2048/2048/6144, 10240/2048/14336, 2048/2048/6144 |
| CR | 32 | 8 / 16 / 24 | 10240/2048/14336 throughout |

CR uses four instruction, two GSM8K and two ARC replay rows throughout, plus eight FinQA rows in stages two and three and eight legal rows in stage three. All arms read the same frozen ordered schedule for each seed/stage. Preserve task IDs, demonstrations, replay targets, slots, balancing policy, exclusions and pool identities in the binding package.

Each binding has 36 training stages, two generation-only identity probes and 21 scoring lanes (three seeds × seven panels). Full four-arm execution still has 453 score cells per binding. Each lane's CPU preparation depends on the binding and every scheduled arm's three stage exports for that seed, so no GPU is allocated while waiting for exports. After allocation, the lane gate checks those prepared dependencies; the lane then scores base, acquisition references and all scheduled arms' checkpoints serially on one physical GPU. The reader requires one measured allocation nonce, GPU UUID and environment across that entire (seed, panel) frontier. A failed or interrupted lane is rescored in full. Deferred arms are omitted from the scheduled frontier; contrasts requiring them remain PENDING, with the arm marked PENDING/DEFERRED. Missing lanes leave their affected evidence PENDING.

Panel lanes cover FinQA-1147, legal-test-100, hard-MATH-100, strict instruction-100, general-maths-100, knowledge-100 and HellaSwag-100. Knowledge and HellaSwag use the incumbent full-precision reference scorers. The reader derives S−D and H−D primary contrasts and S−M and M−D secondary contrasts from validated item vectors. R equally averages stage-three changes on instruction, general maths and knowledge against base, FinQA against that arm's stage-one final, and legal against its stage-two final. Paired three-seed intervals use sample SD and t=4.302653. Benefit requires mean ≥2 pp, lower >0 and three positive differences. Every paired guard requires its lower bound ≥−1 pp. Unmet bars produce INCONCLUSIVE.

Readings retain acquisition gain/AUC, acquired-answer survival denominators, answer transitions, common acquired cohorts, training histories and token costs including incorrect answers. S/M mask ledgers report intersections, arm-only eligibility, neither-eligible counts and Jaccard overlap on matched scheduled IDs. Zero-correct token cost is undefined. The inherited 1.5× serving ceiling is reported at cost. The collector produces numbered path-free return parts.

## Resume and mechanical qualification

Native checkpoints retain model, optimizer, scheduler, RNG, EMA, update counters and history. On resume, active mask-ledger entries newer than the restored checkpoint are copied into a unique superseded-history directory. A durable audit records their hashes, restored checkpoint and reason before those entries leave the active ledger. Earlier entries preserve their bytes. Regenerated updates become the active entries. The stage-row lock protects this operation. A CPU test crashes each M/S/H simulation after update 15, restores checkpoint 10, changes subsequent checker outcomes and completes all 40 steps.

The mechanical-qualification rows are `qualify-D-prepare`, `qualify-D`, `qualify-M-prepare`, `qualify-M` and `qualify-attempt0-read` within each binding. Both probe jobs initialize identical weights and seed 101, generate the complete update-one batch through the normal native entry and stop before the first optimizer update. M also completes its seven siblings. The passive observer records the actual attempt-zero request before dispatch, including prompt token IDs and a deep copy of sampling parameters with no seed. The reader requires 100% request identity and matching scheduled identities over all 28 I0 task slots. CR checks all 32 slots. Matching incoming-weight and schedule digests are mandatory. Response-token identity is advisory and never gates training. An existing hash-bound D control can be supplied as `attempt0_d_control` in the configuration to report D-vs-D agreement alongside D-vs-M agreement. An absent control is reported as UNAVAILABLE and adds no GPU rows.

The M probe mechanically requires all eight finish reasons in every task slot to be `stop` or `length`, and zero rows masked as `unknown_finish_reason` or `invalid_model_output_or_checker_error`. Missing mask-reason transport also fails qualification. A failed transport check leaves M/S/H technically UNVERIFIED with named reasons in the return text and stops their chains before training. D retains its independent chain after request identity passes. All chains depend on this qualification reader and their arm-specific status. No qualification gate uses sampled token agreement, correctness or eligibility rates. The finish-reason and checker-exception checks do inspect sampled outputs, but they are zero-tolerance code-and-transport checks: one occurrence indicates a kit or transport defect, not a model-quality rate, and the `Q|M|UNVERIFIED` line names each triggering exception type as `exception=<type>`. Synthetic probes carry TECHNICAL-ONLY status.

The pinned verl source is available for read-only inspection in the development environment. CPU source-contract checks verify the real plan-keyword dispatch, one-dimensional object-array handling of `extra_fields`, EOS finish gate and async default against six explicitly sealed source hashes. CPU agent/server doubles test concurrent request seeds and passive finish-reason transport. Executing downloaded pinned-source methods requires resolving the standing execution restriction. The gated `tests/test_ts_verl_integration.py` exercises the real dispatch, single-turn and batch-postprocessing methods with CPU dependency doubles. Set `TS_ALLOW_PINNED_CPU_EXECUTION=1` only after an explicit owner exception. Until that exception is granted, this runtime integration remains unverified. Actual vLLM attempt-zero request identity and finish-reason transport, distributed gradients, longest legal shapes and GPU checkpoint/resume remain mechanical-qualification work. CPU results provide no eight-attempt GPU throughput measurement.

## Operator commands

Use an isolated output root. `TS_WORK` holds the selected campaign and `ts-config.json`. A live configuration contains the following fields.

```json
{
  "out": "<absolute isolated output root>",
  "technical_synthetic": false,
  "replay_binding": null,
  "arms": ["D", "M", "S", "H"],
  "binding": {"path": "<send-time binding JSON>", "sha256": "<digest>"},
  "send_approval": null,
  "job_python": "python3"
}
```

The hash-bound binding document contains `binding`, `base_revision`, `recipe_receipts` and `bundle`. The bundle supplies `base`, local `tokenizer`, staged trainer source under `sdpo`, `trainer_python`, `balance_batch`, hash-bound `schedules[seed][stage]` and `panels[panel]`. The native recipe identity follows the bound I0 or CR recipe name. Model records use the incumbent tensor/config digest convention. Schedules supply training rows with `id`, `prompt`, verified `response`, `gold`, `domain`, `split` and replay flags.

```sh
python -I "$KIT/ts_jobs.py" inventory --binding "$REPLAY_BINDING" \
  --arms D M S H --out "$TS_WORK/campaign.json"
python -I "$KIT/runner.py" plan "$TS_WORK/campaign.json"
```

To defer H, pass `--arms D M S` and use that same arms list in the configuration. Inventory generation builds the selected DAG directly. Runtime roots include the replay binding, with distinct I0/CR stage, qualification, plan and score paths.

The operator submits [one row](jobs/ts/run-task.sh) per allocation with the declared resources and dependency options. Training and identity probes use eight GPUs and 32 CPUs on one node. Each scoring lane uses one GPU and eight CPUs. Its timeout is 14,400 seconds per started block of nine expected score cells, capped at the 172,800-second training timeout. At full scope this gives 28,800 seconds for each 13-cell general-panel lane, 72,000 for each 37-cell FinQA lane, and 57,600 for the 33-cell legal and 29-cell hard-MATH lanes. Other rows use zero GPUs and eight CPUs. Stage chains use `afterok`. Readers and collectors use `afterany`. Include `--kill-on-invalid-dep=yes` for dependent allocations. Utilisation and missing telemetry remain advisory. Allocation-end accounting requires scheduler receipt reconciliation.

Reprice using measured eight-attempt throughput, allocation wall time, IO and retries. The approved I0 envelope of about 897–1877 GPU-hours remains a planning forecast. Reprice CR separately. H can be deferred if the revised envelope exceeds the available budget.

## CPU departure ledger

```sh
mkdir -p tmp/tsarms-round2
env -u TS_ALLOW_PINNED_CPU_EXECUTION python -m pytest -q tests/test_ts_*.py \
  --basetemp=./tmp/tsarms-round2/ts-tests
python -m pytest -q tests/test_nm_*.py tests/test_kit_sdft.py \
  tests/test_kit_runner.py tests/test_v7b_fixes.py \
  --basetemp=./tmp/tsarms-round2/incumbent-tests
python -I kit/ts_dryrun.py --out ./tmp/tsarms-round2/fresh-rehearsal
```

Delete each pytest basetemp after its run. The rehearsal uses a fresh directory and executes all 316 rows through the campaign runner with CPU children, including 72 toy-logit training stages, four identity-probe jobs and 42 scoring lanes. It exercises dependency refusals, missing checker output, all-failure batches and idempotent runner replay. Its outputs carry TECHNICAL-ONLY provenance.

| Review finding | Departure probe or correction |
| --- | --- |
| 1 | Real ContractNLI and FinQA prompt fixtures, teacher-only structural checks and preserved verified-solution guard |
| 2 | Crash-after-15 / checkpoint-10 / complete-40 CPU tests for M, S and H with superseded mask audit |
| 3 | Failed-target duplicate-success rule for S/H, named model-output masking and telemetry |
| 4 | Raw finish-reason transport, exact incumbent reward gate and alternate-stop-token fixtures |
| 5 | Seven requests overlap concurrently with independent seeds |
| 6 | Pinned source seam checks, deterministic D/M update-one all-slot request identity and M finish-reason transport qualification, with GPU evidence pending |
| 7 | TS-D = incumbent D plus passive observers. Command comparisons against NM I0/CR and the CR entry cover explicit sampling fields and D mode validation. Runtime entry installs TS request/finish observers |
| 8 | TS resources in their own directory, explicit implementation seal and NM digest equality with main |
| 9 | Whitespace-normalised distinctness and measured near-duplicate numerator/denominator/rate |
| 10 | Literal example-response wrapper also wraps positive feedback |
| 11 | Shared (seed, panel) scoring across base and scheduled arms, arms selection, pending reader evidence and binding-specific paths |
| 12 | Nonfinite masked-row loss retains incumbent failure strictness |

TS explicitly seals its source, row inventory, agent config, token fixtures, real-prompt fixtures, pinned seam hashes and this README. The incumbent NM source-seal manifest and digest equal main. D delegates its actor update directly to the incumbent, and CPU tests compare the resulting actor and EMA weights byte-for-byte when every task mask is one.
