# Rung P boundary step and technical smoke

Registration: `docs/programme-plan-v6-amendment-20261024.md`, Rung P, and
`docs/phase3/package-b-registration-20261014.md` §1. No scientific training,
Modal launch or new spending is authorised by these files.

Run locally against an untouched HF Qwen3 checkpoint and complete frozen pools:

```bash
python kit/p_replace.py utility --checkpoint /work/incoming \
  --pool finqa=/work/finqa-targets.jsonl \
  --pool generic=/work/replay.jsonl --replay-pool generic \
  --output /work/p-utility.json
python kit/p_replace.py replace --checkpoint /work/incoming \
  --utility /work/p-utility.json --seed 101 \
  --output /work/replaced --receipt /work/p-replacement.json
```

Both actions are offline (`local_files_only=True`). JSONL rows have unique `id`,
`prompt` (text or chat messages), `response`, and training provenance. The existing
trainer form `extra_info.index`/`extra_info.demonstration` is also accepted.
Require an explicit `split=train` (or `extra_info.split=train`). Reject non-training
rows, missing targets, duplicate IDs and incomplete
pools. Caller supplies the complete FinQA and admitted replay pools and remains
responsible for their provenance and exclusion audit. There is no implicit pool
search or fallback. `--replay-pool NAME` marks a whole pool as replay; the trainer's
`p3b_replay` row/extra-info flag is also recognised. Mark every replay pool.

Utility observes `down_proj`'s activation input with a forward pre-hook, namely
the actual bf16 `SiLU(gate(x))*up(x)` product. Accumulation and the down-column L1
norm are fp32. It calls D's `kit.v4_teacher.tokenize_student` unchanged: thinking
off, no demonstration in the student's prompt, EOS appended to the response.
Each valid response label contributes equally, across rows. Receipt selection
includes namespaced IDs, ordering hashes, original row hashes and replay flags.
Rows over D's prompt cap are excluded before hash ordering, using
`kit.c_d_prompt.d_prompt_eligible`, the same rule now called by `c_train`.
No generation or scoring precedes freezing the utility receipt.

Replacement writes each original shard separately, preserving tensor dtypes,
shard names, index bytes, config and tokenizer bytes. It never loads the 8B model
in fp32 just to edit it. Receipts hash the HF inference files by relative filename
and SHA256, then hash that canonical mapping. Caches and receipts are excluded.
Receipts also have their own canonical SHA256; changed checkpoint bytes or a
changed utility receipt cause refusal. Use fresh output/receipt paths. An
interruption can leave a partial output directory; it has no valid receipt and
must not be used as a checkpoint.

The manager launches the smoke only after adding the ledger row quoting the
standing $5–10 authority:

```bash
modal run --detach scripts/smoke_rung_p_modal.py
```

One H100, eight CPU cores, 200 GiB host memory, one container, no retries,
75-minute timeout for attempt 1 (attempt 2: 30 minutes, since attempt 1 took 897 s and the total rung-P smoke spend must stay within the $5–10 authority). The image follows
`smoke_common_modal.py`; the model download is pinned to the registered Qwen3-8B
revision. The smoke runs utility, seed-101 replacement, every immediate check,
then paired updates. Only the frozen FinQA training rows are shipped; their v4
training diagnostic probe overlap is recorded below. A JSON
record contains every assertion's value and PASS/FAIL, plus timings, gradients,
per-unit distances, losses and comparisons. Utility and replacement receipts
are separate files under the attempt record's adjacent directory (same stem,
without `.json`); the record contains file hashes, receipt hashes and summaries.
Full receipts are returned separately and never embedded in the `[record]` line. On a
prerequisite failure, unexecuted assertions are explicit FAIL with a reason.
The shared writer saves the returned record under `docs/phase2/evidence/` and the
remote function emits the same record in a `[record]` log line.

`kit/resources/p/smoke-prompts.jsonl` freezes the 32 prompts before execution.
`smoke-settings.json` freezes comparisons, dataset hashes, update count and
thresholds. The committed 256-row smoke pool is the first 256 Rung P hashes of
D's frozen 1,441-row FinQA training pool, reconstructed from the pinned local
FinQA `train.json` and the committed v4 manifest, checking every original row
hash. Selecting from these 256 is identical to selecting from the full pool.
Verified 27B demonstrations and admitted replay responses are not present locally;
the permitted technical stand-in uses FinQA train prompts with executed gold
answers. It does not establish performance at a connected FinQA final.

The connected `m2_train.py` → `c_train.py` → `c_sdft_runner.py` path requires eight
distinct GPUs. On one H100, use the requested minimal full-weight response CE
loop, not a scientific D invocation: 32 prompts/update; response token mean per
item, then batch mean; fresh AdamW (.9/.999, decay .01, clipping 1); LR
0/1e-6/2e-6/3e-6; fresh .01 EMA after each update. The actor runs in bf16; fp32
master weights, AdamW states and EMA live on CPU in the same container (200 GiB
host memory). This preserves small warmup updates. Both arms use identical
frozen rows, ordering, seed and precision. All selected units must pass the
movement/gradient checks individually. The record reports how far each moved.
The teacher-from-replaced-weights ordering must be asserted in the add-on
integration, where the fresh EMA actually supplies teacher targets; this smoke
does not assert that integration.
This loop tests mechanics; it does not test D's on-policy KL, demonstration
context, importance correction or scientific throughput.

Planning estimate: 40–60 minutes, unmeasured until the manager's run. Named
constants in `scripts/smoke_rung_p_modal.py` price the H100 at $3.95/GPU-hour,
CPU at $0.047/core-hour and memory at $0.008/GiB-hour. These are planning
estimates. The reservation costs $3.95 + 8×$0.047 + 200×$0.008 = **$5.926/hour**.
At the 75-minute timeout that is $7.4075, or **$9.31** with the shared writer's
25% allowance plus $0.05. The writer uses this CPU+memory+GPU rate for P.
The timeout bounds function runtime; image-build billing is separate.

Implementation interpretations, frozen before the smoke:

- **Utility positions and loss mask:** align activations to causal response
  prediction positions, as D's `hidden[:, -response_width-1:-1]` does. Apply
  `sdft_objective.loss_mask` (skip first three tokens) to task rows; replay uses
  all EOS-inclusive response labels. This resolves “response tokens” together
  with “stage's … loss mask.” Zero-mask rows remain selected, with zero weight;
  there is no replacement draw. An entirely empty mask fails.
- **Seed conversion and normal draws:** use the first eight SHA256 bytes as an
  unsigned big-endian torch seed. Per unit, CPU fp32 draws consume gate then up
  from one generator, then cast to the original stored dtype. The complete salt,
  SHA256 and integer seed are recorded. This avoids device-dependent redraws.
- **Registered first updates:** four updates, matching the registered first-four
  schedule used by Screen S: zero LR followed by three nonzero warmup updates.
- **Greedy comparison:** exact generated token IDs, EOS included, with greedy
  decoding, one beam, 128 new-token cap, repetition penalty 1, and thinking off.
  Record truncations and both token sequences. This is a bounded feasibility
  check, not an uncapped sequence invariance claim.
- **Gross-damage gates (revised amendment):** teacher-force the same frozen
  contexts on all 32 prompts. Project the final normalised hidden state through
  an fp32 head at response prediction positions only, including EOS. Gate the
  pooled full-vocabulary p99.9 |Δlogit| at ≤1 and the token-weighted mean response
  NLL rise at ≤0.1 nats/token. Report p99, p99.9, max |Δlogit| and greedy-change
  count; max and greedy changes have no gate. Quantiles use exact linear
  interpolation over pooled logit entries. Nonfinite gated measurements fail.
- **Gradient and timing comparison:** require each update's pre-clipping norm
  ratio in [0.5, 2]. Compare median wall step time over updates 2–4, including
  CPU optimiser and EMA; exclude first-update state allocation equally in both
  arms and retain every individual time. Require ratio ≤1.1.
- **Development inputs/fallback:** use only the explicitly permitted FinQA train
  gold-answer stand-in and CE/EMA loop. Real D needs eight GPUs and locally
  unavailable verified targets. No scored panels, held-out responses or online
  response generation substitute for training data.

No smoke has been launched during implementation. Failure defers Rung P for
re-registration; these scripts do not switch methods automatically.

Local input freeze, 25 October 2026:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/check_rung_p_inputs.py \
  --tokenizer references/r100-tokenizer-cache/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
```

The committed `kit/resources/p/smoke-input-check.json` records every row's lengths,
tokenizer file hashes and frozen input hashes. No model weights were loaded or
downloaded. All 256 utility rows pass D's shared prompt-cap rule; all 128 training
rows and 32 comparison prompts pass the prompt/response caps. Maximum prompt
lengths are 1,786 (training) and 1,660 (comparison); maximum response lengths
including EOS are 16 and 13 respectively.

The v4 `finqa-probe` selector is in `kit/v4_finqa_data.py`: the first 20 training
IDs by `sha256("v4-fixed-probe|finqa|"+id)`. Reconstructing it from the committed
1,441-row manifest finds **6/256 pool rows overlapping 6/20 probe IDs**:

- `CDNS/2007/page_30.pdf-1`
- `ETR/2013/page_93.pdf-1`
- `ETR/2013/page_93.pdf-2`
- `LMT/2007/page_55.pdf-4`
- `RSG/2014/page_73.pdf-1`
- `SNPS/2012/page_61.pdf-3`

These are training diagnostic probe rows; v4 scores that probe descriptively in
its mechanism analysis (`kit/v4_auxiliary.py`). This technical smoke does not
score the v4 probe and cannot establish held-out performance. Its training pool
is unchanged; no held-out dev/test/private_test IDs are included. The complete
20 IDs, overlap and manifest hash are committed in the input check.

Local offline validation (fix round):

```text
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q tests/test_p_replace.py --basetemp=<fresh-test-dir>
24 passed, 1 warning in 9.01s
```

The focused C training-command and M2 protocol dependency checks also pass
(`2 passed in 0.18s`). The pricing constants independently evaluate to $5.926/hour
and a $9.31 timeout bound.

The existing pandas/numexpr version warning remains. Python compilation and
`git diff --check` pass. The test basetemp and the review's
`<review-test-dir>` are deleted after validation. Modal was not run.
