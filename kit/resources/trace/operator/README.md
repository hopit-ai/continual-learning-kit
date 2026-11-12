# Offline TRACE operator assets

These assets add no acquisition, API calls, scheduler submission or science.
`registration.md` remains the authority. Paths and resulting evidence stay local.

## Sealed-90 calibration assembly

Stage the already approved local sources with their provenance and notices.
The builder verifies these exact bindings before assembly.

| Panel | Repository | Revision | File | Population |
|---|---|---|---|---|
| Self-Instruct | yizhongw/self-instruct | 0b26ccaa | seed_tasks.jsonl | 175 |
| LIMA | BeyonderXX/TRACE | 462e39f616134f4f819efeb3baea8638c03c7db4 | inference/HHH/helpful/LIMA/eval.json | 300 |
| CoNa | BeyonderXX/TRACE | 462e39f616134f4f819efeb3baea8638c03c7db4 | inference/HHH/harmless/CoNa/eval.json | 178 |

Self-Instruct is JSONL with `instruction` and `instances`, whose first instance
has string `input` and `output`. Its prompt joins instruction and nonempty input
with two newlines and retains the first human output. LIMA and CoNa are JSON
arrays with string `prompt` and empty `answer`. Exact source SHA-256 values are

- Self-Instruct `7779004fa198fdf27cf70a159363879d8a26c53329e11b436af17b3941875f48`
- LIMA `32d358c81ea71b87aed46b17fa43e1fdfe8c3946c1e345873c22ea900650e6aa`
- CoNa `a0d4c70cb26b990d7bd6a0e7c42cab8e5bdfe530e14e7941fcbdd51c4e3d4ec6`

Use the shipped `kit/resources/trace/calibration/selection-v1.json`,
`authored-v1.json` and `authored-v1.sha256`. The selection holds IDs, source
indices and prompt hashes for thirty pairs per panel, globally sorted by ID.
Authored entries contain exactly `{b}` for Self-Instruct and `{a,b}` for the
other panels. The sidecar must equal the authored file's SHA-256 followed by
`  authored-v1.json` and a newline. The builder reselects from the verified
sources using `trace-calibration-20261112-v1`, verifies IDs and prompt hashes,
joins the human/authored answers, and applies the deterministic pair reversal.

On the node, set `calibration` in node-inputs.json to the following structure.
Every file reference requires its local path and exact file SHA-256.

```json
{
  "schema": "trace-calibration-assembly.v1",
  "selection": {"path": "kit/resources/trace/calibration/selection-v1.json", "sha256": "e7cd3143731c765f56e8aabbfeed6c673e8aebbf795e8deacebd1abac95ce64b"},
  "authored": {"path": "kit/resources/trace/calibration/authored-v1.json", "sha256": "a98699d7f6f0bf85dbd3b44ecfe8f052288b68680639cbd565915f608519fa52"},
  "sidecar": {"path": "kit/resources/trace/calibration/authored-v1.sha256", "sha256": "7d349efc4e6945da4218f1f215e0f4c549e499f7741b65f724c7398abda3f867"},
  "self_instruct": {"path": "references/self-instruct-0b26ccaa/seed_tasks.jsonl", "sha256": "7779004fa198fdf27cf70a159363879d8a26c53329e11b436af17b3941875f48"}
}
```

The config's `panels` reference binds a JSON object with the `open` array used
by TRACE prepare. Node assembly verifies the ordered prompt digests in
`trace_calibration_build.PANEL_PROMPTS`. This includes the separate 175-prompt
TRACE Self-Instruct open panel. The human answers come from the bound seed file.
The node reconstructs LIMA and CoNa sources from the bound open panels.

From the public kit root, run the exported assembly command

```bash
python -I kit/trace_calibration_build.py node --config node-inputs.json --out calibration-local
```

It produces `sealed/calibration.jsonl`, `sealed/pairs.json` and `receipt.json`.
The expected 90-pair JSONL SHA-256 is
`b4713433a390203a266b7b9c4e1b6a34fc5a7df5f2c0ea12086886426c054de5`.
Normal TRACE prepare invokes the same `node_assemble` implementation.
Changed bound sources, sidecars, selections or existing seals fail validation.

For the manager's independent reconstruction, keep sources under
`references/self-instruct-0b26ccaa/` and `references/trace-upstream-462e39f6/`
using the exact filenames above. The manager command also validates the local
relation key at `docs/phase4/trace-calibration-LOCAL-KEY-20261112.json`.

```bash
python -I scripts/trace_calibration_assemble_v7h.py --root . --out calibration-manager
python -I kit/trace_calibration.py --pairs calibration-manager/sealed/pairs.json --out rater-packets
```

The manager command produces the same sealed bytes and blinded high-effort
packets for Astra and Fable. Send each rater only its packet's `requests`.
Keep the LOCAL relation key and packet mapping on the manager. Full prompts,
sealed rows and human answers stay in local staging. Return IDs, orders and
labels through the existing return contract. Assembly uses local CPU files.

## Node: file bindings and receipts

The base config fields remain those in README-trace. `asset-schema.json` names
the additional `operator` fields. File references may initially have only a
`path`; `trace_operator.py bind` computes SHA-256 and verifies any supplied hash.
Model/tokenizer directories are inventoried in full, and the base tensor/config
digest uses the inherited convention. No network access is used.

To get the exact input digest before measuring CPU authorities, run
`python -I kit/trace_operator.py assets --config node-inputs.json --out staged`.
It writes fully hashed `assets.json` and `admission-input.json`. Add the measured
operator evidence references to assets.json, then run bind on that config.

`metrics.json` supplies all callable declarations. `fixtures/` supplies twenty
authored cases for each of the eight task metrics, six TRACE general metrics,
and the frozen instruction/maths guards. Expected scores are explicit values,
not scores learned from the scorer under test. Source-dependent cases test
perfect-answer invariants; they do not establish all extraction, normalisation
or partial-credit conventions. Additional upstream edge fixtures are needed
for full Q qualification. The installed source scorer remains authoritative.

`operator.metric_authorities` maps each task/panel to an upstream file reference,
`function`, result `key`, `scale_to_100`, `dependencies` file references and
`uses_source` for SARI. Use the exact admitted TRACE extraction callable, not a
newly approximated metric. Its interface receives prediction/reference lists,
or source/prediction/reference lists with `uses_source`. Preserve staged code
and dependency licences. Bind validates every fixture against those actual
callables with absolute error at most 1e-6. Missing source code or a mismatch
stops binding. Knowledge continues to use the inherited reference scorer.

### Staged task metric authorities

Copy `task-authorities-template.json` into `operator.metric_authorities`.
Fill only staged paths and SHA-256 hashes. Keep its functions, keys, scales,
dependencies and source flags. Stage the upstream tree unmodified, with
`metrics.py` at its root and `evaluations/` beside it. Each dependency's parent
is added after the evaluation directory during the upstream call.
The result key accepts a string or a nonempty JSON list of string/integer steps.
20Minuten uses `["sari", 0, "sari"]` to unwrap upstream's tuple result.

Use a metric environment with rouge 1.0.1, fuzzywuzzy 0.18.0, sacrebleu 2.4.3,
sacremoses 0.1.1 and packaging 26.3 (the vendored SARI script imports it).
Do not install python-Levenshtein: Py150 edit
similarity is pinned to fuzzywuzzy's difflib matcher and the bridge refuses
the other backend. Use nltk 3.5 on Python <=3.11, or nltk 3.9.1 on
Python >=3.12. Upstream also computes BLEU, which is unused by registered keys.
The newer nltk avoids its Python >=3.12 Fraction crash.
`datasets` need not be installed for metrics. During each upstream call, the
bridge supplies only the datasets API needed by its hash-checked, vendored
Hugging Face datasets 2.21.0 SARI script. `load_metric` accepts only `sari`.
The bridge restores the prior import state on success and failure. SARI uses
the local vendored source, with no network fetch.

The authored fixtures test perfect-answer invariants only. They leave
extraction, normalisation and partial-credit edge conventions unqualified.
The six general OpenCompass metrics require separate upstream verification.

On the manager checkout, verify the eight task metrics with a metric venv.
The verification script copies upstream code to temporary staging before use.
It writes per-task fixture counts and failures and exits nonzero on a failure.
The script stays manager-only and is excluded from the public export.

```bash
METRIC_PY=/local/metric-venv/bin/python
"$METRIC_PY" -I scripts/trace_metric_upstream_verify.py \
  references/trace-upstream-462e39f6 tmp/metric-upstream-report.json
```

`operator.admission_evidence` maps the six names in admission-template.json to
measured detailed JSON file references. The template deliberately says invalid;
the receipt generator never turns a missing observation into valid evidence.
`admission-evidence/` contains the six detailed schemas with unmeasured fields
set to null. Copy these locally and fill them from the actual audit, preserving
the underlying per-row observations; nulls cannot produce a valid receipt.
Every detail requires `technical_synthetic: false`, `registration_sha256`,
`input_sha256` and a positive `rows` count. Compute `input_sha256` with
`trace_operator.admission_context` on the fully hash-bound config: it hashes
base, original_tokenizer, datasets and panels. Each probe adds these fields:

| Probe | Detailed evidence fields |
|---|---|
| original_tokenizer_census | same_rows=true, original_revision, by_task_split |
| supervised_masks | wrong_supervised_tokens, max_eos_position_shift |
| prompt_fixtures | cases_per_task, semantic_changes, leaked_targets, missing_roles_or_eos |
| scorer_fixtures | cases_per_metric, max_error, label_or_unit_errors |
| demo_fidelity_gold / 27b | training_only=true, summaries=50, summary_ids, material_mismatches, invalid_targets, leaked_targets |

`by_task_split` covers every requested task and train/eval/test. Each entry has
`rows`, `train_prompt_excess`, `inference_prompt_excess`, `target_loss_excess`,
`wrong_supervised_tokens`, `max_eos_position_shift`. Relative rate units are
fractions. Bounds are 0.05, 0.05, 0.01; token errors are zero and EOS shift at
most one. EOS affected-share excess is diagnostic, never a rejection threshold.
Prompt cases cover every task with at least twenty; scorer cases cover every
bound metric with at least twenty. Error counts are zero; scorer error <=1e-6.
Demo summary IDs are fifty distinct, fixed training-only MeetingBank rows per
source. `trace_operator.fidelity_ids(prepared)` ranks the selected training IDs
by a fixed hash and returns their exact order. At bind, both source audits must
use those IDs; the generated audit must also bind `targets_sha256` to targets.json.
Retain the rendered cases, full census and signed review locally.

The generated-demo audit cannot exist before targets. Its evidence entry can
be `{"deferred_path": "GENERATED_DEMO_AUDIT.json"}`. The binder pins this plan;
prepare validates all available CPU authorities before GPU targets. After
targets, complete that audit at the declared path. CPU bind hashes and validates
it, seals admission-receipt.json, then seals schedules without changing config.
No pending CPU probe other than this generated-demo audit can be deferred.

The native command assets are still an installed-stack integration dependency.
This kit provides their command/receipt contracts, not a replacement native
trainer qualification suite. `operator.native_probes` maps s8/gold/27b to the
eight probe names in asset-schema.json. Each value is a file reference to an
installed-stack command manifest with argv and executable/config file refs.
Bind those existing assets; never invent measurements to satisfy a template.
The wrapper runs probes in listed order, with TRACE_CONFIG, TRACE_STUDY,
TRACE_PROBE_OUT and TRACE_NOISE_OUT. Each command writes its measurement JSON
to TRACE_PROBE_OUT; later probes read the earlier fixed noise bound.

Every native detail requires `technical_synthetic: false`, world_size=8,
registration_sha256, model_sha256 and environment_sha256, identical across
the set and bound to the selected base. Per-probe measurement fields are:

| Probe | Measurement fields and checks |
|---|---|
| noise_floor | student_noise, ema_noise >=0; fixed_tolerance.{student,ema} in [0,3*noise]; fixed_before_freeze=true |
| world_size_reduction | world_sizes=[1,2,8], max_error <=1e-4 |
| ema_recurrence | student_error, ema_error <= fixed tolerance |
| native_resume | student_error, ema_error <= fixed tolerance; optimizer_equal, rng_equal, mask_equal=true |
| export_identity | mismatches=0 |
| freeze_bit_equality | blocks=[0,1,2], changed_coordinates=0, optimizer_coordinates=0 |
| likelihood | max_error <=1e-6 |
| scorer_concurrency | mismatches=0 |

Errors must be finite and nonnegative. Zero noise requires exact equality.
Retain raw tensors, native checkpoints, manifest identities and diagnostics
behind the detailed evidence files. A receipt cannot establish correctness
without those real installed-stack observations. Timing/utilisation remain
diagnostic. The qualification wrapper hashes all details and publishes the
five dispositions required by trace_jobs only after validation.

```bash
python -I kit/trace_operator.py bind --config node-inputs.json --out bindings
python -I kit/trace_jobs.py inventory --config bindings/config.json
```

Use bindings/config.json as TRACE_CONFIG, unchanged across every normal row.
The binder emits metric bindings, fixture references, admission plan and the
three qualification command manifests; the operator need not author those files.
`trace_operator.py admission --config ... --out ...` can separately validate and
write a measured admission receipt, once all audits exist.

## Wall time and queued-message replacement

Schedule from dependencies. Seeds and sources are independent after their own
qualification. Scoring/local judging for a completed seed can overlap training
on other physical GPUs. CPU prepare/bind/collect run outside GPU allocations.
Within a scoring lane, every state and local judge keep the same physical GPU.
Within a seed, S8 uses one persistent optimizer; D/F child stages retain their
common parent. Moving those stages or judge to independent allocations would
change recovery/identity contracts, so it is not a pure edge edit. The existing
inventory already exposes all safe row-level parallelism; no edges changed.

Exact replacement lines for the queued TRACE insert (plain ASCII, each <90):

```text
Calibration: 90 sealed pairs; same blind pairs go to Astra/Fable high.
Order: CPU prepare -> 2-GPU targets -> CPU bind -> qualify -> train/score.
Paste ESSENTIAL return-*.txt only (<=3 parts); retain FULL on node.
```
