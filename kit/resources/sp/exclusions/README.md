# Scientific-freeze inventory attempt

`inventory.available.json` records the historical available-source audit and
node-held dependencies. Its original **PENDING** label is preserved as evidence.
The manager subsequently authorized `FROZEN.json` for this exact available set,
with mandatory node-side VERIFY-ONLY checks described in `../NODE-VERIFY.md`. Public source
supersets are complete for the four requested sources. A complete collection of
accepted responses and programme ancestry is still required. No SP request was
removed, rewritten or replenished based on this audit.

`freeze_inventory.py` stages the pinned sources and available returned pools in
an external cache and runs the existing SP audit. Invoke it with isolated Python:

```sh
PYTHONDONTWRITEBYTECODE=1 python -I kit/resources/sp/freeze_inventory.py \
  --cache CACHE --rlvr RLVR_PARQUET --gsm GSM_TRAIN_PARQUET \
  --arc ARC_CHALLENGE_TRAIN_PARQUET --contract CONTRACT_SOURCE_DIRECTORY \
  --finqa FINQA_DATASET_DIRECTORY --return-next EXTRACTED_NEXT_RETURN --audit
```

The capitalized arguments are local input locations, not shipped locations.
Extract untrusted archives using `tarfile.extractall(..., filter='data')` into
fresh empty directories before supplying them. Never execute their contents.
Public source bytes are checked against existing pins; the FinQA source files
must also equal their pinned Git blobs. Returned files are checked against their
accompanying manifest hashes. Response text is consumed by exclusion code only;
it is not used to write or adapt the panel.

The external `pools.json` intentionally has `complete_inventory: false` and is
rejected by the scientific freeze loader. The available audit is diagnostic:
role coverage does not establish within-role completeness. The manager ruling accepts the author inspection for this exact local set.
Remaining node inventories undergo verification only; no panel edits or rescoring.
Exact canonical reuse is exempt; all content and witnesses, and only paraphrase
constraint clauses, undergo the registered duplicate rules.

The available candidate fingerprints use the SP preparation contract's
`digest(normalized_prompt)` representation. The separate K/CR
`sp-candidate-superset-v1` generator uses SHA256 of raw normalized UTF-8 text.
These are different encodings and must not be substituted for one another.
This attempt does not publish or certify the K/CR generator's final manifest:
its exact frozen FinQA target inputs are missing.
