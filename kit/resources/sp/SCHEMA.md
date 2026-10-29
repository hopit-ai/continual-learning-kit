# SP source and preparation contract v2

The authoritative schema, exact template provenance, replay checker semantics,
compatibility matrix, fixture contract, candidate-superset manifest requirements,
review receipts and commands are documented in [source/README.md](source/README.md).

Consumers use `request_id`, `primary_family`, `constraints[{family,args}]` and
`canonical/paraphrase{clause,clauses,prompt}`. The family is `word_frequency`.
Source v1 author aliases are retired. The canonical arm additionally carries
`template_bindings`; every literal clause must regenerate exactly from the pinned
replay comment template and its recorded parameter substitutions.

Construction receipt: `CONSTRUCTION.json`, schema `sp-construction-v2`.
Scientific freeze: `FROZEN.json`, schema `sp-panel-freeze-v2`, now supplied for
the complete locally available 23-pool set under the explicit manager ruling.
The two independent reviews and 200 author inspections are bound. Unavailable
node inventories are named in `node_verification` and must pass VERIFY-ONLY
checking before scoring; overlap hits flag only implicated pairs. See
[NODE-VERIFY.md](NODE-VERIFY.md). Global inventory completeness is not asserted.

Review receipts are external `sp-review-receipt-v2` documents, with two distinct
independent reviewer identities and every request/resource/fixture hash bound.
Historical reviewer receipts in `reviews/` are immutable. Round-3 changed IDs and
old/new request hashes are in `reviews/changed-pairs-round3.json`; reviewers
recheck those pairs and issue separate receipts with the new binding. An old
receipt does not approve a changed request or the new global binding.

Canonical template reuse is intentional. Content exclusions cover both arms;
parameter-normalized clause exclusions cover only paraphrases. Return rechecks
verify containment in frozen candidate supersets and cannot alter the panel.

The scoring owner must call `checkers.json_fence_failure(raw_text)` separately
for descriptive fence-related failures; strict accuracy always uses raw JSON.
