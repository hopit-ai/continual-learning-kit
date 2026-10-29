# Combined message 2+3: final cross-family review (redacted)

Reviewer: Opus (cross-family), 29 October 2026, read-only confirmation after the final fix round.

**VERDICT: SEND.** Findings 2 and 3 still have to be done, as you planned, before the tag.

Python, hashing commands and several git commands were blocked here. Everything below comes from file reads, grep, and one `git diff --stat` from the PD merge (`ddcf3ee1`) to HEAD. I did not recompute any SHA256 or run any tests.

**(a) P2 smoke attempt 5: confirmed.**
- `docs/phase2/evidence/x-integration-p2-smoke.attempt5.json`: top-level `"passed": true` is at `:228250`, `"mechanics_passed": true` at `:599` and `"resources_passed": true` at `:228257`.
- It is attempt 5 of 5 (`:3`, `:528`), with cost bound $10.53 (`:408`), retries 0, and `scientific_evidence: false`.
- `a_resume_status` is `"UNVERIFIED"` (`:2`). It is recorded honestly, so I am not blocking on it.

**Code map vs current tree: matches, but I could not recompute the hashes.**
- The receipt's `code_sha256` (`:37381-37426`, 45 files) is identical, entry for entry, to the map in the pre-launch CPU stage audit (`docs/phase3/evidence/p2-cpu-stage-audit-attempt5-20261029.json:3-48`).
- Since the PD merge, the 45 bound files changed only in the authority commit `4f83557b`, which came before the pre-launch ledger row `d7acdca8`.
- The two later commits touched none of them. `66d8843f` changed only the PD and README files plus `tests/test_pd.py`; `28189d1a` added only the evidence file and the ledger row.
- The working tree has no tracked changes; only `tmp/combined/` is untracked.
- So the current bytes should equal the smoke's bytes. A `shasum` re-check of the 45 files during the publisher run would close this.

**(b) Finding 1: closed.**
- `kit/resources/pd/scientific/send-binding.json` has `status: "SEALED"` (`:170`) and `identity_sha256` (`:67`).
- It carries the fields the code requires: `pd_chain_sources_sha256` (`:69`), `pd_custodian_receipt_sha256 = 5e86c73c…` (`:70`), and `pd_starts` (`:71-168`). These are the fields `pd_probe.py:679-680` reads.
- `pd_starts` has nine starts (seeds 101–103 × base/stage1/stage2), each with `weights_sha256` and `export_receipt_sha256`. They agree with the table in `tmp/combined/pd-binding-recheck.md:21-29`.
- `plan-inputs.template.json:9` points `send_manifest` at it, and `README-combined.md:32-39` (step 3) tells operators to copy it and run `verify-node`.
- Every start is `live_tensors_verified: false`, and the stage1 original receipt bytes must resolve on the node. This is correct: the file is a commitment, and the node does the live check.

**(c) Finding 4: closed.**
- `resources/pd/README.md:12-16` is now grammatical and documents both exit 3 and exit 1.
- The "still required" sentence is gone (grep finds no hits).
- The topic review is labelled our-side and not shipped in `resources/pd/README.md:33-34` and `scientific/README.md:50`.

**(d) Finding 5: closed.**
- `pd_node.py:110-116` wraps only `_prepare_launch` errors as `PreparationError`.
- Launch and retention failures write a PENDING receipt and re-raise (`:117-126`).
- `main` returns 3 only for `PreparationError` (`:142-145`) and returns 1 with `error_type` for anything else (`:146-149`).
- `tmp/combined/pd-tests.txt:10` reports 149 passed. I did not re-run it.

### Findings

1. **Required, already planned (finding 2):** `docs/send/final-review-redacted.md` still opens with "VERDICT: FIX". Replace it with a redacted copy of this confirmation and put its new SHA256 in the send message.
2. **Required, already planned (finding 3):** do a full `scripts/publish_kit.py` run on the final HEAD and save it to `tmp/combined/full-suite.txt`.
   - Add `tests/test_pd.py` to that run, or record a separate run of it, because it is not in the publisher's `TESTS`.
   - In the same step, re-hash the 45 P2-bound files against receipt `:37382-37426`.
3. **Optional:** record the byte SHA256 of `send-binding.json` (the claimed value is `63cfa302…`) in the send message next to its identity `a15fd56e…`, so the operator can check the file before copying it into plans.

[exited with code 0]
