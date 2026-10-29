# SP template-reliance source — v2

The final registration compares **exact replay-family canonical templates** with
independently authored paraphrases on identical new content. This directory has
200 requests, 20 per primary family, and 400 prompts. Each primary family has ten
single-constraint and ten compatible dual-constraint requests. Each family also
occurs ten times as a secondary. No model was run or its outputs inspected during
construction. The panel and fixtures remain permanently outside replay.

`CONSTRUCTION.json` binds the authored requests, checker resources and fixtures.
It is **not** `FROZEN.json`: independent reviews, complete real candidate-superset
exclusions and manual nearest-clause inspections are still required before the
scientific freeze and warning-only probes. Probe rates cannot trigger rewrites.
Manager scope is base, incoming and final chain checkpoints only, excluding
intermediates, PD forks and S; coverage/scoring remains the SP runner's ownership.

## Authoritative schema and rendering

Every `requests.jsonl` line uses `schema: "sp-source-v2"` and these fields:

- `request_id`: unique portable identifier; `primary_family`: first constraint family.
- `content`: task text, identical in both arms.
- `constraints`: ordered objects `{ "family": ..., "args": {...} }`.
- `canonical`: `clause`, atomic `clauses`, `prompt`, and `template_bindings`.
- `paraphrase`: `clause`, one atomic clause per constraint in `clauses`, and `prompt`.
- `witness`: authored passing response; `mutations`: optional authored mutations.
- `reviews`: empty; inline approvals never satisfy the external review gate.

Both prompts must equal `content + "\n\n" + clause` exactly. Canonical atomic
clauses join with newlines; paraphrase clauses join with spaces. The same
constraint array is scored for both variants; no text or parameter is inferred
from a model response. The witness is never appended to a prompt.

This replaces the initial author's `id`, `parameters`,
`canonical_instruction` and `paraphrase_instruction` fields with the consumer's
`request_id`, `args`, and nested clause/prompt objects. `exact_word_frequency`
becomes `word_frequency`; existing request IDs remain stable. `sp_prepare.py`
consumes v2 directly; it does not accept a silent mixture of schemas. Scored
`panel.jsonl` rows retain the loader/scorer contract: `id`, `request_id`,
`wording`, `primary_family`, `prompt`, `constraints`.

## Canonical provenance and instantiation

`templates.json` records source lines, exact comment bytes, template IDs, source
revision/hash and replay dataset revision. `templates.py` verifies the pinned
local replay source and reconstructs each canonical clause. The source is
[open-instruct if_functions.py at the pinned revision](https://github.com/allenai/open-instruct/blob/11826255077617a46919ce75cadf1f3d53f30dac/open_instruct/if_functions.py),
Apache-2.0, used by `allenai/RLVR-IFeval` revision
`47c03c73621c4aab2b824b7818681117d662770e`. Its full file SHA256 is
`afea345e3bc100e6d5588ca14e7269141e8810e1489e60ad64a59cd57360201c`.
No source was downloaded. This binds the canonical **family templates**, not a
claim that every realized parameter combination occurred in accepted replay.

Template IDs have prefix `open-instruct:` and suffix `:comment`:

| Family | Source function/template ID body | Substitution rule |
| --- | --- | --- |
| required_word | `verify_keywords` | Fill the variadic `{keyword1}, {keyword2}` list span with the singleton keyword |
| forbidden_word | `validate_forbidden_words` | Fill `{forbidden words}` with the singleton keyword |
| word_frequency | `verify_keyword_frequency` | Fill `{word}` and `{N}` |
| word_count_interval | `validate_word_constraint` | Instantiate the source's explicit alternative `at least` at `min`, then `at most` at `max` |
| sentence_count | `verify_sentence_constraint` | Instantiate `at least` and `at most` at the same `count` |
| paragraph_count | `verify_paragraph_count` | Fill `{N}`; preserve the newline before the literal `* * *` divider |
| bullet_count | `verify_bullet_points` | Fill `{N}`; join the source-wrapped prose comment with one space |
| json_only | `validate_json_format` | No substitutions |
| lowercase_only | `validate_lowercase` | No substitutions; join source-wrapped prose with one space |
| required_suffix | `validate_end` | Fill `{end phrase}` with the JSON-quoted literal suffix so terminal punctuation is unambiguous |

Only placeholder/list spans, explicit quantifier alternatives and source comment
line wrapping are resolved. Literal template wording is unchanged. Conjoining
two bound clauses creates one interval or exact-count checker, not an extra
independent constraint. Each row cites every instantiated template and its
substitution map. The validator regenerates and compares exact text and maps;
a hand-edited canonical clause fails provenance validation.

All 200 original everyday task texts remain. After independent review, round 3
revises 182 pairs and leaves 18 unchanged. Thirteen forbidden-word arguments were
replaced consistently in both arms to avoid incidental embedded matches in
ordinary words. The passing witnesses remain unchanged and were regenerated and
rechecked with the new arguments. Paragraph witnesses use `\n* * *\n`;
JSON-string witnesses remain valid under the unchanged raw-JSON checker.

The independently written paraphrases now state only the requested output
requirements. They no longer reveal substring matching, tokenization, case
comparison, code-fence handling, empty-block tests or trailing-whitespace rules
that are absent from canonical wording. Those operational details remain below
for implementers, not in either prompt. Markdown bullets, literal paragraph
dividers, exact counts, English/lowercase and exact suffix text are retained
because the canonical templates specify them. Ambiguous count-production and
lowercase clauses were corrected. No model outputs guided these revisions.

Forbidden-word replacements are fixed in `FORBIDDEN_REPLACEMENTS`:
`front→podium`, `craft→origami`, `clean→spotless`, `tea→cappuccino`,
`book→encyclopedia`, `game→charades`, `mess→squalor`, `box→shoebox`,
`ball→frisbee`, `perfect→flawless`, `urgent→hasty`, `instant→immediate`,
`ultimate→superb`. Regression cases include `tea` in `instead`, `mess` in
`message`, `front` in `confront`, `craft` in `aircraft`, and `ultimate` in
`penultimate`. This changes arguments, not the checker or canonical templates.

The author table and deterministic parameter instantiation remain in
`build_source.py`: primary word intervals use authored whitespace-word length
minus three/plus four; primary count parameters come from authored fixtures;
secondary families and parameters are fixed by index. Nothing depends on model
behavior. Primary templates are now source-bound; independently written
paraphrase variants still use the original ten-position rotation per family.

## Checkers, compatibility and fixtures

`checkers.py` extracts only the needed functions from the hash-bound replay module
without importing its unrelated runtime dependencies. Exact matching semantics:

| Family | Arguments | Executable rule |
| --- | --- | --- |
| required_word | `word` | Case-insensitive substring presence; matches inside longer words count |
| forbidden_word | `word` | Case-insensitive substring absence |
| word_frequency | `word`, `count` | Exact frequency of lowercased `re.findall(r"\b\w+\b", text)` tokens |
| word_count_interval | `min`, `max` | Inclusive bounds on `text.strip().split()` length; standalone bullet markers count |
| sentence_count | `count` | Exact conjunction of replay lower/upper sentence bounds; source regex is `(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s` on raw text |
| paragraph_count | `count` | Nonempty blocks separated by literal `* * *`, with the replay line-whitespace cleanup |
| bullet_count | `count` | Lines whose stripped text starts with `*` or `-`; marker-only lines also count under the pinned checker |
| json_only | none | One complete raw JSON value, rejecting nonfinite constants, wrappers, multiple values and code fences |
| lowercase_only | none | Exact replay `text == text.lower()`; no added cased-letter requirement |
| required_suffix | `suffix` | Raw case-sensitive `text.endswith(suffix)`; trailing whitespace fails |

There is no universal empty-response gate or extra language detector. The English
instruction is inherited from the source template, whose checker only checks
case. These inherited limitations, abbreviation behavior and whitespace edge
cases are explicit in `checker-edge-fixtures.json` (38 cases).

**Declared raw-JSON deviation:** unlike IFEval's fence-stripping variant, a fenced
JSON answer fails. The pinned RLVR JSON checker is already raw; this panel also
rejects nonstandard `NaN`/`Infinity`. `json_fence_failure(raw_text)` supplies a
separate descriptive flag for invalid raw JSON containing a Markdown code fence;
it never converts a failure into a pass. The scoring owner must report that flag
separately. Strings, numbers, booleans and null are valid JSON values.

`compatibility.json` is the frozen symmetric 10-by-10 matrix. Same-family pairs
are forbidden. JSON cannot pair with suffix, bullet, paragraph or sentence
constraints. Paragraph/bullet combinations are also forbidden because the replay
divider itself counts as a bullet. Remaining pairs are conditional on arguments,
a passing witness and independently failing mutations. The schema conservatively
rejects two word constraints with the same lowercased argument and cased
word/suffix arguments paired with lowercase.

`authored-fixtures.jsonl` binds 200 witnesses and 300 deterministic string-edit
mutations, each failing exactly its targeted constraint while passing the other.
These edits use no model. The two wording variants share the checker and fixtures.
Their byte hashes, matrix, edge cases, source templates and checker hashes are
bound by construction, review receipts and the eventual scientific freeze.

## Exclusions and return verification

`sp_prepare.py audit` requires a `sp-pools-v2` manifest. The blank
`pools.template.json` documents all mandatory roles: generic-candidates,
generic-accepted, broad, earlier, k, cr, ancestry and panels. The custodian must
attest `complete_inventory` and `frozen_before_scoring`; every input file needs an
exact SHA256. Candidate roles must use `scope: "candidate-superset"`, independently
of later response acceptance; existing accepted responses and panels use
`scope: "frozen-reference"`. Empty pools or incomplete declarations fail closed.
Input locations are consumed locally and never emitted in receipts.

Both arms are screened through their shared **content** against complete candidate,
accepted-response, ancestry and panel text. Source lines/paragraphs/sentence
segments are also compared to prevent an appended instruction hiding copied
content. Distinct SP contents are compared with each other. Witnesses are also
screened. NFKC/casefold/whitespace normalization precedes equality, token-five-gram
Jaccard >=0.8 and normalized Levenshtein >=0.9. The intended within-pair match is
exempt. Whole canonical templates are intentionally reusable; they cannot cause
a clause-exclusion failure.

Only **paraphrase** clauses undergo parameter-normalized clause exclusion. Each
atomic clause is checked separately, so a second clause cannot dilute a copy.
The audit records five nearest replay clauses for every request, ranked by the
maximum similarity over its paraphrase clauses. A real inspector must attest
those five entries using their `nearest_sha256`; insufficient lists stay PENDING.
`reviews/nearest-inspection.template.jsonl` contains blank receipt slots.

Before scoring, `freeze` requires clean complete exclusions, fixture agreement,
matching construction bindings, two independent reviews and nearest-clause
inspections. It writes a bound, write-once `FROZEN.json` and the immutable requests,
400 scoring prompts, fixtures, audit and completed review receipts.

Return `recheck` requires broad/K/CR coverage and binds each returned pool to its
frozen candidate-superset name and file hash. Every returned prompt must be a
member of that superset. It verifies content/clause exclusions without changing
requests. Extra candidates, changed parent hashes, missing returns, new overlaps
or invalid bindings yield **UNVERIFIED**, never item deletion or rescoring. The
success status remains `CLEAN` for the existing scoring consumer, with explicit
`verification_only: true`, `panel_changed: false`, `rescoring_allowed: false`.

## Independent reviews and commands

The two independent round-2 receipts are preserved byte for byte in
`reviews/reviewer-1.template.json` and `reviews/reviewer-2.template.json` despite
their historical template filenames. `construction` never overwrites an existing
receipt. This author has not changed any reviewer verdict or written approvals.

`reviews/changed-pairs-round3.json` lists the 182 changed pair IDs, their old/new
request hashes, changed fields, parameter replacements and reviewer findings.
It also binds the original receipt bytes and all 18 unchanged request hashes.
Reviewers need to re-examine only the changed pairs for meaning equivalence,
clarity, satisfiability, canonical provenance and information parity. They may
carry forward their unchanged-pair verdicts after verifying the exact request
hashes, then issue separate full current receipts with the manifest's
`new_binding`. Do not modify or reinterpret the historical receipts. Old global
bindings intentionally fail the new freeze gate, even for reviewers whose
unchanged per-pair judgments remain usable. Neither this manifest nor the
construction receipt constitutes approval.

Every request and the whole resource/fixture set remain hash-bound. Null entries,
duplicate reviewers, missing requests and stale bindings cannot satisfy freeze.
Reviewers' new receipt files do not modify the source JSONL.

Run from the shared build repository:

```sh
PYTHONDONTWRITEBYTECODE=1 python kit/resources/sp/source/build_source.py
PYTHONDONTWRITEBYTECODE=1 python kit/sp_prepare.py construction
PYTHONDONTWRITEBYTECODE=1 python kit/resources/sp/source/validate_source.py
```

The CPU validator uses only authored and synthetic fixtures. It tests canonical
provenance, every fixture, matrix symmetry, clause/content exclusion separation,
review bindings, pool-file hashes and verification-only candidate containment.
Temporary synthetic review objects are never saved as real approvals. Real pools
and model responses are not used by this validation.

After the custodian and independent reviewers supply real artifacts:

```sh
python kit/sp_prepare.py audit --pools pools.json --inspection inspection.jsonl --reviews review-a.json review-b.json --out audit-output
python kit/sp_prepare.py freeze --pools pools.json --inspection inspection.jsonl --reviews review-a.json review-b.json
python kit/sp_prepare.py recheck --pools returned-pools.json --out recheck.json
```

`construction` can be regenerated before scientific freeze; changed artifacts
change bindings and require current receipts. Existing receipt files are preserved. It refuses to run once the panel is
scientifically frozen. No validation command launches a probe or other remote job.
