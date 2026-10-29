# Author-performed nearest-clause inspection

The SP author inspected all five ranked replay entries for each of the 200
paraphrases from the available real-pool audit: 1,000 ranked entries, containing
196 distinct source clause texts. `inspection.jsonl` retains every entry, its
pool and row identity, the exact nearest-list hash, the request hash and the pool
inventory digest. `reference-clauses.json` records the author's wording judgments
for the shared display of repeated clause texts. Both atomic paraphrase clauses
were inspected when a request has two constraints.

All 200 available lists were judged wording-distinct. This is an **author
inspection**, not a third independent semantic review and not a replacement for
the two independent round-3 receipts. Those receipts remain untouched.

The largest observed normalized edit similarity among these neighbors is 0.7.
Related instruction templates share ordinary constraint vocabulary; many other
nearest entries are unrelated maths, finance or legal content. Repeated clauses
from different rows or role-specific supersets retain separate ranks, as required
by the existing audit's ordering. No change to thresholds or clause extraction
was made. An independent length-bound exact nearest search reproduced every
list and hash from the exhaustive audit.

**Inspection status: accepted for the FROZEN available set under the manager ruling.**
The receipts retain `inventory_complete: false` because node-held inventories
were unavailable. The freeze explicitly binds this scope and all 200 nearest
lists; the default complete-inventory gate remains strict. Remaining node-held
inventories now receive VERIFY-ONLY exclusion checks before scoring, with
pair-local UNVERIFIED flags and no deletion, rewriting or rescoring. See
`../../NODE-VERIFY.md` for the bound node contract.
No SP request, parameter, witness, checker or independent receipt changed.
