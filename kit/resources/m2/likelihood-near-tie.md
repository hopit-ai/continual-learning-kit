# Likelihood-scorer validation: near-tie ruling

Fixed by the owner/manager on 1 November 2026. This amendment replaces only the
unanimous selected-choice technical validation rule in the frozen message-2
HellaSwag and message-3 knowledge/HellaSwag policies. It changes no scorer,
selected choice, scientific endpoint, seed rule, panel or reference model.

Validation PASSES if every disagreeing item has a top-two length-normalised
log-likelihood gap strictly below **0.001 nats/token in BOTH scorers**. Equality
at 0.001, a larger gap, or an unverifiable gap keeps validation PENDING. Complete
100-item/four-choice coverage, finiteness, exact checkpoint identities and
assignment checks remain mandatory. Agreed choices need no near-tie exception.

The frozen scorers store log likelihood divided by option **character** count.
For this technical validation only, recover each option's token-normalised
likelihood as `stored_score * characters / conditional_tokens`, then subtract
the second-highest from the highest. The pinned `likelihood-token-lengths.json`
contains all 200 frozen HellaSwag/knowledge item vectors. It was derived on CPU
with the exact frozen tokenizer hashes, no inference, using context-plus-space
continuations and the same conditional-token boundary. Existing scores and their
argmax choices are retained verbatim. Counts include the conditional separator
as tokenised; character counts exclude it, exactly as the original scorers do.

New/derived validation receipts record every item's two gaps, units, both
selected option indices (zero-based), disagreeing pair and near-tie status.
The top-two token-normalised gaps need not involve that disagreeing pair.
K/CR HellaSwag validates panel membership in the pinned length table before
GPU work; missing IDs are a configuration error. A larger-gap disagreement is terminal PENDING: no automatic
rescore/retry. Historical raw validation receipts can be reinterpreted on CPU;
when a node writes amended diagnostics it first archives the original bytes.
Reader output explicitly notes any legacy FAIL/PENDING receipt re-evaluated
as PASS under this amendment; its stored status and bytes remain unchanged.
Current-rule receipts are immutable on repeat calls; only legacy records may
be archived and rewritten. No historical training smoke receipt is changed. The exact original unanimous
HellaSwag attempt-2 receipt remains accepted after all its original checks and
all raw score rows are verified. Its original policy/declaration hashes remain.

`likelihood-near-tie.json` pins only the two amended message-2 validators and the
new diagnostic helper, explicitly naming the original implementation hashes it
replaces. The original `format-validity.json` stays byte-for-byte unchanged so
ARC FROZEN and the historical HellaSwag declaration retain their original pins.
The resource freeze pins both this amendment and the derived length table.
