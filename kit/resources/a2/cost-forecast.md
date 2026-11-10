# A2 cost forecast (partner GPU-hours)

No 4A outcome or actual run timing was opened. The historical D+replay proxy is
6.3–6.7 GPU-h/stage, borrowed provisionally for D; the slower 58-minute anchor
is 7.733333 GPU-h. These are not qualified SEC runtimes. Missing node D/replay/
rewrite timing remains PENDING until qualification. The source citations below
identify repository-only documents, absent from a kit-only export: 4A registration
section 11 (`docs/phase4/4a-registration-20261020.md`), the historical kit build
report (`docs/phase4/4a-kit-build-report-20261021.json`, D_replay.stage), and the
long-cap reprice (`docs/phase3/kcr-reprice-20261029.md`).

Every standalone contrast has 18 fresh stages: 113.4–120.6 training GPU-h.
RP has 27, including its nine shared control stages once: 170.1–180.9 GPU-h.
No saving assumes reused 4A training or historical scientific scores.

The registration's inherited formula retains a 60 GPU-h scoring reserve and
80 GPU-h overhead/failure reserve, with 1.5 times training, rounded up to 50.
The operational forecast now prices **every exploratory and confirmation cell**.
The exploratory allowance is max(60, cells × rate + 5), so a cell-based estimate
can increase the inherited reserve. There is no double counting of that 60.
The unqualified rate range is 0.1–0.3 GPU-h/cell, including serial model loads;
the extra 5 GPU-h per scoring phase is a provisional loading/failure reserve.
These are assumptions, not node measurements. `a2_budget.forecast` computes this
arithmetic and retains the registered rounding before confirmation.

| Study | Complete exploratory cells | Extra chronological arrival/collector cells | Total exploratory commands | Confirmation cells |
|---|---:|---:|---:|---:|
| J(b) | 285 | 90 | 375 | 102 |
| P | 285 | 72 | 357 | 102 |
| R | 285 | 72 | 357 | 102 |
| RP | 420 | 108 | 528 | 153 |

Complete scoring remeasures arrival checkpoints and references on its common
allocation; chronological arrival/collector calls are separately charged.
Confirmation includes fixed June October/November scoring and all terminal
holdout/general guards. No exact-prompt or model-load deduplication saving is
assumed. Standalone exploratory scoring is 60–117.5 for J and 60–112.1 for P/R;
RP is 60–163.4. Confirmation is 15.2–35.6 standalone and 20.3–50.9 for RP.

| Other item | Planning GPU-h | Basis/limitation |
|---|---:|---|
| Connected-input CPU verification/export | 0 | Inputs already installed; missing upstream training remains PENDING |
| J current collection | Included above | 18 collector cells are inside the 90 extra chronological cells; no double counting |
| R one shared rewrite pass incl failures | 1–8 | <=303 non-holdouts, 2048-token cap, assumed 25–200 tokens/s |
| Exports, transfers, idle and failed allocations | 80 | Inherited reserve; replace with actual allocated walltime |

J's prior separate 3–25 collection proxy described the same 18 cells. Counting
those cells at the stated rate gives 1.8–5.4 GPU-h within exploratory scoring;
no second collection addition is charged in the revised totals. Additional
connected preparation, if needed, is still separately unmeasured.

| Study | Rounded allowance before confirmation | Provisional total incl confirmation |
|---|---:|---:|
| J(b) | 350–400 | **365.2–435.6** |
| P | 350–400 | **365.2–435.6** |
| R | 350–400 | **365.2–435.6** |
| RP | 400–550 | **420.3–600.9** |

Rounded outward to whole GPU-hours these are **365–436** standalone and
**420–601** RP. J may rise further if connected-input preparation or qualified
collector throughput exceeds the per-cell assumption; for example adding 25
GPU-h of genuinely additional work raises its upper forecast to 485.6.

The frozen single-allocation rule prevents splitting complete comparisons at
the default 600-second command limit: J/P/R are 52:17, RP 77:02; RP confirmation
is 28:05. These long lanes fit the captured seven-day site limit. Submit each
complete comparison in one allocation with its declared ceiling.
Monthly rewrite rows instead use actual admitted counts from the bound ledger
and split into <=24-hour command batches. Smaller rows do not establish a
utilisation qualification; utilisation remains warn-only.

Sensitivity: the common-long-cap 15.49–25.56 stage proxy gives 278.82–460.08
standalone training GPU-h. With the revised cell pricing, R's rounded allowance
is 600–900, or 615.2–935.6 including confirmation. The 7.733333 anchor gives
139.2 training GPU-h; per-cell scoring still must be charged. The registered
60–120 R and 150–300 P ranges omit the full fresh-pair burden.

These are forecasts, not spending ceilings or execution authority. Bind measured
D, generic/rewrite context and scoring timings, connected preparation, rewrite
failures and transfers during qualification under the separately approved
A2 cap. The price row then permits continuation or returns owner escalation.
RP charges shared control, scoring and overhead once to study expenditure; each standalone system
charges shared artifacts fully. Upstream shared raw-ledger/memory costs remain
UNMEASURED without receipts, never zero. Node accounting uses GPUs × actual full
allocation elapsed time, including idle and failed time. Queue gaps cost zero.
