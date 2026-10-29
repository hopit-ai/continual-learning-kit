# Connected continuation: base → FinQA → ContractNLI → MATH

**Conditionally registered 18 October 2026, before any connected-continuation result.** Owner approval covers the Package C amendments in [Astra's sequencing review](sequencing-check-astra-20261018.md). This is a desk registration, conditional on completed unreset [Package B](package-b-registration-20261014.md) evidence and [Package C's](package-c-registration-20261018.md) recipe selection. Build only after C's selected recipe and its selection rationale are recorded; no scripts, paid runs or launches are added now. Deliver plain preparation/training/scoring/readout scripts in a **future consolidated partner batch**. The partner uses his own `sbatch`; batch inclusion follows the existing owner approval process.

## Trajectory and identities

The system claim requires acquired tasks to coexist along **one connected checkpoint trajectory per seed**. Successful scores from separate branches cannot establish that claim. Use **five paired seeds, 101–105**, with one final step-40 checkpoint at each stage:

| Stage | Incoming weights → training → outgoing weights |
|---|---|
| 0 | Frozen Qwen3-8B base, revision `b968826d9c46dd6066d109eabc6255188de91218`; score before training. |
| 1: FinQA | Base → B's selected recipe → FinQA final. If B selects D+replay, use the exact **B tier-1 prevention-trained D+replay final**, not v4 D plus replay added later. Otherwise use the exact base-to-FinQA final for B's selected D or F recipe. |
| 2: ContractNLI | That seed's stage-1 final → **C's selected recipe** → ContractNLI final. |
| 3: MATH | That exact stage-2 final → **the same recipe used for ContractNLI** → MATH final. No recipe reselection after stage 2. |

Bind each seed's base, parent and output identities with export hashes and run IDs; preserve the complete chain in the return. Reuse verified stage-1 finals where available rather than retraining or charging them twice. No reset/recovery branch, best intermediate checkpoint or seed substitution enters this chain. C's core v4 starts remain distinct evidence. If B and C select different recipes, label the chain explicitly as that fixed recipe transition; it is not a single-recipe result across all three stages. A provisional selection stays labelled provisional.

Inherit the selected B/C recipes' full-weight 40-update stages, batch 32, LR `1e-5`, ten-step warmup, fresh optimiser/schedule and fresh EMA where applicable, losses and 2,048-token response limit. Preserve D+replay's fixed 28-task/4-generic positions and checked base-response pool when selected. Use B's frozen FinQA and 500-item MATH training pools/verified targets and C's frozen ContractNLI pool/targets. Check replay/prompt overlap against all three evaluations and general-v1 before build; record exclusions and hashes prospectively. Reuse compatible targets without new selection or scientific retries.

## Scoring and inherited bars

At base and **after every stage**, score all three task panels and general-v1: all 1,147 FinQA questions, C's frozen 100-item ContractNLI test reserve and B's frozen 100-item MATH evaluation. Thus each boundary measures the current task and **every earlier task** on the connected weights; future-task scores supply incoming competence. General-v1 reports IFEval, knowledge and maths separately, with IFEval's case, punctuation, length, keywords, detectable format, content and start/end-wrapper families, denominators, overlapping memberships and lost/recovered IDs. Also inherit B/C's step-20 earlier-task/general diagnostics and 0/10/20/30/40 acquisition curves for the new task. Step 40 remains primary.

Use the frozen task scorers, C's strict/tolerant parsers, deterministic single-physical-GPU scoring and B/C's pinned decoding settings. Score base/incoming comparators on the same scoring setup; retain raw responses, token counts, finish reasons and hardware/software identities. Track survival of answers acquired at each task's own stage (wrong at base, right after acquisition) through **each later boundary**, with denominators, alongside net accuracy.

Bars inherit B and C and are **analysis criteria, never performance stops**:

- **Acquisition:** FinQA gain has a positive one-sided 95% seed-t lower bound. ContractNLI uses C's amended matched-frontier bar: five-seed tolerant mean and its one-sided 95% lower bound ≥ **0.85 × F_test**, positive gain in ≥4/5 seeds and a positive gain lower bound. Show strict accuracy/format compliance separately. MATH inherits B's mean gain ≥ **6/100**, positive in ≥4/5 seeds and a positive one-sided 95% gain lower bound; show gain from incoming weights and from base separately, applying B's standalone acquisition bar to its same-recipe base-start reference.
- **Sequence acquisition:** ContractNLI and MATH final scores each finish within **2 percentage points** of their same-recipe standalone reference under C and B, respectively. Use matched seeds and frozen scoring; if a required comparator is absent or incompatible, leave that comparison pending rather than substitute a branch or silently budget new controls.
- **Earlier-task retention:** FinQA loses **≤1 point** relative to its stage-1 final after both stages 2 and 3. Extend the inherited earlier-task 1-point margin to ContractNLI at stage 3 relative to its stage-2 final. This cumulative comparison cannot be replaced by successive 1-point allowances. Each general-v1 panel remains within **3 points of original base at every stage**.
- **Cost:** output tokens per correct, including tokens spent on failed answers, must be **≤1.5× frozen base** for each task separately. Report this at every scored boundary, total output tokens, finish/truncation rates, preparation/training/scoring cost and charged GPU-hours. Zero correct yields undefined/infinite cost; passing accuracy/retention above this bar is “at cost.”

Apply seed-paired one-sided 95% bounds to retention margins; crossing is inconclusive even if means pass. Show every seed and question-paired bootstrap uncertainty separately, clustering ContractNLI by document. Questions and checkpoints do not add training seeds. The ContractNLI frontier reference is the manager's `claude-opus-5-5` medium-effort run on the identical reserve, archived at `docs/phase4/evidence/legal-headroom/frontier-contract_nli-test/`; reuse it without another API call. The historical panel score 78 and planning bar 66 do not decide this chain's frontier comparison. This chain is a separate readout, not a replacement for C's primary W-standalone test.

## Conditional budget and delivery

| Reservation | GPU-hours |
|---|---:|
| Full-chain slot proxy: 3 stages × 5 seeds × 17.7 | **265.5** |
| Additional all-task/boundary/step-20 scoring and export allowance beyond slot scoring (planning estimate) | **30.0** |
| **Conditional planning total** | **295.5** |

The **17.7 GPU-h** proxy is `407.4/23`, inherited from v4, not measured ContractNLI throughput. The full-chain figure includes stage 1 for transparent accounting; reusing five B FinQA finals removes **88.5 GPU-h** from incremental stage slots, leaving **177 + 30 = 207 GPU-h** as the provisional additional reservation. Do not double-charge prior B work or reused targets/frontier inference. The extra 30 hours is an allowance to refine from B/C timings before batch inclusion, not measured cost or authority to run now. Any new target generation or missing standalone controls require an explicit prospective budget addition. This reservation is outside C's 600-hour core; local desk work is $0, with no paid runs here. Report a revised estimate before exceeding the batch's agreed allocation; do not cut seeds to fit it.

## What the result can and cannot show

This can test whether the selected prevention/acquisition recipes accumulate finance, legal and mathematical benchmark performance on connected weights while retaining measured earlier and general skills, and at what serving/training cost. It closes the gap between C's old v4 start and B's prevention-trained checkpoint. One fixed order, three short stages, five seeds and small evaluation panels cannot establish universal non-forgetting, long-horizon stability, internal plasticity, optimal recipe selection, equal-compute superiority or production legal reliability. Selection on B/C evidence makes this a conditional demonstration, not independent confirmation of recipe superiority. If an earlier task never gains, retention of that task cannot be described as retention of newly learned capability. It does not test goal 2's repeated learning from arriving traces; that requires the separately registered stream work.

## Amendment, 21 October

Written before the partner’s hard-MATH teacher result is known. This supersedes the easy-MATH third-stage specification and old cost proxy. It is a desk registration, not launch authority; other registered rules remain unchanged.

**Decision and binding.** Preserve the frozen screen registration (`91bf4f3d…d7d4`) and amendment 1. The student passed 138/200 (69%). `QUALIFIES` requires verified teacher demonstrations on ≥160/200 items, including ≥30 of the 62 student failures. Use `math_hard_teacher_read.py` to validate evidence and recompute the decision. `INCOMPLETE` leaves stage 3 pending; it is neither outcome below. The quick job supplies screen coverage, not training-pool targets.

Return 1 binds W, C’s recipe/checkpoint identities and the probe decision under registered rules. Tier 2 currently gives **W = D, provisional**. These are binding-at-send rules. Probe success nominates replay components; admission requires connected-final confirmation under plan v6.

**If QUALIFIES.** Use the frozen **500 Level-5 training items** in `data/math_hard/train_pool.jsonl` and **100-item evaluation panel** in `data/math_hard/eval.jsonl`, bound by their manifest. Both exclude the 200-item screen and tier 2’s MATH evaluation; training also excludes the hard evaluation panel. Preserve source, rational-answer eligibility, overlap exclusions and near-duplicate checks. Membership uses proportional subject quotas by largest remainder, then `(sha256(salt + "|" + id), id)`, with salts `hmath-v1-pool` and `hmath-v1-eval`. Never redraw membership or select hardest types after coverage results.

Generate targets only for those 500 training items with pinned Qwen3.6-27B under **package B’s 3D bounded target protocol**: at most four attempts, seeds 27101–27104, temperatures 0/.3/.6/.8, top-p 1, thinking off, 2,560 output tokens and 4,096 input tokens. Accept the first complete reasoning-plus-box demonstration passing the frozen rational verifier and EOS-inclusive 2,048-token student limit. Gold is verifier-only. Archive all attempts; no rewrites or replacement items. Share accepted targets across arms, cycle seed-shuffled IDs and report coverage. Zero accepted targets leaves input undefined.

Run **five paired seeds, 101–105**, from each trajectory’s exact ContractNLI step-40 final: plain W=D and its replay sibling. Preserve each recipe’s own FinQA history, including B’s prevention-trained D+replay finals for the sibling. C’s replay finals starting from v4 D cannot substitute for that history. The **plain-D connected ContractNLI stage is C’s core sequence-D run from the v4 D finals**. Bind parent/output hashes and run IDs; reuse only identical compatible stages.

Each MATH stage has **40 updates**, batch 32, LR `1e-5`, ten-step warmup, fresh optimiser/schedule and EMA, and inherited D objectives and response limits. Plain D remains plain D.

The replay sibling is one configuration from start to finish. If the connected broad-first, then earlier-task rule admits a component, the sibling’s MATH stage starts from the connected ContractNLI final trained with that replay and keeps it during MATH. If none is admitted, both stages use generic replay. If connected admission is still pending when message 2’s MATH stages are built, the sibling’s MATH stage waits and never starts from a pending configuration.

Apply those registered admission rules before MATH. Preserve frozen pools, losses and slot maps: 28 task/4 generic; 24/4/4 for one addition; 20/4/4/4 for both. MATH outcomes never choose replay.

Score the hard panel at base, every boundary and MATH steps 0/10/20/30/40. Acquisition requires mean gain **≥6 percentage points over incoming ContractNLI weights**, positive gain in ≥4/5 seeds and a positive one-sided 95% seed-t lower bound. Call it acquisition only if the five-seed final mean also exceeds the measured base on the hard panel; otherwise label the gain recovery.

Six points is about 19% of the screen’s 31-point headroom. The 69% screen score is not the untouched evaluation baseline: report gains from measured base and incoming weights. The same-recipe standalone comparison remains pending unless a compatible hard-MATH control exists; easy-MATH controls cannot supply it.

Retain all **1,147 FinQA questions**, the **100-pair ContractNLI test reserve**, and general-v1’s separate panels. FinQA may lose ≤1 point from its FinQA final; ContractNLI ≤1 point from its ContractNLI final; each general panel ≤3 points from original base. Preserve inherited scoring, paired uncertainty and retention bounds, boundary/step-20 diagnostics, acquired-answer survival and serving-cost bars. These are analysis criteria, not performance stops.

**If DOES_NOT_QUALIFY.** Drop and record stage 3. The continuation becomes **FinQA → ContractNLI**. Connected finals still report legal acquisition, FinQA retention and acquired-answer survival, general retention and cost. Third-task acquisition and retention of both earlier tasks through another learning stage are lost. A hardest-types screen requires a separate owner decision and prospective registration.

**Budget and order.** Message 2 follows return 1; C and probe bindings gate it, 4A does not. README-message gives 58 minutes × eight GPUs = **7.73 GPU-h/stage**, with D using D+replay’s measured proxy. Ten MATH stages cost **77.3 GPU-h**. Allow **10** for 500-item targets and **30** for scoring/export: **117.3 incremental GPU-h**, plus missing connected legal stages at 7.73 each.

Five missing sibling legal stages give **156.0 GPU-h, a floor**; each component confirmed on connected finals adds five legal stages (**38.7 GPU-h**) plus its scoring. Without MATH, the five missing sibling legal stages plus scoring allowance total **68.7 GPU-h**, before additional component confirmations. Plain-D legal stages are reused from C, explaining why only sibling legal stages are counted as missing.

Level-5 targets are long, so the FinQA-based 58-minute proxy may underestimate cost. Reprice from return 1, explicitly adding replay-confirmation controls, their scoring and allocation overhead. Never double-charge reused stages or silently add standalone controls.

*Drafted by Astra; cross-family review by Opus (one blocking finding on the replay sibling's chain, applied); committed by the manager before the hard-MATH teacher result.*

**Manager note, 24 October (owner: "Go ahead with all 3"; before return 1 and before the hard-MATH result).** Declared exception to "no branch build before entry", as for package C: the message-2 kit is built now as a parametric kit whose W, replay-sibling configuration (registered broad-first, then earlier-task admission on connected finals, pending stays pending) and third-stage outcome (QUALIFIES / DOES_NOT_QUALIFY / INCOMPLETE) bind at send from the returned evidence by the rules above. No choice is left to be made after results; nothing built is wasted under any outcome. No paid run or partner launch follows from this note.

**Manager clarification, 24 October (before return 1; procedure only, no rule changed).** The kit builder asked when the replay sibling's admission binds. Return 1's probe only nominates (its reader reports admission "Pending connected finals"). Message 2 therefore (1) binds at send the nomination state, W and the hard-MATH outcome; (2) runs the sibling's connected ContractNLI confirmation stages; (3) applies the registered broad-first, then earlier-task admission rule mechanically on the node with the kit's frozen reader, writing the result durably before any sibling MATH stage; and (4) runs the sibling MATH stages only with that admitted configuration. If the frozen reader returns pending or incomplete, the sibling MATH stages do not run and the evidence returns for the next message. Plain-D stages never wait for replay admission. No human choice enters between steps 2 and 4.

