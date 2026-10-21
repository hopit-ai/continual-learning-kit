# Package C — ContractNLI, W bound at send

Plain jobs; the partner supplies sbatch. Use one 8×H100 node, serial full-node
training, and physical GPU 0 for every scoring job. No downloads or paid calls.
The core always finishes regardless of weak accuracy, retention, or cost.
Exit 2 means a correctness stop; exit 3 means a prerequisite is pending.

Every after-FinQA arm starts at the exact v4 D final for D/D+replay,
or exact v4 F final for F/F+replay/answer-only, seed matched. Replay is applied
during ContractNLI only. B prevention finals belong to connected continuation.
No recovery/reset checkpoint enters selection or starts. This follows C §3 and
the final 20 October amendments. W needs complete unreset B evidence; B tier 2
is forgetting/recipe comparison only (base MATH 89/100), not acquisition evidence.

The byte-frozen legal panels, reserves, prompts and strict/tolerant parsers ship
under resources/c. Its JSONL reader uses newline framing so Unicode separators inside
documents remain intact; parser and prompt function bytes match the upstream freeze. Test reserve is primary; panel is diagnostic. The matched
frontier is frozen at 82; it is never generated here. Goal 1 requires W
standalone's five-seed tolerant mean **and one-sided 95% seed-t lower bound ≥
69.7**, positive gain in ≥4/5 seeds, and a strictly positive gain lower bound.
Missing cells remain pending; question/document bootstrap intervals are separate.

## 1. Environment and local paths

Use the partner's existing inference venv (vLLM 0.18.0) for 27B targets, and
existing trainer venv for training and one-GPU scoring. Retain the previously
working LD_LIBRARY_PATH for each venv. Python preparation/reading needs the kit's
existing dependencies; no installation is performed by these commands.

```bash
export KIT="$PWD/kit"
export OUT=/fast/disk/package-c
export B=/durable/disk/package-b
export B_READING=/durable/disk/b-reading/reading.json
export B_SOURCE="$B"  # or B's compact return tar.gz
export V4_WORK=/durable/disk/v4-finqa-keep
export V4_EXPORTS="$V4_WORK/exports"
export V4_FINQA_PANEL=/durable/disk/finqa
export BASE=/cached/Qwen3-8B
export TEACHER=/cached/Qwen3.6-27B
export CONTRACT_SOURCE=/cached/contract-nli  # original train.json/dev.json/test.json
export SDPO=/cached/SDPO
export TRAINER_PY=/venvs/trainer/bin/python
export INFERENCE_PY=/venvs/inference/bin/python
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$OUT"
```

Student revision b968826d9c46dd6066d109eabc6255188de91218; teacher revision
6a9e13bd6fc8f0983b9b99948120bc37f49c13e9; SDPO
7c457fc1b1f636ae794eb0362ba37d4743b06fbc. Existing kit B model hashes and original
export receipts are checked. These paths must already be local.

## 2. Bind W and the exact starts, at send (CPU)

First finish B's existing reading, if not already made:

```bash
"$TRAINER_PY" "$KIT/p3b_read.py" --source "$B_SOURCE" --out "$(dirname "$B_READING")" || exit $?
"$TRAINER_PY" - <<'PY'
import json, os
from pathlib import Path
v4 = Path(os.environ['V4_EXPORTS'])
b = Path(os.environ['B']); work = Path(os.environ['V4_WORK'])
starts = {'base': {'model': os.environ['BASE']}}
for recipe in ('D','F'):
    for seed in range(101,106):
        slot = f'{recipe}-finqa-alone-s{seed}'
        matches = [p for p in v4.rglob('hf-step40') if slot in p.parts]
        if len(matches) != 1: raise SystemExit(f'STOP: expected one exact v4 export: {slot}')
        model = matches[0]
        receipt = work / 'runs' / slot / 'env/export-identity-step40.json'
        if not model.is_dir() or not receipt.is_file(): raise SystemExit(f'STOP: missing original final/receipt: {model}')
        starts[f'{recipe}:{seed}'] = {'model': str(model.resolve()), 'receipt': str(receipt.resolve())}
(Path(os.environ['OUT'])/'starts.json').write_text(json.dumps(starts,indent=2)+'\n')
PY
"$TRAINER_PY" "$KIT/c_bind.py" --out "$OUT" --b-reading "$B_READING" --b-source "$B_SOURCE" --checkpoints "$OUT/starts.json" || exit $?
```

The binder recomputes B seed bars from the returned scores and checks them
against its reading. Ranking: confirmed joint acquisition/sequence retention;
number of confirmed retention margins (including both stages' general panels);
mean sequence MATH final; lower maximum of the two task-specific mean per-seed
serving-token ratios. Exact ties: F, D, D+replay, F+replay. A completed weak result
still selects provisional W. Undefined serving ratios rank last in that field.
F is shared when W=F: 20 stages, otherwise 30. Binding cannot change in one OUT.

## 3. Freeze membership (CPU), then targets (2 GPUs)

```bash
"$INFERENCE_PY" "$KIT/c_targets.py" --out "$OUT" --source "$CONTRACT_SOURCE" --finqa-panel "$V4_FINQA_PANEL" --tokenizer "$BASE" --prepare-only || exit $?
"$INFERENCE_PY" "$KIT/c_targets.py" --out "$OUT" --source "$CONTRACT_SOURCE" --finqa-panel "$V4_FINQA_PANEL" --tokenizer "$BASE" --model "$TEACHER" --gpu 0,1 || exit $?
```

Preparation validates the original split hashes, excludes cross-split and held-out
documents by both ID and normalized document equality, checks B prompt overlap
against general-v1, all FinQA test questions and legal evaluations, and excludes chat inputs >10,240 tokens.
No truncation. Hash-order document round robin selects up to 1,024 pairs, at most
four/document, without quotas. Membership, counts, exclusions, prompts and hashes
are durable before generation. Shortfalls are disclosed without replacement.

Targets: first complete reasoning plus strict terminal `Answer: <label>` matching
gold, with ≥8 reasoning words (B's structural completeness screen), fitting 2,048
student tokens **including EOS**. The teacher receives only the prompt and a
reasoning instruction. At most four attempts, seeds 27101–27104, temperatures
0/.3/.6/.8, top-p 1, top-k −1, thinking off, max 2,560 output tokens. The legal
teacher engine uses a 16,384 context to accommodate the different 27B tokenizer;
an overlong rendered teacher input is archived as a rejected attempt. All raw
attempts and rejections remain on storage. Retrying the same command reuses
journalled attempts; no extra scientific attempts. Zero valid targets is undefined
(exit 3), with the evidence retained. All arms share the accepted IDs and five
seed-shuffled 1,280-row schedules, cycling only when necessary.

## 4. Score incoming models, train the stages, score all checkpoints

```bash
ARMS=$("$TRAINER_PY" - <<'PY'
import json, os
b=json.load(open(os.path.join(os.environ['OUT'],'binding.json')))
print(' '.join(dict.fromkeys([b['W'],'F','answer-only'])))
PY
)
# Fixed base denominators, one scoring GPU, before any stage.
for PANEL in panel test finqa general-v1; do
  "$TRAINER_PY" "$KIT/c_score.py" --out "$OUT" --arm base --seed 101 --step 0 --panel-set "$PANEL" --panel-root "$V4_FINQA_PANEL" --gpu 0 || exit $?
done
for RECIPE in $ARMS; do
  for ROUTE in standalone sequence; do
    for SEED in 101 102 103 104 105; do
      ARM="${ROUTE}-${RECIPE}"
      for PANEL in panel test finqa general-v1; do
        "$TRAINER_PY" "$KIT/c_score.py" --out "$OUT" --arm "$ARM" --seed "$SEED" --step 0 --panel-set "$PANEL" --panel-root "$V4_FINQA_PANEL" --gpu 0 || exit $?
      done
      "$TRAINER_PY" "$KIT/c_train.py" --out "$OUT" --recipe "$RECIPE" --route "$ROUTE" --seed "$SEED" --replay "$B/replay" --tokenizer "$BASE" --sdpo "$SDPO" --trainer-python "$TRAINER_PY" --gpu 0,1,2,3,4,5,6,7 || exit $?
      for STEP in 10 20 30 40; do
        for PANEL in panel test; do
          "$TRAINER_PY" "$KIT/c_score.py" --out "$OUT" --arm "$ARM" --seed "$SEED" --step "$STEP" --panel-set "$PANEL" --gpu 0 || exit $?
        done
      done
      for STEP in 20 40; do
        for PANEL in finqa general-v1; do
          "$TRAINER_PY" "$KIT/c_score.py" --out "$OUT" --arm "$ARM" --seed "$SEED" --step "$STEP" --panel-set "$PANEL" --panel-root "$V4_FINQA_PANEL" --gpu 0 || exit $?
        done
      done
    done
  done
done
```

The trainer resolves and archives its exact command/config, uses full weights,
40 optimizer updates, batch 32, LR 1e-5, a fresh optimizer, and ten-step warmup
including update one's zero LR. D keeps full-vocabulary SDFT, fresh EMA at .01,
and the original loss. F keeps B's response-only SFT including EOS; answer-only
is that same loss on `Answer: <gold label>` for the same accepted IDs/schedule.
Replay uses the frozen B checked base responses at positions 8/16/24/32:
28 task + 4 generic. D preserves B's full-pool shuffle/skip order for generic
prompts over 2,048 tokens; C task prompts retain their full 10,240 allowance.
Replay overlap with C evaluations is a correctness stop, never silent filtering.
The C-only SFT context is 12,288; D privileged context allowance is 14,336.
These C adapters leave package A/B files and behavior unchanged.

Scores bind the checkpoint automatically from binding.json/export receipts.
Legal outputs are generated once and graded by both frozen parsers; output token
IDs, format compliance, parseability, finish rates and truncation evidence are kept.
FinQA is all 1,147 questions, general-v1 all 300, using B's unchanged deterministic
scorers at 0/20/40. All score jobs bind to the same machine and physical GPU.
Identical completed commands skip. Interrupted stages preserve partials and repeat
from step 0 with the same seed/input; no checkpoint selection on evaluation.

## 5. Read, collect and return

```bash
"$TRAINER_PY" "$KIT/c_read.py" --out "$OUT/reading" --source "$OUT" || exit $?
"$TRAINER_PY" "$KIT/c_collect.py" --out "$OUT" --archive "$OUT/package-c-return.tar.gz" || exit $?
```

Return **package-c-return.tar.gz only**, plus a brief note with node/GPU identity,
wall times, failed attempts, storage path, and anything unfinished. It is strictly
below 28 MiB. Contains gzipped score/metric CSVs, reading.json/.md, per-seed.csv,
W/start identities, costs, summaries and receipts. bulk-evidence.json lists hashes
and sizes of omitted scientific text. Keep the complete OUT durable, including
teacher raw attempts/rejections, schedules, token IDs, trainer inputs/logs/partials,
checkpoints and exports. Weights never go in the compact return; merged weight
hashes remain in export receipts. Manager re-read:

```bash
"$TRAINER_PY" "$KIT/c_read.py" --source package-c-return.tar.gz --out c-return-reading
```

Reading reports strict/tolerant scores and gains, gain AUC, label denominators,
paired gained/lost IDs, document-cluster bootstrap, answer-only contrasts,
sequence-minus-standalone and seed-paired one-sided 95% retention bounds:
legal ≥−2 points; FinQA ≥−1 from incoming; each general panel ≥−3 from base at
both stages; incoming FinQA acquisition lower bound >0. Crossing bounds are
inconclusive. FinQA acquired-answer survival and overlapping IFEval family
lost/recovered IDs are separate. Tokens/correct includes failed-answer tokens,
separately for legal and FinQA; >1.5× base is at cost, zero correct is undefined.

## 6. Operational forecast

Use the measured-stage repricing in [README-message.md](README-message.md):
F 20–26 min, D+replay 58 min (D uses that proxy), eight GPUs. Legal long-input
throughput and 27B target time remain unmeasured. Keep the 45-GPU-hour target and
24 scoring/startup allowances; C's registered 600-GPU-hour ceiling is unchanged.
Do not cut seeds or raise scientific attempts to fit an allocation.

## 7. After-FinQA replay-breadth probe (required before return 1)

Run after core stages using the frozen HellaSwag/pools from README-next N3/N6.
`NEXT` is the sibling next-message tree; `OUT` here is the C tree. All arms start
from **exact v4 W-family D/F finals**, seeds 101–105. Generic is the replay sibling
of W: share C's sequence W run when W=D+replay/F+replay; otherwise run it fresh.
B prevention/sequence MATH finals never supply C or probe starts.

| Probe arm | ContractNLI | Generic positions | Broad positions | Earlier positions | Legal exposures |
|---|---:|---|---|---|---:|
| generic | 28 | 8/16/24/32 | — | — | 1120 |
| broad | 24 | 8/16/24/32 | 4/12/20/28 | — | 960 |
| earlier | 24 | 8/16/24/32 | — | 4/12/20/28 | 960 |
| both | 20 | 8/16/24/32 | 4/12/20/28 | 2/10/18/26 | 800 |

Broad uses checked base GSM8K-train/ARC-train answers; earlier uses verified 27B
FinQA training targets. Pools/checkers/exclusions/licences are N6's, including
C's legal evaluations. `next_slots` supplies the slot map and D skip-in-place
audit. D uses per-item token-mean legal KL and EOS-inclusive replay CE at slot
fractions; F keeps response-only SFT. Reduced legal exposure is part of each arm.

```bash
rsync -a "$NEXT/hellaswag/" "$OUT/hellaswag/" || exit $?
test -z "$(rsync -rcn --out-format=%n "$NEXT/hellaswag/" "$OUT/hellaswag/")" || exit 2
"$TRAINER_PY" "$KIT/c_score.py" --out "$OUT" --arm base --seed 101 --step 0 --panel-set hellaswag --gpu "$SCORE_GPU" || exit $?
for PROBE in generic broad earlier both; do
  ARM=$("$TRAINER_PY" -c 'import json,sys; w=json.load(open(sys.argv[1]))["W"]; p=sys.argv[2]; print("sequence-"+(w if p=="generic" and "+replay" in w else "probe-"+p))' "$OUT/binding.json" "$PROBE")
  for SEED in 101 102 103 104 105; do
    for PANEL in panel test finqa general-v1 hellaswag; do
      "$TRAINER_PY" "$KIT/c_score.py" --out "$OUT" --arm "$ARM" --seed "$SEED" --step 0 --panel-set "$PANEL" --panel-root "$V4_FINQA_PANEL" --gpu "$SCORE_GPU" || exit $?
    done
    "$TRAINER_PY" "$KIT/c_train.py" --out "$OUT" --recipe W --route sequence --probe-arm "$PROBE" --pool-root "$NEXT/pools" --seed "$SEED" --replay "$B/replay" --tokenizer "$BASE" --sdpo "$SDPO" --trainer-python "$TRAINER_PY" --gpu 0,1,2,3,4,5,6,7 || exit $?
    for STEP in 10 20 30 40; do
      for PANEL in panel test; do
        "$TRAINER_PY" "$KIT/c_score.py" --out "$OUT" --arm "$ARM" --seed "$SEED" --step "$STEP" --panel-set "$PANEL" --gpu "$SCORE_GPU" || exit $?
      done
    done
    for STEP in 20 40; do
      for PANEL in finqa general-v1 hellaswag; do
        "$TRAINER_PY" "$KIT/c_score.py" --out "$OUT" --arm "$ARM" --seed "$SEED" --step "$STEP" --panel-set "$PANEL" --panel-root "$V4_FINQA_PANEL" --gpu "$SCORE_GPU" || exit $?
      done
    done
  done
done
"$TRAINER_PY" "$KIT/c_probe_read.py" --out "$OUT/reading" --source "$OUT" || exit $?
# Recreate C reading/archive after the probe, including all probe rows and receipts.
"$TRAINER_PY" "$KIT/c_read.py" --out "$OUT/reading" --source "$OUT" || exit $?
"$TRAINER_PY" "$KIT/c_collect.py" --out "$OUT" --archive "$OUT/package-c-return.tar.gz" || exit $?
```

Step 40 is primary. ContractNLI test-reserve **tolerant score and gain** are the
acquisition endpoints: candidate-reference mean loss ≤1 point and paired 95%
lower bounds ≥−1 for each. Gain-AUC is diagnostic. Broad first vs generic on
maths retention; earlier vs generic if broad fails, otherwise both vs broad on
acquired-FinQA survival (wrong at base/right at incoming FinQA final). Require
≥2-point primary benefit and positive paired 95% interval; protect the other
retention endpoint and HellaSwag within 1 point with confirming bounds. Incremental
additions also protect generic acquisition. Report FinQA net accuracy, survival
cohorts/denominators, maths/knowledge/IFEval separately, answer transitions,
finish rates and failed-answer tokens/correct. Legal question intervals cluster
by document; seed intervals remain separate. Missing cells stay pending.

Broad maths replay is **in-distribution for general-v1 maths**, near-distribution
for knowledge. HellaSwag-100 is the only out-of-pool secondary transfer endpoint;
ARC shares its answer format. A benefit nominates components; admission waits
for connected finals. No acquisition or retention result stops registered arms.
