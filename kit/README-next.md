# Next-message v0+ and v3 scoring supplement

Use [README-message.md](README-message.md) for the complete ordered message, two
Slurm jobs, returns and totals. C and its after-FinQA replay probe are in
[README-c.md](README-c.md). No FinQA→MATH replay-breadth stages run here.
B tier 2 and the v3 MATH curves are forgetting/recipe diagnostics, limited by
headroom (base 89/100); they support no second-task acquisition claim.

Use the existing offline trainer environment and local pinned model/dataset
cache. Keep scoring serial on the same physical GPU. W needs complete unreset B;
weak complete results choose a provisional W, while incomplete evidence is pending.
These are plain invocations for the partner's own sbatch.

```bash
export CHECKOUT=/absolute/reviewed-next-message-checkout
export KIT="$CHECKOUT/kit"
export NEXT=/absolute/node-local/next-OUT
export NVME=/absolute/node-local/nvme-blends
export TRAINER_PY=/absolute/trainer-venv/bin/python
export B=/absolute/complete-package-B-OUT
export A=/absolute/complete-package-A-OUT
export V4_WORK=/absolute/preserved-finqa-WORK
export V4_EXPORTS=/absolute/v4-finqa-exports
export BASE=/absolute/pinned-Qwen3-8B-snapshot
export FINQA="$V4_WORK/v4/datasets/datasets/v4_finqa"
export FINQA_PANEL="$V4_WORK/v4/report-inputs/finqa-panel.jsonl"
export SDPO=/absolute/pinned-SDPO
export SCORE_GPU=0
export DURABLE=/absolute/durable-storage/next-OUT
export RETURN=/absolute/next-return-1.tar.gz
export RETURN2=/absolute/next-return-2.tar.gz
export RUN_4A=0 # change to 1 only with a present, registered and audited 4A kit
mkdir -p "$NEXT" "$NVME"
```

N1–N6 are the supplement rows. N3 freezes HellaSwag before new outputs;
N4 scores six SEC checkpoints; N5 blends/scores/deletes all 60 available finals
at five lambda points (180 interiors, 120 endpoint aliases). The full screen
has 663 unique panel scorings. N6 prepares pools for C, after the quick screens.
Blend weights remain on NVMe only; preserve tensor hashes and re-creation commands.

N1. Select W with C's amended ranking and B's unchanged one-sided bars. All four
recipes, five seeds, unreset curves and shared base denominators must be complete.
Incomplete evidence exits NOT READY; weak complete evidence selects provisional W.
The exact tie order is F, D, D+replay, F+replay. Retention count includes the
stage-1 and stage-2 general margins plus FinQA survival from the incoming final.

```bash
"$TRAINER_PY" "$KIT/next_w.py" --out "$NEXT" --source "$B" || exit $?
```

N2. The binder checks actual files against receipts, locates unique D/F exports,
records all missing optional tier-2 screen slots, and copies B's frozen MATH
schedules and generic replay unchanged into NEXT. It requires all 20 tier-1 FinQA finals, including W and its replay sibling. Sequence blends require their exact incoming receipt;
base is used only for FinQA/standalone starts. TSV paths must not contain tabs
or newlines. `inputs.json` contains hashes and the W decision, not a path-only list.

```bash
"$TRAINER_PY" "$KIT/next_bind.py" --out "$NEXT" --w-identity "$NEXT/W.json" --base "$BASE" --package-b "$B" --v4-work "$V4_WORK" --v4-exports "$V4_EXPORTS" || exit $?
```

N3. **Local pinned cache required** (pinned HellaSwag download). Freeze the untouched panel and strict final-line parser **before any new model
output**. The only HellaSwag split read is validation; no split supplies replay.
Use only rows with `source_id` starting with `activitynet~` (ActivityNet-derived
items; no WikiHow items). Check duplicate `ind` values within that population.
Select first 100 by `sha256("next-v4-holdout-v1|"+str(ind))`, tie by ind. Preserve original
context/endings labelled A–D, suffix `End with Answer: <letter>`. The score requires
an exact final line `Answer: A`, B, C or D (case-sensitive, no punctuation). It is
frozen before seeing answers, never relaxed. Its manifest carries the panel hash.

Before the offline job, stage this exact parquet revision into the Hugging Face
cache that the job will use (preserve the same `HF_HOME`/`HF_HUB_CACHE`):

```bash
huggingface-cli download Rowan/hellaswag --repo-type dataset --revision 218ec52e09a7e7462a5400043bb9a69a41d06b76 --include 'data/validation-*.parquet' --cache-dir "$HF_HOME/hub"
```

`data/validation-00000-of-00001.parquet` must have SHA256
`899813071e1e95efafec90f856e1987d2150fa4d020fc005df6962c259f660cd`;
a missing file, unexpected validation file or wrong hash STOPs (exit 2).

```bash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
"$TRAINER_PY" "$KIT/next_hellaswag.py" --out "$NEXT" --action prepare || exit $?
```

N4. The binder chooses D+replay if W is D/D+replay, otherwise F+replay. Each job
scores all months memory-off, then builds static k=3 memory from **that same
checkpoint's earlier memory-off successes**, passing historical and explicit-unit
scorers. Holdout, target issuer components, accessions and filing-group exclusions
are inherited unchanged. Current-month answers never enter memory. Each seed has
its own journal; memory manifests bind collector, checkpoint, batch and responses.
Decoding: greedy, thinking off, seed 0, 2,048 output, 12,288 context. Seed 0 is base.

```bash
while IFS=$'\t' read -r SEED MODEL RECEIPT; do
  "$TRAINER_PY" "$KIT/next_sec.py" --out "$NEXT" --seed "$SEED" --checkpoint-path "$MODEL" --checkpoint-receipt "$RECEIPT" --collector-checkpoint-path "$MODEL" --collector-checkpoint-receipt "$RECEIPT" --gpu "$SCORE_GPU" || exit $?
done < "$NEXT/next-inputs/sec.tsv"
```

N5. Every floating tensor is interpolated in FP32 and cast to its original export
dtype (including bf16); keys, shapes and dtype must match. Tokenizer encoding
assets (`tokenizer.json`, `vocab.json`, merges files) must match; differing
export metadata (`tokenizer_config.json`, chat template, added/special token
files) is hash-listed in the receipt and copied from the final export. Nonfloating
buffers must be identical. Lambda 0 and 1 score the exact parent weights without materialising a blend.
Controls are freshly scored once per identity/panel here, covering the required case where the scoring GPU is
not B's. Sequence lambda 0 is its matched FinQA start. No recovery/reset blends.
Only node-local NVMe is accepted (`findmnt`/`lsblk` ancestry); parent weights stay
on durable storage. A failure also deletes temporary weights. Retain tensor hashes,
both parents, metadata differences and an executable `--recreate DEST` command.
Re-creation bypasses completed-score SKIP, writes to a new local NVMe directory,
and verifies every recorded tensor hash without scoring. The operator removes
that verification copy after inspection. NVMe needs space for only one blend plus
scoring scratch. A lambda never becomes incumbent; **v3 proper fixes 0.5 now**.

```bash
while IFS= read -r SLOT; do
  for LAMBDA in 0 0.25 0.5 0.75 1; do
    "$TRAINER_PY" "$KIT/next_blend.py" --out "$NEXT" --slot "$SLOT" --lam "$LAMBDA" --nvme-root "$NVME" --package-a "$A" --finqa "$FINQA" --gpu "$SCORE_GPU" || exit $?
  done
done < "$NEXT/next-inputs/screen.txt"
```

N6. **Local pinned cache required** (pinned GSM8K and ARC files), plus one GPU for
base-response preparation. N3 and N6 read the local pinned Hugging Face cache; model and
existing data paths above must already be staged. Pinned
sources (parquet files staged from these exact Git commits, file hashes retained):

| Source | Split/config | Revision | Notice carried on every pool row |
|---|---|---|---|
| [openai/gsm8k](https://huggingface.co/api/datasets/openai/gsm8k) | main/train | `740312add88f781978c0658806c59bc2815b9866` | Cobbe et al., MIT |
| [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc/commit/210d026faf9955653af8916fad021475a3f00453) | ARC-Challenge/train | `210d026faf9955653af8916fad021475a3f00453` | Clark et al., CC-BY-SA-4.0, attribution/link/changes |
| [Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag/blob/218ec52e09a7e7462a5400043bb9a69a41d06b76/README.md) | validation, ActivityNet-derived items only | `218ec52e09a7e7462a5400043bb9a69a41d06b76` | Zellers et al., MIT |

GSM prompt is exactly the general-v1 maths instruction from
`beds/gsm8k.py:PANEL_PROMPT_SUFFIX`; checker is `scorers.extract_number` (the
extractor used by eval_bed's general scorer) followed by exact rational comparison
with the gold final number. ARC retains original source labels, including numeric
labels where supplied, uses the frozen letter-answer suffix, and requires a strict
final `Answer: <source label>` equal to `answerKey`. No source solution enters the
prompt or replay target. Checks verify final answers, not rationales.

Source keys for hash order are `gsm8k` and `arc`, with original train index/source
ID, respectively: `sha256("next-v4-broad-v1|"+source+"|"+id)`, ID tie-break.
Namespace replay IDs as `gsm8k-train-<idx>` and `arc-c-train-<id>`.
Keep at most 1,024 deduplicated candidates/source **before** one greedy base answer
per candidate; no replenishment. Apply B's source-ID, NFKC/casefold/whitespace,
5-shingle Jaccard >=.8 and edit similarity >=.9 rules, including short prompts,
within/across generic, broad and earlier pools and against all 1,147 FinQA,
300 general-v1, frozen MATH-100, C legal panel/test and HellaSwag-100 evaluation items. Earlier-target
dedup takes place first; within broad, GSM then ARC is fixed before generation.
Equal-source cycling shuffles each accepted source by the paired training seed.

Earlier replay takes unique B COMMON FinQA training targets (all five schedules),
requires original qualified 27B provenance and re-verifies their teacher text with
B's checker. Evaluation items never supply targets. Freeze outputs, IDs, exclusions,
licences and source hashes before training. Empty pools are incomplete inputs.
Before excluding earlier/broad rows, audit B's entire inherited generic pool
against every panel, including the new HellaSwag-100, and within the pool.
An overlap stops preparation without editing B's pool; retain
`pools/generic-exclusion-audit.json`, including the exclusion IDs and hashes.
Generic remains B's original checked RLVR-IFeval pool, excluding refresher rows;
its source/checker pins and `IF_FUNCTIONS_MAP` remain B's.

```bash
TARGETS=()
for SEED in 101 102 103 104 105; do
  TARGETS+=("$V4_WORK/v4/report-main/scheduled/finqa-s${SEED}")
done
"$TRAINER_PY" "$KIT/next_pools.py" --out "$NEXT" --package-a "$A" --finqa-panel "$FINQA_PANEL" --finqa-targets "${TARGETS[@]}" --gpu "$SCORE_GPU" || exit $?
```


N10. Read v0+/v3 results (the C probe has its own reader).

```bash
"$TRAINER_PY" "$KIT/next_read.py" --out "$NEXT/reading" --source "$NEXT" || exit $?
```

v0+: met means seed mean ≥ reference; confirmed also requires seed and issuer
clustered non-inferiority lower bounds ≥ −2 points. July memory-off is compared
to frozen; August/September own-memory to frozen+memory. Report own-memory gains,
trace yield, tokens/correct and B's existing general margins without re-judging.
v3 uses paired two-sided 95% intervals; ≥2-point retention benefit, positive
interval, acquisition/other panels protected within 1 point. Proper lambda is
fixed at 0.5, conditional on v2. No evaluation panel selects lambda or incumbent.
Endpoint controls are freshly scored on the active GPU; identical endpoint
hashes share generations within that scoring epoch.

N11. After C + probe, collect as part of message return 1 (see the main runbook).

```bash
"$TRAINER_PY" "$KIT/next_collect.py" --out "$NEXT" --durable-root "$DURABLE/NEXT" --archive "$RETURN" || exit $?
```

N12. Verify/read the archive on our side:

```bash
"$TRAINER_PY" "$KIT/next_read.py" --out /absolute/returned-reading --source "$RETURN" || exit $?
```

Every identical completed job SKIPs. Interrupted blends are recreated and scored
before deletion, completed scorer outputs remain reusable. Identity changes stop.
The offline `next_dryrun.py --out <fresh-dir> --standin` rehearses these rows;
`message_dryrun.py` rehearses the whole combined message including C and handoff.
