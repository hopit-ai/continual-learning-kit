# One combined next partner message

Use this order for the two follow-on jobs, with a seven-day cap per job. The
partner owns sbatch. No submission wrapper, network, push or paid API is needed.
The final 20 October registration amendments take precedence over plan v6's old
probe-before-C order: replay breadth now runs on C's FinQA→ContractNLI route.
B tier 2 is read as forgetting/recipe comparison only, limited by MATH headroom
(base 89/100). Complete unreset B selects W; recovery/reset evidence never does.

## Stage before job 1

Stage on a network-enabled CPU host into the **same HF cache used by the
partner's offline environment**, before reserving GPUs. The commands below
are staging instructions, not job commands. Set `HF_HOME` to the durable cache;
`--cache-dir "$HF_HOME/hub"` keeps the snapshots visible to the kit's
`local_files_only=True` loaders. Preserve snapshot revision directories and
symlinks when moving a cache. If using `hf download` instead of
`huggingface-cli download`, keep the same arguments. Disable offline flags only
on the staging host:

```bash
export HF_HOME=/absolute/durable/huggingface
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0
```

This inventory follows `next_hellaswag.download/panel_rows`,
`next_pools.job/candidates/earlier_rows`, `c_targets.source_rows`,
`a4_bind`/`a4_train`, and `math_hard_teacher.frozen` and their inherited
`second_task_screen.MODELS` / `p3b_common.identity` pins. The partner **reported**
ContractNLI, GSM8K and ARC staged and verified; that report is not an independent
verification here. Require the manager receipt below before a tag.

| Input | Repository and immutable revision | Files read / rows needing it |
|---|---|---|
| HellaSwag | `Rowan/hellaswag` @ `218ec52e09a7e7462a5400043bb9a69a41d06b76` | `data/validation-00000-of-00001.parquet`; M2 freezes 100 ActivityNet rows from 3,243 unique `ind` (10,042 total source rows); M3/M6/M7 scoring and M5 exclusions reuse it. No train/test or WikiHow replay. |
| GSM8K | `openai/gsm8k` @ `740312add88f781978c0658806c59bc2815b9866` | `main/train-*.parquet`, 7,473 rows; M5 broad replay, used in M7. No kit-pinned file hash; record observed SHA-256. |
| ARC-Challenge | `allenai/ai2_arc` @ `210d026faf9955653af8916fad021475a3f00453` | `ARC-Challenge/train-*.parquet`, 1,119 rows; M5 broad replay, used in M7. No kit-pinned file hash; record observed SHA-256. |
| MATH mirror (source provenance check) | `EleutherAI/hendrycks_math` @ `21a5633873b6a120296cce3e2df9d5550074f4a3` | Seven subjects' `train-*.parquet` and `test-*.parquet`, 14 files, 7,500/5,000 rows. M3/M5/M6/M7 read Package A/B's already-frozen MATH panel/schedules; M3b reads the bundled 200 Level-5 rows. These rows do not fetch the mirror again. The manager check validates this source pin too; do not replace the frozen inputs. |
| Teacher, as named by the kit | `Qwen/Qwen3.6-27B` @ `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` | Full snapshot: 15 `model-*-of-00015.safetensors` shards, index, configuration, tokenizer, template and preprocessing files (exact pinned names/hashes below). M3b teacher coverage and M4 C targets. 4A uses the current collector, not new 27B targets. |
| Student, as named by the kit | `Qwen/Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218` | Full snapshot: five `model-*-of-00005.safetensors` shards, index/configuration/tokenizer files (below). M2 base identity, M3 base scoring/blends, M3b student tokenizer, M4 tokenizer/base, M5 base answers/target checks, M6/M7 base/standalone starts, and M10/M11 4A base/retention/tokenization. Trained checkpoints use their own original receipts. |

Exact cache downloads (models are intentionally complete snapshots for offline
inference; the manager checker never downloads model files):

```bash
huggingface-cli download Rowan/hellaswag --repo-type dataset --revision 218ec52e09a7e7462a5400043bb9a69a41d06b76 --include 'data/validation-*.parquet' --cache-dir "$HF_HOME/hub"
huggingface-cli download openai/gsm8k --repo-type dataset --revision 740312add88f781978c0658806c59bc2815b9866 --include 'main/train-*.parquet' --cache-dir "$HF_HOME/hub"
huggingface-cli download allenai/ai2_arc --repo-type dataset --revision 210d026faf9955653af8916fad021475a3f00453 --include 'ARC-Challenge/train-*.parquet' --cache-dir "$HF_HOME/hub"
huggingface-cli download EleutherAI/hendrycks_math --repo-type dataset --revision 21a5633873b6a120296cce3e2df9d5550074f4a3 --include '*/*.parquet' --cache-dir "$HF_HOME/hub"
huggingface-cli download Qwen/Qwen3.6-27B --revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 --cache-dir "$HF_HOME/hub"
huggingface-cli download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --cache-dir "$HF_HOME/hub"
```

The HellaSwag validation file must have SHA-256
`899813071e1e95efafec90f856e1987d2150fa4d020fc005df6962c259f660cd`.
The resulting frozen panel must have SHA-256
`2b0fc85c6e80854bb47982d35a18d15bc83a5655de6202634547071eccbe487d`.
The superseded `90242dd8` loader-only revision cannot satisfy staging; see the
22 October amendment in [registration.md](resources/next/registration.md).

Other external inputs (preserve original bytes and receipts):

- **ContractNLI**, non-HF original release from
  [Hitachi America / Stanford ContractNLI](https://stanfordnlp.github.io/contract-nli/),
  CC BY 4.0. M4 reads `CONTRACT_SOURCE/{train,dev,test}.json` (17 hypotheses;
  7,191/1,037/2,091 pairs). The kit pins bytes, not an HF repository/revision or
  immutable download URL. Obtain the original archive through that project or
  the partner's existing staged copy; archive SHA-256
  `e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b`.
  The exact split hashes are below. M6/M7 use the bundled frozen legal panels
  and M4's targets, not a different HF adaptation.
- **SDPO**, `lasgroup/SDPO` source tree at
  `7c457fc1b1f636ae794eb0362ba37d4743b06fbc`, checked by
  `git -C "$SDPO" rev-parse HEAD`; M6/M7 and 4A training require it and the
  existing trainer/inference environments. Preserve the working local checkout.
- **Prior Package A**, the frozen `3c/report-inputs/` tree, including
  `prepared/math_screen/heldout.jsonl` (100 rows) and manifests; M3/M5/M6/M7
  need those evaluation rows; Package A input manifest SHA-256
  `c75c3d5a2fcd140f434316fb768f6d590a056e51a3e410d56abb433ef9c64d37`
  is enforced by `p3a_common.frozen`. **Prior Package B**, completed unreset evidence,
  scores/reading, `targets/`, generic `replay/`, and original checkpoint export
  receipts; M2 binds/copies them, M3 scores/blends them, M5 re-verifies B's unique
  27B FinQA training targets, M6/M7 start from exact v4 D/F finals, and 4A binds
  W/replay. Preserve all 20 tier-1 FinQA finals and available tier-2 finals with
  their incoming identities. These are local experiment outputs, not HF pins.
- **FinQA**, original `czyssrs/FinQA` data in the preserved v4 dataset root,
  plus the frozen 1,147-row test exclusion panel and original manifests.
  M3/M5/M6/M7 and 4A diagnostic retention read it; hashes are bound by the
  existing v4/B manifests/receipts, rather than a new upstream revision.
- **SEC EDGAR**, non-HF public filings. M3 uses the already-shipped 193-question
  July–September slice in `resources/next/sec/`, manifest SHA-256
  `3e98419636a327c004464cb31b921db54acf78a06cc2e92f873c6a5e83b13782`;
  all shipped file hashes are in `resources/next/SHA256SUMS`. No live SEC fetch
  occurs. Optional M10/M11 require the separately delivered audited January–June
  4A slice, all 15 files in `resources/a4/slice.json`, manifest SHA-256
  `ed5a8b17357e3e9679f12b653e7a782dfab5af7ac9ddf14452183f26ae4cbcad`.
  Do not rebuild either slice. Source/audit/text bindings travel with it.
- **Bundled evaluation resources**, already part of this checkout:
  `panels/general-v1.jsonl`, `resources/c/contract_nli/`, and
  `resources/math-hard-teacher/` (including the frozen student responses).
  M3/M3b/M5/M6/M7/4A reuse these original rows. Their resource manifests,
  `freeze.json` and `SHA256SUMS` are authoritative; no new MMLU, IFEval, legal,
  or MATH download is needed for the offline rows.

Pinned external-file SHA-256 values follow. These are copied from
`resources/c/contract_nli/manifest.json`,
`resources/math-hard-teacher/data-manifest.json` and
`resources/p3b/model-hashes.json`; the checker reads those same sources.
Model hashes are enforced by the kit against staged local bytes; the manager's
metadata-only check confirms their filenames exist, without claiming byte
verification of models.

ContractNLI:

```text
310af7d661d2ab50ee3700169cef524c75f39fb296bbf5a515c229eb0f42e68e  dev.json
460267b56052a2dc5aead98eb35eadef9e6734d5723d37b4a9790e410f812387  test.json
dbceb356cd6203b35b27be94a5fa85e499a81c34c42c89ad53060b39f0257ba5  train.json
```

MATH mirror:

```text
0e04a850dac9cfc758eac0c80afd2d0cc65e29ea72f916c8ac1045b12af937f5  algebra/test-00000-of-00001.parquet
73347f5b55869f21ff094361025dbd12d291e971aa4d72142eacee10438e449f  algebra/train-00000-of-00001.parquet
c324818949c75a6227c0328a8499d58023cbbf6757cfc1736f5ab24cc8d87966  counting_and_probability/test-00000-of-00001.parquet
d37ae017930df41678edee50dcde18c17fb06afaff9eb505c1b95f1002bfcb31  counting_and_probability/train-00000-of-00001.parquet
98720f3a8134ad9784fd8951195245264495df03ffc5c9485a9a54ff5ae2431a  geometry/test-00000-of-00001.parquet
9dbf5e5b55499e696e0ba72da9d85d4f0be3acbf7eaa6cb014b0fc78bb63784f  geometry/train-00000-of-00001.parquet
e859f72486c951c2017da880541ca729940eaa986b0364eb23e238983df9bfeb  intermediate_algebra/test-00000-of-00001.parquet
0c87f3798ebdfb915b1f9b6fc13c1a8d064c3cb91631ccbe85b83ed5d71150f6  intermediate_algebra/train-00000-of-00001.parquet
ba156c9c9fa2040836ba12135fd964e6e67d6bccd8c469ec4a3ad941f8f6fbe9  number_theory/test-00000-of-00001.parquet
81c08649149b39811f2d1d7ddf5dfccf6c4c8044a2e169cfb24db09160c941f4  number_theory/train-00000-of-00001.parquet
301e5d8ade5c7b5038032532e498bdc9dca799e7f1061a63f2e7d980803240cc  prealgebra/test-00000-of-00001.parquet
765b526fb5c0b8385cb7be2c16bf4f6f28cf71f2bc15c826abfbb5fa7f8616cc  prealgebra/train-00000-of-00001.parquet
da37474ba3761ae5727af3844a7ba963a735bd6a14c65e6137593669a4fe3b9c  precalculus/test-00000-of-00001.parquet
28bb1ff95525c8d32521509bef91de76339659f5467a381e6e8deea204538f0e  precalculus/train-00000-of-00001.parquet
```

Qwen3.6-27B:

```text
e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259  chat_template.jinja
69db4eb7196bc8190813231b3018ca05d8c2e3abc7b1af19d55c157af44a9d9c  config.json
2d4464e2ead06bc9bc718c781309ad1e7baded626d66e8dcdc8b469ba185faf0  configuration.json
e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e  generation_config.json
5f21d4e349aef6c74bedef7b3835dc8c11a16dd5ce72f4437e2284f1e83736e9  model-00001-of-00015.safetensors
03de44dc7e933025498d72f8d7ea32d5cde16eb7a85b35ff1fa49a737f4b2242  model-00002-of-00015.safetensors
5c3a68304dabeaa5a0eb70a5e383d2b8134997ef6d5c1e8afdcd236091c847f2  model-00003-of-00015.safetensors
ba8b0849cb4c4c97e674709bfe56a9c008d7563a68ac57120ceb2855a73a9944  model-00004-of-00015.safetensors
a5abc1d5e9583409193e2cd58a671a4ed98467a6eaf883cd4377e3cc02021c22  model-00005-of-00015.safetensors
160d914e2e4704a401a3eab9e9eec3380e5723a35981c9169bedfb82ea32e6d6  model-00006-of-00015.safetensors
0bcd0ce28c7d2cc6f5ab2c21902cd032fec0cc5f29e0897e3bb0967add388e47  model-00007-of-00015.safetensors
584a0ed8018d3b19ce8e533d08a460454650027901be6323b50b591da143dca6  model-00008-of-00015.safetensors
e7e3e1a17a2673340eb6bba95c3c6f07b12e7746ee738dde56551cbde27656f5  model-00009-of-00015.safetensors
e8934789f4742c11da88f937da84bdbb8764c8a7f7ebe9e7b056bcbd41b14285  model-00010-of-00015.safetensors
44e8fe06d2d609bf20b16b1d2f42348ca2dc99179721f76035ff50db678f9a8a  model-00011-of-00015.safetensors
33c5d7d18e1b3f661334dff736fc7d00561f1f1bb9d2970c669b40d2e582974e  model-00012-of-00015.safetensors
68db2ebb03231238c2a114f3642df967a970a1d96b029adc427838cb5cf7f27a  model-00013-of-00015.safetensors
26c114fb6d5d4131ab227552d489d7a97c8c958a9ed94f6d3d1584f6cfe4b9fb  model-00014-of-00015.safetensors
b84b5b1315e865c9a19a444045d422a73e3e2e31ce3766797cffd3507c68c9c9  model-00015-of-00015.safetensors
a8ad2c26fb707ff8c245806315b03e3b4b74595528492423af5dae0ce39b4d9b  model.safetensors.index.json
27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516  preprocessor_config.json
5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42  tokenizer.json
5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b  tokenizer_config.json
7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13  video_preprocessor_config.json
ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003  vocab.json
```

Qwen3-8B:

```text
f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30  config.json
2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2  generation_config.json
31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f  model-00001-of-00005.safetensors
5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282  model-00002-of-00005.safetensors
c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836  model-00003-of-00005.safetensors
b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a  model-00004-of-00005.safetensors
20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff  model-00005-of-00005.safetensors
f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc  model.safetensors.index.json
aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4  tokenizer.json
d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101  tokenizer_config.json
ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910  vocab.json
```

**Before a tag:** from the manager checkout, with network enabled, run:

```bash
python3 scripts/check_pinned_inputs.py --contract-source "$CONTRACT_SOURCE" --receipt /absolute/receipts/pinned-inputs.json
```

Require exit 0 and an `ok: true` row for all seven inputs. This resolves every
HF revision, checks file listings, downloads only the required small dataset
parquets, verifies pinned hashes, and runs the kit's own row loaders/count/ID
checks. ContractNLI is checked from the supplied local originals. Models receive
revision/listing checks only. Keep the receipt with the export/tag review;
stand-ins alone do not satisfy this gate. The checker is manager-side and is
not shipped inside the public `kit/` export. Return to
`HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1` for GPU jobs.

Use [README-next.md](README-next.md) for v0+/v3 and frozen pools, and
[README-c.md](README-c.md) for C plus its probe. Every script is resumable:
identical completed rows print SKIP, identity mismatches stop, partial training
preserves evidence and restarts the same registered seed from step 0. Never run
two copies of a row or overlap full-node training. Exit 2 means correctness stop,
3 means prerequisite pending; weak scores do not stop the core.

| Row | Ordered work | Concrete commands |
|---|---|---|
| M1 | Restore durable OUT; identify scoring host/GPU | Job boundary block below |
| M2 | Bind W/inputs and freeze HellaSwag before any new output | README-next N1–N3 |
| M3 | v0+ and v3 screen scoring, serial on one GPU | README-next N4–N5, N10 |
| M3b | Quick job: hard-MATH teacher coverage, two GPUs in a free gap | Quick-jobs block below |
| M4 | Freeze C membership; 27B targets, two GPUs in a free gap | README-c §§2–3 |
| M5 | Broad/earlier pools; checked base answers | README-next N6 |
| M6 | C base/incoming controls, then core full-node stages serially | README-c §4 |
| M7 | C probe reference + broad + earlier + both, full-node serial | README-c §7 |
| M8 | Read complete C/probe and screen tables | README-c §5/§7; README-next N10 |
| M9 | Durable save/verify; collect return 1 immediately | Return block below |
| M10 | Explicit optional 4A gate, only a present audited 4A kit | Gate block below |
| M11 | Return 2 after 4A, durable save/verify | Gate block below |
| M12 | Job end save/verify; next job repeats M1 and unfinished rows | Job boundary block |
| M13 | Manager verifies and reads compact returns | README-next N12; README-c §5 plus c_probe_read |

Bind paths from both supplements, then use one parent tree. Keep the same
absolute OUT/NEXT/C/NVME paths across jobs because receipts contain paths.
Inputs/models/v4 exports/B stay at their original durable paths. OUT contains
all newly produced scientific evidence, exports, journals and receipts.

```bash
export CHECKOUT=/absolute/reviewed-checkout
export KIT="$CHECKOUT/kit"
export OUT=/local/nvme/message
export NEXT="$OUT/NEXT"
export C_OUT="$OUT/C"
export NVME=/local/nvme/blends
export DURABLE=/fsx_efa/next-message
export TRAINER_PY=/venvs/trainer/bin/python
export INFERENCE_PY=/venvs/inference/bin/python
export SCORE_GPU=0
export RETURN="$OUT/next-return-1.tar.gz"
export RUN_4A=0
export HF_HOME=/absolute/durable/huggingface   # the same cache used for staging below
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

When executing README-c commands, temporarily set `OUT="$C_OUT"`, then restore
the parent OUT for handoff/return commands. The job-boundary block captures
read-only `PARENT_OUT` and `DURABLE_ROOT` once; C commands must not change them. Set `V4_FINQA_PANEL="$FINQA"` for
c_score/c_targets; it is the validated 1,147-item FinQA dataset, while N6 uses
`FINQA_PANEL`, the exclusion JSONL. Keep the supplements' remaining pinned paths.
Execute C §§2–3 in M4, §4 in M6, §7 in M7, then §5 in M8. N3 must precede M3;
M4 may use two GPUs only during a gap with no full-node training and sufficient
host bandwidth. Scoring stays serial; no allocation overlap is assumed below.

Quick-jobs block before C (M3b). **Hard-MATH teacher coverage**: the pinned
Qwen3.6-27B answers the frozen 200-item Level-5 MATH screen panel with the 3C
teacher protocol (four registered attempts, thinking off, the 3C verifier) and
the frozen rule decides whether this panel can be the connected continuation's
third task: teacher accuracy at least 160/200 **and** at least 30 of the frozen
8B's 62 failures solved by the teacher. Inference venv (vLLM 0.18.0), **two
GPUs** (TP2), no other input than the kit's own frozen resources and the two
pinned snapshots from README-c (`TEACHER`, `BASE`; `BASE` supplies the 8B
tokenizer for 3C's target limits). It can run in any gap with no full-node
training, beside one-GPU scoring on other GPUs; nothing else reads its output,
so C never waits for it. Run it before C if a gap exists, and in any case before
return 1, so its directory travels with the return.

```bash
export TEACHER_GPUS=6,7   # two GPUs not used by any running job
run_stage "$INFERENCE_PY" "$KIT/math_hard_teacher.py" --out "$OUT" --model "$TEACHER" --tokenizer "$BASE" --gpu "$TEACHER_GPUS" || exit $?
run_stage "$TRAINER_PY" "$KIT/math_hard_teacher_read.py" --source "$OUT/math-hard-teacher" || exit $?
run_stage tar -czf "$OUT/math-hard-teacher-return.tar.gz" -C "$OUT" math-hard-teacher
```

Exit 0 prints QUALIFIES or DOES_NOT_QUALIFY; either is a finished job. Exit 3
(incomplete journal) or an interrupted allocation: rerun the same command, which
resumes the journalled attempts and never resamples them; a finished run prints
SKIP. Exit 2 means a frozen input, model or setting differs: stop and report.
Estimated cost from the 3C MATH teacher step's measured 1,548 s on two GPUs:
**about 1.3 h wall, 2.6 GPU-h; allow 2 h wall, 4 GPU-h**. Rehearsal on CPU:
`python "$KIT/math_hard_teacher.py" --out /tmp/mht-standin --standin`.

Job boundary commands (every job and requeue). Make `job-start` the first
executable job-script command, with the path/venv variables above already set:

```bash
"$TRAINER_PY" "$KIT/message_handoff.py" --out "$OUT" --durable "$DURABLE" --action job-start || exit $?
readonly PARENT_OUT="$OUT" DURABLE_ROOT="$DURABLE"
"$TRAINER_PY" "$KIT/message_handoff.py" --out "$PARENT_OUT" --durable "$DURABLE_ROOT" --action restore --move-aside || exit $?
"$TRAINER_PY" "$KIT/message_handoff.py" --out "$PARENT_OUT" --durable "$DURABLE_ROOT" --action identity --gpu "$SCORE_GPU" || exit $?
# Preserve OUT on exits; forward TERM promptly to the running stage.
export TERM_LEAD_SECONDS=1800 SAVE_ALLOWANCE_SECONDS=900
source "$KIT/message_job.sh"
```

At each job end, also run the save command explicitly and check exit status.
The helper locks the durable root with `flock` for save/restore. A second
writer stops with a clear busy message. It compares a size + nanosecond mtime
manifest to CURRENT, reuses cached hashes for unchanged files and skips an
unchanged OUT. New generations use `rsync -a --link-dest=<CURRENT tree>` for
unchanged hard links; only changed/new files are copied and hashed at source and
destination. The receipt reports `bytes_copied` from rsync and `changed_bytes`.
Same-size changes within one second are copied explicitly. Stop all OUT writers
before saving; a size/mtime change during the copy refuses publication. This
assumes completed checkpoints are immutable: deliberately preserving both size
and mtime on a content edit defeats the quick check.

Publication renames the verified generation and atomically replaces CURRENT
using unique pid/random writer temps. CURRENT alone is authoritative, including
if a dead writer left `CURRENT.json.tmp`. After publication, only CURRENT and its
previous committed generation remain; older/unpublished generations and stale
`.partial-*` are removed. Failed saves leave CURRENT untouched. A hard kill
cannot run a trap; the next restore uses the last committed generation.

Restore fully verifies CURRENT. It never deletes local evidence: with
`--move-aside`, differing/local-only entries are renamed into the sibling
`OUT.pre-restore-<UTC>-<unique>` tree before copying, and the receipt lists those
paths. Newer/local-only files print a prominent warning. Without that flag,
a differing local OUT is refused. Inspect preserved newer stages before resuming.
An old in-place durable layout still requires explicit inspection/migration.

Run **every long stage** through the background-child wrapper, including commands
from the supplements and optional command files:

```bash
# Set REGISTERED_C_ARGS to the exact argument array from README-c.
run_stage "$TRAINER_PY" "$KIT/c_train.py" "${REGISTERED_C_ARGS[@]}" || exit $?
# For an existing payload script: run_stage bash -e partner-stages.sh || exit $?
```

The wrapper uses `cmd & child=$!; wait "$child"`, so TERM interrupts the wait
immediately. Each stage gets its own process group. It forwards TERM to that
group (including trainer/inference workers), waits at most `TERM_LEAD_SECONDS -
SAVE_ALLOWANCE_SECONDS`, then kills an unresponsive child and saves on EXIT.
Never detach workers from that process group or launch separate writers into OUT. Save between stages to bound
changed checkpoint volume. Measure incremental copy + changed-file hash time
on the actual durable filesystem and set the allowance above its worst case
plus metadata/cleanup margin. The 1800 s example reserves 900 s for saving and
900 s for child shutdown; it is sufficient only if measured work fits those
budgets. The first save copies the entire starting OUT: do it before long work,
outside the TERM window. A 3 TB initial copy cannot rely on a 30-minute lead.

```bash
# Partner supplies allocation/account/partition and existing payload scripts.
# Request --nodelist=<host> and --dependency=afterany:<prev> for both jobs.
HOST=the-current-host
B_JOB_ID=the-current-B-job-id
J1=$(sbatch --parsable --time=7-00:00:00 --nodelist="$HOST" --dependency="afterany:$B_JOB_ID" --signal=B:TERM@1800 partner-message-job1.sh)
J2=$(sbatch --parsable --time=7-00:00:00 --nodelist="$HOST" --dependency="afterany:${J1}" --signal=B:TERM@1800 partner-message-job2.sh)
```

These are partner submission examples, not commands this kit executes. Job 2
runs the same ordered rows; completed scoring/training rows SKIP. `afterany`
also starts after a failed job, so inspect the saved failure/status note first.
If the scoring **machine, environment or physical GPU UUID differs**, the helper
moves old scores/SEC/blends/aliases into `OUT/scoring-history/<epoch>/` outside
active readers. Re-run M2–M3 (base and λ=0/1 references first), then C/probe
base and incoming/reference scores before any comparison; re-score every compared
checkpoint on this GPU. Finished training and frozen pools/targets remain reusable.
This full refresh avoids mixing epochs; receipts record UUID/host/environment,
and the old raw evidence stays durable. CUDA ordinal alone is insufficient.

Return 1 after C **and** probe, before 4A:

```bash
run_stage "$TRAINER_PY" "$KIT/next_read.py" --out "$NEXT/reading" --source "$NEXT" || exit $?
run_stage "$TRAINER_PY" "$KIT/c_read.py" --out "$C_OUT/reading" --source "$C_OUT" || exit $?
run_stage "$TRAINER_PY" "$KIT/c_probe_read.py" --out "$C_OUT/reading" --source "$C_OUT" || exit $?
"$TRAINER_PY" "$KIT/message_handoff.py" --out "$PARENT_OUT" --durable "$DURABLE_ROOT" --action save || exit $?
COMMITTED_OUT=$("$TRAINER_PY" "$KIT/message_handoff.py" --out "$PARENT_OUT" --durable "$DURABLE_ROOT" --action path) || exit $?
run_stage "$TRAINER_PY" "$KIT/next_collect.py" --out "$NEXT" --durable-root "$COMMITTED_OUT/NEXT" --archive "$RETURN" || exit $?
run_stage "$TRAINER_PY" "$KIT/c_collect.py" --out "$C_OUT" --archive "$PARENT_OUT/package-c-return.tar.gz" || exit $?
"$TRAINER_PY" "$KIT/message_handoff.py" --out "$PARENT_OUT" --durable "$DURABLE_ROOT" --action save || exit $?
# RETURN 1: both compact archives plus kit/env/GPU receipts, job IDs,
# failed/restarted attempts, wall/GPU times, durable path and unfinished inputs.
```

Optional 4A is now present: [README-4a.md](README-4a.md), plan-v6 rung v2,
with its audited primary-slice pins, collector, all arms, reader and compact
return. Historical k4a is unrelated. Complete return 1 first. Set the A2 input,
and measured-timing variables from README-4a before opening
this gate. The payload runs A2–A11 inside the existing process-group wrapper;
monthly durable saves and unchanged-row SKIPs support both chained jobs.

```bash
if [ "$RUN_4A" = 1 ]; then
  test -f "$KIT/README-4a.md" && test -s "$KIT/a4_payload.sh" || exit 3
  export B_REPLAY="${B_REPLAY:-$B/replay}"
  export PARENT_OUT DURABLE_ROOT KIT BASE FINQA SDPO B_REPLAY B_SOURCE
  export TRAINER_PY INFERENCE_PY SCORE_GPU FOUR_A_SLICE FOUR_A_TIMING
  export FOUR_A_STAGE_GPU_HOURS
  # Payload fsyncs this run's marker; continuation charging starts at job-start.
  # Optional bound limits: FOUR_A_ALLOCATION_GPUS, FOUR_A_SCORE_MAX_SECONDS,
  # FOUR_A_NLL_MAX_SECONDS, FOUR_A_HANDOFF_SECONDS (README-4a).
  run_stage bash -e "$KIT/a4_payload.sh" || exit $?
  "$TRAINER_PY" "$KIT/a4_handoff.py" --out "$PARENT_OUT/A4" --parent-out "$PARENT_OUT" \
    --durable "$DURABLE_ROOT" --trainer-python "$TRAINER_PY" || exit $?
  # RETURN 2: 4A kit compact archive, findings and measured costs. Never delay return 1.
fi
```

Operational totals from measured B stage timings: F/F+replay **20–26 minutes**
(2.67–3.47 GPU-h), D/D+replay **58 minutes** (7.73 GPU-h; D uses the replay proxy).
Answer-only is charged a full F slot. Sharing removes only the five probe generic
stages when W is already its replay sibling. No B finals are reused as legal runs.

| W | Core + new probe stages | Training GPU-h | Training wall h | Combined through return 1 GPU-h | Serial wall h |
|---|---:|---:|---:|---:|---:|
| F | 20 + 20 = 40 | 106.7–138.7 | 13.3–17.3 | 265.7–323.7 | 70.8–98.8 |
| F+replay | 30 + 15 = 45 | 120–156 | 15–19.5 | 279–341 | 72.5–101 |
| D | 30 + 20 = 50 | 285.3–301.3 | 35.7–37.7 | 444.3–486.3 | 93.2–119.2 |
| D+replay | 30 + 15 = 45 | 246.7–262.7 | 30.8–32.8 | 405.7–447.7 | 88.3–114.3 |

The M3b quick job adds 2.6–4 GPU-h (at most 2 wall h on two GPUs), not in the
table or totals below.
Combined adds once: 45 GPU-h for 27B targets on two GPUs (22.5 wall h allowance),
24 for C scoring/startup, 6 v0+, 10–20 v3 interiors, 6 fresh controls, 68–84
probe pools/scoring/startup: **159–185 GPU-h overhead**. Serial wall adds target
22.5 h, one-GPU scoring/preparation 24–48 h and 11 h startup/IO margin:
**57.5–81.5 h**. These are operational estimates based on measured B training,
not measured ContractNLI or per-panel timings. General legal inputs may be slower.
No allocation-equivalent GPU total is presented as elapsed time. Measure each
process, include failed attempts and idle reservation, and reprice at send.

Through return 1: **265.7–486.3 GPU-h, 70.8–119.2 serial wall hours**, depending
on W. At send, 4A binds the owner's 21 October approval:
**at least 600 partner GPU-h for F/F+replay; 800 for D/D+replay**, with the
owner's approval reference recorded in the binding. A higher explicitly approved
value is accepted. The registered formula
is a reported forecast only (350–400 for the F anchor, about 600 for the
50-minute D+replay anchor, 700 for the slower 58-minute D anchor).
The stop and worst-case preflight both use `approved_ceiling_gpu_hours`.
At these minimum approvals, combined with the matching return-1 W totals, the budget envelope through
return 2 is **865.7–1,286.3 GPU-h**, plus M3b's 2.6–4.

README-4a covers registered arrivals, permanent holdouts and general-v1:
**688 score cells, 41,575 uncached predictions**, removing 104,377 FinQA
predictions. At 1.22 s/question that is **14.1 scoring wall hours**; with eight
GPUs held, scoring costs **112.7 allocation GPU-h**. Including training and
the 80 GPU-h overhead allowance, the 4A operational forecast is
**314.2–494.2 GPU-h, 39.3–61.8 wall hours** at the 2.7–6.7 GPU-h stage anchors.
Combined through return 2: **579.9–980.5 GPU-h, 110.1–181.0 wall hours**, before
NLL, slower contexts, training contingency and M3b. These are forecasts.

The first job-script command durably records allocation start in
`DURABLE/job-starts/<job-id>-r<restart-count>.json`, before restore and identity
checks. At entry into 4A the payload fsyncs a marker keyed by Slurm job ID and
restart count (default 0), then saves it before binding. The first job that ever
starts 4A charges **8 × time from max(allocation start, marker) to job end**;
return-1 work before that marker costs 4A zero. A continuation whose restored
tree holds a 4A marker from any earlier run (an earlier job, or an earlier run
of the same requeued job) charges from **its own allocation start**:
restore, identity checks, SKIPs and re-scoring after a GPU change all serve 4A.
Use `SLURM_JOB_START_TIME`, else the durable first script record.

Allocation and attempt receipts distinguish requeued runs of the same job ID.
A requeue writes a fresh marker. It charges from its allocation start when any
earlier run already started 4A; otherwise from max(allocation start, its marker).
Its earlier run charges through its recorded end, or provisionally through the
later of its last 4A record and each unfinished attempt's start plus duration
limit, capped at the next run's allocation start. An actual end can replace
that estimate; **queue gaps cost zero**.
For an earlier job with unknown end, local `scontrol`, or `sacct` if `scontrol`
has no record, must report a terminal state and actual end, else the run is
pending (exit 3) before any new attempt.
RUNNING and COMPLETING are non-terminal; scheduled EndTime is never an end.
If scheduler history is unavailable, supply the actual end and restart count
with `a4_allocation.py` as in README-4a. Never charge through the next job's start.
Duration limits and worst-case preflight
enforce the approved ceiling. The recurrence split remains EXPLORATORY under
the verbatim 21 Oct registration deviation. C's registered ceiling remains 600;
no seeds, endpoints or attempts change.

Full offline rehearsal (fresh OUT, CPU fixtures only):

```bash
python "$KIT/message_dryrun.py" --out /tmp/message-standin --standin
```

The existing combined rehearsal retains its deliberately missing-kit gate
fixture and A/B/C/next paths unchanged. Rehearse the present M10/M11 payload too:

```bash
python "$KIT/a4_dryrun.py" --out /tmp/a4-message-D-standin --standin --recipe D+replay
python "$KIT/a4_dryrun.py" --out /tmp/a4-message-F-standin --standin --recipe F+replay
```

That runs every 4A row, 80 h of preceding return-1 work in the same job (excluded
from 4A), failure and cross-job resume paths, and compact archive
readback. Nothing sends partner messages.
