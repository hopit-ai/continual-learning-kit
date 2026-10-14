# Partner package A: plain jobs

Run the frozen 3A Chemistry scores and ordered 3C screen. No training.
Every command is an ordinary program; write your own sbatch files and use your
existing host venvs. Nothing reads a FinQA WORK. Keep one absolute OUT for the
entire run. Models and caches live outside OUT; no model download is performed.

```bash
export KIT=/absolute/path/to/checkout/kit
export OUT=/absolute/path/to/p3a-results
export TRAINER_PY=/absolute/path/to/trainer-venv/bin/python
export INFERENCE_PY=/absolute/path/to/inference-venv/bin/python
export BASE=/absolute/path/to/Qwen3-8B-snapshot
export TEACHER=/absolute/path/to/Qwen3.6-27B-snapshot
export R1=/absolute/path/to/chemistry-r1/hf-step40
export R2=/absolute/path/to/chemistry-r2/hf-step40
export HF_HUB_CACHE=/absolute/path/to/shared/hf/hub
```

BASE must be Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`.
TEACHER must be Qwen3.6-27B revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
Use the inference venv with vLLM 0.18.0 for the teacher, TP2/eager, and the
trainer venv for frozen-8B rewrites, TP1 in a fresh process. The rewrite script
sets `V4_PHASE0_INFERENCE=0`. Teacher RPC/NCCL/Gloo timeouts remain the frozen
3,000/6,000/18,000 seconds. BASE also supplies the teacher's 8B target tokenizer.

1. Run the CPU inputs job on a compute node with network, huggingface_hub and
   pyarrow installed. It downloads only test5/test6 from
   `livecodebench/code_generation_lite@eef204666b9879915189d71010db8aab4e766ed5`
   and fourteen MATH parquets from
   `EleutherAI/hendrycks_math@21a5633873b6a120296cce3e2df9d5550074f4a3`, checks
   source SHA-256s, materialises MATH and runs the unchanged joint freezer.
   It also copies/checks the four frozen 3A panels. No GPU is needed.

```bash
"$TRAINER_PY" "$KIT/p3a_inputs.py" --out "$OUT" --cache "$HF_HUB_CACHE"
```

The rebuilt annex must equal `c75c3d5a2fcd140f434316fb768f6d590a056e51a3e410d56abb433ef9c64d37`.
The real LCB step-1 PASS ships in `resources/3c-screen/code-step1.json`.
For offline preparation, `--sources DIR` reads only `DIR/code/test{5,6}.jsonl`
and `DIR/math_screen/<subject>/{train,test}-00000-of-00001.parquet`.
Retain dataset licences with your cached downloads. Wait for “Joint freeze
checked” (or “SKIP: complete joint freeze”) before running GPU jobs.

2. Run three 3A jobs in parallel: base on GPU 0, r1 on GPU 1, r2 on GPU 2.
   Each job scores original, letters, balanced and rank_preserved sequentially.
   Twelve scorings total; estimate ~8 hours per lane.

```bash
"$TRAINER_PY" "$KIT/p3a_3a.py" --out "$OUT" --lane base --model "$BASE" --gpu 0 & p0=$!
"$TRAINER_PY" "$KIT/p3a_3a.py" --out "$OUT" --lane r1 --model "$R1" --gpu 1 & p1=$!
"$TRAINER_PY" "$KIT/p3a_3a.py" --out "$OUT" --lane r2 --model "$R2" --gpu 2 & p2=$!
status=0; for p in $p0 $p1 $p2; do wait $p || status=1; done; exit $status
```

3. Alongside 3A, run the LCB teacher on GPUs 3,4. When it completes, run rewrites
   on GPU 3 in a separate process, then the CPU decision. Estimate ~2.7 hours
   for teacher plus rewrites per candidate; local throughput is unmeasured.

```bash
"$INFERENCE_PY" "$KIT/p3a_teacher.py" --out "$OUT" --candidate code --model "$TEACHER" --tokenizer "$BASE" --gpu 3,4
"$TRAINER_PY" "$KIT/p3a_rewrite.py" --out "$OUT" --candidate code --model "$BASE" --tokenizer "$BASE" --gpu 3
"$TRAINER_PY" "$KIT/p3a_decide.py" --out "$OUT"
```

4. Read `OUT/3c/report-screen/code/decision.json`. Only **complete FAIL** permits
   MATH. PASS ends the candidate order; INCOMPLETE requires local recovery.
   Run MATH headroom with frozen BASE on GPU 3 (~300–600 seconds estimated).
   Only its complete PASS permits the MATH teacher/rewrite jobs.

```bash
"$TRAINER_PY" "$KIT/p3a_baseline.py" --out "$OUT" --model "$BASE" --gpu 3
# Inspect OUT/3c/report-inputs/math_screen-step1.json; continue only on complete PASS:
"$INFERENCE_PY" "$KIT/p3a_teacher.py" --out "$OUT" --candidate math_screen --model "$TEACHER" --tokenizer "$BASE" --gpu 3,4
"$TRAINER_PY" "$KIT/p3a_rewrite.py" --out "$OUT" --candidate math_screen --model "$BASE" --tokenizer "$BASE" --gpu 3
"$TRAINER_PY" "$KIT/p3a_decide.py" --out "$OUT"
```

One short sbatch example, as comments to copy/adapt (the site supplies resources):

```bash
# #!/bin/bash
# #SBATCH --account=<ACCOUNT>
# #SBATCH --partition=<PARTITION>
# #SBATCH --gres=gpu:<COUNT>
# #SBATCH --time=<SITE_TIME>
# #SBATCH --output=<SLURM_LOG>
# /absolute/venv/bin/python /absolute/kit/p3a_3a.py --out /absolute/OUT --lane base --model /absolute/BASE --gpu <VISIBLE_GPU_ID>
```

After a crash, fix infrastructure and rerun the **same command with the same OUT**.
Completed 3A panels and baseline scores are skipped. Interrupted scoring folders
are retained under `partials/`. 3C completed requests replay from the durable
raw journal with no resampling; completed phases are skipped. Do not run two
copies of the same lane/phase against one OUT. GPU IDs refer to the devices
visible inside your allocation. There is no automatic submission or continuation.
Every job appends stdout/stderr to `OUT/logs/`. Exit codes: 0 done (or SKIP when
the registered order rules the step out); 2 correctness stop (wrong input hash,
wrong annex, missing model path); 3 NOT READY (run the earlier step first); any
other crash exits non-zero with its traceback (a frozen-journal identity
mismatch shows as a `ValueError` traceback: treat it as a correctness stop). Warnings (e.g. model provenance)
never change the exit code.

Do not change the panels, sample/order, prompts, decoding, verifier, teacher
settings, first-verified selection, seeds, attempt count, token limits or
scientific decision rules. Do not hand-edit outputs/verdicts/reports or delete
evidence. If a scientific change is unavoidable, preserve the original evidence
and flag the exact departure for the owner; it is outside this registered run.

After all jobs finish, add `OUT/local/note.md`, your job IDs, source diff and any
local infrastructure helpers. Recompute decisions, then collect on CPU:

```bash
"$TRAINER_PY" "$KIT/p3a_decide.py" --out "$OUT"
"$TRAINER_PY" "$KIT/p3a_collect.py" --out "$OUT" --archive /absolute/path/p3a-return.tar.gz
```

Send back the archive and a short note naming fixes, warnings, missing work and
model provenance. Collection excludes weights, caches, source downloads, build
scratch and full private tests; frozen test hashes and verifier evidence return.
The reader stays on our side: `python3 kit/p3a_read.py --archive RETURN.tar.gz
--out READING` produces `3a/table.md`, both `3c/*/admission-table.md`, step-1/
step-2 decisions and combined `reading.json`. It also accepts an OUT directory.
No kit snapshot is needed. Scientific registrations remain byte-identical in
`resources/3a-shortcut/` and `resources/3c-screen/`; their historical Modal cost
text does not prescribe this partner's job scheduling.
