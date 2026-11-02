# Hard-MATH screen — amendment 1 (teacher coverage), 20 October 2026

Amends the frozen [registration](hard-math-screen-preregistration-20261020.md) (file sha256 `91bf4f3d…d7d4`, unchanged).
**Written on 20 October 2026, before any teacher output existed on these items.** The student result it refers to is in
[the result record](hard-math-screen-result-20261020.md). The text below is unchanged from when it was first written
(it was first appended to the registration file and moved here so the registration stays byte-identical). Kit copies:
`kit/resources/math-hard-teacher/registration.md` (the frozen registration) and
`kit/resources/math-hard-teacher/amendment-1.md` (this file), both in that directory's `SHA256SUMS`.

## Amendment 1 (20 October 2026): teacher coverage, frozen before any teacher output

Written before the pinned teacher has produced any output on these items. Data, the student measurement and its
decision above are unchanged. Do not change these rules after outputs.

**Job.** Pinned `Qwen/Qwen3.6-27B`, revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, answers the frozen
200-item Level-5 screen panel (`data/math_hard/screen_panel.jsonl`, file sha256 `12ca425e…3ceb`, row digest
`7235a1bd…39b6`; data manifest file sha256 `ee2417d5b1b772e994b80b70ea8866b5f49e79e70d241559e96145c55e1a0b8f`)
on the partner's node, through `kit/math_hard_teacher.py`. The 100-item evaluation set and the 500-item training
pool are not read by this job.

**Protocol: the 3C step-2 teacher step, unchanged** ([3C registration](3c-second-task-screen-preregistration.md),
`kit/p3a_jobs.py` teacher phase), reused by import, not copied: `kit.v4_teacher._run_pool` (durable raw journal,
whole round persisted before verification, first verified attempt chosen, no adaptive retries) with
`kit.v4_teacher.VLLMGenerator` in the inference environment (vLLM 0.18.0, tensor parallel 2 on two GPUs, bf16,
eager, batch-invariant, Flash Attention, Triton GDN prefill, no batch-invariance fallback). Seeds
27101/27102/27103/27104, temperatures 0/.3/.6/.8, top-p 1, at most four attempts, 2,560 new teacher tokens, teacher
input cap 4,096, thinking disabled. Prompt: the 3C `math_screen` prompt (the panel row's `prompt`) as the only user
message, exactly the 8B's unaided prompt; no gold, reference solution or student output enters generation. Verifier:
`kit.second_task_screen.verify` for `math_screen`: finished with stop/EOS, no thinking tags, working before the last
balanced `\boxed{}`, that box equal to the gold as an exact rational under `kit/beds/math_screen.py` (sha256
`3869623f…da95`, the file that scored the 8B), and the target within 3C's frozen-8B limits (at most 2,048 tokens
including EOS; input at most 4,096). There is no rewrite step here; rewrites belong to the later registration.

**Per-item outcome.** *Solved by the teacher* = a verified demonstration within the four registered attempts. Four
unverified attempts are a completed failure for that item. A missing attempt, duplicate or foreign record, or a
demonstration whose box the frozen verifier rejects on re-reading is **INCOMPLETE**; an interrupted job resumes the
same journalled requests and never resamples a completed one.

**Student failures.** The 62 items scored wrong for the frozen 8B in
[evidence/math-hard-screen/student-8b/](evidence/math-hard-screen/student-8b/) (`bed-score.json`, file sha256
`c4a969c0…f010`, `per_item` false; responses `responses.jsonl`, sha256 `1e49617e…a4f`), re-derived by the job with the
frozen verifier before generation.

**Decision rule (frozen).**

- *Teacher accuracy* = solved / 200. *Teachable gap* = items wrong for the frozen 8B **and** solved by the teacher.
- **QUALIFIES** as the connected continuation's third task if teacher accuracy **≥ 80% (at least 160 of 200)** AND the
  teachable gap is **at least 30 of the 62** student failures, decided on integer counts.
- Otherwise **DOES_NOT_QUALIFY**, recorded as such. A hardest-types variant may then be considered only as a separate,
  declared decision on this screen panel; the 100-item evaluation set and the 500-item training pool stay untouched.
- Incomplete evidence is **INCOMPLETE**, neither outcome.
- Descriptive only (none rescues or vetoes the decision): greedy first-attempt accuracy, the Wilson 95% interval,
  answers correct but rejected on another check (by reason), per-type rows (n, student correct, teacher solved, gap),
  items the student solved and the teacher did not, finish reasons, teacher tokens and wall time.
- A pass establishes target availability on this panel only; registration of the third task still needs its own
  approved registration and budget. Learnability, retention and acquisition are not tested here.

**Cost.** The 3C MATH teacher step measured **1,548 s wall on two GPUs** on the partner's node (engine start included;
partner log, 17 October), 0.86 GPU-h for 100 Level 1–5 items. Scaled to 200 Level-5 items (twice the items, up to
1.5 times for longer answers and more retry rounds): **about 1.3 h wall, 2.6 GPU-h; allowance 2 h wall, 4 GPU-h**, on
two GPUs in a gap with no full-node training. An estimate, not a measurement; the job is resumable, so a cut
allocation loses no journalled attempt. No paid API or Modal run.

**Freeze.** This registration is copied byte for byte to `kit/resources/math-hard-teacher/registration.md` with the
panel, the data manifest and the student evidence, all listed in that directory's `SHA256SUMS`; the job refuses
any difference before generating.
