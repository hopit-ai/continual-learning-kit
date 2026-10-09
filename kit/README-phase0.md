> Owner-approved 9 October: phase 0 uses one 08:00:00 reservation on eight GPUs, with a 64 GPU-hour allocation cap. The relaxed graph fits this reservation.

# PHASE-0: technical FinQA check before the task decision

**your route: host venv (confirmed 8 Oct)**. Your pilot runs directly as the batch step on the host with `export PATH=/home/<USER>/envs/train/bin:$PATH`, without an enclosing srun or container. Use that same existing native base on the login node and preserve that exact environment line in the copied pilot wrapper. No containerisation or new route question is required.

**Storage: set TASK_ROOT on large shared storage visible from login and compute nodes**, such as `/lustre-storage/fsx_efa/user/$USER/v4-phase0`, and create WORK beneath it. WORK's filesystem needs **at least 869 GiB free** (about 869 GiB under the registered checkpoint/cache policy, increasing for larger model snapshots). CPU **prepare refuses** below that requirement before environment builds or any GPU job; the planner checks it again before submission. Choose your assigned large filesystem, rather than home storage with a smaller quota.
Confirm your quota on WORK's filesystem is at least **869 GiB** (`quota -s`, `lfs quota -u $USER`, or `mmlsquota`); the admission check measures filesystem free space, not your quota.

Send this package after the owner's phase-scoped G5, G6 and G7 checks. The owner's
7 October decision replaces the separate request for pasted settings: the partner's
first phase-0 command captures and checks them on CPU, and the sole return message
carries that evidence whether the check passes or fails. It implements plan v4 **on origin/main, §15.5, §15.6 and §15.8**. The manager merges that
plan before sending; this runbook does not require §§14–15 to be present in the
partner clone. Sections 14/14.5 concern the earlier Chemistry feasibility decision;
§15 declares the conditional Spider change. Neither decision changes this FinQA-only technical package. It makes no
scientific, task-selection, coverage or learning claim. No PHASE-0 demonstration,
rewrite, checkpoint or qualification decision is reused by a scientific campaign.

- *Phase 0 boundary (amends 15.5).* Phase 0 is a bounded technical qualification independent of the Spider decision,
  and an explicit phase-scoped exception to section 12's full-campaign send gate. After the Slurm check and the
  recorded phase-scoped reviews pass, it runs only the listed FinQA technical rows under enforced row and allocation
  caps, then stops and returns its archive. Smoke checkpoints are not scientific initialisations or results. Phase-0
  success cannot authorise full-pool generation, standalone scientific qualification or the training matrix, and cannot
  close Spider-specific gates. Its FinQA timings do not establish Spider feasibility or campaign budget admission. The
  10 to 15 GPU-hour figure is an estimate; the shipped package enforces an allocation cap. Every sbatch carries
  `--no-requeue` and the partner's own account and QOS (partner's Slurm replies, 7 October).

Use a fresh writable `WORK` and capture Slurm settings first. After a passed check,
stage the immutable Qwen3-8B, its pinned tokenizer,
Qwen3.6-27B, the pinned SDPO checkout and public FinQA `train.json`/`test.json` on
CPU before allocating GPUs. Downloads and preprocessing must not write through
links into reference trees. The trainer base is the partner's CUDA-enabled trainer
installation at `~/envs/train`, matching your run3-record.json; the package
builds a trainer venv over that base and a **separate clean 27B inference venv**.
The latter pins vLLM **0.18.0**, torch **2.10.0+cu129**, torchaudio 2.10.0+cu129,
torchvision 0.25.0+cu129, datasets 4.0.0 and pyarrow 21.0.0. Its teacher engine uses
**Triton GDN prefill and explicit FLASH_ATTN**. Plan §13 authorises one teacher-engine restart with `VLLM_BATCH_INVARIANT=0` only if batch-invariant startup fails: it is labelled `batch_invariance: unavailable for this model`. The corpus is frozen once by hash; exact serial equivalence and regeneration on resume are then not guaranteed, and qualification reports measured determinism. Rewrites have no such fallback. It shares no trainer packages.
This follows the inference recipe in `origin/build/v4-27b-check`; this package
never invokes Modal. Network/package downloads are CPU work with finite deadlines.

**TAG arrives in the owner's send message**, as the exact frozen public-kit tag or commit.
On the login node, create an owned staging root. `TAG` is the **exact frozen public
kit tag or 40-character commit that the manager fills at send time**, after the
phase-scoped reviews; `MANAGER_FILLS_AT_SEND` is not runnable and is never “latest”.
Only TAG and your assigned Slurm account, partition and QOS need values from a
person. Discover the latter using `sacctmgr` and your site assignment; do not copy
`normal` or `midpri` from an example.

```sh
export PATH="$HOME/envs/train/bin:$PATH"
export TASK_ROOT="/lustre-storage/fsx_efa/user/$USER/v4-phase0"
# Example: your assigned large shared storage, visible on login and compute nodes.
mkdir -p "$TASK_ROOT/bootstrap/tmp" "$TASK_ROOT/bootstrap/pip" "$TASK_ROOT/bootstrap/xdg" "$TASK_ROOT/hf-cache"
export TMPDIR="$TASK_ROOT/bootstrap/tmp" PIP_CACHE_DIR="$TASK_ROOT/bootstrap/pip" XDG_CACHE_HOME="$TASK_ROOT/bootstrap/xdg"
export HF_HOME="$TASK_ROOT/hf-cache" HF_HUB_CACHE="$TASK_ROOT/hf-cache/hub" HF_ASSETS_CACHE="$TASK_ROOT/hf-cache/assets"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE" TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export TRITON_CACHE_DIR="$TASK_ROOT/bootstrap/triton" VLLM_CACHE_ROOT="$TASK_ROOT/bootstrap/vllm"
export TORCHINDUCTOR_CACHE_DIR="$TASK_ROOT/bootstrap/torchinductor" TORCH_EXTENSIONS_DIR="$TASK_ROOT/bootstrap/torch-extensions"
export TORCH_HOME="$TASK_ROOT/bootstrap/torch" CUDA_CACHE_PATH="$TASK_ROOT/bootstrap/cuda"
export NUMBA_CACHE_DIR="$TASK_ROOT/bootstrap/numba" FLASHINFER_WORKSPACE_BASE="$TASK_ROOT/bootstrap/flashinfer"
export GIT_TERMINAL_PROMPT=0 PIP_DEFAULT_TIMEOUT=30 PIP_RETRIES=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export TAG=MANAGER_FILLS_AT_SEND
if [ "$TAG" = MANAGER_FILLS_AT_SEND ]; then
  echo "STOP: use the frozen TAG from the owner's send message before cloning."
else
if python - "$TASK_ROOT" "$TAG" <<'PY'
import pathlib,shutil,subprocess,sys
root=pathlib.Path(sys.argv[1]);kit=root/'continual-learning-kit'
if shutil.disk_usage(root).free < 1024**3: raise SystemExit('STOP: TASK_ROOT needs 1 GiB before clone; do not use login /tmp')
try:
    subprocess.run(['git','-c','http.lowSpeedLimit=1000','-c','http.lowSpeedTime=30','clone','https://github.com/hopit-ai/continual-learning-kit.git',str(kit)],check=True,timeout=600)
    subprocess.run(['git','-C',str(kit),'checkout','--detach',sys.argv[2]],check=True,timeout=30)
except (subprocess.SubprocessError,OSError) as exc:
    raise SystemExit('STOP: GitHub clone/checkout failed: '+str(exc)+'. If blocked, use the same bounded zero-GPU job pattern after obtaining the kit.')
PY
then
export KIT="$TASK_ROOT/continual-learning-kit/kit"
export V4_KIT_TAG="$(git -C "$KIT" rev-parse HEAD)"
export PYTHONPATH="$KIT/.."
# Use your already assigned site names, not the fixture's partner/midpri names:
export ACCOUNT=YOUR_ASSIGNED_ACCOUNT PARTITION=YOUR_ASSIGNED_PARTITION QOS=YOUR_ASSIGNED_QOS
export WORK="$(mktemp -d "$TASK_ROOT/work.XXXXXX")"
# FIRST phase-0 command: CPU-only, on the login node, before downloads or builds.
if python3 "$KIT/v4_phase0.py" capture --work "$WORK" \
  --partition "$PARTITION" --qos "$QOS" --account "$ACCOUNT"; then
  export SITE_MINUTES="$(python3 -c 'import json,os
from pathlib import Path
p=Path(os.environ["WORK"])/"v4/report-phase0/presend/capture.json"
n=json.loads(p.read_text())["site_minutes"]
if not isinstance(n,int) or n<=0: raise SystemExit("unknown partition MaxTime; stop before planning and return the capture evidence")
print(n)')"
else
  echo "STOP. Send ${WORK}-return.tar.gz and ${WORK}-reading.json to the owner in your one return message."
fi
else
  echo "STOP: public-kit clone or frozen checkout failed; keep evidence and use a fresh staging root."
fi
fi
```

The command captures **version.txt, config.txt, partition.txt, qos.txt and
effective cgroup settings saved as cgroup.conf**; no prior paste, transcription, file request or separate
message is needed. It runs these exact queries with a **120-second CPU command
deadline each**, CUDA selection cleared, and refuses execution inside any existing
Slurm allocation:

```sh
scontrol --version
scontrol show config
scontrol show partition "$PARTITION"
sacctmgr -n -P show qos format=Name,PreemptMode,GraceTime,Preempt
# Read Cgroup Support Configuration from the captured scontrol show config.
# If absent, read local CgroupConf, then cgroup.conf beside SLURM_CONF,
# then /etc/slurm/cgroup.conf. No srun or queued allocation is used.
```

The controller's **Cgroup Support Configuration** section supplies effective
`cgroup.conf` settings first, including configless sites. If that section is absent,
the capture tries local `CgroupConf`, `SLURM_CONF`'s directory and `/etc/slurm`, in that
order. Unreadable settings refuse capture clearly; no CPU job is queued. The script invokes the existing containment **`check --from-file`**
on its own captured files. A failed query, missing settings, timeout, preemptible
QOS or any failed containment check **STOPS before prepare or GPU planning**. It
writes a sealed `containment-presend.json` and `capture.json`, retains all five
files and command outcomes, and automatically creates **`${WORK}-return.tar.gz`
and `${WORK}-reading.json`**. Return these to the owner in one message. Do not
run prepare, sbatch, selftest or reconcile following a capture refusal; there is
no GPU allocation to reconcile. Changed site values or captured bytes also refuse
planning. No GPU job is submitted by this first command, even on success.

The capture step masks only **the values of named user, node, host (including
AccountingStorageHost, ControlMachine and SlurmctldHost) and IP identity keys**
before publishing files into WORK; version lines, paths such as
`/etc/slurm/slurm.conf`, and unrelated scalar values are preserved; original query outputs exist only in a private
temporary directory which is removed on exit. Everything else is retained.
Partition, account and **QOS names MUST NOT be redacted**; the full four-field QOS
inventory, including Preempt relationships and empty fields, is unchanged.
Limits, durations, plugins, cgroup settings and counts stay visible. These masked
raw files, the checker output, per-command results and hashes are collected in
the single return archive. You do not need to edit or send them separately.

For `MaxTime=7-00:00:00`, the captured `SITE_MINUTES` is **10080**. The planner
still requests 480 minutes. If capture succeeds, continue CPU staging:

Before the self-test, both trainer and inference probes require CUDA availability, exactly eight allocated devices, and a synchronized one-element CUDA operation on each device. Each records the driver version beside torch.version.cuda. A failure says “STOP before the self-test”; the measured environment-check deadline is unchanged.


All login-node commands use the existing base env (`python` resolves under `~/envs/train`). GitHub, PyPI and `download.pytorch.org/whl/cu129` remain login-node CPU work, with finite deadlines, 10-second endpoint probes and pip 30-second socket timeouts/one retry. If PyPI or PyTorch wheels are blocked, retain the setup blocker and return it; prepare is login-node work and refuses under SLURM_JOB_ID. A blocked GitHub clone may use the bounded zero-GPU pattern after obtaining the kit. The clean 27B inference venv is built **on the login node from PyPI/PyTorch wheels**; neither pip nor its CPU import check downloads anything from HF. Import/model checks use offline local inputs.

The login node's **1 GB /tmp** is insufficient for large temporary files. Bootstrap sets all temporary/cache variables before git. Scripts also set them for capture, prepare, collect, reconcile, reader and the compute payload. On the login node, TMPDIR uses WORK/phase0-cache/tmp. In both routes, PIP_CACHE_DIR, XDG_CACHE_HOME and Triton/vLLM/torch/CUDA/torch-extension/Numba/FlashInfer caches use WORK/phase0-cache on the large shared filesystem, excluded by collect. HF_HOME remains outside WORK under TASK_ROOT. Prechecks require **30 GiB free at WORK before a large install** and **100 GiB free at HF_HOME before the two whole model downloads**. CPU prepare and the planner separately require **at least 869 GiB free on WORK's filesystem** under the retained-checkpoint policy, increasing this if the two complete downloaded model snapshots exceed the 100-GiB cache reserve. Prepare checks this before environment builds, preserves the parent's `preparation-storage.json`, and records the delegated trainer's capacity recheck in `preparation-storage-trainer.json`; both are integrity-bound by the prepare receipt and collected. A refusal retains a readable `setup-blocker.txt`; the planner records its recheck in `phase0.json` before submission.

Long Lustre TMPDIR paths are allowed **on the login node**. Inside the allocation, the frozen environment instead uses **TMPDIR=/tmp/phase0-tmp**, **VLLM_RPC_BASE_PATH=/tmp** and **RAY_TMPDIR=/tmp/phase0-ray**, under the site's job-private job_container/tmpfs. The payload creates the temporary/session directories. One socket-base checker enforces `len(os.fsencode(TMPDIR)) + 32 <= 107` for multiprocessing's `/pymp-XXXXXXXX/listener-XXXXXXXX`, `len(os.fsencode(VLLM_RPC_BASE_PATH)) + 37 <= 107` for vLLM's `/UUID`, and `len(os.fsencode(RAY_TMPDIR)) + 68 <= 107` for Ray, in CPU planning and before the self-test. The `ipc://` protocol prefix is not part of the Unix pathname. Before creating any Slurm step or CUDA context, the environment check requires **at least 1 GiB free in the job-private TMPDIR**, records that check, and otherwise prints `STOP before the self-test`. What still writes there: multiprocessing listener metadata, torchrun/rendezvous scratch, and temporary FinQA/tokenizer/scorer copies for the live report (the pinned raw sources and tokenizer total about 100 MiB; 1 GiB leaves headroom for overlapping copies and small library scratch). Tensor shared-memory buffers use /dev/shm, not TMPDIR. Wheels, downloads, model checkpoints, merged weights and all large caches stay on shared storage through their explicit paths/cache variables. The login-node first-batch probe uses zero DataLoader workers to avoid sockets there; the allocation's pinned SFT trainer keeps its eight workers. CPU preparation imports Ray/vLLM without starting engines.

The 869-GiB minimum keeps every trainer checkpoint after checks, so the registered state evidence remains available. With two updates and save frequency two, F/R retain one step-2 checkpoint each. S/D can also save at step 1 when the pinned PPO trainer detects capacity expiry; their retention limit keeps both checkpoints, so admission budgets two each. Both S and D save an EMA with each checkpoint through the same state wrapper.

The pinned initial config counts **8,190,735,360 parameters**. Each retained checkpoint reserves 4 bytes/parameter for FP32 FSDP model shards and 8 for Adam's two FP32 moments; each S/D checkpoint additionally reserves 4 for its EMA (conservative even if BF16). Eight ranks partition these states. At the final merge all six budgeted trainer checkpoints, four EMA copies, three previous merged exports and the new 2-byte/parameter export coexist: **96 × 8,190,735,360 = 786,310,594,560 bytes**. The pinned merger concatenates/casts in RAM and creates no additional disk temporary. Add 1 GiB per checkpoint for extra state, shard padding and serialization (6 GiB); at least 100 GiB for the complete 27B/8B snapshots and cache/staging headroom (raised to their measured total size if larger); and 30 GiB for environments and compile caches. This totals **932,339,482,624 bytes = 868.309 GiB**, rounded up to **869 GiB free**, additional to files already resident. HF_HOME remains outside WORK; counting its model-cache allowance again is deliberately conservative and never relies on both paths sharing a filesystem. No scientific or checkpoint policy changes.

```sh
python "$KIT/v4_phase0_site.py" network --work "$WORK" --url https://github.com -- \
  git clone https://github.com/lasgroup/SDPO.git "$TASK_ROOT/SDPO"
python "$KIT/v4_phase0_site.py" network --work "$WORK" -- \
  git -C "$TASK_ROOT/SDPO" checkout --detach 7c457fc1b1f636ae794eb0362ba37d4743b06fbc
export SDPO_DIR="$TASK_ROOT/SDPO"
python "$KIT/v4_phase0_site.py" network --work "$WORK" --url https://github.com -- \
  git clone https://github.com/czyssrs/FinQA.git "$TASK_ROOT/FinQA"
python "$KIT/v4_phase0_site.py" network --work "$WORK" -- \
  git -C "$TASK_ROOT/FinQA" checkout --detach 0f16e2867befa6840783e58be38c9efb9229d742
export FINQA_ROOT="$TASK_ROOT/FinQA/dataset"
export PYTHONPATH="$KIT/..:$SDPO_DIR"
# Separate downloader venv; no install targets the native base.
python "$KIT/v4_phase0_site.py" network --work "$WORK" -- \
  python -m venv "$TASK_ROOT/download-env"
python "$KIT/v4_phase0_site.py" network --work "$WORK" --url https://pypi.org/simple/ --seconds 1800 -- \
  "$TASK_ROOT/download-env/bin/python" -m pip install --timeout 30 --retries 1 'huggingface_hub[cli]==0.34.4'
```

**HF is blocked on the login node (403).** Download both whole repositories at their pinned revisions only in this **zero-GPU compute job**. It runs directly on the batch host, with your own account/QOS/partition, no requeue, two-hour maximum, explicit CPUs and memory. It verifies scheduler zero-GPU assignment and durably records snapshot paths, revision directories, every file's size and JSON hashes. Prepare verifies that receipt, then performs unchanged full model-byte hashing. Zero allocation GPU-hours and zero GPU-block charge are recorded; this job creates no qualification allocation ledger entry.

```sh
python "$KIT/v4_phase0_download.py" script --work "$WORK" --task-root "$TASK_ROOT" \
  --python "$TASK_ROOT/download-env/bin/python" \
  --account "$ACCOUNT" --qos "$QOS" --partition "$PARTITION" \
  --out "$WORK/v4/report-phase0/download/download.sbatch"
sbatch --parsable --no-requeue --gpus=0 --time=02:00:00 --cpus-per-task=8 --mem=32G \
  --account="$ACCOUNT" --qos="$QOS" --partition="$PARTITION" \
  --output="$WORK/v4/report-phase0/download/job.log" --error="$WORK/v4/report-phase0/download/job.err" \
  "$WORK/v4/report-phase0/download/download.sbatch" > "$WORK/v4/report-phase0/download/submission.txt"
# Wait for the CPU job to finish successfully; then verify on the login node:
export MODEL_DIR="$HF_HOME/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
export QWEN3_8B_TOKENIZER="$MODEL_DIR"
export TEACHER_MODEL_DIR="$HF_HOME/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
python "$KIT/v4_phase0_download.py" verify --work "$WORK"
```

Yes, QWEN3_8B_TOKENIZER may equal MODEL_DIR. Snapshot symlinks may point only to blobs in the owned HF cache; no `--local-dir` is used. Keep credentials outside WORK; never put tokens on command lines or enable shell tracing. These repositories are public and need no login. If authentication is ever needed, supply HF_TOKEN in the environment, never logged; do not print it or enable shell tracing. A failed download retains failure.json; return it with the capture evidence. Retry setup with a fresh WORK; the new download job reuses existing HF blobs and writes a new receipt.

**A failed capture or prepare may be retried with a fresh WORK**, retaining the
earlier WORK and archive; repeat capture, run a new zero-GPU download job to reuse the same cached blobs and write the receipt in the fresh WORK, then prepare.
This is a CPU setup retry; never submit a replacement GPU allocation.

CPU prepare consumes the **passed, frozen capture in the same WORK**. There is no
`--paste` handoff. It validates all model/data/tokenizer inputs and the pinned SDPO
checkout before copying or building anything; an input validation failure leaves
that capture reusable. If a build or later preparation fails, keep the failed WORK
and use `export WORK="$(mktemp -d "$TASK_ROOT/work.XXXXXX")"`, then repeat the first
**capture** command, the zero-GPU download job (cached blobs are reused), and prepare. Never rerun prepare into
that WORK after preparation has started. No reference or download tree needs deletion.

Run **prepare and the planner on the login node in the same existing host base env**, using `export PATH="$HOME/envs/train/bin:$PATH"` exactly as your wrapper does. Prepare records/enforces every actual trainer dependency: torch 2.9.0+cu128 (training/autograd/CUDA ABI), vllm 0.12.0 (8B rollouts, rewrites and scoring), verl 0.7.0.dev* (trainer, with the SDPO commit independently pinned), transformers 4.57.1 (models/tokenizers), flash_attn 2.8.3* (training attention/cross-entropy extension ABI), ray 2.53.0 (distributed workers), numpy 1.26.4 (trainer/data numerics and array ABI). Other packages retain existing import checks rather than new site version pins. A mismatch refuses before installation. The trainer uses `--system-site-packages --without-pip`: it invokes the inherited base pip module with the trainer interpreter, so installs target only the trainer prefix and bootstrap pip cannot shadow the base installer. Your micromamba base is not a Python venv (`sys.prefix == sys.base_prefix`), so it needs no `.pth` bridge; only a nested Python venv gets a recorded, hash-checked parent-site bridge. Constraints pin the **full base distribution inventory**, except verl from the independently pinned owned SDPO checkout. After installation, every distribution in the trainer’s OWN site-packages is compared with the base: differing versions are refused except that owned editable verl. Prepare records sorted `python -m pip check` output in the base env without rejecting its pre-existing conflicts. Trainer `pip check` refuses only new lines absent from that baseline before building inference; both raw outputs and the comparison are archived, and the reader re-derives the difference. Nameless or interrupted-pip (`~` prefix) dist-info entries are skipped and recorded; duplicate names with differing versions refuse with a readable setup blocker. Both full trainer inventories (name, version, location, editable provenance) are recorded in environment.json; the prepared SDPO takes precedence on PYTHONPATH. Pip runs only in WORK's trainer/inference venvs, never against the base; no base upgrade is authorised.

```sh
export CUDA_VISIBLE_DEVICES=
export V4_TELEMETRY=1
if python "$KIT/v4_phase0.py" prepare --work "$WORK"; then
  export V4_TEACHER_PYTHON="$WORK/phase0-envs/inference/bin/python"
  export SDPO_DIR="$WORK/phase0-source/SDPO"
  export PYTHONPATH="$KIT/..:$SDPO_DIR"
  # Keep the base PATH active for login-node planning and return commands.
else
  echo "STOP: keep this WORK; retry capture and prepare only with a fresh WORK."
fi
```

If the base env or a required endpoint is unavailable, retain a setup blocker, collect with the base env and stop before GPU planning. The owner can read an unprepared setup archive.

The first verification in the built trainer is the existing CPU import and entry
smoke of **every kit module and launcher**, qwen-vl-utils, and the real SFT loader's
first CPU batch. The separate inference environment gets its own CPU import/version
and EngineArgs check. All command outputs and the environment fingerprint are
archived. Preparation freezes the full FinQA source/pool, the first **20 training**
questions, heldout panel, tokenisation, model/shard identities, code, departures and
environment hashes. It refuses changed prepared bytes before GPU work. Do not run
CPU prepare from a Slurm GPU job or after editing the frozen package.

Generate the sole allocation script on CPU. This plans; it does not submit:

```sh
python "$KIT/v4_allocation.py" --work "$WORK" --phase phase0 --stage phase0 \
  --minutes 480 --site-minutes "$SITE_MINUTES" \
  --partition "$PARTITION" --qos "$QOS" --account "$ACCOUNT" \
  --activate "$WORK/phase0-envs/trainer/bin/activate" \
  --out "$WORK/v4/report-phase0/allocation/phase0.sh"
```

The planner supplies **`phase0.header.sh`**, an `#SBATCH` header block, and
**`phase0.sh`**, a payload to run directly in your existing host-venv batch process. It also
retains `phase0.json`, binding the prepared environment and allocation plan. The
header has `--no-requeue`, `--exclusive`, `--nodes=1`, `--gpus=8`, `--time=08:00:00`
and your captured partition, QOS and account. Scheduler stdout and stderr are
`$WORK/v4/report-phase0/allocation/phase0-%j.out` and `phase0-%j.err` (`%j` is the
job ID); collect includes these files. The payload exports all frozen data/model
paths, the prepared SDPO's PYTHONPATH, telemetry and offline settings.
`SLURM_EXPORT_ENV=ALL` passes these values to steps.

The payload's **first step, before the self-test**, is an **environment
check whose full requirement is max(600 seconds, ceil(5 × (trainer import + inference import + 60 seconds for sixteen CUDA context initialisations))), recorded in the prepare receipt**. Import duration is record-only: slow imports do not themselves refuse prepare. The planner must fit this deadline and every serial row into the unchanged allocation reservation. It first rejects a numeric SLURM_STEP_ID, checks job-private scratch/socket paths, and tests step creation with `srun --overlap -n1 true`; both prepared runtimes then perform the eight-device CUDA and version checks. A real mismatch or expiry prints `STOP before the self-test`, saves receipts/logs, and releases the job without training. Keep the existing native wrapper's base Python and loader search paths.
CPU prepare measures the full prepared-byte rehash. The reviewed allowance is **max(60, ceil(1.5 × measured seconds))**; an 84-second measurement gets 126 seconds. Prepare no longer refuses a measurement above 60 seconds. Planning binds the receipt without rehashing all model bytes again; the allocation still performs full integrity verification.

The v3 relaxed graph requires **27,816 seconds** with a 600-second environment check and the partner's 84-second rehash (126-second allowance), including every serial row and 15 seconds of dispatch per row. It fits the owner's approved **08:00:00 / 480-minute** reservation with **984 seconds** of slack. The planner still checks the full graph against the reservation and the site's time limit; it does not shorten scientific work.
The exact allocation header includes **`--no-requeue --account="$ACCOUNT" --qos="$QOS" --time=08:00:00`**, one exclusive node with **eight GPUs**, and a **64 GPU-hour allocation cap = 8 × 8**. Phase 0's actual elapsed allocation time is charged to the **100 GPU-hour qualification block** and conservative ceiling **560** (the campaign ceiling is 955 per plan 15.11); unused reservation time is not charged against the block. Slurm charges actual elapsed time, so expected phase-0 use stays about **15 GPU-hours**, rather than the 64 GPU-hour worst-case cap.
Phase 0 remains **one allocation, no requeue and no replacement job**. The cap does not authorize extending that allocation; refused jobs and teardown still count, and the reader recomputes actual charges from the returned ledger. Submit only the
header and payload printed by that planner. You previously launched the pilot as
**`sbatch --parsable /home/<USER>/scripts/k8b_pilot_run.sbatch`**. Launch phase 0
exactly as you launched `k8b_pilot_run.sbatch`: copy your wrapper to
`$WORK/v4/report-phase0/allocation/phase0.sbatch`, keep **its exact host-venv environment lines**, including `export PATH=/home/<USER>/envs/train/bin:$PATH`. Replace only the old runner payload with `bash <absolute-WORK>/v4/report-phase0/allocation/phase0.sh`, **directly in the batch step on the batch host**. Add our `phase0.header.sh` lines at the top, replacing conflicting resource/time/account/QOS directives. The payload is bash, not an sbatch submission; submit the copied wrapper. Do not wrap the payload in srun or introduce a container. Its own contained row steps still use srun with every GPU-assignment/cap/termination check. Then,
on the login node the dependency-light helper calls the same `sbatch --parsable <wrapper>` form and durably binds its job ID, WORK, wrapper and plan hash before returning:

```sh
python3 "$KIT/v4_phase0_submission.py" --work "$WORK" \
  --plan "$WORK/v4/report-phase0/allocation/phase0.json" \
  --script "$WORK/v4/report-phase0/allocation/phase0.sbatch"
```

Submit from a plain login shell, not from inside an `srun` or `salloc` session: the environment
check refuses a numeric `SLURM_STEP_ID`, and a value inherited from such a shell would stop the
only allocation. `echo ${SLURM_STEP_ID:-none}` should print `none`; if it does not, submit with
`env -u SLURM_STEP_ID -u SLURM_STEPID python3 "$KIT/v4_phase0_submission.py" ...`.

In every new allocation the **FIRST containment command is the on-node self-test**
with explicit `--block qualification --block-limit 100 --ceiling 560`. Never run
`check`, `reconcile` or runner tracking first inside the allocation. After the environment check the payload
self-tests first; the real runner verifies its plan against Slurm and the
ledger after that test. A failed self-test releases the job with no training.

The 26 serial rows then do exactly this technical work:

1. Verify prepared bytes; reload the initial 8B on the designated physical scoring
   GPU four times: **two independent same-task, same-cap pairs**, each on the same
   first **50 FinQA heldout** questions at **B=2,048**. Engine/configuration failure
   is checked first. Any second-pair difference blocks all generation and training.
2. Load the real 27B in its own environment on four TP2 replicas. Generate the
   phase's **20 FinQA training** questions with the registered four-attempt schedule
   and deterministic first-acceptable verification. Generate frozen-initial-8B
   rewrites with eight TP1 replicas; retain raw attempts, rejections and merge timing.
3. Build the shared demonstration/rewrite intersection and seeded 64-exposure
   technical schedule; repeats are explicitly smoke-only. Zero intersection stops.
   Resolve `--cfg job --resolve` for all four actual launchers and check constants.
4. Train **S, F, R, D**, each from the same real incoming 8B, **two optimizer steps**,
   seed 101, `V4_PROFILE=technical-smoke`, telemetry on. F/D consume this phase's
   27B demonstrations; R consumes its frozen-8B rewrites. Merge/export each final
   checkpoint, reload it on the same scoring GPU and score **20 FinQA heldout**
   questions at B. Archive all eight GPUs' memory, seconds per step, tokens,
   export/merge/reload timing, finite losses, learning rates and D's EMA movement.
5. Recompute the technical report and **PAUSE**, ending the allocation before the
   owner reviews anything. Do not chain another allocation or scientific work.

GPU row containment deadlines are now 1,500 seconds (teacher 600); CPU rows are at least 600 seconds except the measured prepared-byte verifier. The former 300-second GPU caps were unmeasured estimates, so the v3 policy uses five times those estimates. The redundant 120-second wrapper allowance is removed; the Slurm row cap already includes teardown. A row that
cannot fit its actual remaining window refuses before dispatch. Verified row
failures follow the existing typed infrastructure retry admission (at most two
retries, one OOM retry with OFFLOAD=1); there is no outcome-based retry or extension
of this single reservation. Saved-state restoration remains disabled until its
real-stack qualification check passes; PHASE-0 does not authorize restoration.
Verified CANCEL preemption ends the allocation for review. Hard stops require the
owner's plain SHA256-bound acknowledgement and reconciliation; history is retained.
See [README-contain.md](README-contain.md). Never launch a replacement PHASE-0 job
on your own; return the partial evidence.

After Slurm ends the job, reconcile **on the login node**, close its full allocation
accounting, and collect success or failure alike. Owner review happens afterwards:

```sh
python3 "$KIT/p4_contain.py" reconcile --work "$WORK" --out "$WORK/reconcile" --seconds 600
python3 "$KIT/collect.py" --work "$WORK" --out "${WORK}-return.tar.gz"
```

These host `python3` commands are standard-library-only for the unfiltered collection above. Reconciliation also works when the payload or self-test never started: the submission receipt identifies the job; scheduler-verified start, end and eight-GPU width create/close its ledger and charge the complete interval to qualification. Unknown termination or exposure refuses reconciliation and is never charged as zero.

Keep the **same existing host base env** active on the login node for reconcile, collect and the reader. In a new login shell restore PATH, WORK and KIT; do not create a new WORK:

```sh
export PATH="$HOME/envs/train/bin:$PATH"
export WORK=/the/same/absolute/WORK
export KIT=/the/same/absolute/KIT
export SDPO_DIR="$WORK/phase0-source/SDPO"
export PYTHONPATH="$KIT/..:$SDPO_DIR"
python3 "$KIT/read_v4_phase0.py" "${WORK}-return.tar.gz" \
  --out "${WORK}-reading.json"
```

Send **`${WORK}-return.tar.gz`, `${WORK}-reading.json` and the allocation job ID**
to the **owner, who relays them**. Use one return message to the owner; there is no earlier Slurm-file message. The archive includes **`reconcile/containment-reconcile.json`**,
its manifest hash, the allocation ledger, containment receipts, automatically masked five-file
capture, prepare/import/environment receipts, retained sbatch script and plan,
runner starts/verdicts and failed logs, generation attempts/rejections, resolved
configs, metrics/telemetry and scoring records. Keep scheduler stdout/stderr and
send those named job-log files alongside it if the job failed before runner
records. Reconcile refuses a WORK with neither an allocation ledger nor a valid submission receipt; a wrong or never-allocated WORK cannot produce scheduler-verified accounting. A pre-self-test refusal still reports its full allocation hours and remains incomplete. If reconcile
refuses, still collect the partial evidence and send its error to the owner. The reader checks **every member
against the collector manifest**, rebuilds sources/prompts and acceptance, scoring
pairs/verdicts, training dose/recipe/lineage, memory/timing and the one-allocation
charge. Missing or failed evidence refuses a technical pass. A technical pass still
requires owner review and grants no scientific authorization. Weights are excluded. Every exported smoke checkpoint carries immutable
`phase0-technical-only.json`; later preparation and training refuse it as an
initialisation. The frozen campaign labels demonstrations, rewrites, reports and
checkpoints technical, with `usable_as_scientific_initialisations=false`.

For a CPU-only rehearsal of all nine success/failure paths through the real runner,
including host-venv success with a verified download-job receipt, missing capture (the rehearsal case is named `missing_paste`), self-test refusal, second-pair mismatch, OOM, a hung command, Pyxis numeric-step refusal and pre-self-test environment refusal with scheduler accounting:

```sh
python "$KIT/simulate_v4_phase0.py" --out /new/owned/cpu-rehearsal --case all
```

Those stand-ins train nothing, execute no real containment and cannot certify G5,
G7, CUDA memory, model loading or determinism. The archive reader refuses their
synthetic success summaries. Each non-missing-paste rehearsal archives a sealed
**synthetic** `presend/containment-presend.json` plus all five raw files, and the
reader re-parses them, verifies their hashes and checks real runner records. The
reader returns `incomplete` (CLI exit 1) with case-specific reasons: success =
`CPU stand-ins`; missing capture = missing presend receipt; failed self-test =
`selftest refused`; differing second pair = `scoring disagreement`; OOM =
`out_of_memory`; hung command = `wall_time`. The two pre-self-test cases execute the actual environment check, dependency-light reconciliation and collector against owned scheduler stand-ins; their reader remains incomplete and reports the complete eight-GPU, 120-second fictional allocation charge. No case can return a technical pass.

Secondary alternative only if the site route changes: Pyxis/container probe (one minute, zero GPUs)

On the login node, retain the existing pilot wrapper at its original readable path. The probe only inspects its launch lines; it never executes that pilot or creates a Slurm step. It diagnoses a route, not GPU containment or availability of the login-node image. The confirmed host route needs no probe. If that route changes, ask: Does `k8b_pilot_run.sbatch` launch its container with `srun --container-image`, or directly as the batch process, and can that same image run on the login node without an allocation? Any container/Pyxis alternative requires separate qualification; the numeric-step refusal remains. After cloning and making a fresh WORK as above:

```sh
python3 "$KIT/v4_phase0_route.py" --work "$WORK" \
  --pilot-wrapper "$HOME/scripts/k8b_pilot_run.sbatch" \
  --write-probe "$WORK/route-probe.sbatch"
sbatch --parsable --no-requeue --gpus=0 --time=00:01:00 \
  --account="$ACCOUNT" --qos="$QOS" --partition="$PARTITION" \
  --output="$WORK/v4/report-phase0/route-probe.log" \
  --error="$WORK/v4/report-phase0/route-probe.err" "$WORK/route-probe.sbatch"
# After the zero-GPU probe ends:
python3 "$KIT/collect.py" --work "$WORK" --out "${WORK}-route-return.tar.gz"
```

Only if the site route changes, return the route archive and the answer before a GPU submission. For `pyxis_step` or `unknown`, wait for the owner's qualified route decision; no replacement container mechanism is assumed. Collect includes the probe JSON and logs. A probe-only archive is incomplete and has no allocation GPU-hour charge. Preserve this diagnostic WORK and use a fresh WORK for the GPU package after the route is resolved.

Optional CPU-only format capture (one minute, zero GPUs)

This optional one-minute CPU-only parser capture is descriptive. It rides the
**same return archive**: `collect.py` includes `cpu-format-capture.txt` and
`cpu-format-capture.err` automatically. It neither gates phase 0 nor needs an
earlier message. It starts no srun; if no live steps are visible, the format
reading remains incomplete. The earlier fixture's `midpri` QOS is preemptible;
use your own assignment.

```sh
cat > "$WORK/cpu-format-capture.sh" <<'SH'
#!/bin/bash
#SBATCH --no-requeue
#SBATCH --time=00:01:00
#SBATCH --gpus=0
set -eu
printf '%s\n' '--- scontrol show job ---'
scontrol show job "$SLURM_JOB_ID"
printf '%s\n' '--- scontrol show step ---'
scontrol show step "$SLURM_JOB_ID"
printf '%s\n' '--- squeue -s ---'
squeue -s -j "$SLURM_JOB_ID"
printf '%s\n' '--- sacct -n -P --format=JobID,State,ExitCode,Start,End,Elapsed,Timelimit,AllocTRES,ReqTRES ---'
sacct -n -P -j "$SLURM_JOB_ID" --format=JobID,State,ExitCode,Start,End,Elapsed,Timelimit,AllocTRES,ReqTRES
SH
sbatch --no-requeue --account="$ACCOUNT" --qos="$QOS" --partition="$PARTITION" \
  --time=00:01:00 --output="$WORK/cpu-format-capture.txt" \
  --error="$WORK/cpu-format-capture.err" "$WORK/cpu-format-capture.sh"
# After this optional job ends, collect it with the phase-0 return archive above.
# The owner runs check-live-format --from-file on that archived capture.
```

The self-test records live `Requeue=0`, actual QOS, `PreemptExemptTime` and
`UnkillableStepTimeout=500 sec`; every allocation requalification and live row check
rechecks them. REQUEUE is admitted only with `preempt/qos` and live `Requeue=0`.
A QOS named in any other tier's `Preempt` field refuses admission. Normal KillWait
is 40 seconds, but a pathological kill tail can reach **500 seconds**. The row cap
and verified-termination requirement remain unchanged: a late or unknown termination
hard-stops, and accounting charges the entire actual allocation tail. A completed
30-second sleep observed at 41 seconds shows teardown, not timeout enforcement.
Running `sacct` rows with `End=Unknown` never establish termination.

## Questions for us (optional, answer in your return message)

Do you still have the 8B pilot's SDPO Chemistry checkpoints, **sema-chem runs r1
and r2**? If so, please **keep both checkpoints** and mention their locations in
your return message.

The containment self-test proves exactly one CUDA-visible device with a one-element allocation and synchronization while the clients are alive (bounded early-receipt wait up to 180 seconds). It then kills the wrapper, srun client and watchdog. The detached holder attempts a later CUDA re-touch; a missing late receipt is acceptable if Slurm cancelled promptly, every adversarial PID died by kill_by and the GPU is idle. Both client-loss CANCELLED and time-limit TIMEOUT are recorded as valid scheduler cleanup. The test never infers device restriction from CUDA_VISIBLE_DEVICES alone.
