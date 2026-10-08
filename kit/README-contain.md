# Slurm containment for v4

**PHASE 0:** [README-phase0.md](README-phase0.md) is the controlling runbook
(origin/main plan v4 §§15.5–15.6). Its FIRST CPU command automatically captures and selectively masks the five files;
no separate partner settings message is requested. Use its frozen environment,
one 01:50:00 allocation and 22 GPU-hour enforced cap. Historical full-campaign
qualification examples are superseded for phase 0 and grant no admission.

**Historical manual capture reference (superseded for phase 0):**
The commands below describe the settings format for other containment uses. Do not
ask the phase-0 partner to paste or send a separate message; README-phase0.md
automatically captures, checks and archives them. For manual diagnostics, retain the exact outputs below into a directory `PASTES`. These are CPU-only observations; preserve error messages too. Replace `<p>`, and run the last command on the compute node using its effective configuration path:

```sh
scontrol --version                                      # PASTES/version.txt
scontrol show config                                    # PASTES/config.txt
scontrol show partition <p>                             # PASTES/partition.txt
sacctmgr -n -P show qos format=Name,PreemptMode,GraceTime,Preempt # PASTES/qos.txt
cat /actual/path/to/cgroup.conf                         # PASTES/cgroup.conf
```

Run the full settings check on those pastes, without Slurm, torch or GPUs installed:

```sh
python kit/p4_contain.py check --from-file "$PASTES" --qos "$QOS" \
  --work "$WORK" --out "$WORK/presend-review"
```

`containment-presend.json` is a sealed pre-send verdict, **not** a qualification. Choose the actual allocation's QOS explicitly if the paste contains several QOS rows. Missing files, unknown settings and command errors refuse; use a new output directory for corrected pastes. This check covers version, plugins, KillWait, overtime, cgroup device constraints, and effective preemption. Allocation identities and runtime containment still require the live tests.

Inside **each** one-node, exclusive, eight-GPU sbatch allocation, in the kit's torch/CUDA Python environment, the **FIRST containment command must be this self-test with an explicit block**. Never run reconcile, check or runner allocation tracking first. For a scientific allocation replace the block and limit with `--block scientific --block-limit 560`; retain `--ceiling 560`:

```sh
python kit/p4_contain.py selftest --work "$WORK" --out "$WORK/k8b4/containment" \
  --block qualification --block-limit 100 --ceiling 560
```

A CPU-only preflight parses settings and imports torch with attack markers and CUDA selection cleared, retaining site loader variables (`LD_LIBRARY_PATH`, `LD_PRELOAD`, `DYLD_LIBRARY_PATH`, `PATH`). A failed import refuses before freezing an experiment. Then the CPU step requests zero GPUs and leaves a detached, TERM-ignoring grandchild after a five-second launcher. Accounting may lag: the observer allows up to 60 seconds, inside the overall ten-minute bound.

Every caller of `start_step`, including the package-4 runner, requests exactly the GPU width from the allocation in the shared launcher. Slurm selects each step's actual GPU bitmap, independent of the client CVD. The launcher supplies a sealed allocation selector-to-UUID map to the parked step, verifies the returned physical assignment outside the step, and archives it under the admission lock before release. Actual UUIDs must be disjoint from every other live row in the allocation. Abort, cleanup and recovery query the archived owning-step UUIDs. The GPU probe requests one device and adopts Slurm's assignment even when the batch environment exposes `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`. It requests one GPU in a one-minute step, kills its wrapper, srun client and watchdog, and exercises delayed CUDA acquisition, clearing attack markers, setsid, double forks, ignored TERM and stalled discovery. From actual scheduler start S, the observer's bound is **S + 60 + J + W = S + 160 seconds**, with J=60 and W=40. This includes Slurm's 30-second periodic timeout check and its phase jitter. The entire self-test is at most **600 seconds**, and the GPU probe at most **160 seconds on one GPU**. Those exposure bounds are not allocation billing: an eight-GPU allocation can charge up to 8 × 600 / 3600 GPU-hours for the whole qualification interval.

Expected settings: `ProctrackType=proctrack/cgroup`, `TaskPlugin=task/cgroup,task/affinity`, `JobAcctGatherType=jobacct_gather/cgroup`, `KillWait=40 sec`, global `OverTimeLimit=0 min`, and partition `OverTimeLimit=0` or `NONE` inheriting known global zero. Numeric `slurm` and Debian `slurm-wlm` version strings are accepted. `ConstrainDevices=yes` must be explicit; effective settings can come from the controller's **Cgroup Support Configuration** section or readable effective `cgroup.conf` beside `SLURM_CONF`. `SignalChildrenProcesses` is recorded; its default `no` is permitted, with the CPU and GPU probes providing cleanup evidence. Preemption type and global/partition/QOS modes are recorded. OFF and CANCEL are admitted for the real plugins `preempt/none`, `preempt/qos` and `preempt/partition_prio`; REQUEUE with `preempt/qos` is admitted only when the live job has `Requeue=0`, checked at the first self-test, every new allocation requalification and every live row check. Every generated sbatch uses `#SBATCH --no-requeue` and the partner’s own explicit account/QOS. SUSPEND, GANG and unknown policies refuse. The receipt records live QOS and all QOS `Preempt` fields; a QOS listed in any other tier’s Preempt field refuses. A CPU paste check is conditional and cannot prove the live Requeue flag. GraceTime is recorded and does not gate an allocation with preemption off. Actual preemption is typed in the row record; verified CANCEL preemption ends the allocation without a campaign hard stop.

Exactly one frozen verdict is retained per allocation. The first is `$WORK/k8b4/containment/containment-selftest.json`; later allocations use `allocations/<jobid>/containment-selftest.json` beneath that directory. The same `--out` command selects the correct allocation subdirectory automatically. The campaign still pauses after its qualification rows for the review required by plan v4; repeating containment does not replace that review gate. Every new sbatch repeats **both** probes. Automatic admission requires all safety fields to equal the first reviewed receipt; job/node identities belong to the new allocation. A failed verdict cannot be overwritten or rerun to get a pass. Require `ok == true` and `v4_qualified == true` for the **current allocation**, and preserve all receipt/probe/preflight directories in the archive.

Every GPU generation, training and scoring argv goes through a row. Dependent CPU commands, including merge validation, stay deadline-bounded inside the allocation and are charged in its wall time:

```sh
python kit/p4_contain.py row --work "$WORK" --out "$ROW_OUT" \
  --gpus 1 --time-cap 900 --block qualification --block-limit 100 --ceiling 560 \
  -- python kit/your_scorer.py your-arguments
```

Block/ceiling values are campaign inputs, retained consistently in the ledger. Every allocation entering a live containment command is registered and charged before preflight, settings checks or receipt validation; refusals still count. Missing allocation boundaries or width are archived as unknown, never zero. An overrun of any earlier block continues to refuse admission after a phase switch. For the scientific block use `--block scientific --block-limit 560 --ceiling 560` for both its per-allocation self-test and rows. A block cannot change inside an allocation. The ceiling counts **allocation GPU-hours**: allocation GPU width × wall time from scheduler allocation start to scheduler allocation end, including qualification and idle time. **End the qualification sbatch before the owner's pause**; after review the main campaign uses new allocations. If any allocation is left running during a pause, those GPU-hours still count. `allocation-ledger.json` records each allocation's start/end, block intervals and every row's interval. Unknown allocation boundaries refuse admission; they are never zero. After a finished allocation, run `reconcile` while its exact accounting is still available so its end is archived. A later sbatch also queries the preceding allocation's end before admitting anything.

At the self-test and every row admission, reserve allocation width × the entire remaining allocation wall time through **StartTime + TimeLimit** from `scontrol show job`. Spent + reserved must fit the segment's limit and the overall ceiling; overlapping rows share that allocation reservation. Unknown/unlimited TimeLimit refuses. A refused allocation gets one immutable `allocation-admission.json` (and the self-test's failed verdict), with `allowed_sbatch_time` and the exact reason. Advice is rounded down to whole Slurm minutes from the remaining block/ceiling budget **after charging the refused job**; no positive time is advertised when exhausted. It is computed at the recorded timestamp: end the refused job promptly, archive its actual end, and recompute/recheck remaining budget for its replacement. A new allocation repeats both probes. Never treat the failed job's advice as permission to extend it or rerun its verdict.

Every job uses the exact planner-generated `sbatch --time=<HH:MM:00>`, the partner's
own account and QOS and `--no-requeue`. **Superseded examples:** the former
06:00:00 / 48 GPU-hour first qualification and 12:30:00 maximum are historical
full-campaign arithmetic, not phase-0 instructions and not permission to allocate.
Phase 0 uses only README-phase0.md's 01:50:00, eight-GPU reservation (14.666667
GPU-hours), within its enforced 22 GPU-hour cap and the qualification block.
Scientific allocations require their separate owner admission and freshly
recomputed budget; nothing in this containment reference authorizes one.
End the qualification job before the owner's pause and return its accounting.

Only **1-GPU rows** may use `--concurrent`, up to eight at once. Every multi-GPU row runs alone and refuses the concurrency flag or any overlap. Scheduler assignment is checked for UUID overlap under the ledger lock, including rows whose RPCs arrive out of admission order. Each GPU step supplies and archives `--cpus-per-task=floor(job CPUs × GPUs / job GPU width)` and `--mem=floor(job memory MiB × GPUs / job GPU width)M`; a one-GPU step uses at most one eighth of an eight-GPU allocation's CPUs and memory. Unknown CPU/memory settings refuse. Unresolved/dead clients require reconciliation. Row `ledger_charge_gpu_hours` is a non-null exposure bound; **do not sum overlapping row bounds** for allocation billing.

Minimum row cap is **300 seconds**: START_WAIT=120 + one minute + J=60 + KillWait=40 + 20 seconds of cleanup headroom. A 160-second cap has no startup headroom and is refused. Delay that leaves no fitting actual whole-minute grant still refuses before release; admission never promises unlimited startup. For actual scheduler start S, the launcher floors whole minutes m satisfying both `S+60m+J <= limit_until` and `S+60m+J+40 <= kill_by`, with all deadlines fixed at admission + cap. It never submits unlimited `--time=0`, rechecks the grant and GPU UUIDs before release, and cancels only its owning step. Inside a device cgroup, NVML queries the expected UUIDs, avoiding both global and local index spaces. Before release the observer independently resolves `SLURM_STEP_GPUS` outside the step and requires its physical UUIDs to match. This prevents a visible expected UUID from hiding a different actual step assignment.

Plan v4 section 13 has three stop classes. A nonzero row with verified termination and charged exposure writes a failed row, **no** `v4-stop.json`. The runner automatically applies the reviewed, outcome-blind infrastructure admission: a registered reason with its hashed supporting log, unchanged seed/recipe/data/lineage, at most two retries and one OOM retry with `OFFLOAD=1`. An unqualified failure stays failed. A verified CANCEL preemption ends the allocation (runner exit 75), without a manager file; reconcile on the login node, then the next job's **first containment command** is its own self-test with explicit qualification `--block qualification --block-limit 100` or scientific `--block scientific --block-limit 560`, and `--ceiling 560`.

Unknown exposure, termination past the cap, overlapping Slurm GPU assignments, changed safety settings and block/ceiling exhaustion write a durable `v4-stop.json` with `stop_class=hard`. Unknown exposure retains its reservation. Reconcile a killed client explicitly:

```sh
python kit/p4_contain.py reconcile --work "$WORK" --out "$WORK/reconcile" --seconds 60
```

This refuses while any row client is alive; use `--force` only to explicitly cancel a live client. It cancels only recorded owning steps, archives exact step termination plus empty queue/idle GPU evidence, updates allocation accounting, preserves typed failures, and never restarts argv. Reconcile alone cannot clear a hard stop. Each new hard stop has a unique event ID, so an old acknowledgement cannot clear a later stop with the same reason. After acknowledged release, retry admission may use the reconciled termination proof only when the original launch identity is unchanged; the original attempt copy and late charges remain in the archive.

The owner returns a **manager-written, SHA-256-bound JSON acknowledgement** through the reviewed manager return channel. No third-party cryptographic package, signing key or key registration is required. On the returned exact stop bytes, the manager runs `python kit/v4_ack.py make STOP.json`. It creates `STOP.json.ack.json` without overwriting an existing file. Its exact fields are `schema: "kit-v4-hard-stop-ack.v2"`, `stop_sha256: <SHA-256 of exact STOP.json bytes>` and `acknowledged: true`. The partner places the owner-delivered file beside `v4-stop.json`, runs `python kit/v4_ack.py verify STOP.json`, then reconciles on the login node. Missing, malformed, duplicate-key, extra-field, false/numeric-flag, old-schema, edited-stop and mismatched/replayed acknowledgements refuse. SHA-256 verifies byte binding and integrity; it does **not authenticate the sender**. Manager provenance comes from owner delivery, and the partner does not generate an acknowledgement to authorize its own resume.

Only successful reconciliation plus the exact valid acknowledgement moves the stop and acknowledgement into hash-addressed durable history. Accounting, frozen receipts, safety settings, block limits and ceiling remain enforced. Acknowledgement cannot make a failed self-test pass or authorize overspend. The next allocation starts with its own explicit-block self-test. Read `containment-reconcile.json` and return it for review.

The Slurm stand-in emits real unit/NONE/Debian/config-section formats, models the 30-second periodic timeout pass with jitter, finished-step disappearance and accounting lag, and uses an eight-device batch CVD with cgroup-local numbering. Fast clocks affect waiting only; production deadline arithmetic uses real minutes. Tagged stand-in GPU owners respect per-GPU queries so parallel rows can verify their own release; untagged historical owners remain conservatively busy. It assigns the lowest free job-bitmap GPUs independently of client CVD, models CPU/memory availability, and archives assigned IDs. Its process-tree sampling is not kernel cgroups and can miss instantaneous orphan forks. CPU tests and a stubbed CUDA branch do not establish live CUDA containment; a real node receipt and the independent send gates remain required.

The receipt records `PreemptExemptTime` and `UnkillableStepTimeout` (partner: **500 seconds**). A pathological kill tail can reach 500 seconds; normal KillWait remains 40. This does not relax a row cap or prove termination. Late/unknown termination still hard-stops, and actual allocation wall time including teardown is charged. All sacct calls use an explicit field list, never ALL. `check-live-format --from-file CAPTURE.txt --out FORMAT.json` validates the CPU capture’s job, step, squeue and mapped sacct shapes; it is never qualification. See README-phase0.md for the optional one-minute zero-GPU capture.
