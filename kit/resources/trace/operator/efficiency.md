# v7h efficiency adapter

Set `efficiency: true` before operator binding to enable the expanded inventory.
CPU `prep-*` rows produce immutable parquet with tokenizer/cap/profile bindings;
`native-*` rows reserve eight GPUs; CPU `export-*` rows retain 0/10/20/30/40.
Verified individual merge seals survive interruption. The original inventory
remains available for exact-output comparisons. S8 stays in one eight-GPU row;
its CPU hashing worker joins before the complete export inventory is published.

Set `V7H_LOCAL_ROOT` to node-local scratch (or use `SLURM_TMPDIR`). TRACE stages
parquet, S8 token rows and score inputs there; deletion stages score plans and
training/calibration inputs. Score outputs publish verified durable bytes even
on interruption; only the existing completion receipt commits a complete lane.
Native snapshots remain durable. Nothing changes the registered resume cadence.

For native deletion rows run the original command through
`python3 -m deletion_tuning.deletion_efficiency --out "$OUT" --phase train
--gpus 8 -- <original command>` (one shell line). Use phase `prep` or `export`
and zero GPUs for their CPU rows. All attempts, including failures, are charged.
TRACE/deletion ESSENTIAL `phase.<row-type>` arrays are allocated GPU-seconds in
this order: train, score, judge, export, prep, idle-in-allocation. These wall
clocks measure allocation cost, not GPU utilisation. Qualification is prep;
27B target generation is score. Staging/publication overhead outside a measured
phase is idle-in-allocation. CPU rows have six zeros. Old receipts without phase
clocks are not reconstructed or represented as measured zeros.

The optional `judge_throughput_probe: true` adds CPU admission and finalizer rows.
Submit their whole DAG. Later lanes wait on CPU for the single decision before
GPU admission. Each finalizer follows its scorer and admission with `afterany`.
If its lane claimed the probe and ended without a receipt, the finalizer writes
DEFAULT, including after cancellation or process death. It never starts a probe.
The scorer also persists DEFAULT on every exit without a decision, including
PENDING from missing inputs. Finalizers leave an existing decision intact.
The first ready lane uses default throughout. Its first judged state supplies D1
science, followed by D1 calibration, F1, D2, F2 (science then calibration each).
A durable started marker prevents re-probing after interruption. Retrying the
first complete lane uses its persisted DEFAULT after a process kill. All
scientific scoring, physical-GPU checks, denominators and complete-lane rules
remain. A rejected/failed probe cannot be tuned or retried. Probe timing includes
both engine lifetimes per pass. Registered calibration comes only from D1;
probe-only passes never replace registered first-lane numbers. The receipt binds
actual clipped token IDs, request order, seeds, settings, labels, mismatch
matrices, O, optional J and hardware/software identity. Later ineligible hardware
uses default. Probe overhead counts against the existing warn-only ceiling.
Recovered D1 science and calibration must match the accepted receipt's labels,
hashes and execution audit. Rejection persists DEFAULT and regenerates the whole
lane under default, including calibration, on every restart without re-probing.

Generate inventories using the existing inventory commands, then run
`python -I kit/trace_operator.py dag-plan --nodes 1 --trace trace.json --deletion deletion.json
--a2 a2.json` (one shell line; 1/2/4/8 nodes). It only prints sbatch commands.
It preserves every inventory edge and its declared type, including A2 price,
afterany collection and probe finalizer edges. Slurm packs queued jobs onto the
available nodes. Node counts affect only comment estimates assuming equal row
durations. `--node-count` is an alias for `--nodes`. Long-chain ordering hints
appear only in comments. Rolling-window utilisation is not guaranteed.
Every printed row explicitly uses `--kill-on-invalid-dep=yes`, so impossible
dependencies cancel and afterany collection and finalizers can finish.
Only that dependency-policy option is accepted in `dependency_options`;
resource, dependency, wrapper and conflicting policy overrides are rejected
before any commands are printed. CPU counts must be positive integers. GPU
counts are positive integers for GPU rows and zero for CPU rows. Time limits
accept D-HH:MM:SS, HH:MM:SS, MM:SS or minutes. Scalar arguments are shell-quoted.
Identical policy duplicates collapse.

Calibration assembly uses `calibration.schema: trace-calibration-assembly.v1`,
with bound `selection`, `authored`, `sidecar` and `self_instruct` file references.
The node verifies its bound open panels, rejoins authored answers and the approved
Self-Instruct human outputs, and seals locally. No source prompts or full sealed
rows belong in kit exports or return archives. The public node assembly command
is `python -I kit/trace_calibration_build.py node --config node-inputs.json
--out <local-directory>` (one shell line). The operator README records exact
source bindings and the manager reconstruction command. Blinded packets contain
IDs and requests. Assembly uses local CPU files.

Locally assembled calibration: 90 pairs; SHA-256 for the queued message:
`b4713433a390203a266b7b9c4e1b6a34fc5a7df5f2c0ea12086886426c054de5`.

Survey opportunities are conditional guesses, overlap, and are not forecasts:
CPU subset exports 58–288 GPU-h; merge replay reuse 1.1–5.3 per four merges;
S8 hash overlap 1.6–6.4; CPU data prep 7–22 and token prep 0.5–2.5;
local staging 7.58*g GPU-h per comparable stage (13.5x is another workload);
completed target recovery 0.017–0.067 per retry; scheduling about 2*g GPU-h
of avoidable historical allocation gaps. Probe: no measured saving; the survey's
37.5–150 GPU-h at 2x is an unvalidated scenario, not a forecast (judging is already
batched). Timing and calibration wiring themselves claim zero savings. Actual
merge/prep/hash/startup times, TRACE/deletion local-disk speedups, whole-campaign
judge agreement/speed and rolling-window utilisation remain unmeasured on GPUs.


CPU validation passed 288 tests, including CR closure and killed-scorer recovery.
The default dry run kept 43 deletion rows, 25 TRACE rows and 284,868 simulated
GPU-seconds on both 1 and 4 nodes. All 1,268 scientific score/judge JSON files
per layout matched the prior rehearsal byte for byte. The expanded TRACE run
passed 188 rows with local staging and all 24 endpoint values identical to the
default run. These are stand-in results, not measurements of GPU performance.
