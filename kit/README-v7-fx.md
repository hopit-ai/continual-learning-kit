# V7 operator fixes

The partner's approved v6a changes (3a–h) are built into the kit. No overlay or
sitecustomize hook is needed. Scoring formulas, frozen panels, decoding parameters,
seeds and generation caps are unchanged.

- `inference_lifecycle.create_llm` passes the named batch-invariant attention
  backend through the installed `EngineArgs` API (`attention_config` on vLLM
  0.18), and retains `VLLM_ATTENTION_BACKEND` for older versions. All kit LLM
  constructors use it. Batch-invariant mode refuses any resolved backend other
  than `FLASH_ATTN`. Lane engine receipts include the effective named backend.
- `lane_pool.engine_startup` holds a host-local lock keyed by physical GPU UUID
  through the complete constructor. Under that lock it polls measured free memory
  until the requested fraction of total memory is free. Unknown memory is not
  admission. `KIT_ENGINE_START_TIMEOUT_SECONDS` bounds lock/memory admission
  (default 600 s) gives memory admission its full budget after the lock is
  acquired; lock wait has a separate `(K-1)` times that budget. Timeout errors
  distinguish lock contention, measured free/total MiB and unavailable queries.
  The lock directory is fixed under `/tmp`, independent of job `TMPDIR`.
  The controller's existing job timeout still bounds the job.
- Shared vLLM 0.18 engines use the kit's `MemoryProfileWorker`, through the
  supported `worker_cls` API. Only the neighbour-release profiling assertion gets
  one fresh snapshot and re-measure. It retains the original fractional
  reservation and caps the resulting KV budget by current free memory minus
  activation/non-torch growth. Repeated disturbances and other
  errors propagate. Re-measure events are saved under the engine receipt's
  `profiles/` directory.
- Likelihood commands, including a HellaSwag diagnostic followed by likelihood,
  use at most two concurrent lanes at fraction .42. Qualified generation-only
  lanes retain three at .31. The likelihood constructors also enforce this bound.
  M2 and M3 run the HellaSwag diagnostic in a separate process and wait for it
  to exit before the likelihood child starts. The M3 child retains its assignment
  and prepared model identity checks.
- `eval_bed` accepts `math_screen` split `test` as metadata for explicit
  `frozen_items`. Loading it as a dataset split is refused. The M2/M3 hard-MATH
  paths pass frozen items; the older package-A path still uses `heldout`.
- Logging waits five seconds for tee to exit before killing an inherited pipe.
- `util_watch.py --holder-slice --expect-gpus 1` uses the one GPU named by
  `CUDA_VISIBLE_DEVICES`; parent-holder Slurm counts do not widen that scope.
  Empty inventory queries retry with `nvidia-smi --id=<CUDA token>`. Summaries
  retain measured UUIDs and utilisation; missing/ambiguous measurements remain
  unknown. Numeric holder tokens must match the reported index exactly, with
  either a singleton inventory or indexes exactly `0..n-1`. Sparse/renumbered
  inventories require an unambiguous `GPU-` UUID token; the whole-list shortcut
  is disabled in holder mode. The combined, message3 and v6a wrappers enable this for one-GPU rows.
  `UTIL_HOLDER_SLICE=1` is the equivalent environment option.

CPU regressions are in `tests/test_v7_operator_fixes.py`, `test_lane_pool.py`,
`test_util_watch.py` and `test_v6a.py`; the existing scorer replay suite checks
frozen CPU numerical behavior. The installed-vLLM EngineArgs test imports the
real signature but substitutes the constructor, and skips when vLLM is absent.
CPU tests do not establish GPU kernel equality or utilisation on an actual holder.
