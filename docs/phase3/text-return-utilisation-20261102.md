# Utilisation in text returns

The CPU reporter accepts an optional `utilisation` array in its local input map:

```json
{"utilisation": [
  {"family": "core", "roots": ["<core-root>/execution/utilisation"]},
  {"family": "KCR", "roots": ["<KCR-root>/state/utilisation"]},
  {"family": "PD", "roots": ["<PD-root>/seed-101/utilisation"]},
  {"family": "SP", "roots": ["<SP-root>/state/utilisation"]},
  {"family": "preflight", "roots": ["<control-root>/preflight/utilisation"]},
  {"family": "REF", "summaries": ["<reference-attempt>.summary.json"]}
]}
```

Roots are dedicated watcher directories, searched recursively. Include all
attempts, including failures and restarts, exactly once. Duplicate paths are
refused, even across families. Explicit `summaries` paths also work. A JSONL
without its completed summary makes that family's telemetry unknown. Report
from the hosts holding the receipts or copy the small telemetry files to a CPU
reporting job; no models or GPUs are needed. Do not include these local paths,
UUIDs, phase files or scheduler identities in pasted text.

The fixed public families are M, X, S, KCR, PD, SP, 4A, core, preflight and REF.
Use core for mixed M/R/RA/S/X wrapper receipts, or assign disjoint known attempt
sets to M/X/S. Every configured/absent family has one compact row in Part 3:

```text
T|UTIL|family|jobs|gpus|mean_util|worst_window|seconds_below_70
T|UTIL|KCR|3|10|78.52|51.01|22.1
```

`jobs` counts observed attempt summaries (or incomplete JSONL attempts), not
registered scientific tasks. `gpus` sums allocated GPU slots across those
attempts, not unique UUIDs or simultaneous demand. `mean_util` is weighted by
allocated GPU-seconds. `worst_window` is the minimum rolling fleet mean among
GPU attempts, including partial startup windows. `seconds_below_70` sums
job-seconds when a rolling mean is below 70, integrating crossings between
samples. Windows use sample-and-hold activity, not exact device execution times.
All text numbers use the existing four significant figures. CPU attempts count
as jobs with zero GPUs and null activity; they do not dilute GPU means.
An absent, invalid or incomplete GPU summary makes family measurements null.
No telemetry and no observed attempts gives jobs=0 and null measurements.

The watcher writes per-GPU `{t,gpu,util,mem}` samples (Unix seconds, UUID,
activity percent and used MiB), an adjacent `.summary.json`, and optional
`.phases.jsonl`. It freezes the allocated UUID set, checks CUDA/Slurm scope and
the Slurm GPU count when provided, and refuses to claim complete measurements
after missing samples, ambiguous identities or NVML errors. Explicitly empty
CUDA visibility means zero GPUs and never queries host GPUs. Without CUDA
visibility it uses Slurm GPU identifiers, NVIDIA visibility or the allocation's
device cgroup. The scheduler must provide a scoped GPU set/device cgroup;
unrestricted host visibility cannot establish allocation identity. MIG UUIDs
that the GPU query cannot resolve remain unknown.

Defaults: 5-second samples, 300-second rolling window, warning below 75 percent,
at most one low-activity warning per window. Warnings never kill/rebatch work.
TERM/INT/HUP/QUIT go to the complete child process group. Normal child status
passes through; signal exits use shell convention 128+signal. Every invocation
requires a fresh output path. Wrappers create one path per attempt and suppress
nested K/CR watchers under the combined launcher. SIGKILL or host failure may
leave an incomplete JSONL without a summary; retain it for the reporter.

Markers cover coarse preparation, training, export and scoring boundaries in
the M/X/KCR and reference drivers. They are optional best-effort append writes;
unset `UTIL_PHASE_FILE` makes `mark(name)` a no-op. No callback, scorer batching,
model load, optimizer update, numerical setting or scientific rule is changed.
PD/SP drivers and distributed trainer internals remain outside this unit's
marker scope. Model-load/rollout/update detail still needs measured qualification.

ESSENTIAL keeps the existing Parts 1/2/3 semantics and at most one fourth
overflow part; telemetry is never grounds to emit a fifth part. FULL retains
the same utilisation rows. The importer preserves them through arbitrary chat
wraps and later FULL supplements, rejects conflicting/duplicate family rows,
and labels all measurements PARTNER-COMPUTED with independent re-verification
UNVERIFIED. Existing scientific decisions and allowance gates are unchanged.
Right-sizing and monitoring alone do not prove rolling utilisation above 70.
