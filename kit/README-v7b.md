# V7b operator follow-up

For the partner's agent, apply these updates with the existing v7 launch flow.

- OPR generation runs after its scientific qualification succeeds, including
  when utilisation telemetry is missing, incomplete or below 70%. Each generation
  result records the qualification's measured mean, rolling minimum and telemetry
  completeness. The family reading and text return carry these values and `W|`
  advisory lines.
- A one-GPU row passes `--holder-slice` only when `UTIL_HOLDER_SLICE=1` or
  `SLURM_GPUS_ON_NODE`, `SLURM_JOB_GPUS` or `SLURM_STEP_GPUS` reports more than
  one GPU. Plain one-GPU allocations use the non-holder path. Larger and CPU
  rows clear inherited holder-slice settings. Scope failures preserve the child's exit status.
- The combined wrapper runs its child with `UTIL_WATCH_ACTIVE=1` if utilisation
  directory or start-record setup fails. It reports the monitoring failure on
  stderr and launches the child once.
- Scorer regression fixtures use `unittest.mock.patch.dict(os.environ)` to keep
  Python and subprocess environment changes synchronized and restore test state.
- Public exports include `scripts/cr_confirmation_fixture.py`, native FinQA
  regression tests and the runnable 78-row confirmation rehearsal. Install
  `pytest`, `pyyaml`, `pyarrow` and `pandas` in the CPU test environment, then run
  the command below with a fresh output directory.
- The confirmation registration ships byte-for-byte under
  `resources/nm/registrations/`. The archived frozen-gate checklist and feasibility
  receipt ship at their linked paths. The 4A budget link uses its shipped
  registration. The publisher validates relative file links in the root README
  and `kit/*.md` operator runbooks before running exported tests. Frozen resource
  copies retain their original research-tree citations.

The shipped registration copies are frozen originals: their internal links point into the private research tree; where the CR registration says unqualified lanes are PENDING, [README-next-message.md](README-next-message.md)'s warn-only rule governs.

```bash
python scripts/cr_confirmation_dryrun.py --runnable --out /tmp/cr-v7b-runnable
```

The rehearsal returns TECHNICAL-ONLY evidence for all 78 confirmation rows and
checks the recorded FinQA schedule reproduction.
- A failed CR bind settings check writes its error attempt under
  `<dispatch out>/cr-input-errors`, never into the confirmation tree it may have
  refused. A host-policy refusal is FAILED with a receipt; when it is found at
  the later input check, `input-binding.json` still says PENDING for the owner.
