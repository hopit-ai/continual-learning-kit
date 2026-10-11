# Manager-side phase-0 ingest

Run in the manager's CPU environment, with reader dependencies already installed.
No downloads, model loading, allocation or partner operation is added. The tool
is excluded from the partner import-smoke inventory and public test gate.

```sh
python kit/v4_finqa_ingest.py --archive /path/phase0-return.tar.gz \
  --local-diff /path/partner-kit.diff --out /path/new-main-candidate
```

The real phase-0 v7 reader must return a technical pass. Otherwise the command
refuses with its exact reason and produces no campaign. The input archive is
read only; supplied archived kit code and local diffs are never executed. The
hash-pinned scorer is the existing reader dependency. Existing output directories
are refused to preserve evidence; choose a new report directory.

A technical pass invokes the existing C1 hook, validates both generated manifests
with the real runner, carries the closed phase-0 spend into `work/`, and writes
`report.md` and `report.json`. Candidate manifests remain under `campaigns/` for
review and staging. This command does not install or tag the package.

The report follows owner plan 15.21/15.22:

- **TAG BLOCKERS:** phase-0 technical pass, valid generated graph and immutable
  model/data identity. Local patch/source correspondence is recorded as a WARNING,
  with SHA-256 and touched files; a recorded tag does not certify clean bytes.
- **MEASURED NOW:** all phase-0 measurements against qualification/main caps,
  with measured value, extrapolation, cap, margin and PASS/TIGHT/FAIL. These
  labels assess estimates and never prohibit a run. Generation reports both
  the observed whole-row throughput extrapolation and a conservative four-attempt
  serial bound; the full qualification corpus is 1,441 questions too. Full-wave
  shard throughput remains unmeasured.
- **DEFERRED TO QUALIFICATION:** D-R, H/general scoring, eight-trajectory probes,
  full-wave generation, full-state IO, main disk peak and missing timing keys.
  Their absence is expected and never labelled FAIL.

The ceiling is **1,000 GPU-hours actually spent**, including phase-0 carry and
failed/idle allocations. The planned 100-hour qualification block, partial measured
projection (1.5 allowance and 45-hour contingency), and static cap envelopes are
advice. Projection overruns produce WARNING, never a fit refusal. Qualification
runs first and main follows automatically without owner proceed or a PAUSE.

The fit table shows the 720-minute planning example, clipping only by a captured
site window, and which rows would need continuation if all earlier generous caps
were consumed. These are conditional warnings, not row-start projection gates.
Runtime deadlines use the paid allocation window and actual remaining ceiling.
Prepared-byte verification uses `10 * max(60, ceil(1.5 * measured_seconds))`;
measurements themselves are never multiplied.

The **1,879 GiB** disk figure is capacity advice, printed at prepare. Actual free
space is checked against what the next save/export writes. Disk usage and free
space are distinct; compressed archive size is never a main peak measurement.
H100 memory comparisons use nominal 80 GB converted to GiB; real capacity and
40-update state peaks are qualification observations.

Margins are cap / extrapolated requirement: PASS >=1.25, TIGHT >=1 and <1.25,
FAIL <1. The manager status is GO WITH WARNINGS when only observational findings
remain. Exit 0 means the report/candidates were produced; exit 2 retains the exact
technical ingest failure. CPU simulator archives remain technical refusals because
the real reader cannot certify stand-ins. Manager-only synthetic raw-evidence tests
exercise the reader and pricing paths without claiming hardware results.
