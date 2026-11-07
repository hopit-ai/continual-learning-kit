# Commands against one frozen send commit

The manager records the integrated commit and the teacher-batch merge relationship.
The implementation unit does not commit. Run this list from a clean checkout of
that exact commit; do not merge or edit while a check is running. Record `git
rev-parse HEAD`, `git status --porcelain`, Python version, disk/cache availability,
each command, its exit status and full output. The eventual send tag must resolve
to this same commit. A dirty tree cannot produce the final gate receipts.

```sh
git rev-parse HEAD
git status --porcelain
python --version
df -h . /tmp
python -m pytest -q -ra --basetemp "$(mktemp -d /tmp/v4-frozen-suite.XXXXXX)" tests/test_v4_*.py tests/test_g1_*.py \
  tests/test_kit_self_contained.py tests/test_static_undefined_names.py \
  tests/test_kit_sdft.py tests/test_smoke_v4_four_arm.py \
  tests/test_kit_p4_*.py tests/test_kit_runner.py tests/test_kit_collect.py
python scripts/v4_data_audit.py --ship build/v4-g2-round1-ship \
  --dataset-folder build/v4-g2-round1-ship/datasets --source-tree "$PWD" \
  --datasets-manifest docs/phase2/evidence/v4-g2-round1-datasets-manifest.json \
  --out /tmp/v4-frozen-g1
```

G5: on the manager's host with process visibility, run all real runner/collector
and stand-in Slurm failure-path tests above (no sandbox skips accepted as a G5
PASS), then the following explicitly labelled orchestration rehearsals. These
rehearsals substitute operation bodies and do not independently close G5; the
manager's production-entry-point rehearsal and independent Opus/Astra reviews
remain required. Real restoration is closed only by partner qualification.

```sh
for case in success out_of_memory wall_time_kill preempted disk_full coverage D_standalone resume ceiling; do
  python scripts/dryrun_v4_campaign.py --case "$case" --out "/tmp/v4-frozen-g5-$case"
done
```

G6 fresh clone: use a newly owned directory and the same frozen commit, never a
symlink to references. Stage read-only reference data and tokenizer caches through
`V4_REFERENCE_ROOT` and `QWEN3_8B_TOKENIZER`; tests themselves make owned copies.

```sh
frozen_commit=$(git rev-parse HEAD)
fresh_clone=$(mktemp -d /tmp/v4-frozen-g6.XXXXXX)
git clone --no-hardlinks "$PWD" "$fresh_clone/source"
git -C "$fresh_clone/source" checkout --detach "$frozen_commit"
cd "$fresh_clone/source"
python -m pytest -q -ra tests/test_kit_self_contained.py \
  tests/test_static_undefined_names.py tests/test_v4_g0_raw_archives.py \
  tests/test_v4_data_audit.py
```

Partner CPU qualification, before `sbatch`: `python kit/v4_prepare.py --work
"$WORK" --phase qualification`. The bounded CPU import/entry smoke, first SFT
loader batch, resolved configurations, agreement pairs, full-cap waves,
determinism and live restoration must all be present in the returned archive.
Read both verified returns using `kit/v4_read.py qualification ARCHIVE` and
`kit/v4_read.py main ARCHIVE`. A `proceed` reading is separate from owner approval.
