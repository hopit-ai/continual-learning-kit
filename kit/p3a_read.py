#!/usr/bin/env python3
"""Read one partner return into the registered 3A table and both 3C decisions."""
import argparse
import json
import hashlib
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit import chemistry_shortcut_report as chem, second_task_screen as screen
from kit.v4_archive import Archive
from kit.v4_teacher import atomic_json
from kit.p3a_common import INPUTS, OUTPUTS, logging


class LocalEvidence:
    def __init__(self, root):
        self.files = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*')
                      if p.is_file() and not p.is_symlink() and p.name != 'tests.jsonl' and p.suffix in ('.json', '.jsonl', '.md')
                      and not {'sources', 'inputs-build'} & set(p.relative_to(root).parts)}
    def json(self, name): return json.loads(self.files[name])
    def jsonl(self, name): return [json.loads(r) for r in self.files[name].decode().split('\n') if r.strip()]


def read(source, out, *, repeats=10000):
    source, out = Path(source), Path(out)
    evidence = Archive(source) if source.is_file() else LocalEvidence(source)
    try:
        chem_evidence = chem.Evidence(source)
        try: result3a = chem.report(chem_evidence, repeats=repeats)
        finally: chem_evidence.close()
        out3a = out / '3a'; out3a.mkdir(parents=True, exist_ok=True)
        atomic_json(out3a / 'reading.json', result3a)
        (out3a / 'table.md').write_text(chem.markdown(result3a))
        seal = evidence.json(str(INPUTS / 'frozen-annex.json'))
        freeze_hash = screen.digest(seal['manifest'])
        if hashlib.sha256(evidence.files[str(INPUTS / 'preregistration.md')]).hexdigest() != seal['manifest']['preregistration_sha256']:
            raise ValueError('returned 3C design identity differs')
        if freeze_hash != seal['manifest_sha256']:
            raise ValueError('returned 3C annex hash differs')
        decisions, warnings = {}, list(result3a['warnings'])
        for name in screen.ORDER:
            base = INPUTS / 'prepared' / name
            # Tests remain on the node; their frozen hash is in the annex. The
            # reader uses archived verifier results, never generates or resamples.
            for file in ('sample.jsonl', 'train.jsonl', 'heldout.jsonl'):
                raw = evidence.files[str(base / file)]
                if hashlib.sha256(raw).hexdigest() != seal['manifest']['candidates'][name]['files'][file]:
                    raise ValueError('returned 3C data hash differs: ' + name + '/' + file)
            folder = OUTPUTS / name
            path = str(folder / 'decision.json')
            if path not in evidence.files:
                decision = dict(candidate=name, step=2, status='INCOMPLETE', complete=False,
                    selected_for_partner_qualification=False, reason='missing step-2 decision', freeze_sha256=freeze_hash)
            else:
                saved = evidence.json(path)
                if saved.get('freeze_sha256') != freeze_hash:
                    raise ValueError('returned 3C decision input identity differs')
                if saved.get('step2_executed') is False:
                    decision = saved
                else:
                    def rows(relative):
                        key = str(folder / relative)
                        return evidence.jsonl(key) if key in evidence.files else []
                    decision = screen.admission(evidence.jsonl(str(base / 'sample.jsonl')),
                        rows('teacher/demonstrations.jsonl'), rows('rewrite/rewrites.jsonl'),
                        complete=all(str(folder / (phase + '-complete.json')) in evidence.files
                                     for phase in ('teacher', 'rewrite')) and not saved.get('error'),
                        synthetic=saved.get('technical_synthetic', False), pool=evidence.jsonl(str(base / 'train.jsonl')))
                    decision.update(candidate=name, step=2, freeze_sha256=freeze_hash, error=saved.get('error'))
                    for field in ('status', 'complete', 'intersection', 'strata'):
                        if saved.get(field) != decision.get(field): warnings.append('3C saved/recomputed decision differs: ' + name + '/' + field)
            decisions[name] = decision
            destination = out / '3c' / name; destination.mkdir(parents=True, exist_ok=True)
            step1_path = str(INPUTS / (name + '-step1.json'))
            if step1_path in evidence.files:
                step1 = evidence.json(step1_path)
                if (step1.get('candidate') != name or step1.get('step') != 1
                        or step1.get('freeze_sha256') != freeze_hash):
                    raise ValueError('returned 3C step-1 input identity differs: ' + name)
            elif name == 'math_screen':
                skipped = decision.get('step2_executed') is False and decision['status'] == 'SKIPPED'
                step1 = dict(candidate=name, step=1, freeze_sha256=freeze_hash,
                    status='SKIPPED' if skipped else 'INCOMPLETE', complete=skipped,
                    step1_executed=False, reason='MATH step 1 not run: ' + decision.get('reason', 'missing decision'))
            else:
                raise ValueError('returned LCB step-1 input identity missing')
            atomic_json(destination / 'step1-decision.json', step1)
            atomic_json(destination / 'decision.json', decision)
            rows = ['| Stratum | n | Intersection successes | Required |', '|---|---:|---:|---:|']
            rows += [f"| {key} | {row['n']} | {row['successes']} | {row['required']} |"
                     for key, row in decision.get('strata', {}).items()]
            (destination / 'admission-table.md').write_text(
                name + ': **' + decision['status'] + '**\n\n' + decision.get('reason', '') + '\n\n'
                + '\n'.join(rows) + '\n\nIntersection: ' + str(decision.get('intersection', 'not run'))
                + '/100; global requirement 85; every eligible stratum ≥70%.\n')
        chosen = None
        screen_pass = None
        for name in screen.ORDER:
            row = decisions[name]
            if not row['complete']: break
            if row['status'] == 'PASS':
                screen_pass = name
                if row.get('selected_for_partner_qualification'): chosen = name
                break
        result = dict(three_a_complete=result3a['complete'], three_c=decisions,
            screen_pass_candidate=screen_pass, selected_for_partner_qualification=chosen,
            warnings=warnings)
        atomic_json(out / 'reading.json', result)
        return result
    finally:
        if isinstance(evidence, Archive): evidence.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    with logging(a.out, 'read'):
        try: read(a.archive, a.out)
        except ValueError as exc: print('STOP: correctness:', exc); return 2
        # Any other error propagates with its traceback and a non-zero exit.
    return 0
if __name__ == '__main__': raise SystemExit(main())
