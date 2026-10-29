#!/usr/bin/env python3
"""Freeze disjoint hashed partitions, shared across all incoming checkpoints."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.s_common import *


def job(a):
    context(a.out, a.standin)
    audit = read(a.out/'audit.json')
    files = {}
    for domain, proof in audit['domains'].items():
        values = proof['eligible'] if proof['passed'] else []
        for part in ('monitor', 'train'):
            selected = [r for i, r in enumerate(values, 1) if (i % 3 == 0) == (part == 'monitor')]
            path = a.out/'frozen'/f'{domain}-{part}.jsonl'
            write_rows(path, selected)
            files[path.name] = sha(path)
    bind(a.out/'frozen/manifest.json', dict(files=files, salt='screen-s-v1', audit_sha256=sha(a.out/'audit.json'),
        binding_sha256=sha(a.out/'binding.json'), technical_synthetic=a.standin))
    print('Frozen: every third audited row monitors; monitoring never trains')


def partitions(out, domain):
    manifest = read(out/'frozen/manifest.json')
    require(manifest['audit_sha256'] == sha(out/'audit.json'), 'audit changed')
    require(manifest['binding_sha256'] == sha(out/'binding.json'), 'partition binding changed')
    result = {}
    for part in ('train', 'monitor'):
        name = f'{domain}-{part}.jsonl'
        require(sha(out/'frozen'/name) == manifest['files'][name], 'partition changed')
        result[part] = rows(out/'frozen'/name)
    return result

if __name__ == '__main__':
    raise SystemExit(entry(job, lambda p: None))
