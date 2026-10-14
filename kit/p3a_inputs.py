#!/usr/bin/env python3
"""CPU job: download pinned sources, check hashes, rebuild the joint freeze."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.p3a_common import *


def source_identity(root, files):
    for name, meta in files.items():
        check(root / name, meta['sha256'])
        if (root / name).stat().st_size != meta['bytes']:
            raise CorrectnessStop('wrong source size: ' + name)


def job(a):
    """Prepare both datasets before any inference. --sources permits offline reuse."""
    registration()
    from kit import chemistry_shortcut_run as chem
    target = a.out / '3a/report-inputs'
    if not target.exists():
        staging = a.out / '3a-panels.tmp'
        shutil.copytree(chem.DEFAULT, staging, dirs_exist_ok=True)
        target.parent.mkdir(parents=True, exist_ok=True); staging.rename(target)
    from kit.chemistry_shortcut_report import EXPECTED_MANIFEST_SHA256
    check(chem.DEFAULT / 'manifest.json', EXPECTED_MANIFEST_SHA256)
    manifest = json.loads((chem.DEFAULT / 'manifest.json').read_text())
    check(target / 'manifest.json', s.file_hash(chem.DEFAULT / 'manifest.json'))
    for name, expected in manifest['files'].items(): check(target / name, expected)
    for name, expected in manifest['code'].items(): check(KIT.parent / name, expected)
    dest = a.out / INPUTS; dest.mkdir(parents=True, exist_ok=True)
    step1 = (a.sources if a.standin else RESOURCES) / 'code-step1.json'
    if (dest / 'prepared/frozen.json').exists() and (dest / 'frozen-annex.json').exists():
        # A crash may have stopped promotion before the last copies; finish them.
        shutil.copyfile(step1, dest / 'code-step1.json')
        shutil.copyfile(RESOURCES / 'preregistration.md', dest / 'preregistration.md')
        frozen(a.out, a.standin); print('SKIP: complete joint freeze'); return
    if a.standin:
        source = a.sources
        seal = json.loads((source / 'frozen-annex.json').read_text())
        if not seal['manifest'].get('technical_synthetic'): raise CorrectnessStop('stand-in fixture required')
        shutil.copytree(source / 'prepared', dest / 'prepared', dirs_exist_ok=True)
        shutil.copyfile(source / 'frozen-annex.json', dest / 'frozen-annex.json')
    else:
        doc = registration(); sources = a.sources or a.out / 'sources'
        for candidate, meta in doc['sources'].items():
            root = sources / candidate
            for name in meta['files']:
                path = root / name
                if not path.exists():
                    if a.sources: raise CorrectnessStop('missing input: ' + str(path))
                    from huggingface_hub import hf_hub_download
                    cached = hf_hub_download(repo_id=meta['repo'], repo_type='dataset', revision=meta['revision'], filename=name, cache_dir=a.cache)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pending = path.with_name(path.name + '.tmp')
                    shutil.copyfile(cached, pending); pending.replace(path)
            source_identity(root, meta['files'])
        mirror = sources / 'math_screen'
        (mirror / 'REVISION').write_text(doc['sources']['math_screen']['revision'] + '\n')
        build = a.out / 'inputs-build'; build.mkdir(exist_ok=True)
        math = build / 'MATH'
        if not (build / 'materialised.json').exists():
            retain_partial(math)
            subprocess.run([sys.executable, str(KIT / 'materialise_math_mirror.py'), '--mirror', str(mirror), '--out', str(math)], check=True)
            atomic(build / 'materialised.json', {'complete': True})
        # The unchanged freezer hashes scripts/ and docs/ in their original layout.
        tree = build / 'source-tree'
        if not tree.exists(): shutil.copytree(KIT, tree / 'kit', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        (tree / 'scripts').mkdir(exist_ok=True)
        shutil.copyfile(implementation_path('scripts/second_task_screen_modal.py'), tree / 'scripts/second_task_screen_modal.py')
        prereg = tree / 'docs/phase3/3c-second-task-screen-preregistration.md'
        prereg.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(RESOURCES / 'preregistration.md', prereg)
        generated = build / 'frozen-annex.json'; prepared = build / 'prepared'
        if not generated.exists():
            retain_partial(prepared)
            subprocess.run([sys.executable, '-m', 'kit.second_task_screen', 'freeze', '--lcb-root', str(sources / 'code'), '--math-root', str(math), '--out', str(prepared), '--annex', str(generated)], cwd=tree, env={**os.environ, 'PYTHONPATH': str(tree), 'CUDA_VISIBLE_DEVICES': ''}, check=True)
        if json.loads(generated.read_text()).get('manifest_sha256') != EXPECTED or generated.read_bytes() != (RESOURCES / 'frozen-annex.json').read_bytes():
            raise CorrectnessStop('rebuilt annex differs')
        if not (dest / 'prepared').exists(): prepared.rename(dest / 'prepared')
        shutil.copyfile(generated, dest / 'frozen-annex.json')
    shutil.copyfile(step1, dest / 'code-step1.json')
    shutil.copyfile(RESOURCES / 'preregistration.md', dest / 'preregistration.md')
    frozen(a.out, a.standin); print('Joint freeze checked')


def configure(p):
    p.add_argument('--sources', type=Path, help='local code/ and math_screen/ sources; synthetic fixture with --standin')
    p.add_argument('--cache', default=os.environ.get('HF_HUB_CACHE'), help='optional Hugging Face cache outside OUT')

if __name__ == '__main__': raise SystemExit(entry(job, configure, 'inputs'))
