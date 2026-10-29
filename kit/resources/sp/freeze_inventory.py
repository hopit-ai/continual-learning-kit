#!/usr/bin/env python3
"""Stage pinned exclusion data outside the repository; run with python -I.

All arguments are local data locations. Never import or execute archive content.
The caller extracts return archives with tarfile filter='data' into fresh empty
folders. Archive response fields are inspected only by duplicate-checking code.
This inventory deliberately remains incomplete when manifest-listed files or
programme ancestry are absent; having one file per role is not completeness.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from kit import sp_prepare as sp

PINS = {
    'rlvr': ('allenai/RLVR-IFeval', '47c03c73621c4aab2b824b7818681117d662770e',
             '75f1b1f63039044034fe75d22f51f77f257a79291d0084262bb43695d4aac971'),
    'gsm': ('openai/gsm8k', '740312add88f781978c0658806c59bc2815b9866',
            'ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d'),
    'arc': ('allenai/ai2_arc', '210d026faf9955653af8916fad021475a3f00453',
            'e488c1587ffdcfc8443f916c53488a95cd471c5790e0746c6bfe4cecf20962cb'),
}


def stage(a):
    import pyarrow.parquet as pq
    from kit.next_pools import candidates
    from kit.c_targets import source_rows
    from kit.beds import finqa
    root = a.cache.resolve()
    if root.is_relative_to(sp.KIT.parent.resolve()):
        raise ValueError('data cache must be outside the git tree')
    root.mkdir(parents=True, exist_ok=True)
    source_dir = root / 'public'; source_dir.mkdir(exist_ok=True)
    records = []
    sources = {}
    for key, (repo, revision, expected) in PINS.items():
        path = getattr(a, key)
        assert sp.sha256_file(path) == expected, key + ' pin mismatch'
        dest = source_dir / (key + '.parquet'); shutil.copyfile(path, dest)
        sources[key] = pq.read_table(dest).to_pylist()
        records.append(dict(name=key, repository=repo, revision=revision, sha256=expected, rows=len(sources[key])))
    contract_pin = json.loads((sp.KIT/'resources/c/contract_nli/manifest.json').read_text())['source']
    legal_dir = source_dir / 'contract'; legal_dir.mkdir(exist_ok=True)
    for name, expected in contract_pin['source_files_sha256'].items():
        assert sp.sha256_file(a.contract/name) == expected, 'ContractNLI pin mismatch'
        shutil.copyfile(a.contract/name, legal_dir/name)
        records.append(dict(name='contract-'+name.removesuffix('.json'), repository='ContractNLI',
                            sha256=expected, archive_sha256=contract_pin['archive_sha256']))
    legal = source_rows(legal_dir, False)
    pools = []; specs = []; provenance = []
    def add(name, role, rows, origin, complete=True):
        # Only text/ID fields are shipped into the exclusion cache. In
        # particular, private runtime metadata never enters review artifacts.
        clean = [dict(id=str(r.get('id', i)), **{k:r[k] for k in sp.TEXT_FIELDS if k in r}) for i,r in enumerate(rows)]
        assert clean and all(sp.prompt_text(r) for r in clean)
        path = root/(name+'.jsonl'); sp.write_jsonl(path, clean)
        scope = 'frozen-reference' if role in ('panels','generic-accepted') else 'candidate-superset'
        sha = sp.sha256_file(path)
        specs.append(dict(name=name,role=role,path=path.name,sha256=sha,scope=scope))
        pools.append(dict(name=name,role=role,replay=role in sp.REPLAY_ROLES,rows=clean,sha256=sha,scope=scope))
        provenance.append(dict(name=name,role=role,rows=len(clean),sha256=sha,origin=origin,complete=complete))
    instruction = [dict(id='rlvr-'+str(i), prompt=r['messages']) for i,r in enumerate(sources['rlvr'])]
    add('rlvr-candidates','generic-candidates',instruction,'Pinned entire RLVR-IFeval training split, before any acceptance or exclusion.')
    broad = candidates(sources['gsm'], sources['arc'])
    add('broad-candidates','broad',broad,'Entire pinned GSM8K and ARC-Challenge training splits under current strict prompts.')
    add('contract-candidates','cr',legal['train'],'Entire pinned ContractNLI training split rendered by c_targets.source_rows; no acceptance filter.')
    # K shares the frozen generic/broad replay inputs. Their entire public
    # sources are conservative prompt supersets, independently of accepted rows.
    add('k-instruction-candidates','k',instruction,'Same full instruction source superset as generic replay.')
    add('k-broad-candidates','k',broad,'Same full broad source superset; future returned prompts must verify containment.')
    add('cr-instruction-candidates','cr',instruction,'CR instruction component: entire source superset.')
    add('cr-broad-candidates','cr',broad,'CR broad component: entire source superset.')
    # Verify the public FinQA bytes against its already-local pinned Git commit.
    for split in ('train','dev','test'):
        name=split+'.json'
        pinned=subprocess.check_output(['git','-C',str(a.finqa.parent),'show',finqa.DATASET_COMMIT+':dataset/'+name])
        assert (a.finqa/name).read_bytes()==pinned, 'FinQA source differs from pinned commit'
        records.append(dict(name='finqa-'+split,repository='czyssrs/FinQA',revision=finqa.DATASET_COMMIT,
                            sha256=hashlib.sha256(pinned).hexdigest()))
    finqa_train = finqa.load(a.finqa, 'train')
    rendered = [dict(id=r['id'],prompt=finqa.render_prompt(r)) for r in finqa_train]
    add('finqa-source-reference','earlier',rendered,'All source FinQA train content, rendered by the bed; exact B trainer/teacher artifacts missing.',False)
    add('cr-finqa-source-reference','cr',rendered,'Conservative content reference only; exact frozen B training prompts still required.',False)
    for split in ('dev','test'):
        rows = finqa.load(a.finqa,split)
        add('finqa-'+split,'panels',[dict(id=r['id'],prompt=finqa.render_prompt(r)) for r in rows],
            'Full source FinQA '+split+' content; exact programme trainer render still requires binding.',False)
    for name, path in [('general-v1',sp.KIT/'panels/general-v1.jsonl'),
                       ('legal-panel',sp.KIT/'resources/c/contract_nli/panel.jsonl'),
                       ('legal-test',sp.KIT/'resources/c/contract_nli/test.jsonl'),
                       ('hard-math-eval',sp.KIT/'resources/m2/eval.jsonl'),
                       ('math-screen',sp.KIT/'resources/math-hard-teacher/screen_panel.jsonl')]:
        add(name,'panels',sp.read_jsonl(path),'Versioned local panel; bytes bound in this receipt.')
    add('hard-math-candidates','ancestry',sp.read_jsonl(sp.KIT/'resources/m2/train_pool.jsonl'),
        'Versioned complete 500-row hard-MATH training pool; only one component of ancestry.')
    archive_records=[]
    for name, relative, role, manifest in [
        ('return-broad-candidates','pools/candidates.jsonl','broad','pools/manifest.json'),
        ('return-gsm-accepted','pools/gsm8k.jsonl','broad','pools/manifest.json'),
        ('return-arc-accepted','pools/arc.jsonl','broad','pools/manifest.json'),
        ('return-refresher','replay/refresher-64.jsonl','generic-accepted','replay/manifest.json'),
        ('return-math-targets','targets/pool.jsonl','ancestry','targets/manifest.json'),
        ('hellaswag','hellaswag/panel.jsonl','panels','hellaswag/manifest.json')]:
        path=a.return_next/relative; doc=json.loads((a.return_next/manifest).read_text())
        expected=doc['files'].get(path.name)
        assert expected and sp.sha256_file(path)==expected, name+' returned manifest mismatch'
        add(name,role,sp.read_jsonl(path),'Safely extracted return data, verified against accompanying manifest.',name!='return-refresher')
        archive_records.append(dict(name=name,sha256=expected,manifest_sha256=sp.sha256_file(a.return_next/manifest)))
    replay_manifest=json.loads((a.return_next/'replay/manifest.json').read_text())
    missing_replay=[dict(name=n,expected_sha256=replay_manifest['files'][n]) for n in
                    ('candidates.jsonl','accepted.jsonl','replay.jsonl') if not (a.return_next/'replay'/n).is_file()]
    blockers = [
        'Returned generic replay manifest lists files absent from the archive: '+', '.join(x['name'] for x in missing_replay),
        'Exact frozen B FinQA training prompts and accepted teacher responses are absent; full public FinQA content does not close this binding.',
        'Complete programme ancestry and heldout/panel inventory (including Package-A inputs and custodian exclusions) is not available.',
        'Returned ContractNLI prepared rows and accepted teacher journal are absent; the pinned full public prompt superset does not substitute for accepted responses.',
    ]
    spec=dict(schema='sp-pools-v2',complete_inventory=False,frozen_before_scoring=True,pools=specs)
    sp.write_json(root/'pools.json',spec)
    earlier=json.loads((a.return_next/'pools/inputs.json').read_text())['earlier_receipts']
    earlier_hashes=sorted({(x['manifest'],x['train']) for x in earlier.values()})
    receipt=dict(schema='sp-real-pool-inventory-v1',status='PENDING',inventory_complete=False,
                 public_sources=records,pools=provenance,returned_files=archive_records,
                 missing_replay_files=missing_replay,blockers=blockers,
                 missing_finqa_target_bindings=[dict(manifest_sha256=m,train_sha256=t) for m,t in earlier_hashes],
                 isolation='Archive data parsed under python -I; no archive code executed.',
                 model_inference=False,source_mutation=False)
    sp.write_json(root/'inventory.json',receipt)
    print('PASS: pinned public sources: RLVR-IFeval %d; GSM8K %d; ARC-Challenge %d; ContractNLI train %d.' %
          (len(instruction),len(sources['gsm']),len(sources['arc']),len(legal['train'])),flush=True)
    print('PASS: %d available exclusion pools staged; %d rows; inventory explicitly incomplete.' %
          (len(pools),sum(len(p['rows']) for p in pools)),flush=True)
    return pools,receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','rlvr','gsm','arc','contract','finqa','return-next'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--audit',action='store_true')
    a=p.parse_args();pools,receipt=stage(a)
    if a.audit:
        rows,_=sp.load_source(); ck=sp.load_checkers()
        report=sp.audit(rows,pools,ck)
        report.update(inventory_complete=False,inventory_blockers=receipt['blockers'])
        if report['status']=='CLEAN': report['status']='PENDING'
        sp.write_json(a.cache/'audit.json',report)
        print('AUDIT: %d hits; %d rejected pairs; %d five-nearest lists; status %s.' %
              (len(report['hits']),len(report['rejected_request_ids']),len(report['nearest']),report['status']),flush=True)


if __name__=='__main__': main()
