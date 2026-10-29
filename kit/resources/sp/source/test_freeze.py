"""Owned CPU tests for the write-once freeze and node-only verification ruling."""
from pathlib import Path
import copy
import importlib.util
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from kit import sp_prepare as sp

spec = importlib.util.spec_from_file_location('sp_node_test', sp.RESOURCES/'node_verification.py')
node = importlib.util.module_from_spec(spec); spec.loader.exec_module(node)


def fixture(tmp_path):
    requests, _ = sp.load_source()
    requests = requests[:3]
    required = [dict(inventory_id='node-reference',role='ancestry',format='jsonl',sha256=None)]
    policy = dict(unavailable_inventories=required,rules=dict(shingle=5,jaccard=.8,edit=.9))
    doc = dict(node_verification=policy)
    data = tmp_path/'data.jsonl'
    sp.write_jsonl(data,[dict(id='candidate',prompt='Describe an unusual prism-shaped greenhouse.',response='A uniquely colored design with mirrored walls.')])
    manifest = tmp_path/'node.json'
    value = dict(schema='sp-node-inventories-v1',complete_inventory=True,ancestry_closed=True,
                 policy_sha256=sp.digest(policy),inventories=[dict(inventory_id='node-reference',format='jsonl',path=data.name,sha256=sp.sha256_file(data))])
    sp.write_json(manifest,value)
    return requests,doc,data,manifest,value


def test_node_clean_hit_and_missing_are_pair_local(tmp_path):
    rows,doc,data,path,value=fixture(tmp_path);ck=sp.load_checkers();before=sp.digest(rows)
    clean=node.verify(doc,path,rows,ck)
    assert clean['status']=='CLEAN' and clean['scoring_allowed']
    assert set(clean['pair_status'].values())=={'VERIFIED'}
    for field,text in [('prompt',rows[0]['content']),('response',rows[0]['witness']),
                       ('prompt',rows[0]['paraphrase']['clauses'][0])]:
        sp.write_jsonl(data,[dict(id='candidate',**{field:text})])
        value['inventories'][0]['sha256']=sp.sha256_file(data);sp.write_json(path,value)
        result=node.verify(doc,path,rows,ck)
        assert result['status']=='UNVERIFIED' and result['verification_complete'] and result['scoring_allowed']
        assert result['affected_request_ids']==[rows[0]['request_id']]
        assert result['pair_status'][rows[0]['request_id']]=='UNVERIFIED'
        assert all(result['pair_status'][r['request_id']]=='VERIFIED' for r in rows[1:])
        assert len(result['unverified_prompt_ids'])==2
        assert not result['panel_changed'] and not result['rescoring_allowed'] and result['deletes']==0
        assert sp.digest(rows)==before
    value['inventories']=[];sp.write_json(path,value)
    missing=node.verify(doc,path,rows,ck)
    assert not missing['scoring_allowed'] and not missing['verification_complete']
    assert missing['affected_request_ids']==[] and len(missing['unverified_request_ids'])==3


def test_fixed_hash_cannot_be_redeclared_and_journal_must_be_complete(tmp_path):
    rows,doc,data,path,value=fixture(tmp_path);ck=sp.load_checkers()
    doc['node_verification']['unavailable_inventories'][0]['sha256']=sp.sha256_file(data)
    value['policy_sha256']=sp.digest(doc['node_verification'])
    sp.write_jsonl(data,[dict(id='changed',prompt='Another prompt about a blue lighthouse.')])
    value['inventories'][0]['sha256']=sp.sha256_file(data);sp.write_json(path,value)
    assert not node.verify(doc,path,rows,ck)['scoring_allowed']
    raw=tmp_path/'teacher/raw';raw.mkdir(parents=True)
    file=raw/'attempt.json';sp.write_json(file,dict(request=dict(id='item',messages=[dict(role='user',content='An unrelated question.')]),text=rows[0]['witness']))
    producer=tmp_path/'producer.json';sp.write_json(producer,dict(files={'teacher/raw/attempt.json':sp.sha256_file(file)}))
    requirement=dict(inventory_id='journal',role='cr',format='journal',sha256=None)
    entry=dict(inventory_id='journal',format='journal',path=producer.name,sha256=sp.sha256_file(producer))
    loaded=node._load(entry,tmp_path,requirement)
    assert loaded['rows'][0]['response']==rows[0]['witness'] and loaded['rows'][0]['prompt']
    sp.write_json(raw/'omitted.json',dict(text='Not in the manifest'))
    try:node._load(entry,tmp_path,requirement)
    except ValueError:pass
    else:raise AssertionError('omitted journal file accepted')


def test_preflight_runs_before_command_and_annotations_preserve_rows(tmp_path,monkeypatch):
    rows,doc,data,path,value=fixture(tmp_path);ck=sp.load_checkers();events=[]
    result=node.verify(doc,path,rows,ck)
    def recheck(*args):
        events.append('verify');return result
    monkeypatch.setattr(sp,'recheck',recheck)
    def runner(command,check):
        events.append('score');return type('Exit',(),{'returncode':0})()
    assert node.run_command(tmp_path,path,tmp_path/'receipt.json',['score'],runner)==0
    assert events==['verify','score']
    result['scoring_allowed']=False;events.clear()
    assert node.run_command(tmp_path,path,tmp_path/'receipt.json',['score'],runner)==2
    assert events==['verify']
    scores=[dict(request_id=r['request_id'],id=r['request_id']+':canonical',strict=i%2) for i,r in enumerate(rows)]
    original=copy.deepcopy(scores);result['pair_status'][rows[0]['request_id']]='UNVERIFIED'
    annotated=node.annotate(scores,result)
    assert scores==original and len(annotated)==3
    assert [{k:v for k,v in r.items() if k!='verification_status'} for r in annotated]==scores
    assert [r['verification_status'] for r in annotated]==['UNVERIFIED','VERIFIED','VERIFIED']


def test_real_available_freeze_is_write_once(tmp_path):
    """Validate the shipped scientific freeze after the production command runs."""
    if not (sp.RESOURCES/'FROZEN.json').exists():
        # Pre-freeze runs still check authorization is explicit and evidence pins exist.
        ruling=json.loads((sp.RESOURCES/'freeze-ruling.json').read_text())
        assert ruling['required_before_scoring'] and len(ruling['unavailable_inventories'])==12
        assert sp.sha256_file(sp.RESOURCES/'source/requests.jsonl')==ruling['source_sha256']
        return
    frozen=sp.verify_frozen()
    assert frozen['status']=='FROZEN' and frozen['n_requests']==200 and frozen['n_prompts']==400
    assert len(frozen['pools'])==23 and sum(p['rows'] for p in frozen['pools'])==98273
    assert frozen['node_verification']['verification_only'] and frozen['node_verification']['pair_local_hits']
    original=(sp.RESOURCES/'FROZEN.json').read_bytes()
    # The protected receipt cannot be recreated and all shipped files verify.
    try:
        with (sp.RESOURCES/'FROZEN.json').open('x'):pass
    except FileExistsError:pass
    else:raise AssertionError('write-once receipt was overwritten')
    assert (sp.RESOURCES/'FROZEN.json').read_bytes()==original


def test_existing_manifest_hook_uses_node_policy_and_binds_freeze(tmp_path):
    from kit import sp_manifest
    rows,doc,data,path,value=fixture(tmp_path)
    frozen=tmp_path/'frozen';frozen.mkdir()
    sp.write_jsonl(frozen/'requests.jsonl',rows)
    doc.update(schema=sp.SCHEMA,files={'requests.jsonl':sp.sha256_file(frozen/'requests.jsonl')},
               resources=sp.resource_hashes(),code=sp.code_hashes(),checker_sha256=sp.sha256_file(sp.CHECKERS),
               node_verification_implementation_sha256=sp.sha256_file(sp.RESOURCES/'node_verification.py'))
    sp.write_json(frozen/'FROZEN.json',doc)
    value['frozen_sha256']=sp.sha256_file(frozen/'FROZEN.json')
    sp.write_jsonl(data,[dict(id='overlap',prompt='An unrelated candidate about a bell tower.',response=rows[0]['witness'])])
    value['inventories'][0]['sha256']=sp.sha256_file(data);sp.write_json(path,value)
    out=tmp_path/'verification.json'
    assert sp_manifest.main(['verify-pools','--frozen',str(frozen),'--pools-spec',str(path),'--out',str(out)])==0
    result=json.loads(out.read_text())
    assert result['affected_request_ids']==[rows[0]['request_id']] and result['scoring_allowed']
    assert result['frozen_sha256']==value['frozen_sha256'] and result['checked_before_scoring']
    value['frozen_sha256']='0'*64;sp.write_json(path,value)
    assert not sp.recheck(frozen,path)['scoring_allowed']
