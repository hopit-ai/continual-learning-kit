"""Offline CPU source/fixture/protocol validation; no model or real response access."""
from collections import Counter
import copy
import gzip
import itertools
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))
from kit import sp_prepare as prep
from build_source import source_bytes


def must_reject(call):
    try: call()
    except (ValueError, FileNotFoundError, AssertionError): return
    raise AssertionError('expected rejection')


def synthetic_pool(name='synthetic', role='generic-candidates', prompts=None):
    if prompts is None:
        templates = prep.load_templates()
        prompts = [rec['template'] for rec in templates.inventory()['templates'].values()]
    return dict(name=name, role=role, replay=True, sha256='a' * 64, scope='candidate-superset',
                rows=[dict(id='fixture-%d' % i, prompt=p) for i, p in enumerate(prompts)])


def main():
    rows, _ = prep.load_source(ROOT); ck = prep.load_checkers()
    assert (ROOT / 'requests.jsonl').read_bytes() == source_bytes()
    assert not prep.validate_source(rows, ck)
    assert Counter(r['primary_family'] for r in rows) == {f: 20 for f in ck.FAMILIES}
    assert Counter(len(r['constraints']) for r in rows) == {1: 100, 2: 100}
    assert Counter(r['constraints'][1]['family'] for r in rows if len(r['constraints']) == 2) == {f:10 for f in ck.FAMILIES}
    for family in ck.FAMILIES:
        assert Counter(len(r['constraints']) for r in rows if r['primary_family'] == family) == {1:10,2:10}
    assert len({r['request_id'] for r in rows}) == 200
    assert len(prep.panel_rows(rows)) == 400
    for a,b in itertools.combinations(rows,2): assert not prep.near(prep.prepared(a['content']),prep.prepared(b['content']))
    bad=copy.deepcopy(rows[0]); bad['canonical']['clause']+=' changed'
    assert prep.validate_request(bad,ck)
    bad=copy.deepcopy(rows[0]); bad['paraphrase']['prompt']='extra ' + bad['paraphrase']['prompt']
    assert prep.validate_request(bad,ck)
    print('PASS: 200 source-v2 requests; 400 prompts; 20 primary/family; 100 single + 100 dual; exact canonical provenance and unchanged pair content.')

    fixtures=[]
    for row in rows:
        built,errors=prep.fixtures_for(row,ck); assert not errors; fixtures+=built
    assert len(fixtures)==500
    assert Counter(f['kind'] for f in fixtures)=={'witness':200,'mutation':300}
    assert all(f['expected'].count(False)==1 for f in fixtures if f['kind']=='mutation')
    assert prep.check_fixtures(fixtures,rows,ck)==[]
    assert fixtures==prep.read_jsonl(prep.RESOURCES/'authored-fixtures.jsonl')
    edge=json.loads((prep.RESOURCES/'checker-edge-fixtures.json').read_text())
    assert all(ck.check(e['text'],e['constraint'])==e['expected'] for e in edge)
    assert ck.json_fence_failure('```json\n{}\n```')
    assert not ck.json_fence_failure('{}')
    print('PASS: 200 passing witnesses; 300 single-constraint failing mutations; 38 checker edge fixtures; raw JSON rejects fences.')

    matrix=json.loads((prep.RESOURCES/'compatibility.json').read_text())
    assert matrix==ck.compatibility_matrix()
    for a in ck.FAMILIES:
        for b in ck.FAMILIES: assert matrix['matrix'][a][b]==matrix['matrix'][b][a]
    for f in ('required_suffix','bullet_count','paragraph_count','sentence_count'):
        assert matrix['matrix']['json_only'][f]=='forbidden'
    must_reject(lambda: ck.validate_constraints([dict(family='json_only',args={}),dict(family='bullet_count',args={'count':2})]))
    print('PASS: frozen 10x10 compatibility matrix matches checker parameter rules and all 100 authored pairs.')

    # Synthetic template-only pools isolate intentional canonical reuse from content leakage.
    pool=synthetic_pool(prompts=[c for r in rows for c in r['canonical']['clauses']])
    report=prep.audit(rows,[pool],ck)
    assert not report['hits'], report['hits'][:5]
    assert len(report['nearest'])==200 and not report['insufficient_nearest']
    assert all(len(v['nearest'])==5 for v in report['nearest'].values())
    leaked=copy.deepcopy(rows[0]); leaked['paraphrase']['clauses']=[leaked['canonical']['clauses'][0]]
    report=prep.audit([leaked],[pool],ck)
    assert any(h['kind']=='clause-vs-replay' and h['wording']=='paraphrase' for h in report['hits'])
    assert not any(h.get('wording')=='canonical' for h in report['hits'])
    content_pool=synthetic_pool(prompts=[rows[0]['content']+'\n\nAn unrelated instruction.'])
    report=prep.audit([rows[0]],[content_pool],ck)
    assert any(h['kind']=='content-vs-pool' and h['wording']=='both' for h in report['hits'])
    assert report['missing_roles'] and report['status']=='REJECTED'
    print('PASS: canonical reuse exempt; atomic paraphrase copies rejected; content overlap rejects both arms; 200 five-nearest lists built on synthetic replay templates.')

    receipt_paths=[prep.RESOURCES/'reviews'/('reviewer-%d.template.json'%i) for i in (1,2)]
    receipt_hashes={p.name:prep.sha256_file(p) for p in receipt_paths}
    existing=[json.loads(p.read_text()) for p in receipt_paths]
    assert prep.check_reviews(existing,rows,fixtures)  # Stale round-2 reviews cannot approve round 3.
    # Independent receipts are immutable. Build temporary test approvals from
    # current requests rather than editing or copying verdicts from real reviews.
    approvals=[]
    for i in (1,2):
        approvals.append(dict(schema='sp-review-receipt-v2', reviewer_id='synthetic-test-%d'%i,
                              independent_of_author_and_other_reviewer=True,
                              binding=prep.review_binding(rows,fixtures),
                              items=[dict(request_id=r['request_id'],request_sha256=prep.digest(r),
                                          semantic_equivalence=True,clarity=True,satisfiable=True,
                                          canonical_provenance=True,notes='synthetic test only') for r in rows]))
    assert prep.check_reviews(approvals,rows,fixtures)==[]
    same=copy.deepcopy(approvals);same[1]['reviewer_id']=same[0]['reviewer_id'];assert prep.check_reviews(same,rows,fixtures)
    stale=copy.deepcopy(approvals);stale[0]['items'][0]['request_sha256']='b'*64;assert prep.check_reviews(stale,rows,fixtures)
    incomplete=copy.deepcopy(approvals);incomplete[0]['items'].pop();assert prep.check_reviews(incomplete,rows,fixtures)
    assert {p.name:prep.sha256_file(p) for p in receipt_paths}==receipt_hashes
    print('PASS: independent receipts preserved; stale round-2 bindings, duplicate reviewers and missing reviews cannot approve round 3.')
    current=[json.loads((prep.RESOURCES/'reviews'/('reviewer-%d.round3.json'%i)).read_text()) for i in (1,2)]
    assert prep.check_reviews(current,rows,fixtures)==[]
    print('PASS: both independent round-3 receipts approve all 200 current request hashes.')

    # Repeated text must preserve every source identity, first matching segment,
    # per-field witness hit, and deterministic five-nearest tie ordering.
    repeated=synthetic_pool(prompts=[rows[0]['content']+'\n'+rows[0]['canonical']['clause']]*6)
    for item in repeated['rows']: item['response']=rows[0]['witness']
    cached=prep.audit([rows[0]], [repeated], ck)
    assert len([h for h in cached['hits'] if h['kind']=='content-vs-pool' and h['field']=='prompt'])==6
    assert len([h for h in cached['hits'] if h['kind']=='witness-vs-pool' and h['field']=='response'])==6
    brute={}
    for item in repeated['rows']:
        for clause in prep.clauses_of(item['prompt'],ck):
            masked=prep.mask(clause)
            if masked:
                score=max(round(prep.edit_similarity(prep.mask(c,prep.constraint_values(rows[0]['constraints'])),masked),6)
                          for c in rows[0]['paraphrase']['clauses'])
                brute[(repeated['name'],item['id'],clause,masked)]=score
    expected=[dict(similarity=-score,pool=pool,row_id=rid,clause=clause,masked=masked)
              for score,pool,rid,clause,masked in sorted((-score,*key) for key,score in brute.items())[:5]]
    assert cached['nearest'][rows[0]['request_id']]['nearest']==expected
    print('PASS: cached audit preserves repeated-source hits and brute-force five-nearest ordering.')
    inspected=[dict(request_id=rid,nearest_sha256=x['nearest_sha256'],inspector='synthetic-fixture',verdict='distinct')
               for rid,x in cached['nearest'].items()]
    assert not prep.check_inspection(inspected,cached)
    inspected[0]['inventory_complete']=False
    assert any('incomplete inventory' in e for e in prep.check_inspection(inspected,cached))
    inspected[0]['inventory_complete']=True
    inspected[0]['audit_inventory_sha256']='0'*64
    assert any('different pool inventory' in e for e in prep.check_inspection(inspected,cached))
    print('PASS: incomplete-inventory inspection and mismatched inventory bindings cannot authorize a freeze.')

    changed=json.loads((prep.RESOURCES/'reviews'/'changed-pairs-round3.json').read_text())
    assert changed['new_binding']==prep.review_binding(rows,fixtures)
    assert changed['review_receipts_sha256']==receipt_hashes
    assert changed['changed_pair_ids']==sorted(item['request_id'] for item in changed['changes'])
    assert set(changed['reviewer_rejected_pair_ids']) <= set(changed['changed_pair_ids'])
    assert set(changed['reviewer1_count_note_ids']) <= set(changed['changed_pair_ids'])
    indexed={r['request_id']:r for r in rows}
    baseline={item['request_id']:item['request_sha256'] for item in existing[0]['items']}
    assert baseline=={item['request_id']:item['request_sha256'] for item in existing[1]['items']}
    assert set(changed['changed_pair_ids'])=={rid for rid,row in indexed.items() if prep.digest(row)!=baseline[rid]}
    assert set(changed['unchanged_request_sha256'])==set(indexed)-set(changed['changed_pair_ids'])
    for entry in changed['changes']:
        assert entry['old_request_sha256']==baseline[entry['request_id']]
        assert entry['new_request_sha256']==prep.digest(indexed[entry['request_id']])
    for rid,sha in changed['unchanged_request_sha256'].items(): assert prep.digest(indexed[rid])==sha
    for row in rows:
        text=row['paraphrase']['clause'].lower()
        for explanation in ('substring','whitespace','code fence','code block','case-insensitive',
                            'case insensitive','letter case','capitalization','word boundaries',
                            'nonempty','non-whitespace','nonwhitespace','lowercase forms',
                            'split the','split at','splitting','indentation','first character'):
            assert explanation not in text, (row['request_id'],explanation)
        assert all(not clause.startswith('Count ') for clause in row['paraphrase']['clauses'])
        for c in row['constraints']:
            if c['family']=='word_frequency':
                clause=row['paraphrase']['clauses'][row['constraints'].index(c)]
                assert 'word "'+c['args']['word']+'"' in clause
            if c['family']=='forbidden_word':
                word=c['args']['word']
                common=('instead','message','confront','aircraft','cleaning','notebook','endgame',
                        'shoebox','football','imperfect','insurgent','instantaneous','penultimate')
                assert not any(word in other and word != other for other in common)
    print('PASS: %d changed pairs listed; all 12 rejected pairs and all Count-N notes revised; no paraphrase checker-procedure explanations.'%len(changed['changed_pair_ids']))

    parents=[synthetic_pool(name='candidate-'+role,role=role,prompts=['Invent an unusual geometric design for a decorative tile.']) for role in ('broad','k','cr')]
    bound=dict(pools=prep.pool_identities(parents))
    returned=copy.deepcopy(parents)
    for pool in returned:
        pool['candidate_superset_name']=pool['name'];pool['candidate_superset_sha256']=pool['sha256'];pool['name']='returned-'+pool['role']
    before=prep.digest(rows)
    assert prep.verify_returned_pools(bound,returned,rows,ck)['status']=='CLEAN'
    escaped=copy.deepcopy(returned);escaped[0]['rows'].append(dict(id='new',prompt='A newly added candidate outside the frozen list.'))
    result=prep.verify_returned_pools(bound,escaped,rows,ck)
    assert result['status']=='UNVERIFIED' and not result['panel_changed'] and not result['rescoring_allowed']
    assert prep.digest(rows)==before
    mismatch=copy.deepcopy(returned);mismatch[0]['candidate_superset_sha256']='c'*64
    assert prep.verify_returned_pools(bound,mismatch,rows,ck)['status']=='UNVERIFIED'
    assert prep.verify_returned_pools(bound,returned[:2],rows,ck)['status']=='UNVERIFIED'
    with tempfile.TemporaryDirectory(prefix='sp-author-') as temp:
        root=Path(temp); data=root/'pool.jsonl';prep.write_jsonl(data,parents[0]['rows'])
        entry=dict(name='candidate-broad',role='broad',path='pool.jsonl',sha256=prep.sha256_file(data),scope='candidate-superset')
        spec=dict(schema='sp-pools-v2',complete_inventory=True,frozen_before_scoring=True,pools=[entry])
        file=root/'spec.json';prep.write_json(file,spec);assert len(prep.load_pools(file))==1
        spec['pools'][0]['sha256']='d'*64;prep.write_json(file,spec);must_reject(lambda:prep.load_pools(file))
        spec['pools'][0]['sha256']=prep.sha256_file(data);spec['frozen_before_scoring']=False;prep.write_json(file,spec);must_reject(lambda:prep.load_pools(file))
    print('PASS: frozen candidate hashes enforced; contained returns verify; extras, wrong bindings and missing returns are UNVERIFIED without panel edits or rescoring.')
    # Exercise the whole freeze/read path only in a disposable synthetic sandbox.
    # These test attestations never become real source-review receipts.
    with tempfile.TemporaryDirectory(prefix='sp-freeze-fixture-') as temp:
        root=Path(temp)
        pools=[]
        for role in prep.REQUIRED_ROLES:
            path=root/(role+'.jsonl')
            prep.write_jsonl(path,[dict(id='fixture-%d'%i,prompt=rec['template']) for i,rec in enumerate(prep.load_templates().inventory()['templates'].values())])
            pools.append(dict(name=role,role=role,path=path.name,sha256=prep.sha256_file(path),scope='frozen-reference' if role in ('generic-accepted','panels') else 'candidate-superset'))
        spec=root/'pools.json'
        prep.write_json(spec,dict(schema='sp-pools-v2',complete_inventory=True,frozen_before_scoring=True,pools=pools))
        staged=prep.build(ROOT,spec)
        assert staged['audit']['status']=='CLEAN' and staged['blockers']
        must_reject(lambda:prep.freeze(root/'blocked',ROOT,spec))
        inspection=root/'inspection.jsonl'
        prep.write_jsonl(inspection,[dict(request_id=rid,nearest_sha256=value['nearest_sha256'],inspector='synthetic-fixture',verdict='distinct') for rid,value in staged['audit']['nearest'].items()])
        receipts=[]
        for index,doc in enumerate(approvals):
            receipt=root/('review-%d.json'%index);prep.write_json(receipt,doc);receipts.append(receipt)
        frozen=root/'synthetic-freeze'
        doc=prep.freeze(frozen,ROOT,spec,inspection,receipts)
        assert prep.verify_frozen(frozen)==doc and doc['n_prompts']==400
        assert prep.freeze(frozen,ROOT,spec,inspection,receipts)==doc
        panel=frozen/'panel.jsonl';panel.write_text(panel.read_text()+'\n')
        must_reject(lambda:prep.verify_frozen(frozen))
        assert prep.recheck(frozen,spec)['status']=='UNVERIFIED'
    print('PASS: disposable synthetic end-to-end freeze blocks incomplete receipts, verifies 400 prompts, is write-once and detects file tampering.')
    sealed=json.loads((prep.RESOURCES/'CONSTRUCTION.json').read_text())
    assert sealed['binding']==prep.review_binding(rows,fixtures)
    assert {p.name:prep.sha256_file(p) for p in receipt_paths}==receipt_hashes
    available=json.loads(gzip.decompress((prep.RESOURCES/'exclusions/audit.available.json.gz').read_bytes()))
    inventory=json.loads((prep.RESOURCES/'exclusions/inventory.available.json').read_text())
    inspection=prep.read_jsonl(prep.RESOURCES/'reviews/nearest-inspection/inspection.jsonl')
    assert available['status']=='PENDING' and available['inventory_complete'] is False
    assert not available['hits'] and not available['rejected_request_ids']
    assert len(available['pools'])==23 and sum(x['rows'] for x in available['pools'])==98273
    assert len(available['nearest'])==200 and all(len(v['nearest'])==5 for v in available['nearest'].values())
    assert [(x['name'],x['sha256'],x['rows']) for x in available['pools']]==[(x['name'],x['sha256'],x['rows']) for x in inventory['pools']]
    assert inventory['inventory_complete'] is False and len(inventory['blockers'])==4
    assert len(inspection)==200 and all(x['author_performed'] and not x['independent_review'] for x in inspection)
    assert all(x['request_sha256']==prep.digest(indexed[x['request_id']]) for x in inspection)
    assert all(x['nearest']==available['nearest'][x['request_id']]['nearest'] and x['reviewed_ranks']==[1,2,3,4,5] for x in inspection)
    errors=prep.check_inspection(inspection,available)
    assert len(errors)==200 and all('incomplete inventory' in e for e in errors)
    print('PASS: real available-pool audit has 0 hits across 23 pools and 98,273 rows; 200 author inspections match all five nearest entries.')
    assert not prep.check_inspection(inspection,available,available_scope=True)
    ruling=json.loads((prep.RESOURCES/'freeze-ruling.json').read_text())
    assert ruling['required_before_scoring'] and ruling['pair_local_hits'] and not ruling['rescoring_allowed']
    assert ruling['available_pool_count']==23 and ruling['available_row_count']==98273
    assert len(ruling['unavailable_inventories'])==12
    if (prep.RESOURCES/'FROZEN.json').exists():
        doc=prep.verify_frozen()
        assert doc['status']=='FROZEN' and doc['node_verification']==ruling
        assert doc['n_requests']==200 and doc['n_prompts']==400
        assert doc['inspection_sha256']==prep.sha256_file(prep.RESOURCES/'reviews/nearest-inspection/inspection.jsonl')
        assert doc['requests_sha256']==prep.digest(rows)
        print('PASS: scientific FROZEN binds 200 requests, 400 prompts, 23 pools, 98,273 rows and 200 inspected nearest lists.')
    else:
        print('PASS: manager-authorized available-inventory freeze is ready; 12 node-held inventories require VERIFY-ONLY before scoring.')
    print('PASS: historical incomplete-inventory receipts remain truthful; the explicit manager ruling authorizes only the available scope.')


if __name__=='__main__':main()
