"""FinQA standalone arithmetic and raw archive reading; replication cannot rescue primary."""
from fractions import Fraction
import json
import tempfile
from kit import v4_analysis as a, v4_teacher as t, tokens_io
from kit.v4_campaign import slots, PRIMARY_SEEDS, REPLICATION_SEEDS, SEEDS, PANELS
from kit.v4_evidence import require
from kit.v4_finqa_campaign import DEPTH_OMISSION


def general_scoring(archive,phase,checkpoint,cap):
    """Re-grade frozen general-v1 raw text; bind tokens, prompts, weights and GPU."""
    from kit import score_forgetting as f
    from kit.v4_readers import decoded_response, rendered_prompt_hash
    from kit.v4_evidence import weights, scoring_gpu
    root=f'v4/report-{phase}/eval/{checkpoint}-general-v1-{cap}/'
    members=archive.jsonl('v4/report-inputs/general-v1-panel.jsonl')
    require(archive.files['v4/report-inputs/general-v1-panel.jsonl']==f.PANEL_FILE.read_bytes(),'general-v1 frozen panel differs')
    expected={m['panel']+':'+m['id']:m for m in members}
    raw=archive.jsonl(root+'responses.jsonl');token_rows=archive.jsonl(root+'tokens.jsonl')
    key=lambda r:r['panel']+':'+r['id']
    token_map={key(r):tokens_io.unpack(r['ids']) for r in token_rows}
    require(len(token_map)==len(token_rows) and set(token_map)==set(expected),'general token identities differ')
    results=[]
    for row in raw:
        ident=key(row);require(ident in expected,'foreign general identity')
        member=expected[ident];ids=token_map[ident]
        require(row['output_tokens']==len(ids),'general token count differs')
        correct=int(f.scorers.score(member['panel'],row['response'],member)['correct'])
        results.append(dict(id=ident,strict=correct,canonical=correct,tokens=len(ids)))
    result=a.panel(results,expected,cap)
    result['panels']={panel:{k:v for k,v in result['verdicts'].items() if k.startswith(panel+':')} for panel in ('math','knowledge','ifeval')}
    doc=archive.json(root+'forgetting.json')
    require(doc['max_new_tokens']==cap and doc['mode']=='generate','general scoring mode/cap differs')
    require(doc['decoding']=={**f.DECODING,'max_tokens':cap},'general decoding differs')
    require(all(doc['engine'].get(k)==v for k,v in {**f.ENGINE,'max_model_len':12288,'deterministic':True,'eager':True,'batch_invariant':True}.items()),'general engine differs')
    require(doc['generation']==dict(n=1,enable_lora=False,enforce_eager=True,seed=0,chat_template=dict(add_generation_prompt=True,enable_thinking=False)),'general generation differs')
    require(doc['panel_file_sha256']==t.sha(f.PANEL_FILE.read_bytes()),'general panel digest differs')
    require(doc['scorer_sha256']==f.scorer_identity()['scorer_sha256'],'general scorer differs')
    for row in token_rows:
        ident=key(row);member=expected[ident]
        require(row['prompt_sha256']==rendered_prompt_hash(archive,[dict(role='user',content=member['prompt'])]),'general rendered prompt differs')
    for row in raw:require(decoded_response(archive,token_map[key(row)])==row['response'],'general decoded text differs')
    identity=archive.json('v4/report-inputs/model-identities.json')['initial'] if checkpoint=='baseline' else archive.json('runs/'+checkpoint+'/env/export-identity-step40.json')
    require(weights(archive.json(root+'checkpoint-identity.json'))==weights(identity),'general checkpoint differs',identity=True)
    fp=doc['machine'];digest=t.sha(json.dumps({k:fp[k] for k in ('gpus','cuda_visible_devices','versions','deterministic')},sort_keys=True))[:16]
    require(fp['id']==digest and fp['deterministic'] is True and fp['gpus'],'general fingerprint differs')
    result['fingerprint']=digest+':'+scoring_gpu(archive,phase,checkpoint,'general-v1',cap,fp)
    return result


def tables(baseline,scores):
    """Complete per-seed tables plus separate replication/all-five descriptive readings."""
    cells={arm:{} for arm in ('S','F','R','D','D-R')};general={};diagnostics={};failures=[]
    for job in slots():
        ident=job['id']
        if ident not in scores:continue
        arm=job['arm'];seed=str(job['seed']);after=scores[ident]['finqa']['2048'];before=baseline['finqa']['2048']
        transition=a.paired(before['verdicts'],after['verdicts'])
        passes={'acquisition':transition['net']>=58,'deployment_cost':a.cost_pass(after,before)}
        cells[arm][seed]=dict(transitions=transition,passes=passes,met=all(passes.values()),strict=after['strict'],tokens=after['tokens'],deployment_cost=after['deployment_cost'])
        if arm=='D' and job['seed'] in PRIMARY_SEEDS and not all(passes.values()):failures.append(dict(arm=arm,seed=int(seed),criteria=[k for k,v in passes.items() if not v]))
        diagnostics[ident]={cap:{mode:a.paired(baseline['finqa'][cap][field],scores[ident]['finqa'][cap][field]) for mode,field in (('strict','verdicts'),('canonical','canonical_verdicts'))} for cap in ('2048','8192') if cap in baseline['finqa'] and cap in scores[ident]['finqa']}
        general[ident]={}
        for cap in ('2048','8192'):
            if cap not in baseline.get('general-v1',{}) or cap not in scores[ident].get('general-v1',{}):continue
            left=baseline['general-v1'][cap]['verdicts'];right=scores[ident]['general-v1'][cap]['verdicts']
            panels={p:a.paired(left,right,[i for i in left if i.startswith(p+':')]) for p in ('math','knowledge','ifeval')}
            general[ident][cap]=dict(total=a.paired(left,right),panels=panels,within_3_per_panel=all(r['net']>=-3 for r in panels.values()),label='descriptive base-skills health; within 3 per panel')
    def cohort(seeds,record_failures=False):
        rows={};complete=True
        for seed in seeds:
            key=str(seed)
            if key not in cells['D'] or key not in cells['S']:complete=False;continue
            d=scores[f'D-finqa-alone-s{seed}']['finqa']['2048'];s=scores[f'S-finqa-alone-s{seed}']['finqa']['2048']
            diff=Fraction(100*(d['strict']-s['strict']),1147)
            rows[key]=dict(D_minus_S_pp=float(diff),D_minus_S_exact=str(diff),advantage_pass=diff>=5,D_criteria=cells['D'][key],paired=a.paired(s['verdicts'],d['verdicts']))
            if record_failures and diff<5:failures.append(dict(arm='D',seed=seed,criteria=['advantage_over_S']))
        return dict(seeds=list(seeds),complete=complete,per_seed=rows,all_per_seed_bars=complete and all(r['advantage_pass'] and r['D_criteria']['met'] for r in rows.values()),
            totals={arm:dict(correct=sum(cells[arm][str(seed)]['strict'] for seed in seeds if str(seed) in cells[arm]),questions=1147*sum(str(seed) in cells[arm] for seed in seeds)) for arm in ('S','F','R','D')},
            pooled_can_rescue_primary=False)
    primary=cohort(PRIMARY_SEEDS,True);replication=cohort(REPLICATION_SEEDS);all_five=cohort(SEEDS)
    dr={}
    for seed in PRIMARY_SEEDS:
        if str(seed) in cells['D-R'] and str(seed) in cells['D']:
            d=scores[f'D-finqa-alone-s{seed}']['finqa']['2048'];r=scores[f'D-R-finqa-alone-s{seed}']['finqa']['2048']
            paired_diagnostics={cap:{mode:a.paired(scores[f'D-finqa-alone-s{seed}']['finqa'][cap][field],scores[f'D-R-finqa-alone-s{seed}']['finqa'][cap][field]) for mode,field in (('strict','verdicts'),('canonical','canonical_verdicts'))} for cap in ('2048','8192') if cap in scores[f'D-finqa-alone-s{seed}']['finqa'] and cap in scores[f'D-R-finqa-alone-s{seed}']['finqa']}
            dr[str(seed)]=dict(diagnostics=paired_diagnostics,D_R_minus_D_pp=float(Fraction(100*(r['strict']-d['strict']),1147)),paired=a.paired(d['verdicts'],r['verdicts']),label='descriptive; source and content change together; no linguistic-familiarity isolation')
    matches={arm:all(str(seed) in cells[arm] and str(seed) in cells['D'] and cells[arm][str(seed)]['met'] and Fraction(100*(cells['D'][str(seed)]['strict']-cells[arm][str(seed)]['strict']),1147)<Fraction(5,2) for seed in PRIMARY_SEEDS) for arm in ('F','R')}
    return dict(cells=cells,primary=primary,replication=replication,all_five=all_five,D_R_minus_D=dr,finqa_diagnostics=diagnostics,general_v1=general,matches_D=matches,known_failures=failures,primary_arithmetic_met=primary['all_per_seed_bars'])


def _campaign_evidence(archive):
    from kit import v4_readers as r, v4_evidence as e, v4_auxiliary as aux
    report=dict(status='incomplete',reasons=[],tables={},expected_unrun=[],known_failures=[],teacher_depth=dict(status='omitted',reason=DEPTH_OMISSION),historical_finqa_reference=dict(correct=948,label='historical reference'))
    try:
        from kit.v4_phase0_carry import archive_ledger
        ledger,accounting=archive_ledger(archive,'main');report['tables']['allocation_accounting']=accounting
        report['tables']['environment']=e.environment(archive,'main')
        report['tables']['containment']=e.containment(archive)
        from kit.v4_budget import allocation_spend
        actual,blocks=allocation_spend(ledger)
        require(actual<1000,'actual 1000 GPU-hour ceiling reached')
        e.scheduler_rows(archive,'qualification');e.scheduler_rows(archive,'main')
        trace=aux.chronology(archive,'main');report['tables']['chronology']=trace
        with tempfile.TemporaryDirectory(prefix='v4-finqa-read-') as temp:
            tokenizer,mcq=r.scorer_context(archive,temp)
            with e.data_context(archive,tokenizer,mcq):
                coverage={'finqa':r.corpus(archive,'main','finqa',tokenizer,mcq)};report['tables']['coverage']=coverage
                if not coverage['finqa']['pass']:
                    report['reasons'].append('coverage margin missed; trained on surviving verified intersection')
                require(set(archive.json('v4/report-main/coverage.json')['tasks'])=={'finqa'},'coverage code report differs')
                report['tables']['common_data']=e.common_data(archive,'main',tokenizer)
                # The analysis comparison uses valid D-101 training and
                # FinQA strict B, before any diagnostic/full endpoint reading.
                base=r.scoring(archive,'qualification','baseline','finqa',2048,mcq)
                baseline={task:{} for task in PANELS};baseline['finqa']['2048']=base
                fingerprints={base['fingerprint']}
                scores={};records={};invalid={}
                cached={};first=slots()[0];ident=first['id']
                def record(job):
                    ident=job['id']
                    require(not any('train-'+ident in error for error in trace['errors']),'disqualified infrastructure retry')
                    summary=r.run_record(archive,ident,'main');summary['provenance']=e.training_provenance(archive,'main',job,summary,tokenizer)
                    return summary
                try:
                    records[ident]=record(first)
                    d=r.scoring(archive,'main',ident,'finqa',2048,mcq);cached[ident]=d
                    paired=a.paired(base['verdicts'],d['verdicts'])
                    gate=dict(transitions=paired,acquisition_pass=paired['net']>=58,cost_pass=a.cost_pass(d,base));report['tables']['D_qualification']=gate
                    require(base['fingerprint']==d['fingerprint'],'cross-GPU/machine scoring')
                    if not gate['acquisition_pass'] or not gate['cost_pass']:
                        report['reasons'].append('seed-101 D standalone bar not met; analysis finding, all slots continue')
                except (ValueError,KeyError,TypeError) as error:invalid[ident]=str(error)
                for task in PANELS:
                    for cap in (2048,8192):
                        if task=='finqa' and cap==2048:continue
                        try:
                            value=r.scoring(archive,'qualification','baseline',task,cap,mcq)
                            baseline[task][str(cap)]=value;fingerprints.add(value['fingerprint'])
                        except (ValueError,KeyError,TypeError) as error:report['reasons'].append('baseline '+task+' '+str(cap)+': '+str(error))
                def read_job(job):
                    ident=job['id']
                    try:
                        summary=records.get(ident) or record(job)
                        values={task:{} for task in PANELS};errors=[]
                        for task in PANELS:
                            for cap in (2048,8192):
                                try:
                                    value=cached[ident] if ident in cached and task=='finqa' and cap==2048 else r.scoring(archive,'main',ident,task,cap,mcq)
                                    values[task][str(cap)]=value;fingerprints.add(value['fingerprint'])
                                except (ValueError,KeyError,TypeError) as error:errors.append(str(error))
                        # Keep independently established strict-B failures even
                        # when another required endpoint is missing.
                        if '2048' in values['finqa']:scores[ident]=values
                        records[ident]=summary
                        if errors:invalid[ident]='; '.join(errors)
                    except (ValueError,KeyError,TypeError) as error:invalid[ident]=str(error)
                for job in slots():read_job(job)
                report['tables'].update(tables(baseline,scores));report['tables']['scores']=scores;report['tables']['baseline']=baseline;report['tables']['training']=records
                report['known_failures']=report['tables']['known_failures'];report['primary_arithmetic_met']=report['tables']['primary_arithmetic_met'];report['disqualified']=invalid
                if invalid:report['reasons'].append('missing or unreadable slot evidence')
                if len(fingerprints)!=1:report['reasons'].append('cross-GPU/machine scoring')
                report['reasons']+=trace['errors']
                for name,reader in (('mechanism',lambda:aux.mechanism(archive,coverage,mcq,records)),('resources',lambda:aux.resource_costs(archive))):
                    try:
                        value=reader()
                        if name=='mechanism':value,errors=value;report['reasons']+=errors
                        report['tables'][name]=value
                    except (ValueError,KeyError,TypeError) as error:report['reasons'].append(name+': '+str(error))
        _,errors=r.verdicts(archive,'main');report['reasons']+=errors
        if 'k8b4/containment/v4-stop.json' in archive.files:report['reasons'].append('containment or allocation-ceiling stop remains recorded')
        if not report['reasons']:
            met=report['primary_arithmetic_met'];report['status']='primary claim met' if met else 'primary claim not met'
            report['wording']='Under the registered recipes at 40 updates, D '+('met' if met else 'did not meet')+' the FinQA acquisition, deployment cost and D minus S criteria on all primary seeds 101-103. Replication and D-R are descriptive; base-skills retention is measured on general-v1. The sequential claim is retired.'
            resources=report['tables']['resources'];report['additional_machinery_not_justified']=[arm for arm in ('F','R') if report['tables']['matches_D'][arm] and resources['lower_total_cost_than_D'][arm]]
    except (ValueError,KeyError,TypeError,OSError,ImportError) as error:report['reasons'].append(str(error))
    return report


def campaign_evidence(archive):
    from kit.v4_readers import self_monitored
    if not self_monitored(archive):return _campaign_evidence(archive)
    from kit.v4_finqa_policy import observations,correctness
    with observations() as warnings:
        report=_campaign_evidence(archive)
    if 'disqualified' in report:
        report['unreadable_slots']=report.pop('disqualified')
        report.setdefault('reasons',[]).extend(str(reason) for reason in report['unreadable_slots'].values() if correctness(reason))
        warnings.extend({'site':'slot evidence','message':slot+': '+str(reason)} for slot,reason in report['unreadable_slots'].items() if not correctness(reason))
    hard=[r for r in report.get('reasons',[]) if correctness(r)]
    report.update(status='stop: correctness' if hard else 'progress report',owner_proceed_required=False,
                  warnings=warnings+[{'site':'main reader','message':r} for r in report.get('reasons',[]) if r not in hard])
    return report
