#!/usr/bin/env python3
"""Frozen six-row reader. Item vectors determine endpoints; never accept summaries.

Cells: seed/panel/model/step, items={id:0|1}, checkpoint_sha256,
nonce/uuid/environment, scorer_sha256, method. model is base, incoming, C or T.
Each per-(seed,panel) job includes every reference/arm/checkpoint in its frontier.
"""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.nm_common import *
from kit.nm_data import final_cohort


class Hardware(Invalid):pass


class Table:
    def __init__(self,doc,spec):
        self.spec=spec;self.synthetic=doc.get('technical_synthetic',False);self.cells={};self.used={};self.cohorts={};self.cohort_diagnostics={}
        for cell in doc.get('cells',[]):
            key=(cell['seed'],cell['panel'],cell['model'],cell['step'])
            need(key not in self.cells,'duplicate score cell');self.cells[key]=cell
    def cell(self,seed,panel,model,step=40):
        step=0 if model in ('base','incoming','memory') else step
        key=(seed,panel,model,step);ready(key in self.cells,'missing score cell '+str(key));row=self.cells[key]
        ready(row.get('status','COMPLETE')=='COMPLETE','score cell incomplete/FORMAT-CONFOUNDED')
        need(row.get('technical_synthetic') is self.synthetic,'synthetic/scientific score provenance differs')
        need(row.get('method')!='FORMAT-CONFOUNDED','FORMAT-CONFOUNDED')
        ids=self.spec.get('panels',{}).get(panel)
        ready(ids is not None,'frozen panel IDs absent: '+panel)
        need(set(row['items'])==set(ids) and len(row['items'])==len(ids),'panel membership/denominator differs')
        need(all(type(v) is int and v in (0,1) for v in row['items'].values()),'invalid correctness vector')
        expected=self.spec.get('checkpoints',{}).get(str(seed),{}).get(model,{}).get(str(step))
        ready(expected is not None,'checkpoint binding absent')
        need(row.get('checkpoint_sha256')==expected,'checkpoint identity differs')
        scorer=self.spec.get('scorers',{}).get(panel)
        ready(scorer is not None,'scorer authority absent')
        ready(scorer.get('sha256') and scorer.get('method'),'scorer binding incomplete')
        need(row.get('scorer_sha256')==scorer['sha256'] and row.get('method')==scorer['method'],'scorer authority differs')
        if panel in ('knowledge','hellaswag'):
            need(row['method']=='REFERENCE-SCORED' and row.get('precision')=='full','reference precision/method differs')
        identity=tuple(row.get(k) for k in ('nonce','uuid','environment'))
        if not all(identity):raise Hardware('missing measured allocation identity')
        lane=(seed,panel)
        if lane in self.used and self.used[lane]!=identity:raise Hardware('mixed allocation nonce/UUID/environment')
        self.used[lane]=identity
        return row
    def items(self,*args):return self.cell(*args)['items']
    def acc(self,*args):
        values=self.items(*args).values();ready(values,'empty panel');return 100*sum(values)/len(values)
    def delta(self,seed,panel,step=40):return self.acc(seed,panel,'T',step)-self.acc(seed,panel,'C',step)
    def gain(self,seed,panel,model):return self.acc(seed,panel,model)-self.acc(seed,panel,'incoming')
    def auc(self,seed,panel,model):
        vals=[self.acc(seed,panel,model,s) for s in STEPS]
        return sum((a+b)*5 for a,b in zip(vals,vals[1:]))/40-self.acc(seed,panel,'incoming')
    def require_frontier(self,family):
        panels=list(PANELS if family not in ('v2b','grpo') else (*GENERAL,'hellaswag','finqa') if family=='grpo' else
                    (*GENERAL,'hellaswag','july-off','july-on','august-off','august-on','september-off','september-on','september-on-holdout'))
        for seed in SEEDS:
            for panel in panels:
                for model in ('base','incoming','C','T'):
                    self.cell(seed,panel,model,40)
                for model in ('C','T'):self.cell(seed,panel,model,20)
                if panel in ('legal','finqa'):
                    # All FinQA curve points required for GRPO; legal for legal rows.
                    if panel=='legal' or family=='grpo':
                        for model in ('C','T'):
                            for step in STEPS:self.cell(seed,panel,model,step)
                if family=='v2b' and panel not in (*GENERAL,'hellaswag'):self.cell(seed,panel,'memory')
        if family=='v2b':
            for panel,n in [('july-off',21),('july-on',21),('august-off',24),('august-on',24),('september-off',13),('september-on',52),('september-on-holdout',13)]:
                ids=self.spec.get('panels',{}).get(panel);ready(ids is not None,'frozen panel IDs absent: '+panel)
                need(len(ids)==n,'v2b frozen denominator differs')
        if family=='opr':
            ids=self.spec.get('panels',{}).get('finqa');ready(ids is not None,'frozen FinQA panel absent')
            need(len(ids)==1147,'FinQA must contain 1147 IDs')
            for seed in SEEDS:
                pre=self.spec.get('pre_cohort',{}).get(str(seed));ready(pre,'pre-training cohort feasibility absent')
                ready(pre.get('base') is not None and pre.get('incoming') is not None,'pre-training reference vectors absent')
                need(pre.get('before_training') is True,'cohort feasibility not sealed before training')
                # Membership is sealed from references BEFORE final arm vectors are accessed below.
                members,diagnostic=final_cohort(pre['base'],pre['incoming'],self.items(seed,'finqa','base'),self.items(seed,'finqa','incoming'))
                self.cohorts[seed]=members;self.cohort_diagnostics[str(seed)]=diagnostic


def summaries(table,family):
    panels=GENERAL if family=='v2b' else (*GENERAL,'finqa')
    result={}
    for model in ('C','T'):
        parts={}
        for panel in panels:
            reference='incoming' if panel=='finqa' else 'base'
            values=[table.acc(s,panel,model)-table.acc(s,panel,reference) for s in SEEDS]
            parts[panel]=guard(values,-1 if panel=='finqa' else -3,ONE)
        if family not in ('v2b','grpo'):
            parts['legal_absolute']=guard([table.acc(s,'legal',model) for s in SEEDS],69.7,ONE)
            parts['legal_gain']=stats([table.gain(s,'legal',model) for s in SEEDS],ONE)
        result[model]=parts
    return result


def answer_diagnostics(table,family):
    result={}
    panels=(*GENERAL,'finqa') if family!='v2b' else GENERAL
    for seed in SEEDS:
        by_panel={}
        for panel in panels:
            base=table.items(seed,panel,'base');incoming=table.items(seed,panel,'incoming')
            acquired=[i for i in base if not base[i] and incoming[i]]
            by_arm={}
            for arm in ('C','T'):
                final=table.items(seed,panel,arm)
                transitions={str(a)+'->'+str(b):sum(incoming[i]==a and final[i]==b for i in incoming) for a in (0,1) for b in (0,1)}
                by_arm[arm]=dict(transitions=transitions,acquired_count=len(acquired),acquired_survival=100*sum(final[i] for i in acquired)/len(acquired) if acquired else None,
                    incoming_accuracy=100*sum(incoming.values())/len(incoming),final_accuracy=100*sum(final.values())/len(final))
            by_panel[panel]=by_arm
        result[str(seed)]=by_panel
    return result


def correct_counts(table,family):
    """Exact sufficient statistics; order is N, base, incoming, C steps, T steps.

    All five curve steps are retained only where a registered AUC uses them.
    Other panels carry final counts. OPR adds its arm-independent cohort.
    """
    panels=(*GENERAL,'hellaswag','september-off','september-on') if family=='v2b' else (*GENERAL,'hellaswag','finqa') if family=='grpo' else PANELS
    counts={}
    for seed in SEEDS:
        for panel in panels:
            steps=STEPS if panel=='legal' or (family=='grpo' and panel=='finqa') else (40,)
            vectors=[table.items(seed,panel,m) for m in ('base','incoming')]
            vectors += [table.items(seed,panel,m,s) for m in ('C','T') for s in steps]
            counts[str(seed)+'.'+panel]=[len(vectors[0]),*[sum(v.values()) for v in vectors]]
        if family=='opr':
            ids=table.cohorts[seed]
            vectors=[table.items(seed,'finqa',m) for m in ('C','T')]
            counts[str(seed)+'.finqa-cohort']=[len(ids),*[sum(v[i] for i in ids) for v in vectors]]
    return counts


def evaluate(family,doc,spec,binding):
    need(family in FAMILIES,'unregistered family')
    ready(binding.get('status')=='READY','row binding not ready')
    need(binding.get('family')==family,'binding family differs')
    table=Table(doc,spec)
    ready(all(spec.get(k) is not None for k in ('panels','scorers','checkpoints')),'reader spec incomplete')
    # OPR cohort first; do not fetch challenger vectors to form it.
    if family=='opr':
        for seed in SEEDS:
            pre=spec.get('pre_cohort',{}).get(str(seed));ready(pre,'pre-training feasibility absent')
            ready(pre.get('base') is not None and pre.get('incoming') is not None,'pre-training reference vectors absent')
            need(pre.get('before_training') is True,'cohort gate not prospective')
            members,diag=final_cohort(pre['base'],pre['incoming'],table.items(seed,'finqa','base'),table.items(seed,'finqa','incoming'))
            table.cohorts[seed]=members;table.cohort_diagnostics[str(seed)]=diag
    table.require_frontier(family)
    support={}
    if family=='opr':
        coverage=spec.get('opr_refresh',{});ready(set(coverage)==set(map(str,SEEDS)),'OPR refresh evidence absent')
        for r in coverage.values():ready(r.get('coverage',0)>=.8,'OPR coverage below 80%')
        support['opr_refresh']=coverage
    if family=='fisher':
        ready(spec.get('fisher_calibration'),'Fisher calibration evidence absent')
        support['fisher_calibration']=spec['fisher_calibration']
    if family=='isdft':
        persistence=binding.get('binding',{}).get('persistence') or binding.get('persistence')
        ready(persistence,'iSDFT persistence evidence absent');support['persistence']=persistence
    guards={};primary=None;diagnostics={}
    if family=='grpo':
        d_gain=[table.acc(s,'finqa','C')-table.acc(s,'finqa','base') for s in SEEDS]
        d=stats(d_gain,ONE)
        primary=guard([table.delta(s,'finqa') for s in SEEDS])
        acquisition=stats([table.acc(s,'finqa','T')-table.acc(s,'finqa','base') for s in SEEDS],ONE)
        acquisition['pass']=acquisition['lower']>0 and acquisition['positive']==3
        guards['acquisition']=acquisition
        for p in (*GENERAL,'hellaswag'):guards[p]=guard([table.delta(s,p) for s in SEEDS],-3,ONE)
        diagnostics.update(D_sensitivity=d,FinQA_gain_AUC={a:stats([table.auc(s,'finqa',a) for s in SEEDS]) for a in ('C','T')},
            general_retention=stats([sum(table.delta(s,p) for p in GENERAL)/3 for s in SEEDS]),
            historical_D_SD='UNMEASURED',planning='likely INCONCLUSIVE; qualification only')
        interpretation='P3 RL column only; never incumbent admission'
    else:
        if family=='v2b':values=[table.delta(s,'september-off') for s in SEEDS]
        elif family=='isdft':values=[table.delta(s,'instruction') for s in SEEDS]
        elif family=='fisher':values=[sum(table.delta(s,p) for p in GENERAL)/3 for s in SEEDS]
        elif family=='freeze':values=[sum(table.delta(s,p) for p in (*GENERAL,'finqa'))/4 for s in SEEDS]
        else:
            values=[]
            for seed in SEEDS:
                members=table.cohorts[seed]
                values.append(100*sum(table.items(seed,'finqa','T')[i]-table.items(seed,'finqa','C')[i] for i in members)/len(members))
            diagnostics['cohorts']=table.cohort_diagnostics
        primary=benefit(values)
        for panel in (*GENERAL,'hellaswag'):guards[panel]=guard([table.delta(s,panel) for s in SEEDS])
        if family=='v2b':
            guards['september-on-all52']=guard([table.delta(s,'september-on') for s in SEEDS])
            diagnostics['prior_months']={p:stats([table.delta(s,p) for s in SEEDS]) for p in ('july-off','july-on','august-off','august-on','september-on-holdout')}
            diagnostics['pooled_prior_months']={mode:stats([(21*table.delta(s,'july-'+mode)+24*table.delta(s,'august-'+mode))/45 for s in SEEDS]) for mode in ('off','on')}
            diagnostics.update(primary_N=13,memory_on_guard_N=52,item_granularity=100/13,
                power_MDE_sigma=3.26404,matched_SD='UNMEASURED',recovery_ratio='N/A')
            ratios={}
            for model in ('C','T'):
                ratios[model]=[]
                for seed in SEEDS:
                    frozen=table.acc(seed,'september-off','base');memory=table.acc(seed,'september-off','memory');denom=memory-frozen
                    ratios[model].append(dict(value=(table.acc(seed,'september-off',model)-frozen)/denom if denom else None,negative_denominator=denom<0))
            diagnostics['memory_gain_retained']=ratios
        else:
            for panel in ('finqa','legal'):
                guards[panel+'-final']=guard([table.delta(s,panel) for s in SEEDS])
                guards[panel+'-retention' if panel=='finqa' else 'legal-gain']=guard([
                    table.gain(s,panel,'T')-table.gain(s,panel,'C') for s in SEEDS])
            if family!='opr':guards['legal-gain-AUC']=guard([table.auc(s,'legal','T')-table.auc(s,'legal','C') for s in SEEDS])
            else:diagnostics['legal-gain-AUC']=stats([table.auc(s,'legal','T')-table.auc(s,'legal','C') for s in SEEDS])
        interpretation='NON-INCUMBENT BENCHMARK; future comparison against I required' if binding.get('benchmark') and family!='v2b' else 'nomination subject to connected-stack/confirmation'
    diagnostics['answer_transitions']=answer_diagnostics(table,family)
    passed=primary['pass'] and all(g['pass'] for g in guards.values())
    status=('QUALIFIED' if family=='grpo' else 'NON-INCUMBENT BENCHMARK' if binding.get('benchmark') and family!='v2b' else 'PASS') if passed else 'INCONCLUSIVE'
    insensitive=family=='grpo' and not (d['lower']>0 and d['positive']==3)
    if insensitive:status='PENDING/INCOMPLETE'
    result=dict(schema=SCHEMA,family=family,status=status,primary=primary,guards=guards,diagnostics=diagnostics,
        package_summaries=summaries(table,family),interpretation=interpretation,hardware='VERIFIED',
        lanes={str(k):v for k,v in table.used.items()},binding_sha256=digest(binding),spec_sha256=digest(spec),
        incumbent_changed=False,benchmark=binding.get('benchmark',False),return_evidence=support,correct_counts=correct_counts(table,family),
        interval_scope='per contrast, not family-wide',serving=doc.get('serving','UNMEASURED'))
    if insensitive:result['reason']='matched grouped-D control insensitive'
    if doc.get('technical_synthetic') and not insensitive:result.update(status='TECHNICAL-ONLY',simulated_status=status)
    return result


def read(family,doc,spec,binding):
    if binding.get('status')=='ARCHIVED UNREAD':return dict(status='ARCHIVED UNREAD',reason=binding.get('reason','prerequisite archived'),persistence=binding.get('persistence'))
    try:result=evaluate(family,doc,spec,binding)
    except Hardware as e:result=dict(status='UNVERIFIED-HARDWARE',reason=str(e))
    except Pending as e:result=dict(status='PENDING/INCOMPLETE',reason=str(e))
    except (Invalid,KeyError,TypeError,ValueError) as e:
        result=dict(status='PENDING/INCOMPLETE' if 'FORMAT-CONFOUNDED' in str(e) else 'UNVERIFIED',reason=str(e))
    if spec.get('generation_qualification'):
        result['generation_qualification']=spec['generation_qualification']
        result['utilisation_warnings']=[f'generation_qualification_s{seed}_'+warning
            for seed,record in sorted(spec['generation_qualification'].items()) for warning in record['warnings']]
    return result


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('family','scores','spec','binding','out'):p.add_argument('--'+name,required=True)
    a=p.parse_args();binding=load(a.binding)
    result=read(a.family,{}, {},binding) if binding.get('status')=='ARCHIVED UNREAD' else read(a.family,load(a.scores),load(a.spec),binding)
    atomic(a.out,result)
    print('\n'.join(text_lines(a.family,result,os.environ.get('KIT_V6_AUTHORITY_SHA'))))
    return PENDING if result['status'].startswith('PENDING') and not result.get('primary') else 0

if __name__=='__main__':raise SystemExit(main())
