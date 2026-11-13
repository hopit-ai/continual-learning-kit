"""TRACE owner-decision regressions. CPU-only, including torch refusal."""
import copy
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from kit import trace_common as c
from kit.trace_prepare import cohort
from kit.trace_judge import parse, pair_rows, calibration_summary
from kit.trace_read import descriptive, op_bwt, pages, parse_pages, split_return
from kit.trace_sft import suffix_example


class ProtocolTests(unittest.TestCase):
    def test_jsonl_unicode_separators_preserve_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'rows.jsonl'
            rows=[dict(id='one',answer='a'+chr(0x2028)+'b'+chr(0x85)+'c'),
                  dict(id='two',answer='d'+chr(0x2029)+'e')]
            path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),
                            encoding='utf-8')
            self.assertEqual(c.read_rows(dict(path=str(path),sha256=c.sha(path))),rows)

    def test_counts_and_shared_ancestry(self):
        rows = c.inventory()
        self.assertEqual(rows['comparisons'],3)
        self.assertEqual(sum(len(c.states(s,k)) for s in ('s8',*c.SOURCES) for k in c.SEEDS),99)
        self.assertEqual(sum(sum(x['stage']>0 for x in c.states(s,k))
                             for s in ('s8',*c.SOURCES) for k in c.SEEDS),84)
        checkpoints = {x['checkpoint'] for s in ('s8',*c.SOURCES) for k in c.SEEDS
                       for x in c.states(s,k) if x['stage']}
        self.assertEqual(len(checkpoints),78)
        for source in c.SOURCES:
            stages = c.stages(source,101)
            self.assertEqual(len(stages),9)
            self.assertEqual(stages[1]['incoming'],source+'-s101-parent')
            self.assertEqual(stages[5]['incoming'],source+'-s101-parent')
        # Qualification has direct scientific dependants in the same inventory.
        for row in rows['rows']:
            if row['phase']=='train':
                self.assertEqual(row['depends'],['qualify-'+row['study']])

    def test_length_filter_before_ranking_and_exact_schedule(self):
        rows = [dict(id=str(i),task='MeetingBank',prompt=f'transcript {i}',answer='summary')
                for i in range(510)]
        lengths = {r['id']:dict(student=2048,gold_teacher=6144,gold_response=2048) for r in rows}
        lengths['0']['student']=2049
        selected,excluded = cohort(rows,lengths)
        self.assertEqual(len(selected),500)
        self.assertNotIn('0',{r['id'] for r in selected})
        self.assertEqual(excluded[0]['id'],'0')
        ordered = c.schedule(selected,101,'MeetingBank')
        self.assertEqual(len(ordered),1280)
        self.assertEqual(ordered[:500],ordered[500:1000])
        self.assertEqual(ordered,c.schedule(list(reversed(selected)),101,'MeetingBank'))
        self.assertNotEqual(ordered,c.schedule(selected,102,'MeetingBank'))

    def test_judge_orders_and_strict_parser(self):
        self.assertEqual(parse('{"winner":"A"}'),1)
        self.assertEqual(parse('{"winner":"A"}',True),-1)
        self.assertEqual(parse('{"winner":"TIE"}',True),0)
        for value in ('A','[]','{"winner":"A","other":0}','{"winner":null}'):
            self.assertIsNone(parse(value))
        rows=pair_rows([dict(id='x',prompt='q',a='first',b='second')])
        self.assertIn('"answer_A": "first"',rows[0]['messages'][1]['content'])
        self.assertIn('"answer_A": "second"',rows[1]['messages'][1]['content'])

    def test_calibration_uses_original_frontier_agreement(self):
        pairs=[dict(id=str(i),panel=('Self-Instruct','LIMA','CoNa')[i//30]) for i in range(90)]
        labels=[dict(id=str(i),labels=[0,0]) for i in range(90)]
        raters={'gpt-6-astra':{str(i):0 for i in range(90)},'Claude Opus':{str(i):0 for i in range(90)}}
        result=calibration_summary(pairs,labels,raters)
        self.assertEqual(result['all']['H'],1)
        self.assertEqual(result['all']['J_minus_H_interval'],[0,0])
        labels[0]['labels']=[None,None]
        self.assertLess(calibration_summary(pairs,labels,raters)['all']['J'],1)

    def test_bwt_both_conventions(self):
        row=op_bwt([[0,0,0],[60,0,0],[55,70,0],[50,60,80]])
        self.assertAlmostEqual(row['OP'],190/3)
        self.assertAlmostEqual(row['BWT_TRACE'],-20/3)
        self.assertEqual(row['BWT_Tminus1'],-10)
        self.assertIsNone(op_bwt([[0],[1]])['BWT_TRACE'])

    def test_suffix_mask_and_left_clipping(self):
        class Tokenizer:
            eos_token_id=9999
            def apply_chat_template(self,*args,**kwargs): return 'p'*2000
            def encode(self,text,**kwargs): return list(text.encode())
        row=suffix_example(Tokenizer(),dict(prompt='p',answer='target'))
        self.assertEqual(len(row['input_ids']),1536)
        self.assertEqual(row['labels'][-7:],list(b'target')+[9999])
        self.assertEqual(row['labels'].count(-100),1529)

    def test_numeric_transport_and_typo_isolation(self):
        payload=dict(schema='trace-return.v2', reporting='DESCRIPTIVE',
                     transcription='TRANSCRIPTION-UNVERIFIED', checksums='ADVISORY',
                     records={'counts':[1234,0], 'estimate':.123456789, 'hash':'a'*64})
        frames=pages(payload,'test')
        self.assertEqual(parse_pages(frames),payload)
        self.assertTrue(all(len(t.splitlines())<=75 and max(map(len,t.splitlines()))<90
                            and t.isascii() for t in frames))
        frames[0]=frames[0].replace('counts=[1234,0]','counts=[123x,0]')
        imported=parse_pages(frames)
        self.assertEqual(imported['records'],{'estimate':.123456789, 'hash':'a'*64})
        self.assertTrue(imported['invalid_records'])
        self.assertEqual(imported['transcription'],'TRANSCRIPTION-UNVERIFIED')
        # A digit typo is retained as transcribed. Checksums remain advisory.
        frames=pages(payload,'test')
        frames[0]=frames[0].replace('1234','1235').replace('SHA256 ','SHA256 x')
        self.assertEqual(parse_pages(frames)['records']['counts'],[1235,0])

    def test_complete_cpu_rehearsal_idempotency_and_dependencies(self):
        from kit.trace_jobs import run
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            config=root/'config.json'
            c.atomic(config,dict(out=tmp,technical_synthetic=True,estimated_ceiling_gpu_h=-1))
            with self.assertRaises(FileNotFoundError):run(config,'targets',True)
            for row in c.inventory()['rows']:
                receipt=run(config,row['id'],True)
                self.assertEqual(receipt['status'],'COMPLETE')
                self.assertEqual(receipt['gpu_hours'],0)
            first=run(config,'train-gold-101',True)
            self.assertEqual(first,run(config,'train-gold-101',True))
            result=c.load(root/'descriptive.json')
            self.assertEqual(result['reporting'],'DESCRIPTIVE')
            self.assertEqual(set(result['sources']),set(c.SOURCES))
            self.assertFalse(any(word in json.dumps(result) for word in ('"pass"','"verdict"','INCONCLUSIVE')))
            frames=[p.read_text() for p in sorted(root.glob('return-*.txt'))]
            payload=c.load(root/'return-numeric.json')
            self.assertEqual(parse_pages(frames),payload)
            keys=payload['records']
            for prefix in ('departures.','gpu_h.','EOS.','cap_rates.','licence.','recipe.'):
                self.assertTrue(any(k.startswith(prefix) for k in keys),prefix)
            self.assertIn('utilisation.mean',keys)
            self.assertLessEqual(len(frames),3)
            full_frames=[p.read_text() for p in sorted((root/'full').glob('return-*.txt'))]
            full=c.load(root/'full/return-numeric.json')
            self.assertEqual(parse_pages(full_frames),full)
            self.assertEqual(keys['full.sha256'],c.digest(full))
            self.assertEqual(keys['missing.full_spill'],0)
            for prefix in ('SA.','J.','C.','A.','tokens_caps.'):
                self.assertTrue(any(k.startswith(prefix) for k in full['records']),prefix)
                self.assertFalse(any(k.startswith(prefix) for k in keys),prefix)
            for prefix in ('OP.s8.101.','OP.gold.101.D.','OP.27b.103.F.','effect.gold.101.'):
                self.assertTrue(any(k.startswith(prefix) for k in keys),prefix)
            corrupted=list(frames)
            record=next(line for line in corrupted[0].splitlines() if line.startswith('EOS.'))
            corrupted[0]=corrupted[0].replace(record,record.replace('=',':',1))
            imported=parse_pages(corrupted)
            self.assertEqual(len(imported['records']),len(keys)-1)
            self.assertTrue(imported['invalid_records'])

    def test_legacy_output_tokens_remain_unmeasured(self):
        from kit.trace_read import output_diagnostics
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            c.atomic(root/'scores/legacy.json',dict(outputs=[dict(tokens=10,finish_reason='stop')]))
            self.assertEqual(output_diagnostics(root)['score'],[1,None,10,None,0])

    def test_legacy_score_records_keep_diagonal_final_order(self):
        from kit.trace_read import returned_scores
        legacy=dict(records={'S.s8.101.FOMC.0':[2000,120000,160000]})
        self.assertEqual(returned_scores(legacy),
                         {'s8':{'101':{'S2':{'FOMC':60.},'S8':{'FOMC':80.}}}})

    def test_essential_spills_and_full_has_no_page_cap(self):
        from kit.trace_read import score_records, returned_scores
        records={}
        numeric={study:{str(seed):{state['id']:{metric:[199999.123456789,2000]
            for metric in (*(c.TASKS if study=='s8' else c.SUBSET),
                           'instruction','maths','knowledge',
                           *(('MMLU','BBH','GSM8K','TyDiQA','PIQA','BoolQ') if study=='s8' else ()))}
            for state in c.states(study,seed)} for seed in c.SEEDS} for study in ('s8',*c.SOURCES)}
        score_records(records,numeric)
        # Both candidate overflow and unrestricted retry/telemetry evidence.
        for i in range(1000):
            records[f'diagnostic.{i}']=[69.123456789,51.123456789,True]
            records[f'A.{i}']=['score',1.234,'ERROR',1]
            records[f'U.{i}']=[69.123456789,51.123456789,True]
        records['failures']=1000
        payload=dict(schema='trace-return.v2',reporting='DESCRIPTIVE',
                     transcription='TRANSCRIPTION-UNVERIFIED',checksums='ADVISORY',records=records)
        essential,frames,full_frames=split_return(payload,'test')
        self.assertLessEqual(len(frames),3)
        self.assertGreater(len(full_frames),8)
        self.assertGreater(essential['records']['missing.full_spill'],0)
        self.assertEqual(essential['records']['failures'],1000)
        self.assertEqual(parse_pages(frames),essential)
        self.assertEqual(parse_pages(full_frames),payload)
        # FULL carries off-diagonal S8 and standalone general-panel values too.
        reconstructed=returned_scores(payload)
        for study,seeds in numeric.items():
            for seed,cells in seeds.items():
                for state,metrics in cells.items():
                    for metric,(total,n) in metrics.items():
                        self.assertEqual(reconstructed[study][seed][state][metric],total/n)
        for packet,expected in ((frames,essential),(full_frames,payload)):
            record=next(line for line in packet[0].splitlines() if '=' in line)
            corrupted=[packet[0].replace(record,record.replace('=',':',1)),*packet[1:]]
            imported=parse_pages(corrupted)
            self.assertEqual(len(imported['records']),len(expected['records'])-1)
            self.assertTrue(imported['invalid_records'])
            self.assertEqual(imported['checksum_mismatches'],[1])
        self.assertTrue(all(len(t.splitlines())<=75 and max(map(len,t.splitlines()))<90
                            and t.isascii() for t in [*frames,*full_frames]))

    def test_failure_keeps_attempt_without_completion(self):
        from kit.trace_jobs import run
        with tempfile.TemporaryDirectory() as tmp:
            config=Path(tmp)/'config.json'
            c.atomic(config,dict(out=tmp,technical_synthetic=True))
            with patch('kit.trace_jobs.execute',side_effect=RuntimeError('fault')):
                with self.assertRaises(RuntimeError):run(config,'prepare',True)
            self.assertFalse((Path(tmp)/'state/prepare.json').exists())
            receipts=list((Path(tmp)/'attempts').glob('*/*/receipt.json'))
            self.assertEqual(c.load(receipts[0])['status'],'ERROR')
            self.assertEqual(run(config,'prepare',True)['status'],'COMPLETE')

    def test_collect_many_attempts_and_weighted_utilisation(self):
        from kit.trace_jobs import run
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            config=root/'config.json'
            c.atomic(config,dict(out=tmp,technical_synthetic=True,studies=['gold']))
            for row in c.inventory(['gold'])['rows']:
                if row['id'] != 'collect': run(config,row['id'],True)
            for i in range(100):
                c.atomic(root/'attempts/score-gold-101'/f'failed-{i}'/'receipt.json',
                    dict(row='score-gold-101',status='ERROR',gpu_hours=2.,gpus=1,
                         started=0.,ended=7200.,warnings=[],error_type='RuntimeError'))
                c.atomic(root/'utilisation'/f'run-{i}.summary.json',
                    dict(mean_util=50. if i<50 else 100.,min_window_mean=30.,
                         telemetry_complete=True,gpus=1 if i<50 else 2,wall_s=100.))
            c.atomic(root/'utilisation/incomplete.summary.json',
                dict(mean_util=None,min_window_mean=None,telemetry_complete=False))
            c.atomic(root/'utilisation/legacy.summary.json',
                dict(mean_util=99.,min_window_mean=90.,telemetry_complete=True))
            receipt=run(config,'collect',True)
            self.assertEqual(receipt['status'],'COMPLETE')
            essential=c.load(root/'return-numeric.json')['records']
            full=c.load(root/'full/return-numeric.json')['records']
            self.assertEqual(essential['failures'],100)
            self.assertEqual(essential['gpu_h.score'],200.)
            self.assertEqual(essential['utilisation.allocations'],[102,100,2])
            self.assertAlmostEqual(essential['utilisation.mean'],250/3)
            self.assertEqual(essential['utilisation.min_window'],30.)
            self.assertEqual(essential['utilisation.below70'],50)
            self.assertIn('missing.utilisation',essential)
            self.assertEqual(len([k for k in full if k.startswith('U.') and k.count('.')==1]),102)
            self.assertEqual(len([k for k in full if k.startswith('A.') and k.count('.')==1]),
                             100+len(c.inventory(['gold'])['rows'])-1)
            self.assertLessEqual(len(list(root.glob('return-*.txt'))),3)
            self.assertGreater(len(list((root/'full').glob('return-*.txt'))),8)
            self.assertTrue(any(k.startswith('full/return-') for k in receipt['artifacts_sha256']))
            self.assertEqual(run(config,'collect',True),receipt)

    def test_tiny_native_resume_mask_ema_and_exposures(self):
        from kit.trace_rehearsal import tiny_native
        full=tiny_native(101,True)
        prefix=tiny_native(101,True,steps=20)
        resumed=tiny_native(101,True,resume=json.loads(json.dumps(prefix)))
        self.assertEqual(full,resumed)
        self.assertEqual(full['student'][:3],[.1,.2,.3])
        self.assertNotEqual(full['student'][3:],[.4,.5,.6])
        self.assertNotEqual(full['student'],full['teacher'])
        self.assertEqual(full['exposures'],1280)
        self.assertEqual(full['momentum'][:3],[0.,0.,0.])

    def test_short_label_prompt_adapter(self):
        from kit.trace_prompt import check_clean
        check_clean([dict(role='user',content='Classify A versus B')],'A')
        with self.assertRaises(ValueError):
            check_clean([dict(role='user',content='q',demonstration='A')],'A')
        text='complete privileged solution '*4
        with self.assertRaises(ValueError):
            check_clean([dict(role='user',content=text)],text)

    def test_bpe_newline_suffix_is_censused_before_gpu(self):
        from kit.trace_sft import sft_census
        # Actual pair-rank BPE, including the newline/newline merge used by
        # byte-level tokenizers. Unlike a byte fixture, this crosses the boundary.
        class BytePairTokenizer:
            eos_token_id=9999
            def apply_chat_template(self,*args,**kwargs): return 'prompt\n'
            def encode(self,text,**kwargs):
                symbols=list(text)
                ranks={('\n','\n'):0, ('a','b'):1}
                while len(symbols)>1:
                    pairs=[(ranks.get((a,b),100),i)
                           for i,(a,b) in enumerate(zip(symbols,symbols[1:]))]
                    rank,index=min(pairs)
                    if rank==100: break
                    symbols[index:index+2]=[''.join(symbols[index:index+2])]
                return [10000+ord(x) if len(x)==1 else 20000+sum(map(ord,x)) for x in symbols]
        tok=BytePairTokenizer()
        good=dict(id='good',prompt='p',answer='abc')
        bad=dict(id='newline',prompt='p',answer='\nabc')
        encoded,report=sft_census([good,bad],tok)
        self.assertEqual(len(encoded),1)
        self.assertEqual(report['error_ids'],['newline'])
        self.assertEqual(report['suffix_errors'],1)
        self.assertFalse(report['valid'])
        with self.assertRaises(ValueError): suffix_example(tok,bad)

    def test_l_eos_relative_warning_never_gates_share_and_upstream_clipping(self):
        from kit.trace_sft import sft_census, deepspeed_config, eos_comparison
        class Tokenizer:
            eos_token_id=9999
            def apply_chat_template(self,*args,**kwargs): return 'p'*1535
            def encode(self,text,**kwargs): return list(map(ord,text))
        short=dict(id='short',prompt='p',answer='')
        capped=dict(id='capped',prompt='p',answer='a')
        _,report=sft_census([short]*99+[capped],Tokenizer(),Tokenizer())
        self.assertEqual(report['affected'],1)
        self.assertEqual(report['share'],.01)
        self.assertTrue(report['valid'])
        self.assertEqual(report['warnings'],[])
        self.assertEqual(report['max_eos_position_difference'],1)
        self.assertEqual(report['eos_position_errors'],0)
        class Original(Tokenizer):
            def apply_chat_template(self,*args,**kwargs): return 'p'
        _,report=sft_census([capped]*100,Tokenizer(),Original())
        self.assertEqual(report['share'],1.)
        self.assertEqual(report['original_share'],0.)
        self.assertEqual(report['excess_share'],1.)
        self.assertTrue(report['valid'])
        self.assertTrue(report['warnings'])
        _,same=sft_census([capped]*100,Tokenizer(),Tokenizer())
        self.assertTrue(same['valid'])
        self.assertEqual(same['warnings'],[])
        _,missing=sft_census([capped],Tokenizer())
        self.assertTrue(missing['valid'])
        self.assertIsNone(missing['original_share'])
        self.assertIn('UNMEASURED',missing['warnings'][0])
        # The same one-position bound covers uncapped and capped answers.
        for n in (1535,1536,1537,3000):
            encoded,answer_report=sft_census([dict(capped,answer='a'*n)],Tokenizer(),Original())
            self.assertTrue(answer_report['valid'])
            self.assertEqual(answer_report['max_eos_position_difference'],1)
            self.assertEqual(encoded[0]['labels'][-1],Tokenizer.eos_token_id)
            self.assertEqual(eos_comparison([1]*n,[1]*n,9999),(n>=1536,True))
        with patch('kit.trace_sft.eos_comparison',return_value=(True,False)):
            self.assertFalse(sft_census([capped],Tokenizer())[1]['valid'])
        # Check the actual emitted streams, catching a future two-EOS regression.
        from kit.trace_sft import suffix_tokens
        def two_eos(combined,answer,eos):
            result=suffix_tokens(combined,answer,eos)
            result['input_ids']=result['input_ids'][1:]+[eos]
            return result
        with patch('kit.trace_sft.suffix_tokens',side_effect=two_eos):
            invalid=sft_census([capped],Tokenizer(),Original())[1]
            self.assertFalse(invalid['valid'])
            self.assertEqual(invalid['eos_position_errors'],1)
        self.assertEqual(deepspeed_config()['gradient_clipping'],1.)
        self.assertNotIn('G-clip',c.DEPARTURES)

    def test_export_crash_before_and_after_seal(self):
        from kit.trace_sft import export_plan, finish_export
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            crashed=export_plan(root,1)
            (Path(crashed['target'])/'pytorch_model.bin').write_bytes(b'partial')
            retry=export_plan(root,1)
            self.assertNotEqual(crashed['target'],retry['target'])
            target=Path(retry['target'])
            (target/'pytorch_model.bin').write_bytes(b'complete')
            (target/'config.json').write_text('{}')
            record=finish_export(root,1,target)
            # Crash after seal, before the next native checkpoint. Replay
            # reuses the exact sealed bytes instead of overwriting them.
            resumed=export_plan(root,1)
            self.assertTrue(resumed['reuse'])
            self.assertEqual(resumed['record'],record)
            (target/'pytorch_model.bin').write_bytes(b'corrupt')
            with self.assertRaises(ValueError): export_plan(root,1)

    def test_indexed_overlap_matches_brute_force_and_records_review(self):
        from kit.trace_overlap import audit_rows
        from kit.p3b_replay_data import prepared, near
        train=[dict(id='train',task='FOMC',group='train-group',
                    prompt='one two three four five six seven eight nine ten',answer='A')]
        held=[dict(id='held',task='FOMC',group='held-group',
                   prompt='one two three four five six seven eight nine ten!',answer='B'),
              dict(id='far',task='FOMC',group='other-group',prompt='unrelated',answer='A')]
        datasets={'FOMC':{'train':train,'eval':held,'test':[]}}
        report=audit_rows(datasets,[])
        brute=[r['id'] for r in held if near(prepared(train[0]['prompt']),prepared(r['prompt']))]
        self.assertEqual([r['heldout'][1] for r in report['flagged']],brute)
        self.assertEqual(report['unresolved'],1)
        reviews=[dict(id=report['flagged'][0]['id'],disposition='cleared',evidence_sha256='a'*64)]
        self.assertEqual(audit_rows(datasets,[],reviews)['unresolved'],0)
        reviews[0]['disposition']='confirmed_overlap'
        self.assertEqual(audit_rows(datasets,[],reviews)['unresolved'],1)
        # Rotations preserve shingle overlap while failing the edit bound.
        # Rare-prefix screening must keep both word and Chinese candidates.
        words=['word'+str(i) for i in range(100)]
        chinese=''.join(chr(0x4e00+i) for i in range(100))
        for first,second in [(' '.join(words),' '.join(words[40:]+words[:40])),
                             (chinese,chinese[40:]+chinese[:40])]:
            source=dict(train[0],prompt=first)
            evaluation=dict(held[0],prompt=second)
            rotated=audit_rows({'FOMC':{'train':[source],'eval':[evaluation],'test':[]}},[])
            self.assertEqual(rotated['unresolved'],1)

    def test_text_index_prepares_unique_content_once(self):
        from kit import trace_overlap
        from kit.p3b_replay_data import prepared
        rows=[dict(id=str(i),task='FOMC',group='g'+str(i),prompt='same',answer='A')
              for i in range(20)]
        with patch.object(trace_overlap,'prepared',wraps=prepared) as prepare_once:
            result=trace_overlap.audit_rows({'FOMC':{'train':rows,'test':[]}},[])
        self.assertEqual(prepare_once.call_count,2)
        self.assertEqual(result['duplicates']['FOMC/train']['unique'],1)

    def test_missing_grants_scope_and_five_hundred_distinct(self):
        from kit.trace_prepare import licence_blocks
        rights={k:dict(permitted=True,evidence_sha256='a'*64)
                for k in (*c.SUBSET,'original_tokenizer','base','teacher','judge','panels','calibration')}
        blocked=licence_blocks(rights,['s8',*c.SOURCES])
        self.assertEqual(set(blocked),{'s8'})
        self.assertNotIn('C-STANCE',licence_blocks(rights,list(c.SOURCES)))
        rights['FOMC']['permitted']=False
        self.assertEqual(set(licence_blocks(rights,['s8',*c.SOURCES])),{'s8',*c.SOURCES})
        rows=[dict(id=str(i),task='FOMC',prompt=f'content {i}',answer='A') for i in range(501)]
        rows.append(dict(id='duplicate',task='FOMC',prompt='CONTENT   0',answer='a'))
        lengths={r['id']:dict(student=1,gold_teacher=1,gold_response=1) for r in rows}
        chosen,excluded=cohort(rows,lengths)
        self.assertEqual(len(chosen),500)
        self.assertTrue(any(r['reason']=='duplicate_content' for r in excluded))
        self.assertEqual(len({(r['prompt'].casefold(),r['answer'].casefold()) for r in chosen}),500)
        for row in rows: row['prompt']='same'; row['answer']='A'
        with self.assertRaisesRegex(ValueError,'distinct'): cohort(rows,lengths)

    def test_prepare_needs_only_three_cleared_task_datasets(self):
        from kit.trace_prepare import prepare
        from kit.score_forgetting import load_panel
        from types import SimpleNamespace
        general,_=load_panel()
        rights={k:dict(permitted=True,evidence_sha256='a'*64)
                for k in (*c.SUBSET,'original_tokenizer','base','teacher','judge','panels','calibration')}
        datasets={task:{'train':[dict(id=f'{task}-{i}',task=task,group=f'{task}-g{i}',
                     prompt=f'candidate {i}',answer='1') for i in range(500)],
                     'eval':[], 'test':[dict(id=f'{task}-test',task=task,group=f'{task}-test',
                     prompt='held-out prompt',answer='1')]} for task in c.SUBSET}
        panels={'general-v1':general,'open':[dict(id=f'{panel}-{i}',panel=panel,prompt='q')
                    for panel,n in [('Self-Instruct',175),('LIMA',300),('CoNa',178)] for i in range(n)]}
        calibration=[dict(id=f'{panel}-{i}',panel=panel,checkpoint_independent=True)
                     for panel in ('Self-Instruct','LIMA','CoNa') for i in range(30)]
        def lengths(rows,tok):
            return {r['id']:dict(student=1,gold_teacher=1,gold_response=1) for r in rows}
        fake=SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**k:object()))
        with tempfile.TemporaryDirectory() as tmp:
            cfg=dict(out=tmp,base=dict(revision=c.BASE,model='stub'),judge=c.JUDGE,
                teacher=dict(revision='6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'),
                original_tokenizer={},licences=rights,datasets=datasets,panels=panels,
                calibration=calibration,exclusions=[],sec_clearance=dict(
                    reserved_outcomes_opened=False,intersections=0,near_match_clearance=True))
            with patch('kit.trace_prepare.model_record'), \
                 patch('kit.trace_prepare.bound_json',side_effect=lambda x:x), \
                 patch('kit.trace_prepare.read_rows',side_effect=lambda x:x), \
                 patch('kit.trace_prepare.census',side_effect=lengths), \
                 patch('kit.trace_prepare.audit_rows',return_value=dict(flagged=[],unresolved=0)), \
                 patch.dict(sys.modules,transformers=fake):
                result=prepare(cfg)
            self.assertEqual(set(result['selected']),set(c.SUBSET))
            self.assertEqual(result['active_studies'],list(c.SOURCES))
            self.assertEqual(set(result['blocked_studies']),{'s8'})
            self.assertEqual(result['sft_rows'],{})
            self.assertEqual(result['registration_sha256'],c.sha(c.REG))

    def test_prepare_high_eos_share_warns_but_wrong_tokens_block(self):
        from kit.trace_prepare import prepare
        from kit.trace_sft import sft_census
        from kit.score_forgetting import load_panel
        from types import SimpleNamespace
        class Tokenizer:
            eos_token_id=9999
            def apply_chat_template(self,*args,**kwargs): return 'p'*1535
            def encode(self,text,**kwargs): return list(map(ord,text))
        class Original(Tokenizer):
            def apply_chat_template(self,*args,**kwargs): return 'p'
        encoded,report=sft_census([dict(id='capped',prompt='p',answer='a')],
                                 Tokenizer(),Original())
        self.assertEqual(report['share'],1.)
        self.assertTrue(report['valid'])
        general,_=load_panel()
        rights={k:dict(permitted=True,evidence_sha256='a'*64) for k in
                (*c.TASKS,'original_tokenizer','base','judge','panels','calibration')}
        datasets={task:dict(train=[dict(id=str(i)) for i in range(5000)],eval=[],test=[])
                  for task in c.TASKS}
        panels={'general-v1':general,'trace_general':dict.fromkeys(
                ('MMLU','BBH','GSM8K','TyDiQA','PIQA','BoolQ')),
                'open':[dict(id=f'{panel}-{i}',panel=panel,prompt='q')
                    for panel,n in [('Self-Instruct',175),('LIMA',300),('CoNa',178)] for i in range(n)]}
        calibration=[dict(id=f'{panel}-{i}',panel=panel,checkpoint_independent=True)
                     for panel in ('Self-Instruct','LIMA','CoNa') for i in range(30)]
        fake=SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**k:object()))
        with tempfile.TemporaryDirectory() as tmp:
            for valid in (True,False):
                cfg=dict(out=str(Path(tmp)/str(valid)),studies=['s8'],
                    base=dict(revision=c.BASE,model='stub'),judge=c.JUDGE,
                    original_tokenizer=dict(model='original'),licences=rights,datasets=datasets,
                    panels=panels,calibration=calibration,exclusions=[],sec_clearance=dict(
                        reserved_outcomes_opened=False,intersections=0,near_match_clearance=True))
                # Isolate prepare's readiness logic using the actual high-share census.
                summary=dict(report,valid=valid,suffix_errors=0 if valid else 1)
                with patch('kit.trace_prepare.model_record'), \
                     patch('kit.trace_prepare.bound_json',side_effect=lambda x:x), \
                     patch('kit.trace_prepare.read_rows',side_effect=lambda x:x), \
                     patch('kit.trace_general_build.validate', return_value=True), \
                     patch('kit.trace_prepare.audit_rows',return_value=dict(flagged=[],unresolved=0)), \
                     patch('kit.trace_sft.sft_census',return_value=(encoded,summary)), \
                     patch.dict(sys.modules,transformers=fake):
                    result=prepare(cfg)
                self.assertEqual(result['s8_ready'],valid)
                self.assertEqual('s8' in result['blocked_studies'],not valid)
                self.assertTrue(result['warnings'])
                self.assertEqual(set(result['sft_rows']),set(c.TASKS) if valid else set())

    def test_subset_only_rehearsal_and_label_adapter(self):
        from kit.trace_jobs import run
        from kit.trace_entry import compute_score
        from kit.trace_train import converted_trace_rows
        self.assertEqual(compute_score('trace','','')['score'],0.)
        with self.assertRaises(ValueError): compute_score('contract_nli','','')
        with patch('kit.trace_train.nm_converted_rows',return_value=([
                dict(data_source='contract_nli',extra_info=dict(m3_domain='trace_FOMC'))],{})):
            rows,_=converted_trace_rows()
            self.assertEqual(rows[0]['data_source'],'trace')
            self.assertEqual(rows[0]['extra_info']['m3_domain'],'trace_FOMC')
        scoped=c.inventory(list(c.SOURCES))
        self.assertEqual(scoped['comparisons'],2)
        self.assertEqual(scoped['trained_cells'],60)
        with tempfile.TemporaryDirectory() as tmp:
            config=Path(tmp)/'config.json'
            c.atomic(config,dict(out=tmp,technical_synthetic=True,studies=list(c.SOURCES)))
            for row in c.inventory(list(c.SOURCES))['rows']:
                self.assertEqual(run(config,row['id'],True)['status'],'COMPLETE')
            summary=c.load(Path(tmp)/'descriptive.json')
            self.assertTrue((Path(tmp)/'judge/calibration-labels.json').exists())
            self.assertIsNone(summary['baseline'])
            self.assertEqual(set(summary['sources']),set(c.SOURCES))

    def test_shipped_registration_and_all_return_fields(self):
        self.assertEqual(c.REG.relative_to(c.KIT),Path('resources/trace/registration.md'))
        text=c.REG.read_text()
        self.assertIn('≤1 EOS-position shift/row, 0 wrong tokens',text)
        self.assertIn('never blocks S8 at prepare',text)
        self.assertIn('3c2f2322cc93c109adfeda96c2c906fcb0d1192b',text)
        self.assertNotIn('G-clip',text)
        self.assertIn('at most three pages',text)
        self.assertIn('no hard page cap',text)
        # Match the editable registration to the actual runtime bytes in source checkouts.
        draft=c.KIT.parent/'docs/phase4/trace-registration-DRAFT-20261111.md'
        if draft.exists(): self.assertEqual(c.REG.read_bytes(),draft.read_bytes())

    def test_without_torch_import(self):
        code='''
import importlib.abc, sys
class RefuseTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch','transformers','vllm','ray','deepspeed'):
            raise RuntimeError('GPU import attempted: '+fullname)
sys.meta_path.insert(0,RefuseTorch())
from kit import trace_jobs, trace_prepare, trace_score, trace_judge, trace_read, trace_train
assert trace_jobs.inventory()['comparisons']==3
'''
        subprocess.run([sys.executable,'-c',code],check=True)

    def test_registration_owner_record_and_no_scientific_verdicts(self):
        text=c.REG.read_text()
        for quote in ('Can we try both','Report TRACE descriptively','Add general-v1',
                      'B: local open-weight judge','Every state','We can use Astra and Fable in high mode',
                      'Build now; send with the next message'):
            self.assertIn(quote,text)
        for word in ('PASS','FAIL','INCONCLUSIVE','human–human'):
            self.assertNotIn(word,text)
        self.assertEqual(c.estimates()['judge_calls'],84*653*2+180)


if __name__=='__main__':
    unittest.main()
