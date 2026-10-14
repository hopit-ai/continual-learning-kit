#!/usr/bin/env python3
"""Partner job: pinned replay download, exclusions, one base generation, checked freeze."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.p3b_common import *
import importlib.util
import inspect
import unicodedata
SOURCE = '47c03c73621c4aab2b824b7818681117d662770e'
CHECKER = 'afea345e3bc100e6d5588ca14e7269141e8810e1489e60ad64a59cd57360201c'


def normal(text):
    if isinstance(text,list): text='\n'.join(m['content'] for m in text)
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def duplicate(a, b):
    a, b = normal(a), normal(b)
    if a == b: return True
    def shingles(x):
        words = x.split(); return {tuple(words[i:i+5]) for i in range(len(words)-4)}
    x, y = shingles(a), shingles(b)
    if x and y and len(x & y) / len(x | y) >= .8: return True
    # Normalised Levenshtein similarity: 1 - distance / max(lengths).
    limit = max(len(a),len(b)) // 10
    if abs(len(a)-len(b)) > limit: return False
    try:
        from rapidfuzz.distance import Levenshtein
        return Levenshtein.distance(a,b,score_cutoff=limit) <= limit
    except ImportError: pass
    previous = {j:j for j in range(min(len(b),limit)+1)}
    for i, char in enumerate(a,1):
        current = {0:i} if i <= limit else {}
        for j in range(max(1,i-limit),min(len(b),i+limit)+1):
            current[j]=min(previous.get(j,limit+1)+1,current.get(j-1,limit+1)+1,previous.get(j-1,limit+1)+(char!=b[j-1]))
        if min(current.values(),default=limit+1)>limit: return False
        previous=current
    return previous.get(len(b),limit+1)<=limit


def checker(text, ground_truth, *, validate_only=False):
    check(R / 'if_functions.py', CHECKER)
    spec = importlib.util.spec_from_file_location('p3b_if', R / 'if_functions.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    try: constraints = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    except ValueError as exc: raise CorrectnessStop('malformed ground_truth JSON') from exc
    if isinstance(constraints, dict): constraints = [constraints]
    require(isinstance(constraints, list) and bool(constraints), 'malformed ground_truth')
    results = []
    for row in constraints:
        require(isinstance(row, dict) and row.get('func_name') in module.IF_FUNCTIONS_MAP, 'unknown checker function')
        args = {k: v for k, v in row.items() if k != 'func_name' and v is not None}
        function = module.IF_FUNCTIONS_MAP[row['func_name']]
        try: inspect.signature(function).bind(text, **args)
        except TypeError as exc: raise CorrectnessStop('malformed checker arguments: ' + str(exc)) from exc
        if validate_only: continue
        try: result = function(text, **args)
        except getattr(module.langdetect, 'LangDetectException', ()):
            result = False  # Empty/unclassifiable generated text fails the constraint.
        except (TypeError, ValueError) as exc: raise CorrectnessStop('malformed checker arguments: ' + str(exc)) from exc
        require(isinstance(result, bool), 'checker returned non-boolean')
        results.append(dict(func_name=row['func_name'], passed=result))
    return results


def clean(rows, panels):
    kept, exclusions = [], []
    for row in rows:
        match = next((other for other in panels + kept if str(row['id']) == str(other['id']) or duplicate(row['prompt'], other['prompt'])), None)
        if match: exclusions.append(dict(id=row['id'], overlaps=match['id'], prompt_sha256=t.sha(row['prompt'])))
        else: kept.append(row)
    return kept, exclusions


def job(a):
    root = a.out / 'replay'; root.mkdir(parents=True, exist_ok=True)
    require(a.source_revision == SOURCE and a.checker_revision == '11826255077617a46919ce75cadf1f3d53f30dac', 'unpinned replay input')
    frozen_hashes(); check(R / 'if_functions.py', CHECKER)
    if not a.standin:
        from kit.p3a_common import frozen, INPUTS
        require(a.package_a is not None and a.finqa_panel is not None, 'package-A frozen panels and FinQA panel required')
        frozen(a.package_a, False)
        from kit.v4_contract import check_heldout_rows, check_eval_items
        finqa=t.read_jsonl(a.finqa_panel)
        require(len(finqa)==1147,'FinQA exclusions require all 1147 heldout questions')
        if all('extra_info' in r for r in finqa):
            check_heldout_rows(finqa)
            finqa=[dict(id=str(r['extra_info']['index']),prompt=r['prompt']) for r in finqa]
        else: check_eval_items(finqa,'finqa','test')
        require(len({r['id'] for r in finqa})==1147,'duplicate FinQA exclusion IDs')
        panels = t.read_jsonl(KIT / 'panels/general-v1.jsonl') + finqa
        for task in s.ORDER: panels += t.read_jsonl(a.package_a / INPUTS / 'prepared' / task / 'heldout.jsonl')
        model = identity(a.model, 'student')
        if not a.parquet:
            from huggingface_hub import HfApi, hf_hub_download
            names = [n for n in HfApi().list_repo_files('allenai/RLVR-IFeval', repo_type='dataset', revision=SOURCE) if n.endswith('.parquet')]
            require(len(names) == 1, 'expected one train parquet')
            a.parquet = Path(hf_hub_download('allenai/RLVR-IFeval', names[0], repo_type='dataset', revision=SOURCE))
        import pyarrow.parquet as pq
        raw = pq.read_table(a.parquet).to_pylist(); rows = []
        for i, row in enumerate(raw):
            prompt = row.get('prompt', row.get('messages'))
            if isinstance(prompt, list):
                require(len(prompt) == 1 and prompt[0]['role'] == 'user', 'unexpected source chat shape')
                prompt = prompt[0]['content']
            require(isinstance(prompt, str) and bool(prompt.strip()), 'missing source prompt')
            gt = row.get('ground_truth', (row.get('reward_model') or {}).get('ground_truth'))
            checker(None, gt, validate_only=True)  # Bind arguments without scoring.
            rows.append(dict(id=str(row.get('id', row.get('key', i))), prompt=prompt, ground_truth=gt))
        provenance = dict(parquet_sha256=s.file_hash(a.parquet), model=model)
    else:
        require(a.rows is not None, '--rows required for CPU stand-in')
        rows, panels = t.read_jsonl(a.rows), []
        provenance = dict(rows_sha256=s.file_hash(a.rows), technical_synthetic=True)
    bind(root / 'inputs.json', dict(source_revision=SOURCE, checker=CHECKER, panels_sha256=s.digest(panels), **provenance))
    if (root / 'manifest.json').exists(): frozen_files(root); print('SKIP: frozen replay'); return
    eligible, excluded = clean(rows, panels)
    candidates = sorted(eligible, key=lambda r: (t.sha('package-b-replay-v1|' + r['id']), r['id']))[:2048]
    t.write_jsonl(root / 'exclusions.jsonl', excluded); t.write_jsonl(root / 'candidates.jsonl', candidates)
    os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu, V4_PHASE0_INFERENCE='0')
    import time
    started=time.monotonic()
    generator = None
    accepted = []; verdicts=[]
    for row in candidates:
        path = root / 'raw' / (t.sha(row['id']) + '.json')
        require(path.exists() or not (root/'checked'/(t.sha(row['id'])+'.json')).exists(),'completed replay response missing; do not regenerate')
        if path.exists(): output = json.loads(path.read_text())
        elif a.standin:
            output = dict(text=row['fake_response'], finish_reason='stop', technical_synthetic=True); atomic(path, output)
        else:
            if generator is None: generator = t.VLLMGenerator(a.model, 2048, tensor_parallel_size=1, allow_batch_fallback=False)
            output = generator([dict(role='user', content=row['prompt'])], 1, seed=0, temperature=0, top_p=1)
            atomic(path, output)
        verdict=root/'checked'/(t.sha(row['id'])+'.json')
        evidence=dict(id=row['id'],raw_sha256=s.digest(output),ground_truth_sha256=s.digest(row['ground_truth']))
        if verdict.exists():
            checked=json.loads(verdict.read_text())
            require(all(checked[k]==v for k,v in evidence.items()),'checked replay evidence changed')
        else:
            checked=dict(**evidence,checks=checker(output['text'],row['ground_truth']),complete=output['finish_reason'] in ('stop','eos'))
            atomic(verdict,checked)
        verdicts.append(checked); results=checked['checks']
        if checked['complete'] and all(r['passed'] for r in results):
            accepted.append(dict(id=row['id'], prompt=row['prompt'], response=output['text'], checks=results, ground_truth=row['ground_truth']))
    t.write_jsonl(root/'candidate-checks.jsonl',verdicts);t.write_jsonl(root/'accepted.jsonl',accepted)
    require(len(accepted) >= 65, 'fewer than 65 accepted examples; no extra generation')
    for name, values in [('accepted', accepted), ('refresher-64', accepted[:64]), ('replay', accepted[64:])]: t.write_jsonl(root / (name + '.jsonl'), values)
    atomic(root / 'cost.json',dict(kind='replay_preparation',gpu_hours=0 if a.standin else (time.monotonic()-started)/3600,technical_synthetic=a.standin))
    atomic(root / 'manifest.json', files_manifest(root, ['accepted.jsonl', 'refresher-64.jsonl', 'replay.jsonl', 'candidates.jsonl', 'exclusions.jsonl', 'inputs.json', 'candidate-checks.jsonl'], accepted=len(accepted), technical_synthetic=a.standin))


def configure(p):
    device(p); p.add_argument('--package-a', type=Path); p.add_argument('--finqa-panel', type=Path)
    p.add_argument('--parquet', type=Path); p.add_argument('--rows', type=Path)
    p.add_argument('--source-revision', default=SOURCE); p.add_argument('--checker-revision', default='11826255077617a46919ce75cadf1f3d53f30dac')
if __name__ == '__main__': raise SystemExit(entry(job, configure, 'p3b-replay-data'))
