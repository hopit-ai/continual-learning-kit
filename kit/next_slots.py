"""Opt-in registered v4 slot maps. Existing B mixed() remains untouched."""
from kit.next_common import *
MAPS={
 'generic':{'generic':[8,16,24,32]},
 'broad':{'generic':[8,16,24,32],'broad':[4,12,20,28]},
 'earlier':{'generic':[8,16,24,32],'earlier':[4,12,20,28]},
 'both':{'generic':[8,16,24,32],'broad':[4,12,20,28],'earlier':[2,10,18,26]},
}


def draw(pool,seed,n,tokenizer,audit,source,pool_audit=None):
    require(bool(pool),'empty '+source+' pool')
    order=cycle(pool,seed,len(pool));selected=[];index=0
    lengths={str(r['id']):len(t.tokenize_student(tokenizer,r['prompt'] if isinstance(r['prompt'],list) else [dict(role='user',content=r['prompt'])],r['response'])[0]) for r in order} if tokenizer else {}
    require(not tokenizer or any(v<=2048 for v in lengths.values()),'no eligible D '+source+' rows')
    if pool_audit is not None:
        pool_audit[source]=dict(pool_rows=len(pool),prompt_cap=2048 if tokenizer else None,
            eligible_replay_ids=[str(r['id']) for r in pool if not tokenizer or lengths[str(r['id'])]<=2048],
            skipped_ids=[str(r['id']) for r in pool if tokenizer and lengths[str(r['id'])]>2048],
            skipped=[dict(id=str(r['id']),prompt_tokens=lengths[str(r['id'])]) for r in pool if tokenizer and lengths[str(r['id'])]>2048])
    while len(selected)<n:
        r=order[index%len(order)]
        if tokenizer and lengths[str(r['id'])]>2048:
            audit.append(dict(source=source,id=r['id'],draw_index=index,replay_slot=len(selected),prompt_tokens=lengths[str(r['id'])]))
        else:selected.append(dict(r,p3b_replay=True,data_source='p3b_replay',extra_info=dict(r.get('extra_info') or {},p3b_task_source='math_screen',next_pool=source)))
        index+=1
    return selected


def compose(task,generic,root,arm,seed,tokenizer=None, task_name='math'):
    require(arm in MAPS and len(task)==1280,'wrong probe map or schedule')
    frozen_files(root)
    slots=MAPS[arm];audit=[];sources={};counts={};pool_audit={}
    for source,positions in slots.items():
        n=len(positions)*40
        if source=='generic':rows=draw(generic,seed,n,tokenizer,audit,source,pool_audit)
        elif source=='earlier':rows=draw(t.read_jsonl(root/'earlier.jsonl'),seed,n,tokenizer,audit,source,pool_audit)
        else:
            # Equal source cycling, with independent seed-shuffled accepted pools; no outcome-based draws.
            left=draw(t.read_jsonl(root/'gsm8k.jsonl'),seed,n//2,tokenizer,audit,'gsm8k',pool_audit)
            right=draw(t.read_jsonl(root/'arc.jsonl'),seed,n//2,tokenizer,audit,'arc',pool_audit)
            rows=[r for pair in zip(left,right) for r in pair]
        sources[source]=iter(rows);counts[source]=n
    mapped={p:source for source,positions in slots.items() for p in positions}
    require(sum(map(len,slots.values()))==len(mapped),'slot overlap')
    rows=[next(sources[mapped[i%32+1]]) if i%32+1 in mapped else r for i,r in enumerate(task)]
    for row in rows:
        if row.get('p3b_replay'): row['extra_info']['p3b_task_source'] = 'contract_nli' if task_name == 'contract_nli' else 'math_screen'
    return rows,dict(pool_selection=pool_audit,slot_map=slots,exposures={task_name:1280-sum(counts.values()),**counts},skipped_draws=audit,
        skipped_draw_count=len(audit),loss=f'D: per-item token-mean KL on {task_name}; EOS-inclusive token-mean CE on all replay, weighted by actual slot fractions. F: unchanged B SFT reduction.')
