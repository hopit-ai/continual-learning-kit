"""Complete registered P2 verdict from actual fp32 exports before native save."""
from pathlib import Path
from kit.s_common import atomic,read,require,sha,ROOT


def publish_before_checkpoint(run):
    import os
    from kit.x_smoke_memory import Monitor,merge_peak
    from kit.x_integration_smoke import MODEL_REVISION,partial
    from kit.x_p2_export import export,distance
    from kit.x_smoke_batches import paired_receipt
    from kit.x_p2_smoke_read import evaluate
    from kit.p2_smoke import code_hashes
    run=Path(run);root=run.parent
    require(os.environ.get('X_INTEGRATION_SMOKE')=='development-only' and
        os.environ.get('X_P2_MODE')=='train' and os.environ.get('X_P2_EARLY_VERDICT')=='1','early verdict is smoke P2 only')
    context=read(root/'p2/pre-save-context.json')
    require(context['schema']=='p2-pre-save-context.v1' and context['code_sha256']==code_hashes(),'pre-save verdict context/code changed')
    source=run/'live-step4/actor'
    for rank in range(2):
        require(read(source/f'ready-r{rank}.json')==dict(rank=rank,completed_updates=4,precision='fp32',student_only=True),'live student rank incomplete')
    challenger=[read(run/f'smoke-rank{rank}.json') for rank in range(2)]
    require(all(len(v['updates'])==4 for v in challenger),'live P2 dose incomplete')
    # Include export/merge transients as well as construction/update peaks. The
    # parent context retains the complete earlier preparation/X4/control peaks.
    with Monitor() as monitor:
        target=export(run,root/'export-p2-live',source=source)
        exports=dict(context['exports'],p2=distance(target,context['base'],challenger[0]['geometry']['inventory']))
        pairing=paired_receipt(root/'p2/s101',root/'control',run)
        host=merge_peak(context['container_host'],monitor.sample())
        proof=evaluate(context['preparation'],context['control'],challenger,context['coefficient'],exports,MODEL_REVISION,container_host=host)
    proof.update(paired_batches=pairing,verdict_boundary='after-update-4-before-native-checkpoint',
        export_boundary='actual fp32 student-only DTensor export before native optimizer/EMA checkpoint',
        live_export_files={p.name:sha(p) for p in target.iterdir() if p.is_file()})
    # Both durable receipt and stdout precede returning to the trainer, which is
    # the only point at which its native final checkpoint can start.
    partial(root,'P2-smoke-result',dict(p2_smoke=proof))
    atomic(run/'pre-save-verdict.json',dict(passed=proof['passed'],verdict_boundary=proof['verdict_boundary']))
    return proof
