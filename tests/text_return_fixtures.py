"""Text-return fixture support, copied from existing PD/SP CPU test stand-ins.

These construct synthetic evidence only. Native readers and checks are unchanged.
The PD functions below retain the existing test_pd fixture's full seal/live-export
checks, including the 45 streamed-and-deleted probe exports.
"""
from datetime import date
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from collections import Counter
import importlib
from unittest.mock import patch
from kit import pd_contract as contract, pd_probe as probe, pd_read as reader, pd_smoke_modal as smoke


def d(value):
    return contract.digest(value)

def token_fixture(row):
    return dict(id=row["id"], candidate_lengths=[3], input_ids=[1, 2, 3], labels=[-100, 2, 3],
                prefix_ids=[1], suffixes={"HUM:ind": [2]}, content_positions=[0])

def curve(counts, initially_correct=0, panel_size=384):
    return {t: {f"q{i}": i < initially_correct or initially_correct <= i < initially_correct+n
                for i in range(panel_size)} for t, n in zip(contract.UPDATES, counts)}

def _checkpoint(tmp_path, seed, start, step, parent_id, *, role="chain_start"):
    checkpoint_id = (f"Qwen-Qwen3-8B@seed{seed}" if role == "chain_start" and start == "base"
                     else f"fixture-{seed}-{start}-{role}-{step}")
    root = tmp_path/checkpoint_id
    root.mkdir()
    tensor = root/"weights.safetensors"
    tensor.write_bytes(checkpoint_id.encode())
    export_receipt = root/"export.json"
    export_receipt.write_text(json.dumps(dict(checkpoint_id=checkpoint_id)))
    files = {"weights.safetensors": contract.sha256_file(tensor)}
    checkpoint = dict(checkpoint_id=checkpoint_id, seed=seed, start=start, step=step, parent_id=parent_id,
                      export_dtype="bfloat16" if role == "probe" else "float32", role=role, recipe="label-CE" if role == "probe" else "base" if start == "base" else "plain-D",
                      base_revision=probe.BASE_REVISION, tokenizer_sha256=d("tokenizer"), send_binding_sha256=d("placeholder"),
                      tensor_files=files, weights_sha256=d(files), export_receipt_sha256=contract.sha256_file(export_receipt))
    return checkpoint, dict(tensors={"weights.safetensors": tensor}, receipt=export_receipt, directory=root)

def fixture_vectors(predictions):
    vectors = {i: {"gold": -1. if correct else -2., "wrong": -2. if correct else -1.} for i, correct in predictions.items()}
    covariates = {i: dict(gold_log_probability=v["gold"], gold_rank=1 if predictions[i] else 2) for i, v in vectors.items()}
    return vectors, covariates

def rebind_fixture_scores(doc):
    """Build a different coherent synthetic scoring scenario, not a tamper bypass."""
    identities = {s["score_id"]: s for s in doc["manifest"]["scores"]}
    checkpoints = {c["checkpoint_id"]: c for c in doc["manifest"]["checkpoints"]}
    for run in doc["runs"]:
        for cell in run["scores"].values():
            vectors, covariates = fixture_vectors(cell["predictions"])
            cell.update(summed_scores=vectors, gold_covariates=covariates)
            identity = identities[cell["score_id"]]
            identity.update(predictions_sha256=d(cell["predictions"]), summed_scores_sha256=d(vectors), covariates_sha256=d(covariates))
            identity = contract.seal(identity)
            identities[cell["score_id"]] = identity
            checkpoint = checkpoints[identity["checkpoint_id"]]
            if checkpoint["role"] == "probe":
                proof = contract.seal(dict(schema="pd-stream-verification-v2", checkpoint_id=checkpoint["checkpoint_id"],
                    tensor_files=checkpoint["tensor_files"], weights_sha256=checkpoint["weights_sha256"],
                    export_receipt_sha256=checkpoint["export_receipt_sha256"], score_sha256=d(cell),
                    predictions_sha256=identity["predictions_sha256"], summed_scores_sha256=identity["summed_scores_sha256"],
                    covariates_sha256=identity["covariates_sha256"], gpu_identity=identity["gpu_identity"], disposition="verified-scored-deleted"))
                doc["stream_verifications"][checkpoint["checkpoint_id"]] = proof
                checkpoints[checkpoint["checkpoint_id"]] = contract.seal({**checkpoint, "stream_verification_sha256": proof["identity_sha256"]})
    doc["runs"] = [contract.seal(r) for r in doc["runs"]]
    doc["manifest"] = contract.make_manifest(checkpoints.values(), identities.values(),
        preparation_sha256=doc["manifest"]["preparation_sha256"], code_sha256=doc["manifest"]["code_sha256"])

def pd_evidence(tmp_path):
    panel_size = 384
    checkpoints, resolvers, expected, starts = [], {}, {}, {}
    fields = ("checkpoint_id", "weights_sha256", "export_receipt_sha256", "parent_id", "recipe", "step", "base_revision", "tokenizer_sha256")
    for seed in contract.SEEDS:
        starts[seed] = {}
        parent_id = None
        for start in contract.STARTS:
            checkpoint, resolver = _checkpoint(tmp_path, seed, start, 0 if start == "base" else 40, parent_id)
            starts[seed][start] = checkpoint
            resolvers[checkpoint["checkpoint_id"]] = resolver
            parent_id = checkpoint["checkpoint_id"]
        expected[str(seed)] = {start: {k: c[k] for k in fields} for start, c in starts[seed].items()}
    send_binding = contract.seal(dict(pd_starts=expected))
    for seed in contract.SEEDS:
        for start in contract.STARTS:
            starts[seed][start]["send_binding_sha256"] = send_binding["identity_sha256"]
            starts[seed][start] = contract.seal(starts[seed][start])
            checkpoints.append(starts[seed][start])
    scores, runs, cohorts, exposures = [], [], {}, {}
    training_ids = [f"train:{i}" for i in range(1280)]
    tokenized = {i: token_fixture(dict(id=i)) for i in training_ids}
    for seed in contract.SEEDS:
        curves = {"base": curve([0, 32, 64, 96, 128, 160], panel_size=panel_size),
                  "stage1": curve([0, 24, 48, 72, 96, 120], panel_size=panel_size),
                  "stage2": curve([0, 16, 32, 48, 64, 80], panel_size=panel_size)}
        cohorts[str(seed)] = reader.freeze_common_errors({s: c[0] for s, c in curves.items()})
        exposure = probe.exposure_receipt(probe.seed_batches(training_ids, seed), tokenized)
        exposures[str(seed)] = exposure["identity_sha256"]
        for start in contract.STARTS:
            cells = {}
            for t in contract.UPDATES:
                if t == 0:
                    checkpoint = starts[seed][start]
                else:
                    checkpoint, resolver = _checkpoint(tmp_path, seed, start, t, starts[seed][start]["checkpoint_id"], role="probe")
                    checkpoint = contract.seal({**checkpoint, "send_binding_sha256": send_binding["identity_sha256"]})
                    checkpoints.append(checkpoint)
                    # Simulate the registered post-score deletion of fixture-only exports.
                    __import__("shutil").rmtree(resolver["directory"])
                vectors, covariates = fixture_vectors(curves[start][t])
                score_id = f"score-{seed}-{start}-{t}"
                score = contract.seal(dict(score_id=score_id, seed=seed, update=t, checkpoint_id=checkpoint["checkpoint_id"],
                    panel_sha256=d("preparation"), prompt_sha256=d("prompt"), scorer_sha256=d("scorer"),
                    decoding_sha256=d("decoding"), tokenizer_sha256=d("tokenizer"), suffix_sha256=d("suffix"),
                    gpu_identity=dict(uuid=f"fixture-device-{seed}", devices=[f"fixture-device-{seed}"], environment=d("environment")),
                    summed_scores_sha256=d(vectors), covariates_sha256=d(covariates), scoring_rule="sum-label-plus-eos",
                    predictions_sha256=d(curves[start][t]), exposure_sha256=exposure["identity_sha256"]))
                scores.append(score)
                cells[t] = dict(score_id=score_id, predictions=curves[start][t], summed_scores=vectors, gold_covariates=covariates)
            runs.append(contract.seal(dict(seed=seed, start=start, scores=cells, exposure=exposure,
                       training_spec=contract.TRAINING_SPEC, training_world_size=8, fresh_optimizer=True, initial_optimizer_states=0,
                       scheduler="constant-no-state", losses=[1.]*40,
                       preparation_sha256=d("preparation"), cohort_sha256=cohorts[str(seed)]["identity_sha256"],
                       checkpoint_sha256=starts[seed][start]["identity_sha256"],
                       updates=[dict(update=i, lr=1e-5, examples=32, preclip_norm=.1) for i in range(1, 41)])))
    manifest = contract.make_manifest(checkpoints, scores, preparation_sha256=d("preparation"), code_sha256=d("code"))
    qualification = contract.seal({**smoke_receipt(), "code_sha256": d("code")})
    document = dict(manifest=manifest, runs=runs, cohorts=cohorts, send_binding=send_binding, stream_verifications={},
                    smoke_receipt=qualification, return_verification=contract.seal(dict(status="VERIFIED",
                    preparation_sha256=d("preparation"), candidate_supersets_sha256=d("supersets"))))
    rebind_fixture_scores(document)
    trusted = dict(send_binding_sha256=send_binding["identity_sha256"], preparation_sha256=d("preparation"),
                   code_sha256=d("code"), exposure_sha256=exposures, evaluation_ids=[f"q{i}" for i in range(panel_size)],
                   smoke_receipt_sha256=qualification["identity_sha256"], candidate_supersets_sha256=d("supersets"))
    return document, resolvers, trusted

def quote():
    # Synthetic fixture rates, explicitly not a claim about current prices.
    return contract.seal(dict(source="https://modal.com/pricing", date=date.today().isoformat(),
                              rates_per_second=dict(H100=.001, cpu=.000001, memory_gib=.000001),
                              ceilings=dict(startup=.1, storage=.1, egress=.1)))

def smoke_receipt():
    pricing = quote()
    scores = {f"cal:{i}": {"a": -1., "b": -2.} for i in range(128)}
    return contract.seal(dict(reservation=smoke.RESERVATION, quote=pricing, date=date.today().isoformat(),
                    model="Qwen3-8B", seed=101, training_world_size=2,
                    export_dtype="bfloat16", export_scoring_setup="unsharded-fp32-autocast-bf16",
                    scoring_gpu_identity=dict(uuid="fixture", devices=["fixture"], environment=d("env")),
                    full_exposure_sha256=d("full"), expected_exposure_sha256=d("full"), calibration_before=65., calibration_after=71.,
                    initially_wrong_improvement=15., gradient_reference=probe.gradient_reference_check(),
                    determinism=dict(algorithms=True, workspace=":4096:8", attention="eager"),
                    updates=40, losses=[1.]*40, repeated_losses=[1., 1.], initial_optimizer_states=[0, 0],
                    first_two_exposure_sha256=d("exposure"), repeat_exposure_sha256=d("exposure"),
                    first_two_batch_sha256=d("batch"), repeat_batch_sha256=d("batch"),
                    finite_gradients_weights_diagnostics=True,
                    diagnostics={t: dict(dormant_layer_mean=0., layers={"fixture": dict(dormant_fraction=0., zero_layer=False,
                                 effective_rank=1., normalized_effective_rank=.5)}) for t in (0, 40)},
                    initial_weights_sha256=d("before"), final_weights_sha256=d("after"),
                    original_export_scores=scores, reloaded_export_scores=scores, export_agreement=smoke.export_agreement(scores, scores),
                    base_before_sha256=d("base"), base_after_sha256=d("base"), gpu_peak_fractions=[.8, .8],
                    aggregate_host_peak_bytes=80*1024**3, oom=False, seconds_including_receipts=1000.,
                    receipts_complete=True, reservation_cost_ceiling=smoke.validate_price(pricing)))


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value, default=str))
    return str(path)


def m3_cells():
    """Same make_cells/winning geometry as test_m3_protocol, without score IO doubles."""
    from kit import m3_read as r
    cells={};assignment={f'm3|{s}':dict(uuid=f'fixture-{s}',environment='env') for s in r.SEEDS}
    panels={'general-v1':('ifeval','math','knowledge_strict'),'knowledge-ll':('knowledge',),
            'hellaswag':('hellaswag',),'finqa':('finqa',),'test':('test',),'math_hard':('math_hard',)}
    for arm in ('base',*r.ARMS):
        for seed in r.SEEDS:
            for stage in (('finqa',) if arm=='base' else r.STAGES):
                for step in ((0,) if arm=='base' else r.STEPS):
                    for panel, metrics in panels.items():
                        bonus=lambda metric: (5+(seed-101)/2) if arm=='K' and stage=='math_hard' and step==40 and metric=='ifeval' else (4+(seed-101)/2) if arm=='CR' and stage=='math_hard' and step==40 and metric in r.GENERAL else 0
                        values={m:50+bonus(m) for m in metrics}
                        cells[arm,seed,stage,step,panel]=dict(metrics=values,items={m:{f'q{i}':int(i<v) for i in range(100)} for m,v in values.items()},
                            uuid=assignment[f'm3|{seed}']['uuid'],environment='env',format_failures={m:0 for m in metrics},output_tokens=1000,correct=50,tolerant={},pending=None)
    return cells,assignment


def sp_tree(root):
    """Existing test_sp.write_panel_result format; real read_sidecar/score seals."""
    from kit import sp_score as sc, sp_read as rd
    entries=[]
    for seed in (101,102,103):
        for arm in ('C0','K','G','CR'):
            for step in (0,40):
                comparison=arm+'.finqa'
                use=dict(source='message3',family='m3',seed=seed,role='incoming' if step==0 else 'final',arm=comparison,comparison=comparison,step=step,
                         incoming_key=['message3','m3',comparison,seed,0,'finqa'])
                entry=dict(entry_id=f'sp-{arm}-{seed}-{step}',assignment=f'sp:m3-s{seed}',weights_sha256=contract.digest([arm,seed,step]),uses=[use]);entries.append(entry)
                scored=[dict(id=f'r{i}:{wording}',request_id=f'r{i}',wording=wording,primary_family='required_word',strict=int(i==0 or (step==40 and wording=='canonical')),
                             strict_all_constraints_pass=int(i==0 or (step==40 and wording=='canonical')),
                             instructions=[dict(family='required_word',passed=int(i==0 or (step==40 and wording=='canonical')))]) for i in range(2) for wording in ('canonical','paraphrase')]
                raw=[dict(id=r['id'],response='word',output_tokens=5,finished=True,truncated=False) for r in scored]
                cell=root/'panel'/entry['entry_id']
                sc.write_jsonl(cell/'raw.jsonl',raw);sc.write_jsonl(cell/'scores.jsonl',scored)
                sc.write_json(cell/'length.json',sc.length_metrics(raw,{r['id']:r['strict'] for r in scored}))
                sc.write_json(cell/'identity.json',dict(entry=entry,gpu_identity=dict(uuid='fixture-gpu',environment='env'),frozen_sha256='a'*64,
                    decoding_sha256=contract.digest(sc.DECODING),timing=dict(load=1,score=1,receipt=1),technical_synthetic=True))
                sc.write_json(cell/'manifest.json',dict(files={n:contract.sha256_file(cell/n) for n in ('identity.json','raw.jsonl','scores.jsonl','length.json')}))
    inventory=contract.seal(dict(sp_entries=entries,planned=[]))
    write(root/'inventory.json',inventory)
    return dict(source=str(root),inventory=str(root/'inventory.json'))


def a4_tree(root):
    """Use a4_sim, real bind/run jobs, with existing in-process dispatch stand-ins."""
    from kit import a4_sim, a4_bind, a4_run, a4_train, a4_common as c
    import argparse
    sl,b,rp=a4_sim.fixture(root/'fixtures','D+replay');out=root/'out'
    a=NS(out=out,standin=True,slice=sl,b_source=b,base='synthetic-base',finqa=None,replay=rp,stage_gpu_hours=3,
         approved_ceiling=800,approval_reference=c.APPROVAL_REFERENCE,gpu='0')
    a4_bind.job(a);c.start_4a(out,True)
    def dispatch(command,**kwargs):
        module=importlib.import_module('kit.'+Path(command[1]).stem);p=argparse.ArgumentParser()
        p.add_argument('--out',type=Path,required=True);p.add_argument('--standin',action='store_true');module.configure(p)
        module.job(p.parse_args(command[2:]));return NS(returncode=0)
    with patch('subprocess.run',side_effect=dispatch),patch.object(a4_train,'gpu_run',side_effect=dispatch):
        a4_run.job(NS(out=out,standin=True,through_month=6,score_gpu='0',gpu='0,1,2,3,4,5,6,7',sdpo=None,tokenizer=None,
                      trainer_python=sys.executable,inference_python=sys.executable,fail_once=False))
    return dict(source=str(out))


def full_inputs(root):
    """Real native result-tree stand-ins for all seven registered readers."""
    from kit import message2_dryrun, addon_dryrun, m3_manifest
    # Native stage/score/collector/read-back, one representative full branch.
    message2_dryrun.branch(root/'M','none','QUALIFIES',Counter())
    addon_dryrun.branch(root/'X','both',tier2=True)
    x=root/'X/fixtures/x-out'
    # addon_dryrun's bound root is fixtures/x-out; its synthetic S archive is
    # a frozen three-seed Screen S stand-in analyzed by the registered code.
    screen=root/'screen.tar.gz';addon_dryrun.synthetic_screen_return(screen)
    m3=root/'KCR';m3_manifest.init_root(m3,core_submitted=True)
    from kit.m3_pools import pool_root
    for arm in ('C0','K','G','CR'):write(pool_root(m3,arm)/'availability.json',{a:dict(status='READY') for a in ('C0','K','G','CR')})
    pdroot=root/'PD';pdroot.mkdir()
    evidence,exports,trusted=pd_evidence(pdroot)
    pd=dict(evidence=write(pdroot/'evidence.json',evidence),exports=write(pdroot/'exports.local.json',exports),trusted=write(pdroot/'anchors.json',trusted),
            launcher_timings=[write(pdroot/'launcher-timing.json',dict(wall_seconds=1800,gpu_hours=4))])
    jobs=[]
    for family in ('M','X','S','KCR','PD','SP','4A'):
        state=root/'job-fixtures'/family
        from kit.text_return import job_inventory
        labels,_=job_inventory()
        prefix={'M':'M','X':'X','S':'S','KCR':'M3-','PD':'PD-','SP':'SP-','4A':'no-native-4A-manifest'}[family]
        known=[x for x in labels if x.startswith(prefix)][:3]
        tasks=[dict(id=known[i-1] if len(known)>=i else 'synthetic-private-task-'+str(i)) for i in range(1,4)]
        for i,status in ((1,'DONE'),(2,'FAILED')):write(state/(tasks[i-1]['id']+'.json'),dict(status=status,technical_synthetic=True))
        jobs.append(dict(family=family,manifest=write(state/'launch.local.json',dict(tasks=tasks)),state=str(state)))
    return {'jobs':jobs,'M':[dict(source=str(root/'M/out'))],'X':[dict(source=str(x),screen=str(screen))],
            'S':[dict(source=str(screen))],'KCR':[dict(source=str(m3))],'PD':[pd],
            'SP':[sp_tree(root/'SP')],'4A':[a4_tree(root/'4A')]}
