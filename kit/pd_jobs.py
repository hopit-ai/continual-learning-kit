"""Right-sized PD jobs under the approved 2 November scorer amendment.

CPU preparation/finalisation/collection surround independent 1/8/1-GPU jobs.
The qualified numerical core is imported unchanged; no optimizer resume exists.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kit import pd_probe as probe
from kit import pd_scientific as scientific
from kit import pd_materialize as materialize
from kit.pd_contract import (SEEDS, STARTS, UPDATES, digest, require, seal, sha256_file,
    verify_seal, verify_live_export, validate_checkpoint, validate_score, make_manifest)
from kit.pd_prepare import write_frozen
from kit.pd_read import freeze_common_errors
from kit.pd_smoke_modal import smoke_gates, export_agreement

PHASES = ("prep", "pre", "train", "finalize", "post", "collect")
GPUS = dict(prep=0, pre=1, train=8, finalize=0, post=1, collect=0)
# 15 curve points per seed, with a native BF16 snapshot and HF export each.
EXPORTS = len(STARTS)*(len(UPDATES)-1)
MIN_CHECKPOINT_BYTES = 18*1024**3
HEADROOM_BYTES = 64*1024**3


def read(path):
    value = json.loads(Path(path).read_text())
    verify_seal(value)
    return value


def implementation_identity():
    root = Path(__file__).parent
    from kit.v6_infra import runtime_identity
    return digest(dict(files={n:sha256_file(root/n) for n in ('pd_jobs.py','pd_jobs_read.py','pd_node.py')},lane_runtime=runtime_identity()))


def resource_rows(seed):
    require(seed in SEEDS, "unregistered PD seed")
    cpus = dict(prep=8, pre=8, train=32, finalize=16, post=8, collect=4)
    return [dict(id=f"PD-{seed}-{phase}", phase=phase, seed=seed, nodes=1,
        gpus=GPUS[phase], cpus=cpus[phase], may_share_node=True,
        dependency="afterany" if phase == "collect" else "afterok",
        requires=[] if i == 0 else [f"PD-{seed}-{PHASES[i-1]}"],
        gpu_sharing=False if phase == "train" else None)
        for i,phase in enumerate(PHASES)]


def resource_inventory():
    rows = [r for seed in SEEDS for r in resource_rows(seed)]
    rows.append(dict(id="PD-read", phase="read", nodes=1, gpus=0, cpus=4, may_share_node=True,
        requires=[f"PD-{s}-collect" for s in SEEDS], dependency="afterany"))
    return dict(schema="pd-job-resources-v1", rows=rows)


def allocation_tokens(visible, count):
    from kit.pd_node import _gpu_uuid
    require(isinstance(visible, str) and visible.strip(), "PENDING: explicit GPU allocation required")
    tokens = [x.strip() for x in visible.split(",")]
    tokens = [str(int(x)) if x.isdecimal() else _gpu_uuid(x) for x in tokens]
    require(len(tokens) == count and len(set(tokens)) == count,
            f"PENDING: exactly {count} distinct allocated GPUs required")
    return tokens


def cpu_only():
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in ("", "-1", "NoDevFiles"),
            "PENDING: CPU phase requires a 0-GPU allocation")
    require(int(os.environ.get("SLURM_GPUS_ON_NODE", "0")) == 0,
            "PENDING: CPU phase must not hold Slurm GPUs")


def isolated_tree(root, plan):
    root = Path(root).resolve()
    for value in plan["exports"].values():
        source = Path(value["directory"]).resolve()
        require(root != source and root not in source.parents and source not in root.parents,
                "PENDING: PD state must be separate from source checkpoints")
    return root


def verify_directory(checkpoint, directory, receipt):
    directory = Path(directory)
    require(directory.is_dir(), "PENDING: checkpoint directory missing")
    tensors = {p.name:p for p in directory.iterdir() if p.suffix in (".bin", ".safetensors")}
    return verify_live_export(checkpoint, tensors, receipt, directory=directory)


def spool_budget(source_bytes):
    require(type(source_bytes) is int and source_bytes > 0, "invalid source checkpoint size")
    checkpoint = max(MIN_CHECKPOINT_BYTES, source_bytes)
    return dict(exports_per_seed=EXPORTS, checkpoint_bound_bytes=checkpoint,
        native_snapshot_bytes=EXPORTS*checkpoint, hf_export_bytes=EXPORTS*checkpoint,
        spool_bytes=2*EXPORTS*checkpoint, headroom_bytes=HEADROOM_BYTES,
        required_free_bytes=2*EXPORTS*checkpoint+HEADROOM_BYTES)


def prepare(plan_file, root, *, disk_usage=None):
    cpu_only()
    plan = json.loads(Path(plan_file).read_text())
    require(plan["seed"] in SEEDS, "unregistered PD seed")
    root = isolated_tree(root, plan)
    require(not root.exists() or (root.is_dir() and not any(root.iterdir())),
            "PENDING: failed seed restarts from update 0 in a clean PD tree")
    scientific.require_plan(plan)
    pin = materialize.require_plan_pin(plan)
    require(smoke_gates(plan["smoke_receipt"])["status"] == "QUALIFIED" and
            plan["smoke_receipt"]["code_sha256"] == probe.code_identity(),
            "PENDING: qualified unchanged numerical core required")
    bindings = probe.bind_chain(plan["checkpoints"], plan["send_binding"])
    sizes = []
    for start in STARTS:
        export = plan["exports"][start]
        verify_directory(bindings[start], export["directory"], export["receipt"])
        sizes.append(sum((Path(export["directory"])/n).stat().st_size for n in bindings[start]["tensor_files"]))
    budget = spool_budget(max(sizes))
    root.mkdir(parents=True, exist_ok=True)
    observed = (disk_usage or shutil.disk_usage)(root).free
    require(observed >= budget["required_free_bytes"], "PENDING: insufficient PD spool free space")
    prepared = read(plan["preparation"])
    require(prepared["code_sha256"] == probe.code_identity() and
            prepared["identity_sha256"] == pin["preparation_sha256"], "PENDING: preparation pin changed")
    # All tokenisation and source hashing happens in this CPU job.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(plan["tokenizer"], local_files_only=True)
    require(probe.tokenizer_identity(plan["tokenizer"]) == prepared["tokenizer_sha256"], "tokenizer changed")
    require(all(probe.encode_row(tokenizer, row) == prepared["tokenized"][row["id"]] for row in prepared["rows"]),
            "frozen tokenization/masks changed")
    training_ids = prepared["partitions"]["training"]
    batches = probe.seed_batches(training_ids, plan["seed"])
    train = seal(dict(batches=batches, tokenized={i:prepared["tokenized"][i] for i in training_ids},
                     exposure=probe.exposure_receipt(batches, prepared["tokenized"])))
    ids = set(prepared["partitions"]["evaluation"]+prepared["partitions"]["calibration"])
    scoring = seal(dict(rows=[r for r in prepared["rows"] if r["id"] in ids],
        tokenized={i:prepared["tokenized"][i] for i in ids}, partitions=prepared["partitions"],
        preparation_sha256=prepared["identity_sha256"], prompt_sha256=prepared["prompt_sha256"],
        tokenizer_sha256=prepared["tokenizer_sha256"]))
    write_frozen(root/"training-inputs.json", train)
    write_frozen(root/"scoring-inputs.json", scoring)
    receipt = seal(dict(schema="pd-job-preparation-v1", seed=plan["seed"],
        hostname=socket.gethostname(), input_plan_sha256=digest(plan),
        code_sha256=probe.code_identity(), split_job_sha256=implementation_identity(),
        preparation_sha256=prepared["identity_sha256"], preparation_pin_sha256=pin["identity_sha256"],
        training_inputs_sha256=sha256_file(root/"training-inputs.json"),
        scoring_inputs_sha256=sha256_file(root/"scoring-inputs.json"),
        budget=budget, observed_free_bytes=observed, resources=resource_rows(plan["seed"])))
    write_frozen(root/"prep.json", receipt)
    return receipt


def gate(plan_file, root):
    plan = json.loads(Path(plan_file).read_text())
    root = isolated_tree(root, plan)
    require((root/"prep.json").is_file(), "PENDING: CPU preparation missing")
    prepared = read(root/"prep.json")
    require(prepared["hostname"] == socket.gethostname(), "PENDING: PD node-local hostname changed")
    require(prepared["input_plan_sha256"] == digest(plan) and
            prepared["code_sha256"] == probe.code_identity() and
            prepared["split_job_sha256"] == implementation_identity(), "UNVERIFIED: PD job binding changed")
    return plan, root, prepared


def inputs(root, prepared, kind):
    filename = root/(kind+"-inputs.json")
    require(sha256_file(filename) == prepared[kind+"_inputs_sha256"], "UNVERIFIED: CPU inputs changed")
    return read(filename)


def validate_initial(initial, *, seed, preparation_sha256):
    verify_seal(initial)
    require(initial["schema"] == "pd-pre-training-v1" and initial["seed"] == seed and
            initial["preparation_sha256"] == preparation_sha256, "UNVERIFIED: initial scoring binding changed")
    require(set(initial["scores"]) == set(STARTS), "PENDING: three initial scores required")
    verify_seal(initial["job"])
    require(initial["job"]["phase"] == "pre" and initial["job"]["seed"] == seed and
            initial["job"]["preparation_sha256"] == preparation_sha256 and
            initial["job"]["split_job_sha256"] == implementation_identity(), "UNVERIFIED: initial job changed")
    identities = []
    for cell in initial["scores"].values():
        verify_seal(cell)
        require(cell["job_sha256"] == initial["job"]["identity_sha256"] and
                cell["gpu_identity"] == initial["job"]["gpu_identity"] and
                cell["gpu_identity"]["devices"] == [cell["gpu_identity"]["uuid"]], "UNVERIFIED: mixed pre jobs")
        require(set(cell["summed_scores"]) == set(cell["predictions"]) == set(cell["gold_covariates"]),
                "UNVERIFIED: initial score coverage changed")
        identities.append(digest(cell["gpu_identity"]))
    require(len(set(identities)) == 1 and initial["job"]["phase"] == "pre", "UNVERIFIED: mixed initial hardware")
    cohort = freeze_common_errors({s:initial["scores"][s]["predictions"] for s in STARTS})
    require(cohort == initial["cohort"], "UNVERIFIED: pre-training cohort changed")
    return cohort


@contextmanager
def gpu_leases(identity):
    # UUID keys share one host-wide namespace across independent PD allocations.
    directory = Path("/tmp")/"kit-gpu-leases"
    directory.mkdir(exist_ok=True)
    with ExitStack() as stack:
        for device in sorted(identity["devices"]):
            from kit.pd_node import _gpu_uuid
            name = _gpu_uuid(device)
            stream = stack.enter_context(open(directory/(name+".lock"), "a"))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("PENDING: allocated PD GPU already leased") from None
        yield


def job_identity(phase, prepared, identity):
    return seal(dict(schema="pd-gpu-job-v1", phase=phase, seed=prepared["seed"],
        job_nonce=os.urandom(16).hex(), gpu_identity=identity,
        preparation_sha256=prepared["preparation_sha256"], split_job_sha256=implementation_identity()))


def score_cell(plan, pack, checkpoint, directory, receipt, start, update, job):
    """The qualified exact-label scorer and diagnostics, with CPU inputs prechecked."""
    import torch
    from transformers import AutoTokenizer
    verify_directory(checkpoint, directory, receipt)
    require(probe.gpu_identity() == job["gpu_identity"], "UNVERIFIED-HARDWARE: scoring GPU changed")
    tokenizer = AutoTokenizer.from_pretrained(plan["tokenizer"], local_files_only=True)
    require(probe.tokenizer_identity(plan["tokenizer"]) == pack["tokenizer_sha256"], "tokenizer changed")
    started = time.monotonic()
    model = probe.load_local_model(directory, device_map={"":0})
    torch.cuda.synchronize()
    loaded = time.monotonic()-started
    started = time.monotonic()
    ids = set(pack["partitions"]["evaluation"])
    rows = [r for r in pack["rows"] if r["id"] in ids]
    result = probe.score_rows(model, rows, pack["tokenized"], generated=True, tokenizer=tokenizer)
    if update in (0, 40):
        ids = set(pack["partitions"]["calibration"])
        calibration = [r for r in pack["rows"] if r["id"] in ids]
        result["diagnostics"] = probe.representation_diagnostics(model, calibration, pack["tokenized"])
    result.update(checkpoint_id=checkpoint["checkpoint_id"], gpu_identity=job["gpu_identity"],
        job_sha256=job["identity_sha256"], live_load_verification=probe.live_load_verification(checkpoint),
        load_seconds=loaded, score_seconds=time.monotonic()-started, receipt_seconds=0.)
    del model
    torch.cuda.empty_cache()
    return seal(result)


def scoring_cells(plan,root,prepared,job,requests,pooled=False):
    """Independent unchanged HF scorers; all cells inherit one allocation/job proof."""
    pack=inputs(root,prepared,"scoring")
    if not pooled:
        return [score_cell(plan,pack,*request,job) for request in requests]
    from kit.lane_pool import execute
    directory=root/job['phase']/'cells';directory.mkdir()
    commands=[]
    for i,(checkpoint,path,receipt,start,update) in enumerate(requests):
        task=directory/(str(i)+'.json')
        write_frozen(task,dict(checkpoint=checkpoint,directory=str(path),receipt=str(receipt),start=start,update=update,job=job))
        commands.append([sys.executable,'-I',str(Path(__file__).resolve()),'--score-cell',str(task),
            '--plan',str(root/'plan.local.json'),'--out',str(root)])
    # The prepared plan is copied from the checked in-memory plan, never substituted.
    write_frozen(root/'plan.local.json',plan)
    code=execute(commands,directory/'pool',env=dict(os.environ))
    require(code==0,"PENDING: PD scoring lane incomplete; clean whole-seed restart required")
    return [read(directory/(str(i)+'.result.json')) for i in range(len(requests))]


def cell_worker(plan_file,root,task_file):
    plan,root,prepared=gate(plan_file,root)
    task=json.loads(Path(task_file).read_text());job=task['job'];verify_seal(job)
    require(read(root/job['phase']/'job.json')==job,"UNVERIFIED: scoring job changed")
    probe.configure_determinism()
    cell=score_cell(plan,inputs(root,prepared,"scoring"),task['checkpoint'],task['directory'],task['receipt'],task['start'],task['update'],job)
    write_frozen(Path(task_file).with_suffix('.result.json'),cell)


def pre_score(plan, root, prepared, identity, pooled=False):
    pack = inputs(root, prepared, "scoring")
    destination = root/"pre"
    require(not destination.exists(), "PENDING: failed seed requires clean whole-seed restart")
    destination.mkdir()
    job = job_identity("pre", prepared, identity)
    write_frozen(destination/"job.json", job)
    requests=[(plan["checkpoints"][start],plan["exports"][start]["directory"],plan["exports"][start]["receipt"],start,0) for start in STARTS]
    scores=dict(zip(STARTS,scoring_cells(plan,root,prepared,job,requests,pooled)))
    for start in STARTS:
        write_frozen(destination/(start+"-0.json"), scores[start])
    cohort = freeze_common_errors({s:scores[s]["predictions"] for s in STARTS})
    result = seal(dict(schema="pd-pre-training-v1", seed=plan["seed"], job=job, scores=scores,
        cohort=cohort, preparation_sha256=prepared["preparation_sha256"]))
    write_frozen(destination/"complete.json", result)
    return result


class NativeSpool:
    """One immutable CPU snapshot in flight; no scorer or another job is awaited."""
    def __init__(self, root, budget):
        self.root, self.budget = Path(root), budget
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.records = []

    def ready(self):
        if self.future is not None:
            self.records.append(self.future.result())
            self.future = None

    def submit(self, state, config, parent, update):
        require(self.future is None, "checkpoint writer still owns a snapshot")
        # Gather ownership has already passed to this private, detached CPU state.
        self.future = self.pool.submit(self._write, state, config, parent, update)

    def _write(self, state, config, parent, update):
        import torch
        destination = self.root/(parent["start"]+"-"+str(update))
        temporary = destination.with_name(".pending-"+destination.name)
        require(not destination.exists() and not temporary.exists(), "native checkpoint already exists")
        temporary.mkdir()
        torch.save(state, temporary/"state.pt")
        require((temporary/"state.pt").stat().st_size <= self.budget["checkpoint_bound_bytes"],
                "PENDING: native checkpoint exceeds reserved spool bound")
        config.save_pretrained(temporary)
        record = seal(dict(schema="pd-native-snapshot-v1", parent=parent, update=update,
            tensor_sha256=sha256_file(temporary/"state.pt"),
            config_sha256=sha256_file(temporary/"config.json"), complete=True))
        write_frozen(temporary/"complete.json", record)
        temporary.rename(destination)
        return record

    def close(self):
        try:
            self.ready()
        finally:
            self.pool.shutdown(wait=True)


def gather_snapshot(model, spool, parent, update):
    """Only immutable BF16 CPU copies leave the checkpoint barrier."""
    import copy
    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
    # Bound host memory: finish the previous save before taking another snapshot.
    probe._master_call(spool.ready)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        state = model.state_dict()
    def enqueue():
        snapshot = {n:v.detach().to(device="cpu", dtype=torch.bfloat16 if v.is_floating_point() else v.dtype,
                                    copy=True) for n,v in state.items()}
        state.clear()
        spool.submit(snapshot, copy.deepcopy(model.module.config), parent, update)
    probe._master_call(enqueue)
    del state
    torch.cuda.empty_cache()
    torch.distributed.barrier()


def train_worker(plan_file, root):
    import torch
    from datetime import timedelta
    plan, root, prepared = gate(plan_file, root)
    initial = read(root/"pre/complete.json")
    cohort = validate_initial(initial, seed=plan["seed"], preparation_sha256=prepared["preparation_sha256"])
    train = inputs(root, prepared, "training")
    probe.configure_determinism()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=30))
    require(torch.distributed.get_world_size() == 8, "eight training ranks required")
    require(all("H100" in torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())), "H100s required")
    root = root/"train"
    spool = NativeSpool(root/"native", prepared["budget"])
    try:
        runs = []
        for start in STARTS:
            parent = plan["checkpoints"][start]
            def factory():
                export = plan["exports"][start]
                # Live checking precedes model load; no source is ever modified.
                probe._master_call(lambda: verify_directory(parent, export["directory"], export["receipt"]))
                return probe.load_sharded_model(export["directory"])
            def on_update(model, optimizer, update, receipt):
                if update in UPDATES[1:]:
                    gather_snapshot(model, spool, parent, update)
            model, optimizer, receipt = probe.train_fork(factory, train["batches"], train["tokenized"],
                seed=plan["seed"], evaluate=None, on_update=on_update)
            require(receipt["exposure"] == train["exposure"], "UNVERIFIED: fork exposure changed")
            receipt.update(start=start, checkpoint_sha256=parent["identity_sha256"],
                preparation_sha256=prepared["preparation_sha256"], cohort_sha256=cohort["identity_sha256"])
            runs.append(seal(receipt))
            probe._master_call(lambda: write_frozen(root/(start+"-probe.json"), runs[-1]))
            del model, optimizer
            torch.cuda.empty_cache()
        # Outstanding native checkpoint saves are part of training, never HF export/score/publication.
        probe._master_call(spool.ready)
        def complete():
            require(len(spool.records) == EXPORTS, "PENDING: incomplete native spool")
            result = seal(dict(schema="pd-training-complete-v1", seed=plan["seed"],
                pre_training_sha256=initial["identity_sha256"], runs=runs,
                snapshots=spool.records, job=read(root/"job.json"),
                split_job_sha256=implementation_identity(), complete=True))
            write_frozen(root/"complete.json", result)
        probe._master_call(complete)
    finally:
        try:
            spool.close()
        finally:
            torch.distributed.destroy_process_group()


def validate_training(training, initial, plan):
    verify_seal(training)
    require(training["schema"] == "pd-training-complete-v1" and training["complete"] is True and
            training["seed"] == plan["seed"] and training["pre_training_sha256"] == initial["identity_sha256"],
            "PENDING: training did not complete from the frozen cohort")
    require(training["split_job_sha256"] == implementation_identity(), "UNVERIFIED: training implementation changed")
    verify_seal(training["job"])
    require(training["job"]["phase"] == "train" and training["job"]["seed"] == plan["seed"] and
            len(set(training["job"]["gpu_identity"]["devices"])) == 8,
            "UNVERIFIED: training job requires eight GPUs")
    expected = {(s,u) for s in STARTS for u in UPDATES[1:]}
    require(len(training["snapshots"]) == EXPORTS and
            {(r["parent"]["start"],r["update"]) for r in training["snapshots"]} == expected,
            "PENDING: incomplete native spool")
    require(len(training["runs"]) == 3 and {r["start"] for r in training["runs"]} == set(STARTS),
            "PENDING: missing training forks")
    for run in training["runs"]:
        verify_seal(run)
        require(run["scores"] == {}, "UNVERIFIED: training job must not score")
    for record in training["snapshots"]:
        verify_seal(record)
        require(record["complete"] is True and record["parent"] == plan["checkpoints"][record["parent"]["start"]],
                "UNVERIFIED: wrong native checkpoint parent")
    return training


def finalize(plan, root, prepared):
    cpu_only()
    initial = read(root/"pre/complete.json")
    validate_initial(initial, seed=plan["seed"], preparation_sha256=prepared["preparation_sha256"])
    training = validate_training(read(root/"train/complete.json"), initial, plan)
    destination = root/"exports"
    require(not destination.exists(), "PENDING: failed seed requires clean whole-seed restart")
    destination.mkdir()
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    checkpoints = []
    for record in training["snapshots"]:
        start, update = record["parent"]["start"], record["update"]
        native = root/"train/native"/(start+"-"+str(update))
        require(read(native/"complete.json") == record and
                sha256_file(native/"state.pt") == record["tensor_sha256"] and
                sha256_file(native/"config.json") == record["config_sha256"], "PENDING: native snapshot incomplete or changed")
        state = torch.load(native/"state.pt", map_location="cpu", weights_only=True)
        config = AutoConfig.from_pretrained(native, local_files_only=True)
        # Meta construction creates no second full parameter allocation.
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32, attn_implementation="eager")
        path = destination/(start+"-"+str(update))
        checkpoint = probe.export_probe(model, path, record["parent"], update, state=state)
        write_frozen(path/"checkpoint.json", checkpoint)
        require(sum(p.stat().st_size for p in path.iterdir() if p.is_file()) <= prepared["budget"]["checkpoint_bound_bytes"],
                "PENDING: HF checkpoint exceeds reserved spool bound")
        checkpoints.append(checkpoint)
        del model, state
    result = seal(dict(schema="pd-spool-complete-v1", seed=plan["seed"], checkpoints=checkpoints,
        training_sha256=training["identity_sha256"], pre_training_sha256=initial["identity_sha256"], complete=True))
    write_frozen(destination/"complete.json", result)
    return result


def ready_spool(plan, root, prepared):
    require((root/"pre/complete.json").is_file() and (root/"train/complete.json").is_file() and
            (root/"exports/complete.json").is_file(), "PENDING: all PD exports must exist before GPU admission")
    initial = read(root/"pre/complete.json")
    validate_initial(initial, seed=plan["seed"], preparation_sha256=prepared["preparation_sha256"])
    training = validate_training(read(root/"train/complete.json"), initial, plan)
    spool = read(root/"exports/complete.json")
    require(spool["schema"] == "pd-spool-complete-v1" and spool["complete"] is True and spool["training_sha256"] == training["identity_sha256"] and
            spool["pre_training_sha256"] == initial["identity_sha256"], "PENDING: incomplete spool binding")
    require(len(spool["checkpoints"]) == EXPORTS and
            {(c["start"],c["step"]) for c in spool["checkpoints"]} == {(s,u) for s in STARTS for u in UPDATES[1:]},
            "PENDING: incomplete curve export inventory")
    for checkpoint in spool["checkpoints"]:
        validate_checkpoint(checkpoint)
        path = root/"exports"/(checkpoint["start"]+"-"+str(checkpoint["step"]))
        require((path/"checkpoint.json").is_file() and read(path/"checkpoint.json") == checkpoint and
                (path/"pd-export.json").is_file() and
                all((path/n).is_file() for n in checkpoint["tensor_files"]), "PENDING: missing/incomplete curve export")
    return initial, training, spool


def complete_export(root, checkpoint, scored):
    """Deletion follows the same bound live-load proof as the qualified stream."""
    verify_seal(scored)
    proof = scored["live_load_verification"]
    verify_seal(proof)
    require(scored["checkpoint_id"] == checkpoint["checkpoint_id"] and
            proof == probe.live_load_verification(checkpoint), "UNVERIFIED: scorer did not verify bound export")
    validate_checkpoint(checkpoint)
    root = Path(root).resolve()
    require((root/"exports").resolve() == root/"exports" and
            (root/"train/native").resolve() == root/"train/native", "refusing symlinked spool roots")
    path = Path(root)/"exports"/(checkpoint["start"]+"-"+str(checkpoint["step"]))
    require(path.parent.resolve() == (Path(root)/"exports").resolve() and not path.is_symlink(),
            "refusing deletion outside PD-owned spool")
    require(read(path/"checkpoint.json") == checkpoint, "owned export identity changed")
    verification = seal(dict(schema="pd-stream-verification-v2", checkpoint_id=checkpoint["checkpoint_id"],
        tensor_files=checkpoint["tensor_files"], weights_sha256=checkpoint["weights_sha256"],
        export_receipt_sha256=checkpoint["export_receipt_sha256"], score_sha256=digest(scored),
        predictions_sha256=digest(scored["predictions"]), summed_scores_sha256=digest(scored["summed_scores"]),
        covariates_sha256=digest(scored["gold_covariates"]), gpu_identity=scored["gpu_identity"],
        disposition="verified-scored-deleted"))
    write_frozen(Path(root)/"retained"/(checkpoint["checkpoint_id"]+".json"),
                 dict(checkpoint=checkpoint, verification=verification, metrics=scored))
    shutil.rmtree(path)
    native = Path(root)/"train/native"/(checkpoint["start"]+"-"+str(checkpoint["step"]))
    require(not native.is_symlink(), "refusing symlinked native checkpoint")
    shutil.rmtree(native)
    return verification


def score_identity(checkpoint, cell, pack, exposure, seed, start, update):
    return validate_score(seal(dict(score_id=f"pd-{seed}-{start}-{update}", checkpoint_id=checkpoint["checkpoint_id"],
        seed=seed, update=update, panel_sha256=pack["preparation_sha256"], prompt_sha256=pack["prompt_sha256"],
        scorer_sha256=sha256_file(probe.__file__), decoding_sha256=digest(dict(thinking=False, generated_cap=32,
        label_rule="sum-label-plus-eos-lexical-ties", generated_greedy=True)), tokenizer_sha256=pack["tokenizer_sha256"],
        suffix_sha256=digest(next(iter(pack["tokenized"].values()))["suffixes"]),
        gpu_identity=cell["gpu_identity"], predictions_sha256=digest(cell["predictions"]),
        summed_scores_sha256=digest(cell["summed_scores"]), covariates_sha256=digest(cell["gold_covariates"]),
        exposure_sha256=exposure["identity_sha256"], scoring_rule="sum-label-plus-eos")))


def reload_agreement(initial, post_scores):
    cohort = initial["cohort"]
    result = {}
    for start in STARTS:
        before, after = initial["scores"][start], post_scores[start]
        require(set(before["predictions"]) == set(after["predictions"]) == set(cohort["panel_ids"]),
                "UNVERIFIED: update-0 panel coverage changed")
        result[start] = export_agreement({i:before["summed_scores"][i] for i in cohort["ids"]},
                                        {i:after["summed_scores"][i] for i in cohort["ids"]})
    return seal(dict(schema="pd-cross-job-reload-v1", starts=result,
        qualified=all(r["qualified"] for r in result.values()), cohort_sha256=cohort["identity_sha256"]))


def post_score(plan, root, prepared, identity, pooled=False):
    initial, training, spool = ready_spool(plan, root, prepared)
    pack = inputs(root, prepared, "scoring")
    destination = root/"post"
    require(not destination.exists(), "PENDING: failed seed requires clean whole-seed restart")
    destination.mkdir()
    job = job_identity("post", prepared, identity)
    write_frozen(destination/"job.json", job)
    requests=[(plan["checkpoints"][start],plan["exports"][start]["directory"],plan["exports"][start]["receipt"],start,0) for start in STARTS]
    requests += [(c,root/"exports"/(c["start"]+"-"+str(c["step"])),root/"exports"/(c["start"]+"-"+str(c["step"]))/"pd-export.json",c["start"],c["step"]) for c in spool["checkpoints"]]
    scored={(r[3],r[4]):cell for r,cell in zip(requests,scoring_cells(plan,root,prepared,job,requests,pooled))}
    post_initial = {}
    cells = {s:{} for s in STARTS}
    checkpoints, scores, stream = [], [], {}
    for start in STARTS:
        checkpoint = plan["checkpoints"][start]
        export = plan["exports"][start]
        post_initial[start] = scored[start,0]
        cells[start][0] = post_initial[start]
        write_frozen(destination/(start+"-0.json"), post_initial[start])
    agreement = reload_agreement(initial, post_initial)
    write_frozen(destination/"reload-agreement.json", agreement)
    # Still score every registered cell if the cross-job check fails. The seed
    # remains UNVERIFIED and no alternate cohort, seed or scorer is substituted.
    for start in STARTS:
        checkpoints.append(plan["checkpoints"][start])
        for checkpoint in [c for c in spool["checkpoints"] if c["start"] == start]:
            update = checkpoint["step"]
            path = root/"exports"/(start+"-"+str(update))
            cell = scored[start,update]
            cells[start][update] = cell
            write_frozen(destination/(start+"-"+str(update)+".json"), cell)
            proof = complete_export(root, checkpoint, cell)
            stream[checkpoint["checkpoint_id"]] = proof
            checkpoints.append(seal({**checkpoint, "stream_verification_sha256":proof["identity_sha256"]}))
    runs = []
    for run in training["runs"]:
        start = run["start"]
        curve = {}
        for update in UPDATES:
            cell = cells[start][update]
            checkpoint = next(c for c in checkpoints if c["start"] == start and
                (c["step"] if c.get("role") == "probe" else 0) == update)
            score = score_identity(checkpoint, cell, pack, run["exposure"], plan["seed"], start, update)
            scores.append(score)
            curve[update] = seal({**cell, "score_id":score["score_id"]})
        seconds = sum(c["load_seconds"]+c["score_seconds"]+c["receipt_seconds"] for c in curve.values())
        runs.append(seal({**run, "scores":curve, "timing":dict(serial_scoring_wall_seconds=seconds,
            charged_gpu_count=1, scoring_gpu_hours=seconds/3600, wall_time_risk=seconds > 24*3600)}))
    seed = str(plan["seed"])
    result = dict(manifest=make_manifest(checkpoints, scores, preparation_sha256=prepared["preparation_sha256"],
        code_sha256=probe.code_identity()), runs=runs, cohorts={seed:initial["cohort"]},
        send_binding=plan["send_binding"], smoke_receipt=plan["smoke_receipt"], return_verification=None,
        stream_verifications=stream, split_job_sha256=implementation_identity(),
        pre_training={seed:initial}, training_receipts={seed:training},
        post_jobs={seed:dict(job=job, reload_agreement=agreement)},
        status="PROCESS_COMPLETED" if agreement["qualified"] else "UNVERIFIED")
    write_frozen(destination/"seed-evidence.json", result)
    return result


def validate_seed_amendment(record, *, require_agreement=True):
    """Reader recomputes agreement; a caller's PASS flag cannot bless a seed."""
    require(record["split_job_sha256"] == implementation_identity(), "UNVERIFIED: split-job code changed")
    agreements = {}
    for seed, initial in record["pre_training"].items():
        cohort = validate_initial(initial, seed=int(seed), preparation_sha256=record["manifest"]["preparation_sha256"])
        require(record["cohorts"][seed] == cohort, "UNVERIFIED: cohort differs from pre-training freeze")
        training = record["training_receipts"][seed]
        starts = {p["start"]:p for p in record["manifest"]["checkpoints"] if p["seed"] == int(seed) and p.get("role", "chain_start") == "chain_start"}
        validate_training(training, initial, dict(seed=int(seed), checkpoints=starts))
        job = record["post_jobs"][seed]["job"]
        verify_seal(job)
        require(job["phase"] == "post" and job["seed"] == int(seed) and
                job["preparation_sha256"] == record["manifest"]["preparation_sha256"] and
                job["split_job_sha256"] == record["split_job_sha256"], "UNVERIFIED: post job binding changed")
        post_initial = {}
        for run in [r for r in record["runs"] if r["seed"] == int(seed)]:
            original = next(r for r in training["runs"] if r["start"] == run["start"])
            require({k:v for k,v in run.items() if k not in ("scores", "timing", "identity_sha256")} ==
                    {k:v for k,v in original.items() if k not in ("scores", "timing", "identity_sha256")},
                    "UNVERIFIED: training evidence changed after GPU job")
            for cell in run["scores"].values():
                verify_seal(cell)
                require(cell["job_sha256"] == job["identity_sha256"] and cell["gpu_identity"] == job["gpu_identity"],
                        "UNVERIFIED-HARDWARE: curve combines separate jobs or GPUs")
            post_initial[run["start"]] = run["scores"].get(0, run["scores"].get("0"))
        agreement = reload_agreement(initial, post_initial)
        require(agreement == record["post_jobs"][seed]["reload_agreement"] and
                (agreement["qualified"] or not require_agreement),
                "UNVERIFIED: pre/post update-0 reload agreement failed")
        agreements[seed] = agreement
    return agreements


def validate_amendment(evidence, trusted):
    require(trusted is not None and trusted.get("split_job_sha256") == implementation_identity(),
            "UNVERIFIED: independently frozen split-job implementation required")
    expected = {str(s) for s in SEEDS}
    require(all(k in evidence and set(evidence[k]) == expected for k in ("pre_training", "training_receipts", "post_jobs")),
            "INCOMPLETE: three split-job seed receipts required")
    return validate_seed_amendment(evidence)


def collect(plan, root, prepared):
    cpu_only()
    # Full source/inventory/approval checks and evidence publication occur here.
    scientific.require_plan(plan)
    result = json.loads((root/"post/seed-evidence.json").read_text())
    validate_seed_amendment(result, require_agreement=False)
    verified = read(plan["scientific"]["verification"])
    result.update(return_verification=verified["return_verification"],
                  scientific_node_verification_sha256=verified["identity_sha256"])
    write_frozen(root/"seed-evidence.verified.json", result)
    return result


def prerequisites(phase, plan, root, prepared):
    # Missing prerequisites exit before CUDA discovery, model load or rank launch.
    if phase == "pre":
        require(not (root/"pre").exists(), "PENDING: clean whole-seed restart required")
    elif phase == "train":
        require((root/"pre/complete.json").is_file(), "PENDING: initial scores missing")
        initial = read(root/"pre/complete.json")
        validate_initial(initial, seed=plan["seed"], preparation_sha256=prepared["preparation_sha256"])
        require(not (root/"train").exists(), "PENDING: clean whole-seed restart required")
    else:
        ready_spool(plan, root, prepared)
        require(not (root/"post").exists(), "PENDING: clean whole-seed restart required")


def worker(phase, plan_file, root):
    if phase == "prep":
        return prepare(plan_file, root)
    plan, root, prepared = gate(plan_file, root)
    if phase == "finalize":
        return finalize(plan, root, prepared)
    if phase == "collect":
        return collect(plan, root, prepared)
    allocation_tokens(os.environ.get("CUDA_VISIBLE_DEVICES"), GPUS[phase])
    prerequisites(phase, plan, root, prepared)
    probe.configure_determinism()
    identity = probe.gpu_identity()
    require(len(identity["devices"]) == GPUS[phase], "PENDING: CUDA allocation differs")
    with gpu_leases(identity):
        if phase == "pre":
            return pre_score(plan, root, prepared, identity, pooled=True)
        if phase == "post":
            return post_score(plan, root, prepared, identity, pooled=True)
        destination = root/"train"
        destination.mkdir()
        (destination/"native").mkdir()
        write_frozen(destination/"job.json", job_identity("train", prepared, identity))
        command = [sys.executable, "-I", "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            "--nproc-per-node=8", "--max-restarts=0", str(Path(__file__).resolve()),
            "--rank-worker", "--plan", str(plan_file), "--out", str(root)]
        env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
        started = time.monotonic()
        try:
            result = subprocess.run(command, env=env)
            require(result.returncode == 0 and (destination/"complete.json").is_file(), "PENDING: PD training failed; restart whole seed")
        finally:
            elapsed = time.monotonic()-started
            write_frozen(destination/"launcher-timing.json", seal(dict(charged_gpu_count=8,
                wall_seconds=elapsed, gpu_hours=8*elapsed/3600, includes_worker_startup=True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES)
    parser.add_argument("--score-cell", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--rank-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        if args.score_cell:
            cell_worker(args.plan,args.out,args.score_cell)
        elif args.rank_worker:
            train_worker(args.plan, args.out)
        else:
            require(args.phase in PHASES, "explicit PD phase required")
            result = worker(args.phase, args.plan, args.out)
            if isinstance(result, dict) and result.get("status") == "UNVERIFIED":
                print(json.dumps(dict(status="UNVERIFIED", reason="pre/post update-0 agreement failed")))
                return 3
    except (ValueError, KeyError, TypeError, OSError) as error:
        status = "UNVERIFIED-HARDWARE" if "UNVERIFIED-HARDWARE:" in str(error) else "UNVERIFIED" if "UNVERIFIED" in str(error) else "PENDING"
        print(json.dumps(dict(status=status, reason=str(error))))
        return 3
    print(json.dumps(dict(status="PROCESS_COMPLETED", phase=args.phase)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
