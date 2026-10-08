"""CPU disk admission for phase0's retained, full-parameter checkpoint policy."""
import hashlib
import json
from pathlib import Path

GIB=1024**3


def qwen3_parameters(config):
    """Count the pinned dense Qwen3 architecture without importing torch or allocating weights."""
    if config.get('model_type')!='qwen3':raise ValueError('phase0 disk config must describe the initial Qwen3 student')
    keys=('hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','head_dim','vocab_size')
    if any(type(config.get(k)) is not int or config[k]<=0 for k in keys):
        raise ValueError('phase0 disk config is missing positive Qwen3 model dimensions')
    h,i,layers,heads,kv_heads,head,vocab=(config[k] for k in keys)
    q=heads*head;kv=kv_heads*head
    # Embedding, untied output head, final RMSNorm. Each layer has Q/K/V/O,
    # SwiGLU's three matrices, two RMSNorms and Q/K's head-sized RMSNorms.
    embedding=h*vocab*(1 if config.get('tie_word_embeddings',False) else 2)
    layer=2*h*q+2*h*kv+3*h*i+2*h+2*head
    if config.get('attention_bias',False):layer+=q+2*kv
    return embedding+h+layers*layer


def snapshot_bytes(path):
    """Include whole downloaded repositories, including non-weight files, without re-reading tensors."""
    path=Path(path)
    if not path.is_dir():raise ValueError('phase0 disk snapshot directory is missing: '+str(path))
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def phase0_disk_requirement(launch):
    """Retain evidence; reserve the maximum simultaneous trainer-state/export/cache footprint."""
    try:
        initial=Path(launch['MODEL_DIR']);teacher=Path(launch['TEACHER_MODEL_DIR'])
        raw=(initial/'config.json').read_bytes();parameters=qwen3_parameters(json.loads(raw))
    except (KeyError,OSError,ValueError,TypeError) as exc:
        raise ValueError('phase0 disk admission cannot read a valid initial model config: '+str(exc)) from exc
    snapshots={'initial':{'path':str(initial),'bytes':snapshot_bytes(initial)},
               'teacher':{'path':str(teacher),'bytes':snapshot_bytes(teacher)}}
    arms={}
    for arm in ('S','F','R','D'):
        # V4_CHECKPOINT_STATE=1; steps=save_freq=2. F/R save only at step 2.
        # PPO's should_save_ckpt_esi can also force a step-1 checkpoint near
        # capacity expiry; S/D retain both (max_actor_ckpt_to_keep=4).
        checkpoints=2 if arm in 'SD' else 1
        # F/R explicitly load fp32; the pinned PPO actor also defaults to fp32.
        # FSDP partitions state across eight ranks: eight shards, not eight copies.
        arms[arm]={'retained_checkpoints':checkpoints,'fsdp_ranks':8,'model_bytes':4*parameters*checkpoints,
                   'adam_moments_bytes':8*parameters*checkpoints,'extra_and_serialization_bytes':GIB*checkpoints,
                   'ema_bytes':4*parameters*checkpoints if arm in 'SD' else 0,'merged_export_bytes':2*parameters}
    # The pinned FSDP merger casts to BF16 in RAM, creates no disk temporary,
    # and writes a distinct hf-step2 beside the original model/optimizer shards.
    # At the last merge all retained trainer states, S/D EMA, three older exports and
    # the entire new export coexist. Reserve fp32 S/D EMA even if saved in bf16.
    peak=sum(sum(row[k] for k in ('model_bytes','adam_moments_bytes','ema_bytes','merged_export_bytes')) for row in arms.values())
    extra=sum(row['extra_and_serialization_bytes'] for row in arms.values())
    # Keep the existing 30-GiB environment/compile-cache headroom. Additionally
    # reserve both full HF repos (at least the download admission's 100 GiB),
    # even though they already live outside WORK: conservative cache/staging
    # headroom on WORK's filesystem, without relying on a shared mount or /tmp.
    models=max(100*GIB,sum(row['bytes'] for row in snapshots.values()))
    caches=30*GIB
    required=peak+extra+models+caches
    return {'schema':'v4-phase0-disk.v1','initial_parameters':parameters,
            'initial_config_sha256':hashlib.sha256(raw).hexdigest(),'snapshots':snapshots,'arms':arms,
            'checkpoint_policy':'retain model, optimizer, extra and S/D EMA after checks',
            'checkpoint_count_basis':'F/R step 2; S/D steps 1 and 2 if capacity expiry triggers an early save',
            'peak_merge_bytes':peak,'extra_and_serialization_bytes':extra,
            'model_cache_bytes':models,'environment_and_compile_cache_bytes':caches,
            'required_bytes':required,'required_gib':(required+GIB-1)//GIB}
