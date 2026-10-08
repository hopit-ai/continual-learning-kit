"""Partner report, 8 October: Qwen3.6-27B is multimodal; vocab_size lives under text_config, not at the top level."""
import json
from pathlib import Path
import pytest
from kit import v4_phase0 as phase0

# Verbatim config.json of Qwen/Qwen3.6-27B at revision 6a9e13bd (fetched 8 October 2026), embedded so the public kit ships it.
REAL_CONFIG=json.loads(r'''{"architectures": ["Qwen3_5ForConditionalGeneration"], "image_token_id": 248056, "language_model_only": false, "model_type": "qwen3_5", "text_config": {"attention_bias": false, "attention_dropout": 0.0, "attn_output_gate": true, "bos_token_id": 248044, "dtype": "bfloat16", "eos_token_id": 248044, "full_attention_interval": 4, "head_dim": 256, "hidden_act": "silu", "hidden_size": 5120, "initializer_range": 0.02, "intermediate_size": 17408, "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention"], "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_value_head_dim": 128, "mamba_ssm_dtype": "float32", "max_position_embeddings": 262144, "model_type": "qwen3_5_text", "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": false, "num_attention_heads": 24, "num_hidden_layers": 64, "num_key_value_heads": 4, "output_gate_type": "swish", "pad_token_id": null, "partial_rotary_factor": 0.25, "rms_norm_eps": 1e-06, "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"}, "tie_word_embeddings": false, "use_cache": true, "vocab_size": 248320}, "tie_word_embeddings": false, "transformers_version": "4.57.1", "video_token_id": 248057, "vision_config": {"deepstack_visual_indexes": [], "depth": 27, "hidden_act": "gelu_pytorch_tanh", "hidden_size": 1152, "in_channels": 3, "initializer_range": 0.02, "intermediate_size": 4304, "model_type": "qwen3_5", "num_heads": 16, "num_position_embeddings": 2304, "out_hidden_size": 5120, "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2}, "vision_end_token_id": 248054, "vision_start_token_id": 248053}''')
REV='6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'


def snapshot(tmp_path,config):
    path=tmp_path/'hub'/'models--Qwen--Qwen3.6-27B'/'snapshots'/REV;path.mkdir(parents=True)
    (path/'config.json').write_text(json.dumps(config))
    (path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':'model-00001.safetensors'}}))
    (path/'model-00001.safetensors').write_bytes(b'x')
    return path


def teacher_errors(path,monkeypatch=None):
    import os
    old=os.environ.get('TEACHER_MODEL_DIR');os.environ['TEACHER_MODEL_DIR']=str(path)
    try:
        try:phase0.validate_prepare_inputs(path.parent,None,None)
        except ValueError as error:return [line for line in str(error).split('\n') if 'TEACHER_MODEL_DIR' in line and ('architecture' in line or 'model input invalid' in line or 'revision' in line)]
        return []
    finally:
        if old is None:os.environ.pop('TEACHER_MODEL_DIR',None)
        else:os.environ['TEACHER_MODEL_DIR']=old


def test_real_config_is_nested():
    config=json.loads(json.dumps(REAL_CONFIG))
    assert config.get('vocab_size') is None and config['text_config']['vocab_size']==248320


def test_real_27b_config_is_admitted(tmp_path):
    assert teacher_errors(snapshot(tmp_path,json.loads(json.dumps(REAL_CONFIG))))==[]


@pytest.mark.parametrize('config',[{'text_config':{'vocab_size':151936}},{'vocab_size':151936},{}])
def test_wrong_or_missing_vocab_still_refuses(tmp_path,config):
    assert teacher_errors(snapshot(tmp_path,config))


def test_scientific_identity_accepts_the_real_nested_config(monkeypatch):
    from kit import v4_teacher as t, v4_contract
    weights={'model-00001.safetensors':'a'*64}
    monkeypatch.setattr(t,'REGISTERED_TEACHER_REVISION',REV);monkeypatch.setattr(t,'REGISTERED_TEACHER_WEIGHT_HASHES',weights)
    monkeypatch.setattr(t,'INITIAL_8B_WEIGHT_HASHES',{'model.safetensors':'b'*64})
    monkeypatch.setattr(v4_contract,'technical_synthetic',lambda docs:False)
    common=dict(stand_in_teacher=False,attempts=4,thinking=False,response_cap=t.RESPONSE_CAP,immutable_revision_known=True)
    teacher=dict(common,generator_role='external_teacher',max_new_tokens=t.TEACHER_NEW_TOKENS,model_revision=REV,
                 model_file_hashes=weights,model_config=json.loads(json.dumps(REAL_CONFIG)))
    rewriter=dict(common,generator_role='frozen_initial_student',max_new_tokens=t.REWRITE_NEW_TOKENS,
                  model_revision=t.INITIAL_8B_REVISION,model_file_hashes={'model.safetensors':'b'*64})
    assert t.scientific_identity_allowed([teacher],[rewriter])
    teacher['model_config']={'text_config':{'vocab_size':151936}}
    assert not t.scientific_identity_allowed([teacher],[rewriter])


def test_archive_reader_accepts_the_real_nested_config(monkeypatch):
    from kit import v4_evidence as ev
    identities={'teacher':{'model':'Qwen/Qwen3.6-27B','model_revision':REV,'model_config':json.loads(json.dumps(REAL_CONFIG))},
                'initial':{'model':'Qwen/Qwen3-8B','model_revision':ev.INITIAL_8B_REVISION,'model_config':{'vocab_size':151936}}}
    env={'kit_tag':'kit-v4-phase0-v2','trainer_commit':__import__('kit.v4_datasets',fromlist=['x']).SDPO_COMMIT,'inference_package_identity_sha256':'c'*64,'inference':{'package_sha256':'c'*64}}
    class Archive:
        def json(self,name):return identities if name.endswith('model-identities.json') else env
    monkeypatch.setattr(ev,'weights',lambda doc:None)
    ev.environment(Archive(),'phase0')
    identities['teacher']['model_config']={'text_config':{'vocab_size':151936}}
    with pytest.raises(Exception):ev.environment(Archive(),'phase0')
