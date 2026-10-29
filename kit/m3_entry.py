"""M3 CLI shim; no inputs fetched and no job launched by importing this module."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import threading
from contextlib import contextmanager
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.m3_contract import ARMS, SEEDS, BASE_REVISION, CONFIG_KEYS, validate_config, STAGES, stage_number, COMMON_OVERRIDES, REWARD_ENTRYPOINT, PROFILES, find_m3_root, PREFLIGHT_TEMPLATE



# Smoke-only single-container settings. Scientific/node profiles keep their
# registered networking. Reserve 20 seconds below the manager's 60-second bound
# for error propagation and local Ray cleanup.
SMOKE_INIT_TIMEOUT = 40


def smoke_loopback_settings():
    return dict(
        env=dict(MASTER_ADDR='127.0.0.1', MASTER_PORT='29500',
                 NCCL_SOCKET_IFNAME='lo', GLOO_SOCKET_IFNAME='lo',
                 VLLM_HOST_IP='127.0.0.1', RAY_ADDRESS='127.0.0.1:6379',
                 RAY_USAGE_STATS_ENABLED='0',
                 # Pinned Ray uses this switch on Linux too: local-only mode
                 # prevents resolve_ip_for_localhost from discovering a NIC.
                 RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER='0',
                 RAY_gcs_rpc_server_connect_timeout_s=str(SMOKE_INIT_TIMEOUT),
                 RAY_gcs_rpc_server_reconnect_timeout_s=str(SMOKE_INIT_TIMEOUT)),
        ray_init=dict(address='127.0.0.1:6379', _node_ip_address='127.0.0.1'),
        ray_head=['ray', 'start', '--head', '--node-ip-address=127.0.0.1',
                  '--port=6379', '--num-cpus=8', '--num-gpus=2',
                  '--include-dashboard=false', '--disable-usage-stats'],
        init_timeout_seconds=SMOKE_INIT_TIMEOUT, cleanup_timeout_seconds=10,
        worker_network_open=True,
        offline_env=dict(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1'))


def smoke_initialization_timer(stage, *, ready=None, seconds=SMOKE_INIT_TIMEOUT, exit_process=None):
    """Fail the child even when a native socket call ignores Python timeouts."""
    def expired():
        if ready is not None and ready.is_file():
            return
        print('[m3-init-failure] '+json.dumps(dict(stage=stage, timeout_seconds=seconds)),
              file=sys.stderr, flush=True)
        (exit_process or os._exit)(70)
    timer=threading.Timer(seconds, expired)
    timer.daemon=True
    timer.start()
    return timer


@contextmanager
def smoke_initialization_deadline(stage):
    timer=smoke_initialization_timer(stage)
    try:
        yield
    finally:
        timer.cancel()


def smoke_runner_ready_path(receipt):
    return Path(receipt).with_suffix('.ray-runner-ready')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--stage', choices=(*STAGES,'1','2','3'), required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--profile', choices=tuple(PROFILES), default='scientific')
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--base-revision', default=BASE_REVISION)
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--preflight', help='8B/eight-H100 qualification receipt; required for scientific launch')
    parser.add_argument('--preflight-phase',choices=('uninterrupted','prefix','resume'),default='uninterrupted')
    parser.add_argument('--input-manifest')
    parser.add_argument('--host-reservation-bytes',type=int)
    args, remaining = parser.parse_known_args(argv)
    args.stage = STAGES[stage_number(args.stage)-1]
    args.receipt=str(Path(args.receipt).resolve())
    if args.preflight is None:
        args.preflight=str(find_m3_root(args.receipt)/PREFLIGHT_TEMPLATE.format(arm=args.arm))
    else: args.preflight=str(Path(args.preflight).resolve())
    if args.input_manifest: args.input_manifest=str(Path(args.input_manifest).resolve())
    config = validate_config({key: getattr(args, key) for key in CONFIG_KEYS})
    if any(value.lstrip('+').split('=',1)[0].startswith('m3.') for value in remaining):
        parser.error('M3 identity must be supplied through explicit CLI flags')
    return config, remaining


def hydra_arguments(config, overrides):
    """Forward offline/finish/profile settings explicitly into Ray worker envs."""
    runtime = dict(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                   KIT_FINISH_GATE='1', WANDB_MODE='disabled', V4_PROFILE='technical-smoke' if config['profile']=='preflight' else config['profile'])
    if config['profile']=='technical-smoke':
        runtime.update(smoke_loopback_settings()['env'])
    phase_dir=Path(config['receipt']).parent
    if config['profile']=='preflight':
        runtime.update(VERL_FILE_LOGGER_PATH=str(phase_dir/'metrics.jsonl'),CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       PYTHONHASHSEED=str(config['seed']))
    os.environ.update(runtime)
    common = {**COMMON_OVERRIDES, 'custom_reward_function.path': str(Path(__file__).resolve()),
              'custom_reward_function.name': REWARD_ENTRYPOINT}
    if config['profile']=='technical-smoke':
        common['actor_rollout_ref.nccl_timeout']=SMOKE_INIT_TIMEOUT
        common.update({'data.custom_cls.path':str(Path(__file__).with_name('m3_runner.py')),
                       'data.custom_cls.name':'M3SmokeJSONLDataset'})
        common.update({'ray_kwargs.ray_init.'+k:v for k,v in smoke_loopback_settings()['ray_init'].items()})
    if config['profile']=='preflight':
        common.update({'ray_kwargs.ray_init.address':'local','ray_kwargs.ray_init.include_dashboard':False,
                       'trainer.rollout_data_dir':str(phase_dir/'rollouts'),
                       'trainer.validation_data_dir':str(phase_dir/'validation'),
                       'vars.log_dir':str(phase_dir/'logs'),'vars.ckpt_dir':str(phase_dir/'checkpoints')})
    return [*overrides, *['++'+key+'='+json.dumps(value) for key,value in common.items()], *['+m3.'+key+'='+json.dumps(value) for key,value in config.items()],
            *['++ray_kwargs.ray_init.runtime_env.env_vars.'+key+'='+json.dumps(value) for key,value in runtime.items()]]


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Diagnostic rewards only; replay has zero reward and is supervised by M3."""
    extra=dict(extra_info or {})
    replay=extra.get('m3_replay') is True
    if data_source=='p3b_replay':
        if not replay: raise ValueError('replay data_source requires explicit M3 metadata')
        result=dict(score=0.,acc=0.,pred='',incorrect_format=0.)
    else:
        if replay or data_source not in ('finqa','contract_nli','math_screen'):
            raise ValueError('M3 task/replay reward source differs')
        if data_source=='contract_nli':
            from kit.c_reward import compute_score as score
        else:
            from kit.p3b_reward import _task_score as score
        result=score(data_source,solution_str,ground_truth,dict(extra,p3b_replay=False))
    # Stable keys across mixed domains; preserve all original task diagnostics.
    return {**dict(acc=0.,pred='',incorrect_format=0.,feedback='',
                   truncated=int(bool(extra.get('truncated',False))),score_before_gate=0.),
            **result, 'm3_replay':int(replay), 'p3b_replay':int(replay)}


def main(argv=None):
    config, overrides = parse_args(argv)
    arguments = hydra_arguments(config, overrides)
    from kit.m3_runner import M3TaskRunner, check_pinned, main_ppo
    check_pinned()
    main_ppo.TaskRunner = M3TaskRunner
    sys.argv = [sys.argv[0], *arguments]
    dispatch_timer=None
    if config['profile']=='technical-smoke':
        # Connect explicitly to the head started by execute(), preserving the
        # pinned driver's default runtime env as well as our worker settings.
        import ray
        from omegaconf import OmegaConf
        runtime=main_ppo.get_ppo_ray_runtime_env()
        env={key:os.environ[key] for key in (*smoke_loopback_settings()['env'],
             *smoke_loopback_settings()['offline_env'], 'KIT_FINISH_GATE','WANDB_MODE','V4_PROFILE')}
        with smoke_initialization_deadline('ray-connect'):
            ray.init(**smoke_loopback_settings()['ray_init'],
                     runtime_env=OmegaConf.to_container(OmegaConf.merge(runtime, {'env_vars':env})))
        ready=smoke_runner_ready_path(config['receipt'])
        ready.parent.mkdir(parents=True,exist_ok=True)
        ready.unlink(missing_ok=True)
        dispatch_timer=smoke_initialization_timer('ray-task-dispatch',ready=ready)
    try:
        main_ppo.main()
    finally:
        if dispatch_timer is not None:dispatch_timer.cancel()
        if config['profile']=='preflight':
            import ray
            if ray.is_initialized():ray.shutdown()  # Only this child's address=local session.


if __name__ == '__main__':
    main()
