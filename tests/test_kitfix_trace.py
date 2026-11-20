"""Offline regressions for the v7i TRACE prompt and command repairs."""
from copy import deepcopy
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from kit import nm_prepare, sdft_objective, trace_prepare, trace_prompt, trace_train
from kit.trace_common import KIT, atomic, digest, load

REAL_PREFLIGHT = trace_train.trainer_preflight
HYDRA_SKIP = 'local Hydra absent; node CPU prepare requires trainer Hydra preflight before GPU admission'


class Tokenizer:
    eos_token = '\x00'
    eos_token_id = 0

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, **kwargs):
        return self.encode('\n'.join(m['content'] for m in messages))


def test_released_agenda_target_census_preserves_membership_and_lengths():
    target = 'Agenda title read aloud by the clerk, with its complete wording.'
    rows = [dict(id=str(i), task='MeetingBank', prompt='Transcript: '+target,
                 answer=target) for i in range(5000)]
    original = deepcopy(rows)
    lengths = trace_prepare.census(rows, Tokenizer())
    assert len(lengths) == 5000 and rows == original
    student = [dict(role='user', content=rows[0]['prompt'])]
    teacher = trace_prompt.teacher_messages(student, target, released_prompt=rows[0]['prompt'])
    assert teacher == [dict(role='user', content=sdft_objective.TEACHER_TEMPLATE.substitute(
        orig_content=rows[0]['prompt'], output_text=target))]
    assert lengths['0'] == dict(student=len(rows[0]['prompt']),
        gold_teacher=len(teacher[0]['content']), gold_response=len(target)+1)
    assert trace_prepare.verbatim_targets(rows) == dict(count=5000, ids=list(map(str, range(5000))))


@pytest.mark.parametrize('messages,released', [
    ([dict(role='user', content='question + injected answer')], 'question'),
    ([dict(role='user', content='q', demonstration='answer')], 'q'),
    ([dict(role='user', content=['q'])], 'q'),
    ([dict(role='user', content='<demonstration>')], '<demonstration>'),
    ([dict(role='user', content='extra_info.demonstration')], 'extra_info.demonstration'),
    ([], 'q'),
])
def test_changed_or_structured_student_inputs_are_refused(messages, released):
    with pytest.raises(ValueError):
        trace_prompt.teacher_messages(messages, 'answer', released_prompt=released)


def test_native_schedule_binding_and_cpu_adapter_do_not_weaken_historical_check(monkeypatch):
    target = 'The agenda title occurs in the released transcript verbatim.'
    prompt = 'Transcript: '+target
    messages = [dict(role='user', content=prompt)]
    original_checker = sdft_objective.check_student_prompt_clean
    original_builder = sdft_objective.teacher_messages
    with pytest.raises(ValueError):
        original_builder(messages, target)
    with monkeypatch.context() as scoped:
        scoped.setattr(sdft_objective, 'check_student_prompt_clean', original_checker)
        trace_prompt.install([dict(prompt=prompt, response=target)])
        assert sdft_objective.teacher_messages is original_builder
        assert sdft_objective.teacher_messages(messages, target)
        with pytest.raises(ValueError):
            sdft_objective.teacher_messages([dict(role='user', content=prompt+' changed')], target)
        with pytest.raises(ValueError):
            sdft_objective.teacher_messages(messages, 'different demonstration')
        # Reinstallation binds a fresh schedule and cannot keep old permissions.
        trace_prompt.install([dict(prompt='other', response='target')])
        with pytest.raises(ValueError):
            sdft_objective.teacher_messages(messages, target)
    assert sdft_objective.check_student_prompt_clean is original_checker
    with pytest.raises(ValueError):
        original_builder(messages, target)


def test_native_entry_points_load_the_stage_schedule(tmp_path, monkeypatch):
    prompt = 'Released transcript with its agenda title repeated verbatim.'
    target = 'agenda title repeated verbatim.'
    messages = [dict(role='user', content=prompt)]
    atomic(tmp_path/'schedule.json', [dict(prompt=prompt, response=target)])
    config = SimpleNamespace(nm=SimpleNamespace(receipt=str(tmp_path/'native-receipt.json')))
    atomic(tmp_path/'prepared.json', dict(native=dict(receipt=config.nm.receipt),
        audit=dict(schedule_sha256=digest(load(tmp_path/'schedule.json')))))
    def inherited_call():
        assert sdft_objective.teacher_messages(messages, target)
        with pytest.raises(ValueError):
            sdft_objective.teacher_messages([dict(role='user', content=prompt+' injected')], target)
        return 'inherited result'
    class BaseWorker:
        def init_model(self):
            return inherited_call()
    class BaseRunner:
        def run(self, config):
            return inherited_call()
    tree = ast.parse((KIT/'trace_native.py').read_text())
    classes = ast.Module(body=[n for n in tree.body if isinstance(n, ast.ClassDef)], type_ignores=[])
    namespace = dict(nm_native=SimpleNamespace(NMWorker=BaseWorker, NMTaskRunner=BaseRunner),
        install_prepared=trace_prompt.install_prepared,
        register=lambda **kwargs: lambda fn: fn, Dispatch=SimpleNamespace(ONE_TO_ALL='all'))
    exec(compile(classes, str(KIT/'trace_native.py'), 'exec'), namespace)
    monkeypatch.setattr(sdft_objective, 'check_student_prompt_clean',
                        sdft_objective.check_student_prompt_clean)
    worker = namespace['TraceWorker']()
    worker.config = config
    assert worker.init_model() == 'inherited result'
    trace_prompt.install()  # Neither entry point can rely on a previous binding.
    assert namespace['TraceTaskRunner']().run(config) == 'inherited result'
    original = load(tmp_path/'schedule.json')
    for changed in ([], [dict(original[0], prompt=prompt+' changed')],
                    original+[dict(prompt='extra prompt', response=target)]):
        atomic(tmp_path/'schedule.json', changed)
        with pytest.raises(ValueError, match='schedule differs'):
            worker.init_model()
        with pytest.raises(ValueError, match='schedule differs'):
            namespace['TraceTaskRunner']().run(config)


def test_installed_list_prompts_are_detached_and_duplicate_targets_preserved(monkeypatch):
    monkeypatch.setattr(sdft_objective, 'check_student_prompt_clean',
                        sdft_objective.check_student_prompt_clean)
    rows = [dict(prompt=[dict(role='user', content='released')], response='target'),
            dict(prompt=[dict(role='user', content='second released')], response='target')]
    original = deepcopy(rows)
    trace_prompt.install(rows)
    rows[0]['prompt'][0]['content'] += ' target injected'
    rows[1]['prompt'].append(dict(role='user', content='target'))
    for row in original:
        assert sdft_objective.teacher_messages(row['prompt'], row['response'])
    for row in rows:
        with pytest.raises(ValueError, match='released schedule'):
            sdft_objective.teacher_messages(row['prompt'], row['response'])
    with pytest.raises(ValueError, match='released schedule'):
        sdft_objective.teacher_messages(original[0]['prompt'], 'missing pair')


@pytest.mark.parametrize('prompt', [[], None, {}, [dict(role='user', content=[])],
    [dict(role='assistant', content='q')],
    [dict(role='unknown', content='q')], [dict(role='user', content='<demonstration>')],
    [dict(role='user', content='q', demonstration='target')]])
def test_install_validates_prompts_before_replacing_binding(prompt, monkeypatch):
    checker = sdft_objective.check_student_prompt_clean
    with pytest.raises(ValueError):
        trace_prompt.install([dict(prompt=prompt, response='target')])
    assert sdft_objective.check_student_prompt_clean is checker


def test_trace_conversion_keeps_targets_and_restores_inherited_builder():
    target = 'An agenda title longer than forty characters, spoken in the transcript.'
    rows = [dict(id='agenda', prompt='Transcript: '+target, response=target, gold=target)]
    before = deepcopy(rows)
    builder = sdft_objective.teacher_messages
    checker = sdft_objective.check_student_prompt_clean
    converted, audit = trace_train.converted_trace_rows(
        rows, Tokenizer(), 'freeze', 'C', dict(prompt=2048, teacher=6144), 32)
    assert rows == before and audit['rows'] == 1
    assert converted[0]['prompt'] == [dict(role='user', content=rows[0]['prompt'])]
    assert converted[0]['extra_info']['demonstration'] == target
    assert converted[0]['data_source'] == 'trace'
    assert sdft_objective.teacher_messages is builder
    assert sdft_objective.check_student_prompt_clean is checker
    with pytest.raises(ValueError):
        nm_prepare.converted_rows(rows, Tokenizer(), 'freeze', 'C', dict(prompt=2048, teacher=6144), 32)


@pytest.fixture(params=['gold', '27b'])
def prepared_stage(tmp_path, monkeypatch, request):
    # Model/tokenizer stand-ins only; both real converters, Parquet writer and
    # inherited command builder execute. No model is acquired or run.
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer())))
    monkeypatch.setattr(nm_prepare, 'verify_model', lambda record: record)
    from kit import nm_model, trace_score
    monkeypatch.setattr(nm_model, 'metadata', lambda *a: {'fixture': [1, 2, 3]})
    monkeypatch.setattr(trace_score, 'tokenizer_identity', lambda *a: 'fixture-tokenizer')
    monkeypatch.setattr(trace_train, 'trainer_preflight', lambda prepared, python:
        dict(hydra='test-standin', omegaconf='test-standin',
             overrides=len(prepared['command'][5:]), **trace_train.preflight_binding(prepared, python)))
    model = dict(model=str(tmp_path/'base'), files={'weights.safetensors': 'a'*64}, sha256='b'*64)
    target = 'Released agenda title that is longer than forty characters.'
    demonstration = target if request.param == 'gold' else 'Verified teacher demonstration.'
    schedule = [dict(id=str(i), prompt='Transcript: '+target, response=demonstration,
                     gold=target) for i in range(1280)]
    cfg = dict(base=model, sdpo=str(tmp_path/'sdpo'), out=str(tmp_path), source=request.param)
    stage = dict(id='stage', task='MeetingBank', freeze=False)
    atomic(tmp_path/'schedules'/f'{request.param}-101-MeetingBank.json', schedule)
    root = tmp_path/'stage'
    prepared = trace_train.d_stage(cfg, stage, 101, root, model, mode='prepare')
    assert load(root/'prepared.json') == prepared
    assert load(root/'prep-key.json')['trainer_preflight'] == load(root/'trainer-preflight.json')
    assert prepared['audit']['rows'] == 1280
    assert load(root/'schedule.json') == [dict(r, domain='trace_MeetingBank') for r in schedule]
    assert trace_train.d_stage(cfg, stage, 101, root, model, mode='prepare') == prepared
    # Produce the inherited argv with the same real builder and prepared inputs.
    original = nm_prepare.converted_rows
    monkeypatch.setattr(nm_prepare, 'converted_rows', trace_train.converted_trace_rows)
    bundle = dict(base=model, tokenizer=model['model'], sdpo=cfg['sdpo'],
                  base_binding=str(root/'base-binding.json'), base_metadata={'fixture': [1, 2, 3]})
    binding = dict(recipe=dict(name='D', starts={'101':model},
        caps=dict(prompt=2048, teacher=6144, response=2048), balance_batch=False))
    # A second directory would change registered path values, so remove just the
    # test-owned receipt and regenerate in the same directory for value parity.
    (root/'prepared.json').unlink()
    inherited = nm_prepare.training_command(bundle, binding,
        [dict(r, domain='trace_MeetingBank') for r in schedule], root, 'freeze', 'C', 101)
    monkeypatch.setattr(nm_prepare, 'converted_rows', original)
    assert prepared['command'] == trace_train.trace_command(inherited['command'])
    assert {k:v for k,v in prepared.items() if k != 'command'} == {
        k:v for k,v in inherited.items() if k != 'command'}
    # Cached v7i preparation must launch the translated command too.
    assert trace_train.d_stage(cfg, stage, 101, root, model, mode='prepare') == prepared
    # Existing v7i CPU receipts are translated in memory on the native path.
    # Intercept the launch itself: no trainer, model or allocation is started.
    from kit import trace_common
    launched = []
    with monkeypatch.context() as scoped:
        scoped.setattr(trace_common, 'stage_command', lambda command, files: command)
        scoped.setattr(trace_train.subprocess, 'run', lambda command, **kw: launched.append(command))
        assert trace_train.d_stage(cfg, stage, 101, root, model, mode='native') is None
    assert launched == [prepared['command']]
    assert load(root/'prepared.json') == inherited
    return prepared, inherited


def test_real_stage_command_and_reward_adapter(prepared_stage):
    from kit.trace_entry import compute_score
    prepared, inherited = prepared_stage
    assert str(KIT/'trace_entry.py') in prepared['command']
    assert str(KIT/'nm_entry.py') not in prepared['command']
    assert compute_score('trace', 'trajectory', '', {}) == dict(score=0., acc=0.)
    assert prepared['native'] == inherited['native']


def test_real_hydra_parses_every_override_with_identical_values(prepared_stage):
    pytest.importorskip('hydra', reason=HYDRA_SKIP)
    from hydra.core.override_parser.overrides_parser import OverridesParser
    from omegaconf import OmegaConf
    parser = OverridesParser.create()
    prepared, inherited = prepared_stage
    actual_args = prepared['command'][5:]
    old_args = inherited['command'][5:]
    parsed = parser.parse_overrides(actual_args)
    assert len(parsed) == len(old_args)
    for actual, old in zip(parsed, old_args):
        key, value = old.split('=', 1)
        assert actual.key_or_group == key.lstrip('+')
        if key.startswith('+nm.'):
            expected = json.loads(value)
        elif key.lstrip('+') == 'custom_reward_function.path':
            expected = str(KIT/'trace_entry.py')
        else:
            expected = parser.parse_override(old).value()
        assert json.dumps(actual.value(), sort_keys=True) == json.dumps(expected, sort_keys=True)
        # Exercise OmegaConf resolution too, including nested dictionaries.
        resolved = OmegaConf.to_container(OmegaConf.create({'value': actual.value()}), resolve=True)['value']
        assert resolved == expected


@pytest.mark.parametrize('value', [
    {'caps': {'prompt': 2048, 'teacher': 6144}, 'flags': [True, False, None, 1.0]},
    'quotes " and backslash \\ and Unicode é, : [] {}',
    '${literal.value}', 'line one\nline two\tend',
    'trailing backslash '+chr(92), 'slash before quote '+chr(92)+'"',
])
def test_hydra_literal_roundtrip(value):
    pytest.importorskip('hydra', reason=HYDRA_SKIP)
    from hydra.core.override_parser.overrides_parser import OverridesParser
    from omegaconf import OmegaConf
    arg = trace_train.trace_command(['+nm.fixture='+json.dumps(value)])[0]
    parsed = OverridesParser.create().parse_override(arg).value()
    resolved = OmegaConf.to_container(OmegaConf.create({'value': parsed}), resolve=True)['value']
    assert resolved == value


@pytest.mark.parametrize('slashes', range(7))
@pytest.mark.parametrize('nested', [False, True])
def test_hydra_literal_interpolation_backslash_runs(slashes, nested):
    value = chr(92)*slashes+'${literal.value}'
    value = {'inner': [value, 'prefix '+value+' suffix', value+'"'+chr(92)]} if nested else value
    test_hydra_literal_roundtrip(value)


@pytest.mark.parametrize('key', ['1', '1.0', 'true', 'False', 'null', 'nan', 'inf', 'a:b', '${x}', ''])
def test_hydra_rejects_dictionary_keys_that_cannot_roundtrip(key):
    with pytest.raises(ValueError, match='dictionary key cannot round-trip'):
        trace_train.trace_command(['+nm.fixture='+json.dumps({'nested': {key: 1}})])


def test_trainer_preflight_checks_real_gold_and_27b_commands(prepared_stage, monkeypatch):
    pytest.importorskip('hydra', reason=HYDRA_SKIP)
    import hydra
    import omegaconf
    prepared, _ = prepared_stage
    receipt = REAL_PREFLIGHT(prepared, sys.executable)
    assert receipt == dict(hydra=hydra.__version__, omegaconf=omegaconf.__version__,
        overrides=len(prepared['command'][5:]), **trace_train.preflight_binding(prepared, sys.executable))
    # Exercise the CPU integration with the actual trainer interpreter too.
    monkeypatch.setattr(trace_train, 'trainer_preflight', REAL_PREFLIGHT)
    existing = Path(prepared['native']['receipt']).parent
    cfg = dict(out=str(existing.parent), base=prepared['incoming'], sdpo=prepared['cwd'],
        source='gold' if list((existing.parent/'schedules').glob('gold-*.json')) else '27b')
    root = existing/'real-trainer'
    trace_train.d_stage(cfg, dict(id='stage', task='MeetingBank', freeze=False),
                        101, root, prepared['incoming'], mode='prepare')
    cpu_receipt = load(root/'prep-key.json')['trainer_preflight']
    assert cpu_receipt == load(root/'trainer-preflight.json')
    assert (cpu_receipt['hydra'], cpu_receipt['omegaconf']) == (hydra.__version__, omegaconf.__version__)


@pytest.mark.parametrize('change', ['value', 'reward', 'syntax', 'missing', 'extra', 'duplicate'])
def test_trainer_preflight_refuses_mismatches(prepared_stage, change):
    pytest.importorskip('hydra', reason=HYDRA_SKIP)
    prepared, _ = prepared_stage
    prepared = deepcopy(prepared)
    args = prepared['command']
    if change == 'missing':
        args.remove(next(a for a in args if a.startswith('+nm.seed=')))
    elif change in ('extra', 'duplicate'):
        args.append('+nm.unregistered=1' if change == 'extra' else '+nm.seed=101')
    else:
        prefix = '++custom_reward_function.path=' if change == 'reward' else '+nm.seed='
        index = next(i for i, arg in enumerate(args) if arg.startswith(prefix))
        args[index] = prefix+dict(value='999', reward='"nm_entry.py"', syntax='{broken')[change]
    with pytest.raises((ValueError, KeyError)):
        REAL_PREFLIGHT(prepared, sys.executable)


def test_trainer_preflight_missing_hydra_fails_closed(prepared_stage, tmp_path, monkeypatch):
    prepared, _ = prepared_stage
    prepared = deepcopy(prepared)
    stub = tmp_path/'no-hydra-python'
    stub.write_text('#!'+sys.executable+'\nimport os, sys\n'
                    'os.execv(sys.executable, [sys.executable, "-S", *sys.argv[1:]])\n')
    stub.chmod(0o755)
    prepared['command'][0] = str(stub)
    with pytest.raises(ValueError, match='trainer Hydra/OmegaConf required'):
        REAL_PREFLIGHT(prepared, str(stub))
    monkeypatch.setattr(trace_train, 'trainer_preflight', REAL_PREFLIGHT)
    existing = Path(prepared['native']['receipt']).parent
    cfg = dict(out=str(existing.parent), base=prepared['incoming'], sdpo=prepared['cwd'],
        trainer_python=str(stub),
        source='gold' if list((existing.parent/'schedules').glob('gold-*.json')) else '27b')
    root = existing/'missing-hydra'
    with pytest.raises(ValueError, match='trainer Hydra/OmegaConf required'):
        trace_train.d_stage(cfg, dict(id='stage', task='MeetingBank', freeze=False),
                            101, root, prepared['incoming'], mode='prepare')
    assert not (root/'prep-key.json').exists()
    assert not (root/'trainer-preflight.json').exists()


def test_native_refuses_missing_or_changed_cpu_preflight(prepared_stage, monkeypatch):
    prepared, _ = prepared_stage
    root = Path(prepared['native']['receipt']).parent
    cfg = dict(out=str(root.parent), base=prepared['incoming'], sdpo=prepared['cwd'],
               source='gold' if list((root.parent/'schedules').glob('gold-*.json')) else '27b')
    stage = dict(id='stage', task='MeetingBank', freeze=False)
    receipt = load(root/'trainer-preflight.json')
    atomic(root/'trainer-preflight.json', dict(receipt, command_sha256='0'*64))
    with pytest.raises(ValueError, match='preflight binding changed'):
        trace_train.d_stage(cfg, stage, 101, root, prepared['incoming'], mode='native')
    (root/'trainer-preflight.json').unlink()
    with pytest.raises(ValueError, match='CPU trainer Hydra preflight required'):
        trace_train.d_stage(cfg, stage, 101, root, prepared['incoming'], mode='native')


@pytest.mark.parametrize('efficiency', [None, False])
@pytest.mark.parametrize('studies', [None, ['gold'], ['27b'], ['s8', 'gold']])
def test_legacy_production_inventory_refused_before_wrapper_or_execution(
        tmp_path, monkeypatch, efficiency, studies):
    from kit import trace_jobs as jobs
    cfg = dict(out=str(tmp_path/'out'))
    if efficiency is not None:
        cfg['efficiency'] = efficiency
    if studies is not None:
        cfg['studies'] = studies
    config = tmp_path/'config.json'
    atomic(config, cfg)
    monkeypatch.setattr(jobs, 'execute', lambda *a: pytest.fail('allocation executed'))
    monkeypatch.setattr(jobs.subprocess, 'call', lambda *a, **kw: pytest.fail('GPU wrapper started'))
    monkeypatch.delenv('UTIL_WATCH_ACTIVE', raising=False)
    with pytest.raises(ValueError, match='set efficiency: true'):
        jobs.configured_inventory(cfg)
    with pytest.raises(ValueError, match='set efficiency: true'):
        jobs.run(config, 'train-gold-101')
    for args in (['inventory'], ['run', '--row', 'train-gold-101']):
        monkeypatch.setattr(sys, 'argv', ['trace_jobs', *args, '--config', str(config)])
        with pytest.raises(ValueError, match='set efficiency: true'):
            jobs.main()
    assert not Path(cfg['out']).exists()


def test_unconfigured_inventory_cannot_publish_legacy_production_rows(monkeypatch):
    from kit import trace_jobs as jobs
    monkeypatch.setattr(sys, 'argv', ['trace_jobs', 'inventory'])
    with pytest.raises(ValueError, match='set efficiency: true'):
        jobs.main()


@pytest.mark.parametrize('efficiency', [None, False, True])
@pytest.mark.parametrize('source', ['gold', '27b'])
def test_direct_legacy_training_cannot_preflight_inside_gpu_allocation(
        tmp_path, monkeypatch, efficiency, source):
    cfg = dict(out=str(tmp_path/'out'), source=source)
    if efficiency is not None:
        cfg['efficiency'] = efficiency
    monkeypatch.setattr(trace_train, 'trainer_preflight',
        lambda *a: pytest.fail('combined path reached preflight'))
    with pytest.raises(ValueError, match='set efficiency: true'):
        trace_train.run(cfg, source, 101)
    root = tmp_path/'stage'
    for mode in ({}, dict(mode='all')):
        with pytest.raises(ValueError, match='separate CPU preparation'):
            trace_train.d_stage(cfg, {}, 101, root, {}, **mode)
    assert not root.exists() and not Path(cfg['out']).exists()


def test_registered_inventories_and_every_cpu_training_dependency_are_preserved():
    from kit import trace_jobs as jobs, trace_common as common
    for efficiency in ({}, dict(efficiency=False)):
        assert jobs.configured_inventory(dict(technical_synthetic=True, **efficiency)) == common.inventory()
        assert jobs.configured_inventory(dict(studies=['s8'], **efficiency)) == common.inventory(['s8'])
    value = jobs.configured_inventory(dict(efficiency=True))
    assert value == trace_train.efficiency_inventory()
    rows = {row['id']: row for row in value['rows']}
    for source in common.SOURCES:
        for seed in common.SEEDS:
            for stage in common.stages(source, seed):
                cpu = rows['prep-'+stage['id']]
                gpu = rows['native-'+stage['id']]
                assert cpu['phase'] == 'prep-stage' and cpu['gpus'] == 0
                assert gpu['phase'] == 'train-stage' and gpu['gpus'] == 8
                assert gpu['depends'] == [cpu['id']]
    assert not any(row['phase'] == 'train' and row['study'] in common.SOURCES
                   for row in rows.values())


@pytest.mark.parametrize('efficiency', [None, False, True])
@pytest.mark.parametrize('failure', ['missing-hydra', 'missing-python', 'syntax'])
def test_cpu_preflight_failure_prevents_gpu_admission(
        prepared_stage, tmp_path, monkeypatch, efficiency, failure):
    from kit import trace_jobs as jobs, trace_common as common
    prepared, _ = prepared_stage
    existing = Path(prepared['native']['receipt']).parent
    source = 'gold' if list((existing.parent/'schedules').glob('gold-*.json')) else '27b'
    cfg = dict(out=str(existing.parent), base=prepared['incoming'], sdpo=prepared['cwd'],
               studies=[source])
    if efficiency is not None:
        cfg['efficiency'] = efficiency
    config = tmp_path/'production.json'
    executions = []
    original_execute = jobs.execute
    def execute(cfg, row, *args):
        executions.append(row)
        return original_execute(cfg, row, *args)
    monkeypatch.setattr(jobs, 'execute', execute)
    monkeypatch.setattr(trace_train, 'trainer_preflight', REAL_PREFLIGHT)
    if efficiency is not True:
        atomic(config, cfg)
        with pytest.raises(ValueError, match='set efficiency: true'):
            jobs.run(config, f'train-{source}-101')
        assert executions == [] and not (Path(cfg['out'])/'attempts').exists()
        return
    if failure == 'missing-hydra':
        python = tmp_path/'no-hydra-python'
        python.write_text('#!'+sys.executable+'\nimport os, sys\n'
            'os.execv(sys.executable, [sys.executable, "-S", *sys.argv[1:]])\n')
        python.chmod(0o755)
        cfg['trainer_python'] = str(python)
    elif failure == 'missing-python':
        cfg['trainer_python'] = str(tmp_path/'absent-python')
    else:
        pytest.importorskip('hydra', reason=HYDRA_SKIP)
        original_command = trace_train.trace_command
        def malformed(command):
            return ['+nm.seed={broken' if arg.startswith('+nm.seed=') else arg
                    for arg in original_command(command)]
        monkeypatch.setattr(trace_train, 'trace_command', malformed)
    atomic(config, cfg)
    stage = common.stages(source, 101)[0]
    schedule = load(existing/'schedule.json')
    atomic(Path(cfg['out'])/'schedules'/f'{source}-101-{stage["task"]}.json', schedule)
    runtime = dict(cfg, _config_path=str(config.resolve()))
    original_receipt = jobs.row_receipt
    def receipt(out, row_id):
        if row_id == 'qualify-'+source:
            return dict(config_sha256=digest(runtime), technical_synthetic=False)
        return original_receipt(out, row_id)
    monkeypatch.setattr(jobs, 'row_receipt', receipt)
    cpu_id, gpu_id = 'prep-'+stage['id'], 'native-'+stage['id']
    with pytest.raises(ValueError, match='trainer Hydra/OmegaConf required'):
        jobs.run(config, cpu_id)
    root = trace_train.stage_root(cfg, source, 101, stage)
    assert not (root/'prep-key.json').exists()
    assert not (root/'trainer-preflight.json').exists()
    cpu_attempts = list((Path(cfg['out'])/'attempts'/cpu_id).glob('*/receipt.json'))
    assert len(cpu_attempts) == 1
    assert load(cpu_attempts[0])['status'] == 'ERROR'
    assert load(cpu_attempts[0])['gpus'] == 0
    with pytest.raises(FileNotFoundError):
        jobs.run(config, gpu_id)
    assert [row['id'] for row in executions] == [cpu_id]
    assert not (Path(cfg['out'])/'attempts'/gpu_id).exists()
    assert not (Path(cfg['out'])/'state'/cpu_id).exists()
