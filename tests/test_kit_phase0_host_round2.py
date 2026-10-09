"""Closed host-venv round-2 regressions, also executed in the public export."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from test_kit_phase0_host import VERSIONS

BASE_ERROR='opencv-python requires numpy>=2, but you have numpy 1.26.4.'
NEW_ERROR='new-package requires missing-dependency, which is not installed.'


def result(code=0):return {'returncode':code,'failure_type':'cpu_exit' if code else None,'seconds':.1}


@pytest.fixture
def builder(tmp_path,monkeypatch):
    from kit import v4_phase0 as p,v4_phase0_site as s,runner
    work=tmp_path/'work';outputs={'base':BASE_ERROR+'\n','trainer':BASE_ERROR+'\n'};calls=[]
    monkeypatch.delenv('SLURM_JOB_ID',raising=False);monkeypatch.setenv('SDPO_DIR',str(tmp_path/'source'))
    monkeypatch.setattr(s,'installed_versions',lambda:VERSIONS.copy())
    monkeypatch.setattr(s,'distribution_inventory',lambda paths=None,**kw:[])
    monkeypatch.setattr(s,'check_network',lambda urls:None)
    def base(argv,**kw):
        calls.append(('base',argv));kw['log'].parent.mkdir(parents=True,exist_ok=True)
        kw['log'].write_text(outputs['base']);return result(1)
    def command(argv,**kw):
        calls.append(('trainer',argv));kw['log'].write_text(outputs['trainer'] if argv[1:]==['-m','pip','check'] else 'offline command\n')
        if 'audit-trainer' in argv:s.write_trainer_inventory(work,[],[],str(work/'phase0-envs/trainer'))
        return result(1 if argv[1:]==['-m','pip','check'] and outputs['trainer']!= 'No broken requirements found.\n' else 0)
    monkeypatch.setattr(runner,'bounded_command',base);monkeypatch.setattr(p,'bounded_command',command)
    return work,s,p,outputs,calls


def test_base_pip_check_records_existing_failures_without_refusal(builder):
    """The working base's pre-existing dependency conflicts must be observed, not block prepare."""
    work,s,p,outputs,calls=builder;outputs['base']='z conflict\na conflict\n'
    doc=s.base_environment(work)
    assert doc['pip_check']['output']==['a conflict','z conflict'] and doc['pip_check']['returncode']==1
    assert calls==[('base',[sys.executable,'-m','pip','check'])]
    assert (work/'v4/report-phase0/environment-build/base-pip-check.log').read_text()==outputs['base']


@pytest.mark.parametrize('state',['unchanged','repaired','new'])
def test_real_builder_refuses_only_new_pip_check_lines(builder,state):
    """Trainer pip check may preserve or repair known base conflicts, but must refuse a new conflict."""
    work,s,p,outputs,calls=builder
    if state=='repaired':outputs['trainer']='No broken requirements found.\n'
    if state=='new':outputs['trainer']+=NEW_ERROR+'\n'
    if state=='new':
        with pytest.raises(ValueError,match='new pip check'):p.environment_build(work)
        assert not any('vllm==0.18.0' in argv for role,argv in calls)
    else:p.environment_build(work)
    doc=json.loads((work/'v4/report-phase0/environment-build/trainer-pip-check.json').read_text())
    assert doc['new_lines']==([NEW_ERROR] if state=='new' else [])


def test_reader_rederives_pip_check_difference_from_both_raw_logs():
    """A forged empty delta cannot hide a new trainer conflict in the archive's raw pip output."""
    from kit import v4_phase0_site as s
    base={'output':[BASE_ERROR]};trainer={'output':[BASE_ERROR,NEW_ERROR],'new_lines':[]}
    with pytest.raises(ValueError,match='new pip check'):
        s.verify_pip_checks(base,trainer,(BASE_ERROR+'\n').encode(),(BASE_ERROR+'\n'+NEW_ERROR+'\n').encode())


def test_frozen_ray_socket_path_fits_even_with_long_owned_cache(tmp_path):
    """Ray must use a short job-private path independently of the owned wheel/compile cache length."""
    from kit import v4_phase0 as p,v4_phase0_site as s
    work=(tmp_path/('long-work-'+'x'*60)).resolve();folder=work/'v4/report-inputs';folder.mkdir(parents=True)
    env={**s.storage_environment(work,allocation=True),'WORK':str(work),'KIT':str(p.KIT),'V4_TELEMETRY':'1','V4_KIT_TAG':'offline','PYTHONPATH':str(p.KIT.parent),
         **{k:str(work/k) for k in ('MODEL_DIR','TEACHER_MODEL_DIR','QWEN3_8B_TOKENIZER','V4_TEACHER_PYTHON','SDPO_DIR','FINQA_ROOT')}}
    (folder/'launch-inputs-phase0.json').write_text(json.dumps({'environment':env}))
    frozen=p.frozen_launch_environment(work)
    path=frozen['RAY_TMPDIR']+'/ray/session_2026-10-08_12-34-56_123456_1234567/sockets/plasma_store'
    assert len(os.fsencode(path))<=107 and len(os.fsencode(frozen['RAY_TMPDIR']))+68<=107
    assert 'RAY_TMPDIR' not in s.CACHE_KEYS


@pytest.mark.parametrize('where',['planner','payload'])
def test_long_ray_path_refuses_before_gpu_rows(tmp_path,monkeypatch,capsys,where):
    """Both CPU planning and the payload must reject a Ray socket path longer than 107 bytes."""
    from kit import v4_phase0_site as s
    path='/'+('r'*39)
    if where=='planner':
        monkeypatch.setattr(s,'free_space',lambda *a:10**12)
        with pytest.raises(ValueError,match='Ray.*107'):s.planning_storage(tmp_path,{'TMPDIR':'/tmp/safe','RAY_TMPDIR':path})
    else:
        monkeypatch.setenv('RAY_TMPDIR',path)
        assert s.main(['ray-check','--work',str(tmp_path)])==2
        assert 'Ray' in capsys.readouterr().err


def fake_dist_info(folder,name,version):
    folder.mkdir(parents=True)
    (folder/'METADATA').write_text('Metadata-Version: 2.1\n'+('Name: '+name+'\n' if name is not None else '')+'Version: '+version+'\n')


@pytest.mark.parametrize('kind',['no-name','tilde-name','tilde-folder'])
def test_invalid_base_metadata_is_skipped_and_recorded(tmp_path,monkeypatch,kind):
    """Nameless or interrupted-pip dist-info must not crash prepare or become constraints."""
    from kit import v4_phase0_site as s,runner
    fake_dist_info(tmp_path/'site/valid-1.dist-info','valid','1')
    fake_dist_info(tmp_path/'site'/('~orch-2.dist-info' if kind=='tilde-folder' else 'broken-2.dist-info'),
        None if kind=='no-name' else '~orch' if kind=='tilde-name' else 'torch','2')
    distributions=list(s.metadata.distributions(path=[str(tmp_path/'site')]))
    monkeypatch.setattr(s.metadata,'distributions',lambda **kw:iter(distributions))
    monkeypatch.setattr(s,'installed_versions',lambda:VERSIONS.copy())
    def command(argv,**kw):kw['log'].write_text('No broken requirements found.\n');return result()
    monkeypatch.setattr(runner,'bounded_command',command)
    doc=s.base_environment(tmp_path/'work')
    assert [r['name'] for r in doc['inventory']]==['valid']
    assert len(doc['skipped_distributions'])==1 and doc['skipped_distributions'][0]['reason']
    assert s.base_constraints(doc['inventory'])=={'valid':'1'}


@pytest.mark.parametrize('reverse',[False,True])
def test_differing_duplicate_base_versions_leave_readable_blocker(tmp_path,monkeypatch,reverse):
    """Differing duplicate versions must refuse regardless of metadata discovery order."""
    from kit import v4_phase0_site as s
    fake_dist_info(tmp_path/'site/some_pkg-1.dist-info','Some_Pkg','1')
    fake_dist_info(tmp_path/'site/some_pkg-2.dist-info','some.pkg','2')
    distributions=list(s.metadata.distributions(path=[str(tmp_path/'site')]))
    if reverse:distributions.reverse()
    monkeypatch.setattr(s.metadata,'distributions',lambda **kw:iter(distributions))
    monkeypatch.setattr(s,'installed_versions',lambda:VERSIONS.copy())
    with pytest.raises(ValueError,match='duplicate.*some-pkg.*differing versions'):s.base_environment(tmp_path/'work')
    assert 'some-pkg' in (tmp_path/'work/v4/report-phase0/setup-blocker.txt').read_text()


@pytest.mark.parametrize('slowest,expected',[(1,600),(50,600),(80,705),(201,1310),(1000,5305)])
def test_import_deadline_adds_cuda_allowance_within_registered_budget(slowest,expected):
    """Sixteen CUDA contexts need 60 extra seconds; the full derivation must expose requirements above the budget."""
    from kit.v4_phase0_environment import environment_timing
    doc=environment_timing({'trainer':slowest,'inference':1})
    assert doc['deadline_seconds']==expected


def test_visible_owned_verl_exception_survives_duplicate_base_guard(tmp_path):
    """Base duplicate refusal must preserve the sole allowed trainer shadow: verl from owned SDPO."""
    from kit import v4_phase0_site as s
    owned=tmp_path/'owned/SDPO'
    fake_dist_info(tmp_path/'base/verl-1.dist-info','verl','1')
    fake_dist_info(tmp_path/'trainer/verl-2.dist-info','verl','2')
    (tmp_path/'trainer/verl-2.dist-info/direct_url.json').write_text(json.dumps({'url':owned.as_uri(),'dir_info':{'editable':True}}))
    for paths in [[tmp_path/'base',tmp_path/'trainer'],[tmp_path/'trainer',tmp_path/'base']]:
        with pytest.raises(ValueError,match='duplicate.*verl'):s.distribution_inventory(paths)
        rows=s.distribution_inventory(paths,allow_owned_verl=owned)
        assert {r['version'] for r in rows}=={'1','2'}


@pytest.fixture(autouse=True)
def native_cpu_runtime(monkeypatch):
    """Old ML stand-ins also provide the independently observed native torch identity."""
    from kit import v4_phase0_site as site
    monkeypatch.setattr(site,'torch_runtime_identity',lambda:{'version':'2.9.0+cu128','cuda':'12.8'})
