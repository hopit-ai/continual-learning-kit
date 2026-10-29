"""Separate message-2 add-on identities. Never opens the M tree for writing."""
from kit.s_common import *
from kit.m2_seed_policy import SEEDS
from kit import m2_common as m

ARMS = ('control', 'p', 'drift', 'plasticity')
STEPS = (0, 10, 20, 30, 40)


def domains(configuration):
    return (['ifeval'] + (['gsm8k', 'arc'] if configuration in ('broad', 'both') else [])
            + (['finqa'] if configuration in ('earlier', 'both') else []))


def implementation():
    paths = [*sorted((ROOT/'kit').glob('x_*.py')), ROOT/'kit/m2_train.py',
             ROOT/'kit/c_sdft_runner.py', ROOT/'kit/m2_score.py', ROOT/'kit/addon_dryrun.py',
             *sorted((ROOT/'kit').glob('p2_*.py')),ROOT/'scripts/smoke_rung_p2_modal.py',ROOT/'kit/p_replace.py',ROOT/'kit/c_d_prompt.py',ROOT/'kit/p_smoke.py',ROOT/'kit/resources/p/smoke-settings.json',ROOT/'kit/resources/p2/node-limits.json',ROOT/'scripts/smoke_x_integration_modal.py']
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def context(out, standin):
    registration()  # S registration/analysis bytes remain untouched.
    doc = read(out/'binding.json')
    require(doc['technical_synthetic'] is standin, 'technical_synthetic requires --standin')
    require(doc['implementation'] == implementation(), 'add-on code changed after binding')
    if doc.get('scoring_protocol'):require(doc['scoring_protocol']==m.protocol(),'add-on scoring protocol changed')
    # These are read-only checks; continuing the core cannot change this admission.
    core = Path(doc['message2'])
    require(sha(core/'binding.json') == doc['message2_sha256'], 'M2 binding changed')
    require(sha(core/'admission.json') == doc['admission_sha256'], 'M9 admission changed')
    require(sha(core/'m2-scoring-machine.json') == doc['core_scoring_machine']['sha256'], 'core scoring machine changed')
    for ref in doc['assets']['inputs'].values():
        require(sha(Path(ref['path'])/'manifest.json') == ref['manifest'], 'M3 input changed')
        m.frozen_files(Path(ref['path']))
    return doc


def incoming(doc, seed):
    return doc['spec']['nodes'][f'b-s{seed}']['checkpoint']


def checkpoint(out, doc, arm, seed, step):
    from kit.x_tier2 import ARMS2
    require(arm in (*ARMS, *ARMS2) and seed in SEEDS and step in STEPS, 'unregistered arm/seed/step')
    if step == 0:
        require(arm!='p' or doc.get('p_variant')=='p2-l2-init','closed P variant cannot supply a stage-start checkpoint')
        return incoming(doc,seed)
    root = out/'runs'/f'x-{arm}-s{seed}'
    require((root/f'export-{step}.json').exists(), 'export pending')
    return dict(model=str(root/f'hf-step{step}'), receipt=str(root/f'export-{step}.json'))


def independent(*roots):
    roots = [Path(p).resolve() for p in roots]
    require(all(a != b and a not in b.parents and b not in a.parents
                for i, a in enumerate(roots) for b in roots[i+1:]), 'state trees must be separate and unnested')


def supported(out, signal):
    receipt = read(out/'support.json')
    return receipt[signal]['status'] == 'READY'


def entry(job, configure):
    """Persist failures for the job-end handoff; keep S's offline CLI contract."""
    from kit.s_common import entry as original
    def recorded(a):
        try: return job(a)
        except Exception as exc:
            import time
            atomic(a.out/'failures'/f'{time.time_ns()}.json', dict(
                row=Path(sys.argv[0]).name, error_type=type(exc).__name__, error=str(exc),
                technical_synthetic=a.standin))
            raise
    return original(recorded, configure)


def disposition(out, arm):
    doc = read(out/'binding.json')
    from kit.x_tier2 import ARMS2, identity, readiness
    if arm in ARMS2:
        return readiness(out, doc, identity(arm)['comparison'])
    if arm=='p':
        if doc.get('p_variant')!='p2-l2-init':return 'DEFERRED_P_VARIANT'
        if (out/'dispositions/p.json').exists() and read(out/'dispositions/p.json').get('status')=='DEFERRED':return 'DEFERRED'
    if arm in ('drift', 'plasticity'):
        if len(doc['domains']) == 1:
            return 'DEGENERATE'
        if not supported(out, arm):
            return 'UNINFORMATIVE'
    return 'READY'
