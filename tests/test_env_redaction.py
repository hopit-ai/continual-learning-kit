"""Exercise the public report hooks in isolated child pytest sessions."""
import hashlib
import os
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

import pytest

import env_redaction
from env_redaction import RedactedEnv
from _pytest._code.code import (ExceptionChainRepr, ReprEntry, ReprExceptionInfo,
                               ReprFileLocation, ReprFuncArgs, ReprLocals,
                               ReprTraceback)


def _child_environment(**settings):
    env = RedactedEnv(settings)
    runtime_names = ('HOME', 'PYTHONUSERBASE', 'LD_LIBRARY_PATH')
    if sys.platform == 'darwin':
        runtime_names += ('DYLD_LIBRARY_PATH',)
    env.update({name: os.environ[name] for name in runtime_names if name in os.environ})
    env.setdefault('PATH', os.defpath)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    return env


def test_native_environment_repr_preserves_str_and_mapping_behavior(monkeypatch):
    native = os.environ
    monkeypatch.setenv('KIT_NATIVE_SENTINEL', 'u7x9p2q')
    assert repr(native) == '<environment: values redacted>'
    assert os.environ is native
    assert native['KIT_NATIVE_SENTINEL'] == 'u7x9p2q'
    assert RedactedEnv(native.copy())['KIT_NATIVE_SENTINEL'] == 'u7x9p2q'
    # Check explicit str with a synthetic native mapping, keeping inherited
    # values out of assertion diagnostics even if this regression fails.
    synthetic = type(native)({'KIT_NATIVE_SENTINEL': 'u7x9p2q'}, str, str, str, str)
    assert str(synthetic) == "environ({'KIT_NATIVE_SENTINEL': 'u7x9p2q'})"
    assert repr(synthetic) == '<environment: values redacted>'


@pytest.mark.parametrize('platform', ['linux', 'darwin'])
def test_child_environment_preserves_loader_and_user_site_settings(monkeypatch, platform):
    monkeypatch.setattr(sys, 'platform', platform)
    names = ('HOME', 'PYTHONUSERBASE', 'LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH')
    for name in names:
        monkeypatch.delenv(name, raising=False)
    assert all(name not in _child_environment() for name in names)
    for name in names:
        monkeypatch.setenv(name, '/synthetic/' + name.lower())
    env = _child_environment()
    assert type(env) is RedactedEnv
    assert all(env[name] == '/synthetic/' + name.lower() for name in names[:3])
    assert ('DYLD_LIBRARY_PATH' in env) == (platform == 'darwin')
    if platform == 'darwin':
        assert env['DYLD_LIBRARY_PATH'] == '/synthetic/dyld_library_path'


def test_redacted_env_preserves_dict_operations():
    env = RedactedEnv(KIT_MAPPING_SENTINEL='mapping-value', OVERRIDE='before')
    expected = dict(KIT_MAPPING_SENTINEL='mapping-value', OVERRIDE='before')
    assert isinstance(env, dict)
    assert repr(env) == str(env) == '<environment: 2 variables, values redacted>'
    assert dict(env) == {**env} == json.loads(json.dumps(env)) == expected
    copied = env.copy()
    copied.update(ADDED='copy-only')
    assert type(copied) is RedactedEnv
    assert copied == {**expected, 'ADDED': 'copy-only'}
    assert env == expected
    for merged in (env | {'OVERRIDE': 'after'}, {'OVERRIDE': 'after'} | env):
        assert type(merged) is RedactedEnv
        assert merged['KIT_MAPPING_SENTINEL'] == 'mapping-value'
    assert (env | {'OVERRIDE': 'after'})['OVERRIDE'] == 'after'
    assert ({'OVERRIDE': 'after'} | env)['OVERRIDE'] == 'before'
    original = env
    env |= {'OVERRIDE': 'after'}
    env |= [('ADDED', 'in-place')]
    assert env is original and type(env) is RedactedEnv
    assert env == {**expected, 'OVERRIDE': 'after', 'ADDED': 'in-place'}
    assert env.__or__([]) is NotImplemented
    assert env.__ror__([]) is NotImplemented
    done = subprocess.run(
        [sys.executable, '-c',
         'import os; assert os.environ["KIT_MAPPING_SENTINEL"] == "mapping-value"'],
        env=_child_environment(**env), capture_output=True, text=True, timeout=180)
    assert done.returncode == 0


@pytest.mark.parametrize('fixture_source', ['synthetic', 'util_watch'])
def test_quiet_fixture_arguments_hide_short_values_and_path_fragments(tmp_path, fixture_source):
    source = Path(__file__).resolve().parent
    public_conftest = source/'public_conftest.py'
    if not public_conftest.is_file():
        public_conftest = source/'conftest.py'
    shutil.copyfile(source/'env_redaction.py', tmp_path/'env_redaction.py')
    shutil.copyfile(public_conftest, tmp_path/'conftest.py')
    ini = tmp_path/'pytest.ini'
    ini.write_text('[pytest]\n')
    home = '/synthetic-home/' + 'long-private-directory-7e928f63/' * 12 + 'home-tail-834b70a1'
    short = 'u7x9p2q'
    fragments = {home[index:index + 12] for index in range(len(home) - 11)}
    marker = 'INTENTIONAL-QUIET-FIXTURE-FAILURE'
    if fixture_source == 'synthetic':
        fixture = ('@pytest.fixture\n'
                   'def raw_env():\n'
                   '    return RedactedEnv(os.environ)\n\n')
    else:
        fixture = (f'sys.path.insert(0, {str(source)!r})\n'
                   'from test_util_watch import env as raw_env\n\n')
    child_env = _child_environment(A_USER=short, B_HOME_PATH=home)
    command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
               '-c', str(ini), '--rootdir', str(tmp_path),
               '--basetemp', str(tmp_path/'child-temp'), 'test_failure.py']
    # The plain-dict control keeps the report hooks enabled, exposing the gap
    # between value matching and pytest's earlier argument abbreviation.
    for protected in (False, True):
        (tmp_path/'test_failure.py').write_text(
            'import hashlib\nimport os\nimport sys\nimport pytest\n'
            'from env_redaction import RedactedEnv\n\n' + fixture +
            '@pytest.fixture\n'
            'def env(raw_env):\n'
            f'    return {"raw_env" if protected else "dict(raw_env)"}\n\n'
            'def test_failure(env):\n'
            f'    assert hashlib.sha256(env["B_HOME_PATH"].encode()).hexdigest() == {hashlib.sha256(home.encode()).hexdigest()!r}\n'
            f'    assert hashlib.sha256(env["A_USER"].encode()).hexdigest() == {hashlib.sha256(short.encode()).hexdigest()!r}\n'
            f'    assert False, "{marker}"\n')
        done = subprocess.run(command, cwd=tmp_path, env=child_env, capture_output=True,
                              text=True, timeout=180)
        output = done.stdout + done.stderr
        assert done.returncode == 1
        assert f'AssertionError: {marker}' in output
        assert '1 failed' in output
        if protected:
            assert home not in output
            assert all(fragment not in output for fragment in fragments)
            assert short not in output
            assert 'env = <environment: ' in output
            assert 'variables, values redacted>' in output
        else:
            assert any(fragment in output for fragment in fragments)
            assert short in output


@pytest.mark.parametrize('operation', ['environ', 'subprocess-launch', 'subprocess-exit', 'mock'])
def test_quiet_environment_objects_hide_short_values_and_path_fragments(tmp_path, operation):
    source = Path(__file__).resolve().parent
    public_conftest = source/'public_conftest.py'
    if not public_conftest.is_file():
        public_conftest = source/'conftest.py'
    shutil.copyfile(source/'env_redaction.py', tmp_path/'env_redaction.py')
    ini = tmp_path/'pytest.ini'
    ini.write_text('[pytest]\n')
    home = '/synthetic-home/' + 'long-private-directory-7e928f63/' * 12 + 'home-tail-834b70a1'
    short = 'u7x9p2q'
    fragments = {home[index:index + 12] for index in range(len(home) - 11)}
    env = _child_environment(A_USER=short, B_HOME_PATH=home)
    bodies = {
        'environ': '    assert "KIT_MISSING_ASSERTION_SENTINEL" in os.environ\n',
        'subprocess-launch': '    subprocess.run([str(Path(__file__).parent/"missing-executable")], env=env)\n',
        'subprocess-exit': '    subprocess.run([sys.executable, "-c", "raise SystemExit(37)"], env=env, check=True)\n',
        'mock': '    recorded = Mock()\n'
                '    recorded(env=env)\n'
                '    recorded.assert_called_once_with(env={})\n',
    }
    exception = {'subprocess-launch': 'FileNotFoundError',
                 'subprocess-exit': 'CalledProcessError'}.get(operation, 'AssertionError')
    command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
               '-c', str(ini), '--rootdir', str(tmp_path),
               '--basetemp', str(tmp_path/'child-temp'), 'test_failure.py']
    for protected in (False, True):
        if protected:
            shutil.copyfile(public_conftest, tmp_path/'conftest.py')
        imports = ('import os\nimport subprocess\nimport sys\n'
                   'from pathlib import Path\nfrom unittest.mock import Mock\n')
        if protected:
            imports += 'from env_redaction import RedactedEnv\n'
        mapping = 'RedactedEnv(os.environ)' if protected else 'dict(os.environ)'
        (tmp_path/'test_failure.py').write_text(
            imports + '\ndef test_failure():\n' +
            (f'    env = {mapping}\n' if operation != 'environ' else '') + bodies[operation])
        done = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True,
                              text=True, timeout=180)
        output = done.stdout + done.stderr
        assert done.returncode == 1
        assert exception in output and '1 failed' in output
        if protected:
            assert home not in output
            assert all(fragment not in output for fragment in fragments)
            assert short not in output
            assert 'values redacted>' in output
        else:
            assert any(fragment in output for fragment in fragments)
            assert short in output


@pytest.mark.parametrize('case', list('ABCDEFGHI'))
@pytest.mark.parametrize('traceback', ['long', 'short', 'line'])
def test_environment_values_are_redacted(tmp_path, case, traceback):
    sentinel = 'kit-private-sentinel-7e928f63'
    secret = 'sentinel2-scrubbed-834b70a1'
    escaped = 'private-backslash-\\-and-\"-quote-23c1b9'
    marker = f'INTENTIONAL-{case}'
    source = Path(__file__).resolve().parent
    public_conftest = source/'public_conftest.py'
    if not public_conftest.is_file():
        public_conftest = source/'conftest.py'  # Already in an exported kit.
    shutil.copyfile(source/'env_redaction.py', tmp_path/'env_redaction.py')
    ini = tmp_path/'pytest.ini'
    ini.write_text('[pytest]\n')
    # Keep the complete sentinel out of source lines displayed in tracebacks.
    reached = 'assert os.environ["KIT_REDACTION_SENTINEL"] == "kit-private-" + "sentinel-7e928f63"\n'
    bodies = {
        'A': 'def test_failure():\n'
             f'    {reached}'
             f'    assert dict(os.environ) == {{"KIT_REDACTION_SENTINEL": "different"}}, "{marker}"\n',
        'B': '@pytest.fixture\n'
             'def env():\n'
             '    return dict(os.environ)\n\n'
             'def test_failure(env):\n'
             f'    {reached}'
             '    print(os.environ["KIT_REDACTION_SENTINEL"])\n'
             f'    assert False, "{marker}"\n',
        'C': 'def test_failure():\n'
             f'    {reached}'
             f'    recorded = Mock(name="{marker}")\n'
             '    recorded(env=dict(os.environ))\n'
             '    recorded.assert_called_once_with(env={})\n',
        'D': 'def test_failure():\n'
             f'    {reached}'
             f'    missing = "{marker}-" + os.environ["KIT_REDACTION_SENTINEL"]\n'
             '    subprocess.run([str(Path(__file__).parent/missing)], env=dict(os.environ))\n',
        'E': reached + f'raise RuntimeError("{marker}: " + str(dict(os.environ)))\n',
        'F': 'import conftest\n'
             'if hasattr(conftest, "EARLIER_SECRET"):\n'
             '    assert "KIT_TEST_API_KEY" not in os.environ\n'
             'def test_failure():\n'
             '    if hasattr(conftest, "EARLIER_SECRET"):\n'
             '        assert "KIT_TEST_API_KEY" not in os.environ\n'
             '        child = subprocess.check_output([sys.executable, "-c",\n'
             '            \'import os; print("KIT_TEST_API_KEY" in os.environ)\'])\n'
             '        assert child.strip() == b"False"\n'
             '        assert conftest.env_redaction._SCRUBBED_NAME_COUNT >= 1\n'
             f'        raise RuntimeError("{marker}: " + conftest.EARLIER_SECRET)\n'
             f'    raise RuntimeError("{marker}: " + os.environ["KIT_TEST_API_KEY"])\n',
        'G': 'def test_failure():\n'
             f'    {reached}'
             '    assert "\\\\" in os.environ["KIT_REDACTION_ESCAPED"]\n'
             '    assert chr(34) in os.environ["KIT_REDACTION_ESCAPED"]\n'
             f'    assert dict(os.environ) == {{}}, "{marker}"\n',
        'H': 'def test_warning():\n'
             f'    {reached}'
             f'    warnings.warn_explicit("{marker}: " + os.environ["KIT_REDACTION_SENTINEL"],\n'
             '        UserWarning, os.environ["KIT_REDACTION_SENTINEL"] + ".py", 1)\n',
        'I': 'def test_xfail():\n'
             f'    {reached}'
             f'    pytest.xfail("{marker}: " + os.environ["KIT_REDACTION_SENTINEL"])\n',
    }
    (tmp_path/'test_failure.py').write_text(
        'import os\nimport subprocess\nimport sys\nimport warnings\nfrom pathlib import Path\n'
        'from unittest.mock import Mock\nimport pytest\n\n' + bodies[case])
    # Insert the sentinel first so even abbreviated fixture/kwargs reprs expose it
    # in the negative control. No inherited pytest settings or third-party plugins.
    env = RedactedEnv({'KIT_REDACTION_SENTINEL': sentinel, **os.environ})
    env['KIT_REDACTION_SENTINEL'] = sentinel
    env['KIT_TEST_API_KEY'] = secret
    env['KIT_REDACTION_ESCAPED'] = escaped
    for name in ('PYTEST_ADDOPTS', 'PYTEST_PLUGINS'):
        env.pop(name, None)
    env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    command = [sys.executable, '-m', 'pytest', '-vv', '-p', 'no:cacheprovider',
               '-c', str(ini), '--rootdir', str(tmp_path), '--tb=' + traceback,
               '-rA', '--capture=fd', 'test_failure.py']
    for protected in (False, True):
        if protected:
            shutil.copyfile(public_conftest, tmp_path/'conftest.py')
            if case == 'F':
                # Save a value before the plugin import to verify deleted values
                # remain redacted even when another component cached them first.
                original = (tmp_path/'conftest.py').read_text()
                (tmp_path/'conftest.py').write_text(
                    'import os\nEARLIER_SECRET = os.environ["KIT_TEST_API_KEY"]\n'
                    'import env_redaction\n' + original)
        elif case == 'F':
            (tmp_path/'conftest.py').write_text('')
        done = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True,
                              text=True, timeout=180)
        output = done.stdout + done.stderr
        # Do not put the unprotected output in the parent's assertion message.
        expected = 0 if case in ('H', 'I') else 2 if case == 'E' else 1
        assert done.returncode == expected
        assert marker in output
        if case not in ('H', 'I'):
            summary = output.split('short test summary info')[-1]
            exception = {'C': 'AssertionError', 'D': 'FileNotFoundError',
                         'E': 'RuntimeError', 'F': 'RuntimeError'}.get(case, 'AssertionError')
            assert exception in summary
            assert marker in summary or case == 'C'
            if protected and case == 'F':
                assert 'RuntimeError: INTENTIONAL-F: [ENV REDACTED]' in summary
            if traceback == 'line':
                assert any(exception in line for line in output.splitlines()
                           if line.startswith('E   '))
        if protected:
            assert sentinel not in output
            assert secret not in output
            assert all(form not in output for form in
                       (escaped, repr(escaped)[1:-1], json.dumps(escaped)[1:-1]))
            assert '[ENV REDACTED]' in output
        else:
            if case == 'F':
                assert secret in output
            else:
                assert sentinel in output
            if case == 'G':
                assert repr(escaped)[1:-1] in output


def test_remembered_values_sections_and_metadata(monkeypatch):
    earlier = 'earlier-private-value'
    current = 'current-private-value'
    component = '/private/path-component'
    monkeypatch.setattr(env_redaction, '_VALUES', {earlier})
    env_redaction._remember_value(current)
    env_redaction._remember_value(component + os.pathsep + '/another/private-component')
    report = SimpleNamespace(longrepr=f'{earlier} {current} {component}',
                             sections=[(current, f'{earlier} {component}')],
                             outcome='failed', duration=0.125, nodeid='test_module.py::test_case')
    env_redaction._redact_report(report)
    assert report.longrepr == '[ENV REDACTED] [ENV REDACTED] [ENV REDACTED]'
    assert report.sections == [('[ENV REDACTED]', '[ENV REDACTED] [ENV REDACTED]')]
    assert (report.outcome, report.duration, report.nodeid) == ('failed', 0.125, 'test_module.py::test_case')
    report.longrepr = current
    env_redaction._redact_report(report)
    assert report.longrepr == '[ENV REDACTED]'
    report.outcome = 'skipped'
    report.longrepr = (component, 12, current)
    env_redaction._redact_report(report)
    assert report.longrepr == ('[ENV REDACTED]', 12, '[ENV REDACTED]')
    assert report.outcome == 'skipped'


@pytest.mark.parametrize('chain', [False, True])
@pytest.mark.parametrize('sentinel', ['structured-private-sentinel', 'ENV REDACTED'])
def test_structured_traceback_fields(monkeypatch, chain, sentinel):
    monkeypatch.setattr(env_redaction, '_VALUES', {sentinel})
    crash = ReprFileLocation(sentinel, 12, 'RuntimeError: ' + sentinel)
    args = ReprFuncArgs([(sentinel, sentinel)])
    locals_repr = ReprLocals([sentinel])
    entry = ReprEntry([sentinel], args, locals_repr, crash, 'long')
    traceback = ReprTraceback([entry], sentinel, 'long')
    longrepr = (ExceptionChainRepr([(traceback, crash, sentinel)]) if chain
                else ReprExceptionInfo(traceback, crash))
    longrepr.addsection(sentinel, sentinel)
    report = SimpleNamespace(longrepr=longrepr, sections=[], wasxfail=sentinel,
                             nodeid=sentinel, location=(sentinel, 1, sentinel))
    env_redaction._redact_report(report)
    assert report.longrepr is longrepr
    assert report.longrepr.reprcrash is crash
    assert crash.message == 'RuntimeError: [ENV REDACTED]'
    assert crash.path == '[ENV REDACTED]'
    assert entry.lines == locals_repr.lines == ['[ENV REDACTED]']
    assert args.args == [('[ENV REDACTED]', '[ENV REDACTED]')]
    assert traceback.extraline == '[ENV REDACTED]'
    assert report.wasxfail == '[ENV REDACTED]'
    assert report.nodeid == sentinel
    assert report.location == (sentinel, 1, sentinel)
    assert sentinel not in str(longrepr).replace('[ENV REDACTED]', '')


def test_escaped_values_and_components(monkeypatch):
    monkeypatch.setattr(env_redaction, '_VALUES', set())
    component = 'private-\\-\"-component\nwith-unicode-\u2603'
    env_redaction._remember_value(component + os.pathsep + 'another-component')
    for value in (component, component + os.pathsep + 'another-component'):
        for form in (value, repr(value)[1:-1], json.dumps(value)[1:-1]):
            assert env_redaction._redact(form) == '[ENV REDACTED]'
    env_redaction._remember_value('short')
    assert 'short' not in env_redaction._VALUES


def test_import_scrubs_secret_names_and_preserves_nonsecret_settings(tmp_path):
    shutil.copyfile(Path(env_redaction.__file__), tmp_path/'env_redaction.py')
    secret_names = [
        'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACE_TOKEN',
        'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN', 'GITHUB_TOKEN', 'GH_TOKEN',
        'WANDB_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY',
        'MODAL_TOKEN_ID', 'MODAL_TOKEN_SECRET',
        'kit_test_apikey', 'kit_accesskey_end', 'secret', 'X_PASSWORD',
        'X_PASSWD_END', 'X_CREDENTIAL', 'X_CREDENTIALS_END', 'COOKIE',
        'X_PRIVATEKEY_END',
    ]
    preserved_names = [
        'KIT_V6_AUTHORITY_SHA', 'TRACE_TOKEN_CACHE', 'A4_TOKEN_LOG',
        'HF_HUB_DISABLE_IMPLICIT_TOKEN', 'TOKENIZERS_PARALLELISM', 'X_TOKENIZED',
    ]
    env = RedactedEnv(os.environ)
    env.update({name: 'synthetic-secret-' + str(index)
                for index, name in enumerate(secret_names)})
    env.update({name: 'synthetic-setting-' + str(index)
                for index, name in enumerate(preserved_names)})
    env['PYTEST_CURRENT_TEST'] = 'bookkeeping-excluded-from-values'
    script = (
        'import os, sys\n'
        'sys.path.insert(0, sys.argv[1])\n'
        'earlier = dict(os.environ)\n'
        'import env_redaction as redaction\n'
        f'secrets = {secret_names!r}\n'
        f'preserved = {preserved_names!r}\n'
        'assert all(name not in os.environ for name in secrets)\n'
        'assert all(os.environ[name] == earlier[name] for name in preserved)\n'
        'assert redaction._SCRUBBED_NAME_COUNT == len(secrets)\n'
        'assert all(redaction._redact(earlier[name]) == "[ENV REDACTED]" for name in secrets)\n'
        'assert earlier["PYTEST_CURRENT_TEST"] not in redaction._VALUES\n'
    )
    done = subprocess.run([sys.executable, '-c', script, str(tmp_path)],
                          env=env, capture_output=True, text=True, timeout=180)
    assert done.returncode == 0


def test_reports_do_not_capture_per_test_values_or_recompile(monkeypatch):
    sentinel = 'performance-private-sentinel'
    monkeypatch.setattr(env_redaction, '_VALUES', {sentinel})
    env_redaction._redact(sentinel)
    compiled = env_redaction._PATTERN
    started = perf_counter()
    for index in range(3000):
        monkeypatch.setenv('PYTEST_CURRENT_TEST', f'test_synthetic_{index} (call)')
        monkeypatch.setenv('KIT_PER_TEST_VALUE', f'changing-per-test-{index}')
        report = SimpleNamespace(
            longrepr=ReprExceptionInfo(
                ReprTraceback([], None, 'line'),
                ReprFileLocation('test_synthetic.py', 1, 'RuntimeError: ' + sentinel)),
            sections=[('stdout', sentinel)], wasxfail=sentinel)
        env_redaction._redact_report(report)
        assert report.longrepr.reprcrash.message == 'RuntimeError: [ENV REDACTED]'
        assert report.sections == [('stdout', '[ENV REDACTED]')]
        assert report.wasxfail == '[ENV REDACTED]'
    elapsed = perf_counter() - started
    assert elapsed < 1.0
    assert env_redaction._PATTERN is compiled
    assert env_redaction._VALUES == {sentinel}
    env_redaction._remember_value('newly-remembered-value')
    assert env_redaction._redact('newly-remembered-value') == '[ENV REDACTED]'
    assert env_redaction._PATTERN is not compiled


def test_warning_record_is_mutated_in_place(monkeypatch):
    import warnings

    sentinel = 'warning-private-sentinel'
    monkeypatch.setattr(env_redaction, '_VALUES', {sentinel})
    warning = warnings.WarningMessage(UserWarning(sentinel), UserWarning,
                                      sentinel + '.py', 5, line=sentinel)
    env_redaction.pytest_warning_recorded(warning)
    assert warning.message == '[ENV REDACTED]'
    assert warning.filename == '[ENV REDACTED].py'
    assert warning.line == '[ENV REDACTED]'
    assert warning.category is UserWarning
