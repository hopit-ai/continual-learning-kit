"""Scrub inherited secrets before test imports and redact pytest diagnostics.

Secret-looking names are removed at import, so children inherit a scrubbed
environment. The deletion count stays only in memory. Remaining inherited values
and deleted values are remembered, including path-list components and their
Python repr / JSON escaped forms. Literal values and components shorter than
eight characters are excluded to avoid replacing common diagnostic words.

Capture happens only at import, excluding PYTEST_* bookkeeping. Per-test changes
never grow the set. Later environment additions, truncations, other encodings
and transformations are outside this coverage. Structured tracebacks, captured
sections, xfail reasons and recorded warning messages/filenames are redacted.
Environment-bearing fixtures and subprocess mappings use RedactedEnv to hide values before pytest
abbreviates argument representations, including short values and path fragments.
The native environment's repr also hides values while its str stays unchanged.
Nodeid and location are preserved. A grep of every publisher-exported test's
environment reads and parametrization found no environment-derived test ids.

Direct terminal writes, disabled capture/live logging, startup/configuration
errors, internal errors and KeyboardInterrupt bypass these report hooks. Plugins
that subsequently modify or print diagnostics can also bypass redaction.
"""
import json
import os
import re


# object.__str__ delegates to repr. Bind the original native repr as str first
# so changing assertion diagnostics preserves explicit str(os.environ) behavior.
if type(os.environ).__str__ is object.__str__:
    type(os.environ).__str__ = type(os.environ).__repr__
type(os.environ).__repr__ = lambda self: '<environment: values redacted>'


class RedactedEnv(dict):
    """Preserve dict behavior while hiding environment values in diagnostics."""

    def __repr__(self):
        return f'<environment: {len(self)} variables, values redacted>'

    __str__ = __repr__

    def copy(self):
        return type(self)(self)

    def __or__(self, other):
        result = super().__or__(other)
        if result is NotImplemented:
            return NotImplemented
        return type(self)(result)

    def __ror__(self, other):
        result = super().__ror__(other)
        if result is NotImplemented:
            return NotImplemented
        return type(self)(result)


_VALUES = set()
_PATTERN_VALUES = frozenset()
_PATTERN = None
_SECRET_NAME = re.compile(
    r'(^|_)(API_?KEY|ACCESS_?KEY|SECRET|TOKEN|PASSWORD|PASSWD|'
    r'CREDENTIALS?|COOKIE|PRIVATE_?KEY)(_|$)', re.IGNORECASE)
_SECRET_NAMES = frozenset({
    'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACE_TOKEN',
    'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN', 'GITHUB_TOKEN', 'GH_TOKEN',
    'WANDB_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY',
    'MODAL_TOKEN_ID', 'MODAL_TOKEN_SECRET',
})
# Audited exported tests and all kit Python/shell sources, including dynamic
# environment-name constants. These settings contain no authentication secrets.
_EXEMPT_NAMES = frozenset({
    'KIT_V6_AUTHORITY_SHA',  # Public infrastructure authority digest.
    'TRACE_TOKEN_CACHE',  # Directory of cached model token IDs.
    'A4_TOKEN_LOG',  # Directory of training token-count journals.
    'HF_HUB_DISABLE_IMPLICIT_TOKEN',  # Boolean disabling implicit HF auth.
})


def _remember_value(value):
    for part in (value, *value.split(os.pathsep)):
        if len(part) >= 8:
            _VALUES.update((part, repr(part)[1:-1], json.dumps(part)[1:-1]))


def _scrub_environment():
    deleted_count = 0
    for name, value in tuple(os.environ.items()):
        upper = name.upper()
        if upper not in _EXEMPT_NAMES and (
                _SECRET_NAME.search(name) or upper in _SECRET_NAMES):
            _remember_value(value)
            del os.environ[name]
            deleted_count += 1
        elif not upper.startswith('PYTEST_'):
            _remember_value(value)
    return deleted_count


_SCRUBBED_NAME_COUNT = _scrub_environment()

# Import pytest only after scrubbing, before any test module is imported.
import pytest
from _pytest._code.code import TerminalRepr


def _redact(text):
    global _PATTERN_VALUES, _PATTERN
    values = frozenset(_VALUES)
    if values != _PATTERN_VALUES:
        _PATTERN_VALUES = values
        ordered = sorted(values, key=lambda value: (-len(value), value))
        _PATTERN = re.compile('|'.join(map(re.escape, ordered))) if values else None
    # One pass keeps replacement text from being redacted again.
    return _PATTERN.sub('[ENV REDACTED]', text) if _PATTERN else text


def _redact_repr(value, seen=None):
    """Keep pytest's repr objects and mutable containers in place."""
    if isinstance(value, str):
        return _redact(value)
    if seen is None:
        seen = set()
    if isinstance(value, (list, TerminalRepr)):
        if id(value) in seen:
            return value
        seen.add(id(value))
    if isinstance(value, list):
        value[:] = [_redact_repr(part, seen) for part in value]
    elif isinstance(value, tuple):
        return tuple(_redact_repr(part, seen) for part in value)
    elif isinstance(value, TerminalRepr):
        for name, part in vars(value).items():
            setattr(value, name, _redact_repr(part, seen))
    return value


def _redact_report(report):
    if isinstance(report.longrepr, (TerminalRepr, str, tuple)):
        report.longrepr = _redact_repr(report.longrepr)
    elif report.longrepr is not None:
        report.longrepr = _redact(str(report.longrepr))
    report.sections = _redact_repr(report.sections)
    if hasattr(report, 'wasxfail'):
        report.wasxfail = _redact(report.wasxfail)


@pytest.hookimpl(tryfirst=True)
def pytest_warning_recorded(warning_message):
    # WarningMessage is mutable. Preserve its category and source metadata while
    # replacing the message text, including custom Warning.__str__ output.
    warning_message.message = _redact(str(warning_message.message))
    warning_message.filename = _redact(warning_message.filename)
    if warning_message.line is not None:
        warning_message.line = _redact(warning_message.line)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    result = yield
    _redact_report(result.get_result())


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_make_collect_report(collector):
    result = yield
    _redact_report(result.get_result())
