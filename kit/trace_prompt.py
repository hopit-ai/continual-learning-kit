"""TRACE checks released student inputs; historical SDFT checks stay unchanged."""
from collections.abc import Mapping
from copy import deepcopy
from types import FunctionType
from kit.sdft_objective import teacher_messages as _original_teacher_messages


def check_clean(messages, demonstration, released_prompt):
    """A released transcript can legitimately contain its agenda-title target."""
    expected = ([dict(role='user', content=released_prompt)]
                if isinstance(released_prompt, str) else released_prompt)
    if not isinstance(messages, list) or not messages:
        raise ValueError('student prompt must be a nonempty message list')
    for message in messages:
        if not isinstance(message, Mapping) or set(message) != {'role', 'content'}:
            raise ValueError('unexpected student role fields')
        if message['role'] not in ('system', 'user', 'assistant'):
            raise ValueError('unexpected student role')
        content = message['content']
        if not isinstance(content, str):
            raise ValueError('student content must be plain text')
        if '<demonstration>' in content or 'extra_info.demonstration' in content:
            raise ValueError('structured demonstration leaked into student prompt')
    if not expected or messages != expected:
        raise ValueError('student prompt differs from released prompt')
    if messages[-1]['role'] != 'user':
        raise ValueError('student prompt must end with a user message')


def teacher_messages(messages, demonstration, *, released_prompt):
    """Run the unchanged teacher template with a private TRACE checker binding."""
    original = _original_teacher_messages
    builder = FunctionType(original.__code__, dict(original.__globals__,
        check_student_prompt_clean=lambda m, d: check_clean(m, d, released_prompt)))
    return builder(messages, demonstration)


def install(rows=None):
    """Bind native workers to the prepared schedule, never to mutable raw prompts.

    The inherited native interface supplies prompt and demonstration, but no row
    ID. Both must match a released schedule pair. Missing schedule fails closed.
    CPU callers pass each row's released prompt explicitly instead.
    """
    from kit import sdft_objective as objective
    pairs = {}
    for row in rows or ():
        target = row.get('response', row.get('answer'))
        if not isinstance(target, str) or not target.strip():
            raise ValueError('released demonstration must be nonempty text')
        prompt = deepcopy(row['prompt'])
        expected = [dict(role='user', content=prompt)] if isinstance(prompt, str) else prompt
        check_clean(expected, target, prompt)
        pairs.setdefault(target, []).append(prompt)

    def check(messages, demonstration):
        for prompt in pairs.get(demonstration, ()):
            expected = [dict(role='user', content=prompt)] if isinstance(prompt, str) else prompt
            if messages == expected:
                return check_clean(messages, demonstration, prompt)
        raise ValueError('student prompt differs from released schedule prompt')

    objective.check_student_prompt_clean = check


def install_prepared(receipt):
    """Authenticate the schedule against the stage's preparation audit first."""
    from pathlib import Path
    from kit.nm_common import digest, load, need
    root = Path(receipt).parent
    prepared = load(root/'prepared.json')
    rows = load(root/'schedule.json')
    need(prepared['native']['receipt'] == str(receipt), 'TRACE preparation receipt path differs')
    need(digest(rows) == prepared['audit']['schedule_sha256'],
         'TRACE schedule differs from bound preparation audit')
    install(rows)
