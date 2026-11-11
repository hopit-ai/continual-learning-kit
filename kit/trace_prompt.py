"""Task-aware demonstration separation for TRACE's short labels."""
from collections.abc import Mapping


def check_clean(messages, demonstration):
    """Student fields are source prompts only. Short natural labels can recur."""
    needle = demonstration.strip()
    for message in messages:
        if not isinstance(message,Mapping) or set(message) != {'role','content'}:
            raise ValueError('unexpected student role fields')
        content = message['content']
        if not isinstance(content,str):
            raise ValueError('student content must be plain text')
        if len(needle)>40 and needle in content:
            raise ValueError('worked demonstration leaked into student prompt')
        if '<demonstration>' in content or 'extra_info.demonstration' in content:
            raise ValueError('structured demonstration leaked into student prompt')


def install():
    # TRACE-specific entry processes install this adapter. Historical entry
    # points retain their original substring-based leakage check.
    from kit import sdft_objective
    sdft_objective.check_student_prompt_clean = check_clean
