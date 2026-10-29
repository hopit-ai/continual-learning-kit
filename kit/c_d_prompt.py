"""D replay prompt eligibility for rung P: mirrors kit/c_train.py's skip rule (len(student prompt ids) > 2048 is skipped). c_train.py is left byte-identical to the reviewed message-2 kit."""

D_PROMPT_CAP = 2048


def d_prompt_eligible(prompt_ids):
    return len(prompt_ids) <= D_PROMPT_CAP
