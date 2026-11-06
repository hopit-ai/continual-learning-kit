"""Explicit, process-isolated CR confirmation policy; exploratory constants stay frozen."""
import os

CONFIRMATION_SEEDS = (104, 105, 106)
CONFIRMATION_ARMS = ('G', 'CR')


def confirmation():
    mode = os.environ.get('KIT_M3_MODE', 'exploratory')
    if mode not in ('exploratory', 'cr-confirmation'):
        raise ValueError('unregistered M3 mode')
    return mode == 'cr-confirmation'


def seeds():
    from kit.m3_schedule import SEEDS
    return CONFIRMATION_SEEDS if confirmation() else SEEDS


def arms():
    from kit.m3_schedule import ARMS
    return CONFIRMATION_ARMS if confirmation() else ARMS


def qualification_seed():
    return seeds()[0]
