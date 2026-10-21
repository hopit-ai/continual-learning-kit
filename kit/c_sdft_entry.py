"""C-only entry: widen input allowances, retain every other v4 D validation."""
from copy import deepcopy
from kit.c_common import require


def legacy_config(config):
    require(config['data']['max_prompt_length'] == 10240, 'C D prompt cap drift')
    require(config['actor_rollout_ref']['actor']['self_distillation']['max_reprompt_len'] == 14336, 'C D teacher context drift')
    result = deepcopy(config)
    result['data']['max_prompt_length'] = 2048
    result['actor_rollout_ref']['actor']['self_distillation']['max_reprompt_len'] = 6144
    return result


def main():
    from kit import sdft_verl as v
    from kit.c_sdft_runner import CTaskRunner
    v.check_pinned()
    v.main_ppo.TaskRunner = CTaskRunner
    v.main_ppo.main()


if __name__ == '__main__': main()
