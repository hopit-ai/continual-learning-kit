"""One complete table for v4 arm identities and checkpoint/export routing."""
from pathlib import Path

ARM_TABLE = {
    'S': {'checkpoint_dir':'tool-sdpo', 'actor_subpath':'actor', 'launcher':'run_sdpo_toolalpaca.sh', 'target':None},
    'F': {'checkpoint_dir':'train', 'actor_subpath':'', 'launcher':'run_sft_f.sh', 'target':'extra_info.demonstration'},
    'R': {'checkpoint_dir':'train', 'actor_subpath':'', 'launcher':'run_sft_r.sh', 'target':'extra_info.rewrite'},
    'D': {'checkpoint_dir':'sdft', 'actor_subpath':'actor', 'launcher':'run_sdft.sh', 'target':'extra_info.demonstration'},
    'D-R': {'checkpoint_dir':'sdft', 'actor_subpath':'actor', 'launcher':'run_sdft.sh', 'target':'extra_info.rewrite'},
}
ALL_ARMS=tuple(ARM_TABLE)
TARGETS={arm:spec['target'] for arm,spec in ARM_TABLE.items()}
LAUNCHERS={arm:spec['launcher'] for arm,spec in ARM_TABLE.items()}


def checkpoint_path(run,arm,step,*,actor=False):
    spec=ARM_TABLE[arm]
    root=Path(run)/spec['checkpoint_dir']/f'global_step_{step}'
    return root/spec['actor_subpath'] if actor else root
