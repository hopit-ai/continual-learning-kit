"""Hash checks for TS-owned integration seams in the pinned source tree."""
from pathlib import Path

from kit.nm_common import KIT, load, need, sha


def check(root):
    for relative, expected in load(KIT / 'resources/ts/verl-seams.json').items():
        path = Path(root) / relative
        need(path.is_file() and sha(path) == expected, 'TS pinned verl seam differs: ' + relative)
