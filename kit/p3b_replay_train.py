#!/usr/bin/env python3
"""One full mixed or ordinary stage, using the v4 D/F entry points directly."""
from pathlib import Path
import sys
if __package__ in (None, ''): sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.p3b_train import job, configure as ordinary_configure

def configure(p):
    ordinary_configure(p)
    p.add_argument('--slot-map', choices=('generic','broad','earlier','both'))
    p.add_argument('--pool-root', type=Path)
    p.add_argument('--start-recipe', choices=('D','F','D+replay','F+replay'))
from kit.p3b_common import entry

class ReplayLogName(str):
    def format(self, **values):
        return super().format(**values)+(('-'+values['slot_map']) if values.get('slot_map') else '')
def replay_job(a):
    if not a.slot_map:
        del a.start_recipe
    else:
        from kit.next_common import frozen_next, read_inputs, require, checkpoint, SEEDS, frozen_files
        frozen_next(); inputs=read_inputs(a.out,a.standin)
        require(a.route=='sequence' and a.recipe==inputs['w'][0]+'+replay' and a.start_recipe==inputs['w'], 'wrong W probe recipe')
        require(a.pool_root==a.out/'pools', 'use frozen next-message pools')
        require(frozen_files(a.pool_root)['technical_synthetic']==a.standin, 'pool type mismatch')
        expected=inputs['probe_starts'][SEEDS.index(a.seed)]
        observed=checkpoint(a.model,a.incoming_receipt,a.standin)
        require(observed['model_file_hashes']==expected['identity']['model_file_hashes'], 'probe must start at exact W FinQA final')
    return job(a)

if __name__ == '__main__': raise SystemExit(entry(replay_job, configure, ReplayLogName('p3b-{route}-{recipe}-{seed}')))
