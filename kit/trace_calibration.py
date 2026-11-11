#!/usr/bin/env python3
"""Our-side blind frontier-rater packets and descriptive calibration reading."""
from pathlib import Path
import sys
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from kit.trace_common import *
from kit.trace_judge import calibration_summary, RUBRIC


def packets(pairs):
    from collections import Counter
    need(Counter(p['panel'] for p in pairs)==
         Counter({'Self-Instruct':30,'LIMA':30,'CoNa':30}), 'calibration panels differ')
    requests = []
    for pair in pairs:
        need(pair.get('checkpoint_independent') is True, 'checkpoint-linked calibration')
        reverse = int(digest(pair['id'])[:8],16)%2 == 1
        a,b = (pair['b'],pair['a']) if reverse else (pair['a'],pair['b'])
        requests.append(dict(id=pair['id'],reverse=reverse,messages=[dict(role='system',content=RUBRIC),
             dict(role='user',content=canonical(dict(instruction=pair['prompt'],A=a,B=b)).decode())]))
    # Reverse mapping remains local. Raters receive only ID and quoted messages.
    blind = [dict(id=r['id'],messages=r['messages']) for r in requests]
    return dict(mapping={r['id']:r['reverse'] for r in requests},
                raters={name:dict(model=name,effort='high',requests=blind)
                        for name in ('gpt-6-astra','Fable')})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pairs',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    pairs=load(args.pairs)
    value=packets(pairs)
    seal(args.out/'mapping.json',value['mapping'])
    for name,packet in value['raters'].items():
        seal(args.out/(name+'-high.json'),packet)
    print('Prepared 180 blind frontier-rater requests. GPU-h 0, paid calls 0.')


if __name__=='__main__':
    main()
