#!/usr/bin/env python3
"""Dependency-free, byte-bound manager acknowledgement of a hard stop.

The owner delivers the manager-written file through the reviewed return channel.
SHA-256 binds the exact stop bytes; it does not authenticate the sender. No key,
signature or third-party package is required. Accounting is never changed here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

SCHEMA = 'kit-v4-hard-stop-ack.v2'


def _object(pairs):
    doc = {}
    for key, value in pairs:
        if key in doc:
            raise ValueError('duplicate acknowledgement/stop key: ' + key)
        doc[key] = value
    return doc


def payload(stop):
    raw = Path(stop).read_bytes()
    doc = json.loads(raw, object_pairs_hook=_object)
    if not isinstance(doc,dict) or doc.get('stop_class') != 'hard':
        raise ValueError('acknowledgements apply only to hard stops')
    from kit.p4_frozen import seal
    if doc!=seal(doc):raise ValueError('changed hard stop record')
    return {'schema': SCHEMA, 'stop_sha256': hashlib.sha256(raw).hexdigest(),
            'acknowledged': True}


def make(stop):
    doc = payload(stop)
    target = Path(str(stop)+'.ack.json')
    with target.open('x') as handle:
        json.dump(doc, handle, sort_keys=True, indent=2); handle.write('\n')
        handle.flush(); os.fsync(handle.fileno())
    from kit import p4_watchdog as wd
    wd._fsync_dir(target.parent)
    return target


def verify(stop, ack=None):
    doc = json.loads(Path(ack or str(stop)+'.ack.json').read_bytes(), object_pairs_hook=_object)
    expected = payload(stop)
    if (not isinstance(doc,dict) or set(doc) != set(expected) or
            doc.get('acknowledged') is not True or any(doc[k] != v for k,v in expected.items())):
        raise ValueError('acknowledgement does not bind this hard stop')
    return doc


def gate(work, *, release=False):
    """Keep the active stop until its exact hash verifies; retain immutable history."""
    root = Path(work)/'k8b4/containment'; stop = root/'v4-stop.json'
    if not stop.exists():
        return
    ack = Path(str(stop)+'.ack.json')
    doc = verify(stop, ack)
    if not release:
        raise ValueError('hard stop acknowledged; reconcile before the allocation self-test')
    history = root/'acknowledged-stops'/doc['stop_sha256']
    history.mkdir(parents=True, exist_ok=True)
    from kit import p4_watchdog as wd
    for source,name in ((stop,'stop.json'),(ack,'ack.json')):
        target=history/name
        if target.exists():
            if target.read_bytes()!=source.read_bytes():raise ValueError('hard-stop history changed')
        else:
            # Preserve the original durable record inode. Atomic no-overwrite
            # publication adds its history name before removing the active name;
            # the stop record itself is never deleted or replaced.
            os.link(source,target)
            wd._fsync_dir(history)
    stop.unlink(); ack.unlink()
    wd._fsync_dir(root)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('make','verify'))
    parser.add_argument('stop', type=Path)
    args=parser.parse_args(argv)
    try:
        if args.mode=='make':
            print(make(args.stop))
        else:print(json.dumps(verify(args.stop),sort_keys=True))
    except (OSError,ValueError,KeyError,TypeError,ImportError) as exc:
        print('acknowledgement refused: '+str(exc),file=sys.stderr);return 2
    return 0

if __name__=='__main__':raise SystemExit(main())
