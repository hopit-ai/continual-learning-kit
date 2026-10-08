#!/usr/bin/env python3
"""CPU-only calibration of the complete main runner graph, never model work.

The parent measures the fresh process including interpreter/import startup.
Owned temporary records exercise the real prepare, dependency and publication
paths. Commands are replaced by a zero-duration CPU no-op for this calibration;
actual qualification dispatch receipts supply an independent runtime maximum.
"""
from __future__ import annotations
import time
import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile

if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def profile():
    from kit import runner, v4_campaign
    import yaml
    with tempfile.TemporaryDirectory(prefix='v4-runner-profile-') as folder:
        old=os.environ.get('WORK');os.environ['WORK']=folder
        try:
            campaign=runner.load_campaign(Path(__file__).with_name('campaigns')/'v4-main.yaml')
            started=time.monotonic()
            if runner.implicit_edges(campaign):raise ValueError('calibration graph has implicit dependencies')
            with redirect_stdout(io.StringIO()):
                if runner.cmd_prepare(campaign,campaign['rows'],True):raise ValueError('runner calibration preparation failed')
            prepared=time.monotonic()-started
            # Keep the complete graph and pilot dependencies; disable only model
            # dispatch and bars. These synthetic records never leave this tree.
            campaign.pop('v4')
            for row in campaign['rows']:
                row['command']=['cpu-profile-no-op'];row['bars']=[]
            original=runner.bounded_command
            def noop(command,*,log=None,**kwargs):
                if log is not None:Path(log).write_text('')
                return {'returncode':0,'failure_type':None}
            runner.bounded_command=noop
            observations=[]
            try:
                with redirect_stdout(io.StringIO()):
                    for row in campaign['rows']:
                        started=time.monotonic()
                        if runner.run_row(campaign,row,None):raise ValueError('runner calibration dispatch failed')
                        observations.append({'row':row['id'],'seconds':time.monotonic()-started})
            finally:runner.bounded_command=original
            return {'schema':'v4-runner-profile.v1','cpu_profile_only':True,
                    'campaign_sha256':__import__('hashlib').sha256(json.dumps(v4_campaign.build('main'),sort_keys=True,separators=(',',':')).encode()).hexdigest(),'rows':len(campaign['rows']),
                    'prepare_seconds':prepared,'dispatch':observations}
        finally:
            if old is None:os.environ.pop('WORK',None)
            else:os.environ['WORK']=old


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    from kit.runner import write_durably
    write_durably(args.out,profile())
    return 0

if __name__=='__main__':raise SystemExit(main())
