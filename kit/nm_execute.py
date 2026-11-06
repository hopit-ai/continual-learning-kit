#!/usr/bin/env python3
"""Allocation child dispatch. All data-heavy preparation runs in CPU rows."""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import os
from kit.nm_common import *
from kit.p3b_common import NotReady


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--task',required=True);p.add_argument('--attempt',type=Path,required=True);a=p.parse_args()
    from kit.nm_jobs import row_for
    cfg=load(a.config);row=row_for(a.task)
    if row['family']=='cr':
        from kit.cr_jobs import settings
        cfg=settings(cfg)
    out=Path(cfg['out'])
    need(os.environ.get('UTIL_WATCH_ACTIVE')=='1','allocation child requires watcher')
    try:
        if row['family']=='cr':
            from kit.cr_jobs import operation
            result=operation(cfg,row,a.attempt,out)
        elif cfg.get('technical_synthetic'):
            from kit.nm_sim import operation
            result=operation(cfg,row,a.attempt,out)
        else:
            from kit.nm_pipeline import operation
            result=operation(cfg,row,a.attempt,out,a.config)
        atomic(a.attempt/'output.json',result);return 0
    except (Pending,FileNotFoundError) as error:
        atomic(a.attempt/'output.json',dict(status='PENDING',reason=str(error)));print('PENDING:',error);return PENDING
    except NotReady as error:
        if row['family']=='cr':
            atomic(a.attempt/'output.json',dict(status='PENDING',reason=str(error)));print('PENDING:',error);return PENDING
        atomic(a.attempt/'output.json',dict(status='FAILED',reason=type(error).__name__+': '+str(error)));print('STOP:',error);return 2
    except Exception as error:
        atomic(a.attempt/'output.json',dict(status='FAILED',reason=type(error).__name__+': '+str(error)));print('STOP:',error);return 2
if __name__=='__main__':raise SystemExit(main())
