"""Optional, best-effort phase telemetry; never changes scientific execution."""
import json
import os
import time


def mark(name):
    path=os.environ.get('UTIL_PHASE_FILE')
    if not path:return
    try:
        # One append write also keeps markers from independent workers intact.
        with open(path,'a') as stream:stream.write(json.dumps(dict(phase=str(name),t=time.time()))+'\n')
    except OSError:pass
