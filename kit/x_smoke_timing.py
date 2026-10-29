"""Parse unique real trainer timing rows; repeated Ray/console lines count once."""
import re


def parse_attempt3(text):
    groups={}
    for line in text.splitlines():
        pid=re.search(r'AddonTaskRunner pid=(\d+)',line)
        if not pid:continue
        if 'Training Progress:' not in line and not re.search(r'\bstep:\d+',line):continue
        group=groups.setdefault(pid[1],dict(updates={},progress_seconds=0.))
        progress=re.findall(r'\d+/\d+ \[(\d+):(\d+)',line)
        if progress:group['progress_seconds']=max(group['progress_seconds'],max(60*int(m)+int(s) for m,s in progress))
        step=re.search(r'\bstep:(\d+)',line)
        if step:
            metrics={k:float(v) for k,v in re.findall(r'(timing_s/(?:step|update_actor|save_checkpoint)):([0-9.eE+-]+)',line)}
            if metrics:group['updates'][step[1]]=metrics
    if len(groups)!=4:raise ValueError('attempt-3 log must contain exactly four trainer stages')
    result={}
    for name,(pid,values) in zip(('preparation','measure','control','p2'),groups.items()):
        result[name]=dict(pid=int(pid),**values)
    if any(len(result[k]['updates'])!=n for k,n in (('preparation',4),('measure',0),('control',4),('p2',3))):
        raise ValueError('attempt-3 measured/truncated stage inventory changed')
    return result
