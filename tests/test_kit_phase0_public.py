"""Public-export-only phase-0 closure; no private repository dependencies."""
from pathlib import Path
import re
import subprocess
import sys


def test_phase0_public_paths_and_entry_help():
    """A published runbook must never point at an unshipped partner entry point."""
    kit=Path(__file__).resolve().parents[1]/'kit'
    text=(kit/'README-phase0.md').read_text()
    for name in re.findall(r'\$KIT/([^\s"`\\]+?\.(?:py|sh|md|yaml))',text):
        assert '..' not in Path(name).parts and (kit/name).is_file(),name
    for name in ('v4_phase0.py','v4_allocation.py','read_v4_phase0.py','simulate_v4_phase0.py','v4_phase0_environment.py'):
        done=subprocess.run([sys.executable,str(kit/name),'--help'],cwd=kit.parent,capture_output=True,text=True,timeout=20)
        assert done.returncode==0,done.stderr
    assert (kit/'sim/bin/slurm_sim.py').is_file()
