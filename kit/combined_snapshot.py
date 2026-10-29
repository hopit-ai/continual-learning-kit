"""Read completed core target snapshots without creating a lease in the core tree."""
from pathlib import Path
import shutil
import tarfile
import tempfile
from kit import m2_snapshot as snap


def restore_targets(durable, task, destination):
    """Copy/verify immutable bytes locally; missing or racing objects fail, never wait.

    Completed core snapshots follow the owner's retention policy. Opening the
    source for reading never creates a lease or requests retention extension.
    Only targets-math members are materialized in the tier's scratch tree.
    """
    pointer=snap.read_pointer(durable,task,completed=True)
    source=Path(durable)/'snapshots'/snap.safe_key(task)/pointer['tar']
    destination=Path(destination)
    snap.require(not destination.exists(),'restore destination must be fresh')
    destination.parent.mkdir(parents=True,exist_ok=True)
    header=snap.header(source,pointer['manifest_sha256'])
    selected={name:row for name,row in header['files'].items() if name.startswith('targets-math/')}
    snap.require('targets-math/manifest.json' in selected,'core snapshot lacks committed targets-math')
    snap.space_guard(destination.parent,source.stat().st_size+sum(r['bytes'] for r in selected.values()))
    with tempfile.TemporaryDirectory(prefix='combined-target-copy-',dir=destination.parent) as scratch:
        local=Path(scratch)/'snapshot.tar'
        snap.copy_hashed(source,local,pointer['sha256'])
        doc=snap.verify_tar(local)
        snap.require(snap.digest(doc)==pointer['manifest_sha256'] and doc['complete'] is True,'snapshot manifest/pointer mismatch')
        staging=Path(scratch)/'tree';staging.mkdir()
        with tarfile.open(local,'r:') as archive:
            for name in selected:
                target=staging/name;target.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile('tree/'+name) as stream,target.open('wb') as output:shutil.copyfileobj(stream,output)
        staging.replace(destination)
    return doc,pointer
