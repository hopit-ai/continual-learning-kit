"""Bind the original three-seed return without relabelling old protocol evidence."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import tarfile
from kit.p3b_common import require, s

DECISIONS=Path(__file__).resolve().parent/'resources/m2/return1-decisions.json'
REVIEWED_DECISIONS_SHA256='472774171ca234fe6f69eee9ca8a1ce6d8f99eb2adf416c4c8b256bf15f4a62e'
PRIVATE_ORIGINAL_DECISIONS_SHA256='df285e5c7f831b79a778e28ba082b3fb14089aae63519a73c02df99147133860'


def decisions(path=DECISIONS):
    path=Path(path)
    require(s.file_hash(path)==REVIEWED_DECISIONS_SHA256,'return-1 decisions file differs from reviewed ruling')
    doc=json.loads(path.read_text())
    require(doc['seeds']==[101,102,103] and doc['B_selection']['W']=='D' and doc['teacher']['status']=='QUALIFIES',
            'reviewed return-1 seeds/W/qualification changed')
    teacher=doc['teacher']
    require(teacher['n']==200 and teacher['teacher_solved']>=160 and teacher['teachable_gap']>=30 and teacher['complete'] is True,'hard-MATH qualification counts differ')
    require(doc['public_redaction']['private_original_sha256']==PRIVATE_ORIGINAL_DECISIONS_SHA256,'private original decisions identity differs')
    return doc


def archives(root, ruling):
    require(root is not None,'original return-1 archive directory required')
    result={}
    for name in ('package-c-return.tar.gz','next-return-1.tar.gz','math-hard-teacher-return.tar.gz'):
        path=Path(root)/name;expected=ruling['provenance']['input_fingerprints'][name]
        require(path.is_file() and path.stat().st_size==expected['bytes'] and s.file_hash(path)==expected['sha256'],
                'original archive fingerprint differs: '+name+'; do not bind the dot-only reading copy')
        result[name]=expected['sha256']
    return result


def files(source):
    """Read regular payloads only; a dot directory does not change their identities."""
    source=Path(source)
    if source.is_dir():
        return {str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file() and
            p.suffix in ('.json','.gz','.csv') and not any(x.startswith('hf-') for x in p.relative_to(source).parts)}
    values={}
    with tarfile.open(source,'r:*') as archive:
        for member in archive:
            name=PurePosixPath(member.name)
            require(not name.is_absolute() and '..' not in name.parts,'unsafe return member')
            if member.isdir():continue
            require(member.isfile(),'return contains a link or special member')
            key=str(name).removeprefix('return/')
            require(key not in values,'duplicate return member')
            values[key]=archive.extractfile(member).read()
    if 'collect-manifest.json' in values:
        for record in json.loads(values['collect-manifest.json'])['files']:
            name=str(PurePosixPath(record['path']))
            raw=values.get(name)
            require(raw is not None and len(raw)==record['bytes'] and hashlib.sha256(raw).hexdigest()==record['sha256'],
                    'returned collector manifest differs: '+name)
    return values


def binding(source,ruling):
    values=files(source);raw=values['binding.json']
    require(hashlib.sha256(raw).hexdigest()==ruling['continuation']['binding_sha256'],'C binding differs from original return')
    doc=json.loads(raw)
    require(doc['W']=='D' and not doc['technical_synthetic'],'original C W/type differs')
    for export in ruling['continuation']['exports']:
        require(export['receipt'] in values and hashlib.sha256(values[export['receipt']]).hexdigest()==export['receipt_sha256'],
                'C continuation export receipt differs: '+export['receipt'])
        receipt=json.loads(values[export['receipt']])
        require(receipt['model_file_hashes']==export['model_hashes'],'C continuation hashes differ')
    return doc,values


def rescore_refs(mapping,ruling):
    """Paths come only from the prospective binding file; content identities come from the ruling."""
    refs={}
    for package in ruling['rescoring']['packages']:
        for call in package['calls']:
            key=call['checkpoint']
            if key in refs:continue
            require(key in mapping and mapping[key].get('model') and (mapping[key].get('receipt') or __import__('kit.m2_bind',fromlist=['tier_a_base']).tier_a_base(call)),
                    'missing original export path/receipt for '+key)
            refs[key]=dict(mapping[key],model_hashes=call['model_hashes'])
    return refs
