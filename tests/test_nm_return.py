import io
import json
import tarfile
import pytest
from kit.nm_common import *
from kit.nm_sim import fixture
from kit.nm_read import read
from kit.nm_return import reproduce


def archive(path,broken=False):
    scores,spec,binding=fixture('freeze');reading=read('freeze',scores,spec,binding)
    docs={name:canonical(v) for name,v in [('scores.json',scores),('spec.json',spec),('binding.json',binding),('reading.json',reading)]}
    manifest=dict(family='freeze',files={name:raw_hash(raw) for name,raw in docs.items()},scores='scores.json',spec='spec.json',binding='binding.json',reading='reading.json')
    if broken:docs['scores.json']=canonical({})
    docs['meta/archive-manifest.json']=canonical(manifest)
    with tarfile.open(path,'w:gz') as tar:
        for name,raw in docs.items():
            info=tarfile.TarInfo(name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
    return reading


def test_cpu_return_reproduces_full_reader(tmp_path):
    p=tmp_path/'return.tar.gz';expected=archive(p)
    assert reproduce(p)==expected


def test_return_hash_tamper_is_unverified(tmp_path):
    p=tmp_path/'return.tar.gz';archive(p,True)
    with pytest.raises(Invalid):reproduce(p)
