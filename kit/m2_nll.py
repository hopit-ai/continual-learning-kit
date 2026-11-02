#!/usr/bin/env python3
"""Core probe NLL entry with CPU-staged hashes; a4_nll arithmetic is unchanged."""
from pathlib import Path
import sys
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kit.m2_model_cache import install
install()
from kit.a4_nll import job,configure,entry
if __name__=='__main__':raise SystemExit(entry(job,configure,'a4-nll'))
