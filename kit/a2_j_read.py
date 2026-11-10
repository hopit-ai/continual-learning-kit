#!/usr/bin/env python3
"""Standalone J(b) CPU archive/text importer."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kit.a2_return import main
if __name__ == '__main__':
    sys.argv += ['--arm', 'J']
    raise SystemExit(main())
