#!/usr/bin/env python3
"""CPU-only adapter from surviving immutable owner artifacts to SP metadata.

No model inference, checkpoint copying, extraction, snapshot publication, leases or retention
changes. Source locations stay in local map files; derived receipts contain hashes only.
"""
from pathlib import Path
import argparse
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from kit.sp_manifest import adapt_owner_exports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--entry-id')
    args = parser.parse_args(argv)
    result = adapt_owner_exports(args.sources, args.out, args.entry_id)
    print('SP durable adapter:', result['status'])
    # Missing owners leave missing SP entries, and never fail or mutate a parent.
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
