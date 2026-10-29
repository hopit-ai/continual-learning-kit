#!/usr/bin/env python3
"""Fail before any audit if either offline runtime lacks rapidfuzz."""
import argparse
import subprocess


def check_runtime(python):
    result = subprocess.run([str(python), '-c',
        'from rapidfuzz.distance import Levenshtein; assert Levenshtein.distance("ab", "ac") == 1'],
        capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'{python}: rapidfuzz required; install with: {python} -m pip install rapidfuzz==3.14.6')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trainer-python', required=True)
    p.add_argument('--inference-python', required=True)
    a = p.parse_args()
    for runtime in (a.trainer_python, a.inference_python):
        check_runtime(runtime)
    print('Both runtimes: rapidfuzz OK')


if __name__ == '__main__':
    main()
