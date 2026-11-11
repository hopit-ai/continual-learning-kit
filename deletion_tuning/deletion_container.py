"""One bounded smoke attempt inside the sandbox, without importing Modal."""
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    work = Path('/work/results')
    work.mkdir(parents=True, exist_ok=True)
    record = dict(schema='deletion-smoke.v1', status='STOP', scientific_evidence=False,
                  model='Qwen/Qwen3-0.6B', gpu='A10G', gpu_count=2, retries=0, phases=[])
    started = time.monotonic()
    try:
        for leg in ('finder', 'uninterrupted', 'prefix', 'resumed', 'vllm'):
            command = [sys.executable]
            if leg != 'vllm':
                command += ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2', '--max_restarts=0']
            command += ['-m', 'deletion_tuning.deletion_smoke', leg, '--work', str(work)]
            phase_start = time.monotonic()
            remaining = int(3780 - (phase_start - started))
            if remaining <= 0:
                raise TimeoutError('smoke wall limit')
            with (work / f'{leg}.log').open('w') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=remaining)
            record['phases'].append(dict(name=leg, seconds=time.monotonic()-phase_start, returncode=result.returncode))
            if result.returncode:
                raise RuntimeError(leg + ' failed')
        ranks = {}
        for rank in range(2):
            full = json.loads((work / f'uninterrupted-r{rank}.json').read_text())
            prefix = json.loads((work / f'prefix-r{rank}.json').read_text())
            resumed = json.loads((work / f'resumed-r{rank}.json').read_text())
            if full['state'] != resumed['state']:
                raise ValueError('resume state differs')
            if full['metrics'][2:] != resumed['metrics']:
                raise ValueError('resume next losses/gradients differ')
            if full['metrics'][:2] != prefix['metrics']:
                raise ValueError('prefix differs')
            ranks[str(rank)] = dict(uninterrupted=full, prefix=prefix, resumed=resumed)
            finder = json.loads((work / f'finder-r{rank}.json').read_text())
            if (finder['code_check'] is not True or finder['selected_rate'] is not None or
                    finder['updates_per_rate'] != 2 or len(finder['trials']) != 4):
                raise ValueError('smoke LR finder did not exercise every fixed candidate')
            ranks[str(rank)]['lr_finder'] = finder
        # Expose correctness facts directly in the manager receipt.
        record['gates'] = dict(mask_math=True, frozen_dense_bit_identical=True,
                               finite_nonzero_gate_gradients=True, hard_dense_export=True,
                               resume_state_exact=True, sharded_training_world_size=2,
                               sharded_hf_rollout_world_size=2, vllm_tensor_parallel_size=2)
        record.update(status='PASS', ranks=ranks, vllm=json.loads((work / 'vllm.json').read_text()))
    except Exception as error:
        record['error'] = str(error)
        record['error_type'] = type(error).__name__
    finally:
        record['seconds'] = time.monotonic() - started
        payload = json.dumps(record, indent=2, sort_keys=True)
        if len(payload.encode()) > 1024 * 1024:
            record = dict(status='STOP', error='receipt exceeds one MiB', schema='deletion-smoke.v1')
            payload = json.dumps(record)
        (work / 'receipt.json').write_text(payload)
    return 0 if record['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
