"""Prepared 0.6B smoke. Default prints the price. --launch is manager-only.

One existing image, one two-A10G Sandbox (L4 had no capacity on 11 Nov), one attempt, no automatic retry.
No imports of Modal occur in the code uploaded to the container.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from deletion_tuning.deletion_logic import smoke_price

FILES = ('kit/sdft_actor.py', 'kit/sdft_objective.py', 'kit/p3b_loss.py', 'kit/sdft_verl.py',
         'deletion_tuning/__init__.py', 'deletion_tuning/deletion_actor.py',
         'deletion_tuning/deletion_logic.py', 'deletion_tuning/deletion_mask.py', 'deletion_tuning/deletion_smoke.py',
         'deletion_tuning/deletion_container.py', 'deletion_tuning/deletion_worker.py',
         'deletion_tuning/deletion_prepare.py', 'deletion_tuning/deletion_lr.py',
         'deletion_tuning/vendor/deleting_weights/differentiable_mask.py',
         'deletion_tuning/vendor/deleting_weights/LICENSE', 'deletion_tuning/vendor/deleting_weights/PROVENANCE.json')


def launch(receipt, log_path=None, *, image_id=None):
    image_id = image_id or os.environ.get('DELETION_MODAL_IMAGE_ID')
    if not image_id:
        raise ValueError('provide --image-id or DELETION_MODAL_IMAGE_ID before launch')
    import asyncio
    import hashlib
    import signal
    import threading
    log_path = Path(log_path or str(receipt).replace('.json', '.container.log'))
    price = smoke_price()
    full_bound = (price['seconds'] + price['startup_seconds']) * price['rate_usd_second'] + price['receipt_egress_usd']
    if not full_bound <= price['total_cap_usd'] <= 10:
        raise ValueError('smoke reservation exceeds owner cap')
    # Exclusive receipt claim prevents a repeat using the same launch command.
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open('x') as handle:
        json.dump(dict(status='STARTING', price=price), handle)
    record = dict(status='STOP', price=price, image_id=image_id, image_build_seconds=0)
    sandbox = None
    started = time.monotonic()
    def _term(signum, frame):
        raise KeyboardInterrupt('signal %d' % signum)
    signal.signal(signal.SIGTERM, _term)
    try:
        import modal  # host only
        record['code_sha256'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
        # from_id performs no image build. Missing image is a STOP.
        image = modal.Image.from_id(image_id)
        app = modal.App('deletion-06b-smoke')
        # detach=True: a dropped local connection cannot stop the app; the Sandbox timeout bounds cost.
        with app.run(detach=True):
            async def create():
                return await asyncio.wait_for(modal.Sandbox.create.aio(
                    'sleep', str(price['seconds']), app=app, image=image,
                    gpu='A10G:2', cpu=(4., 4.), memory=(32768, 32768), timeout=price['seconds'],
                    env={'PYTHONPATH': '/work:/root/SDPO', 'WANDB_MODE': 'disabled', 'TOKENIZERS_PARALLELISM': 'false',
                         'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}),
                    timeout=price['startup_seconds'])
            sandbox = asyncio.run(create())
            record['sandbox_id'] = sandbox.object_id
            (receipt.parent / (receipt.stem + '.sandbox_id')).write_text(sandbox.object_id + '\n')
            timer = threading.Timer(price['seconds'], sandbox.terminate)
            timer.daemon = True
            timer.start()
            try:
                # Installed Modal 1.5.3, modal/sandbox.py:2283 and
                # modal/sandbox_fs.py:275,551,584. write_* creates parents.
                for name in FILES:
                    sandbox.filesystem.write_bytes((ROOT / name).read_bytes(), '/work/' + name)
                sandbox.filesystem.write_text('', '/work/kit/__init__.py')
                process = sandbox.exec('python', '-m', 'deletion_tuning.deletion_container', timeout=3820)
                def _stream(stream, tag):
                    with log_path.open('a') as out:
                        for line in stream:
                            out.write('%s %.0fs %s' % (tag, time.monotonic() - started, line if line.endswith('\n') else line + '\n'))
                            out.flush()
                readers = [threading.Thread(target=_stream, args=(process.stdout, 'OUT'), daemon=True),
                           threading.Thread(target=_stream, args=(process.stderr, 'ERR'), daemon=True)]
                for r in readers:
                    r.start()
                process.wait()
                for r in readers:
                    r.join(timeout=30)
                payload = sandbox.filesystem.read_bytes('/work/results/receipt.json')
                if len(payload) > 1024 * 1024:
                    raise ValueError('receipt exceeds one MiB egress limit')
                record['smoke'] = json.loads(payload)
                record['status'] = record['smoke']['status']
            finally:
                sandbox.terminate(wait=True)
                timer.cancel()
    except (Exception, KeyboardInterrupt) as error:
        record['error_type'] = type(error).__name__
        record['error'] = str(error)[:500]
    finally:
        elapsed = time.monotonic() - started
        record['client_seconds'] = elapsed
        record['resource_time_ceiling_usd'] = price['total_cap_usd']
        receipt.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    if record['status'] != 'PASS':
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--image-id', default=os.environ.get('DELETION_MODAL_IMAGE_ID'))
    parser.add_argument('--receipt', type=Path, default=ROOT / 'tmp/deletion-smoke-receipt.json')
    args = parser.parse_args()
    if args.launch:
        launch(args.receipt, image_id=args.image_id)
    else:
        print(json.dumps(dict(status='PRICED, UNLAUNCHED', image_id=args.image_id,
                              source='https://modal.com/pricing', rate_date='2026-11-11', **smoke_price()), indent=2))


if __name__ == '__main__':
    main()
