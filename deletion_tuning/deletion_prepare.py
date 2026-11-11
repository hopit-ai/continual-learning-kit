"""Prepare Wanda inputs from the frozen ordered I0 training schedule, CPU only."""
import argparse
import hashlib
import json
from pathlib import Path

from deletion_tuning.deletion_logic import calibration_batches


def prepare(rows, tokenizer, schedule_sha256, seed):
    tasks = [row for row in rows if not row.get('p3b_replay') and not (row.get('extra_info') or {}).get('p3b_replay')][:32]
    if len(tasks) != 32:
        raise ValueError('frozen schedule has fewer than 32 task prompts')
    ids, encoded = [], []
    for row in tasks:
        extra = row.get('extra_info') or {}
        identity = row.get('id', extra.get('index'))
        if identity is None:
            raise ValueError('task row has no identity')
        prompt = row['prompt']
        if isinstance(prompt, str):
            prompt = json.loads(prompt) if prompt.startswith('[') else [dict(role='user', content=prompt)]
        text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        encoded.append(tokenizer.encode(text, add_special_tokens=False)[:1024])
        ids.append(str(identity))
    batches = []
    for start in range(0, 32, 8):
        batch = tokenizer.pad({'input_ids': encoded[start:start + 8]}, padding=True, return_attention_mask=True)
        batches.append(dict(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']))
    document = dict(schema='deletion-calibration.v1', seed=seed, schedule_sha256=schedule_sha256,
                    task_ids=ids, batches=batches, prompt_truncation='right', padding_side=tokenizer.padding_side)
    document['sha256'] = hashlib.sha256(json.dumps(document, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    calibration_batches(document)
    return document


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--schedule', type=Path, required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--seed', type=int, choices=(101, 102, 103), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.schedule.suffix == '.parquet':
        import pyarrow.parquet as pq
        rows = pq.read_table(args.schedule).to_pylist()
    else:
        rows = [json.loads(line) for line in args.schedule.read_text().splitlines() if line.strip()]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    document = prepare(rows, tokenizer, hashlib.sha256(args.schedule.read_bytes()).hexdigest(), args.seed)
    with args.out.open('x') as handle:
        json.dump(document, handle, indent=2)


if __name__ == '__main__':
    main()
