"""C-only long-input dataset; B's response-only, EOS-inclusive masking convention."""
from kit.c_common import require, t


def encode(tokenizer, messages, response, max_length=12288):
    pids, tids = t.tokenize_student(tokenizer, messages, response)
    require(0 < len(pids) <= 10240 and 0 < len(tids) <= 2048, 'C input/response cap exceeded; never truncate')
    require(max_length == 12288, 'C SFT context must be 12288')
    ids = pids + tids; pad = max_length - len(ids)
    return dict(input_ids=ids+[tokenizer.pad_token_id]*pad, attention_mask=[1]*len(ids)+[0]*pad,
                position_ids=list(range(len(ids)))+[0]*pad,
                loss_mask=[0]*len(pids)+[1]*len(tids)+[0]*pad)


class CSFTDataset:
    def __init__(self, parquet_files, tokenizer, config, max_samples=-1):
        import pyarrow.parquet as pq
        paths = [parquet_files] if isinstance(parquet_files, str) else list(parquet_files)
        self.rows = [r for p in paths for r in pq.read_table(p).to_pylist()]
        self.tokenizer = tokenizer
        require(max_samples <= 0 and config.get('max_length') == 12288 and config.get('truncation') == 'error', 'C data cannot be subsetted or truncated')
        require(config.get('response_key') == 'extra_info.demonstration', 'wrong C SFT target field')

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        import torch
        row = self.rows[index]
        return {k:torch.tensor(v, dtype=torch.long) for k,v in encode(self.tokenizer, row['prompt'], row['extra_info']['demonstration']).items()}
