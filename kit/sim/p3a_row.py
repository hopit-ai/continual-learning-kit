"""Gold-aware CPU stand-ins; never a scientific candidate qualification."""
class Tokenizer:
    eos_token = '<eos>'
    def encode(self, text, **kwargs): return list(range(len(text) // 4 + 1))
    def apply_chat_template(self, messages, **kwargs): return '\n'.join(r['content'] for r in messages)


class Generator:
    def __init__(self, candidate, scenario):
        self.candidate, self.scenario = candidate, scenario
    def generate_many(self, requests):
        outputs = []
        for row in requests:
            # Four registered attempts are exercised. In the MATH branches, LCB has
            # only 80 intersections; otherwise all strata clear 85 globally.
            index = int(str(row['id']).rsplit('-', 1)[1])
            good = row['attempt'] == 4 and (self.candidate != 'code' or self.scenario == 'lcb-pass' or index % 5 != 0)
            if self.candidate == 'code':
                text = '```python\nprint(int(input()) + ' + ('1' if good else '2') + ')\n```'
            else:
                text = 'Divide one into two equal parts. \\boxed{' + ('1/2' if good else '2/3') + '}'
            outputs.append(dict(text=text, finish_reason='stop', technical_synthetic=True))
        return outputs
