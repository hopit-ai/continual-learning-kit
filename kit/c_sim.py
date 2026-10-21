"""CPU tokenizer and B evidence fixtures for C's technical rehearsal only."""
from kit.sim.p3a_row import Tokenizer as BaseTokenizer


class Tokenizer(BaseTokenizer):
    pad_token_id = 0
    def apply_chat_template(self, messages, **kwargs):
        text = super().apply_chat_template(messages, **kwargs)
        return self.encode(text) if kwargs.get('tokenize') else text


def b_rows(winner='F+replay'):
    rows = []
    def add(arm,seed,step,panel,correct):
        for i in range(10):
            rows.append(dict(arm=arm,seed=seed,step=step,task=panel,panel=panel,cap=2048,
                question_id=str(i),correct=int(i<correct),output_tokens=10,finished=1))
    for p in ('finqa','ifeval','knowledge','math','math_screen'): add('base',101,0,p,5)
    for recipe in ('D','F','D+replay','F+replay'):
        for seed in range(101,106):
            tier1 = recipe if recipe in ('D','F') else 'finqa-'+recipe
            for p in ('finqa','ifeval','knowledge','math'): add(tier1,seed,40,p,7 if p=='finqa' else 5)
            for route in ('standalone','sequence'):
                for step in (0,10,20,30,40):
                    score = 5 if step==0 else 10 if recipe==winner else 9
                    add(route+'-'+recipe,seed,step,'math_screen',score)
                for step in (0,20,40):
                    for p in ('finqa','ifeval','knowledge','math'):
                        add(route+'-'+recipe,seed,step,p,7 if route=='sequence' and p=='finqa' else 5)
    return rows
