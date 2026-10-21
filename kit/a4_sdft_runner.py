"""Driver-side token accounting only; no changes to W sampling, masks or losses."""
from kit.c_sdft_runner import CTaskRunner


class A4TaskRunner(CTaskRunner):
    def run(self,config):
        from kit import sdft_verl as v
        from kit.v4_teacher import atomic_json
        from pathlib import Path
        import os,time
        original=v.build_sdft_teacher_batch
        def counted(trainer,batch,*args,**kwargs):
            result=original(trainer,batch,*args,**kwargs)
            built=result[0].batch
            # Includes sampled replay rollouts and the actual padded teacher forwards.
            row=dict(step=int(trainer.global_steps),student_tokens=int(batch.batch['attention_mask'].sum()),
                student_output_tokens=int(batch.batch['response_mask'].sum()),
                teacher_tokens=int(built['teacher_attention_mask'].sum()),
                replay_target_tokens=sum(len(x.get('p3b_target_ids',[])) for x in batch.non_tensor_batch['extra_info'] if x.get('p3b_replay')))
            atomic_json(Path(os.environ['A4_TOKEN_LOG'])/(str(time.time_ns())+'.json'),row)
            return result
        v.build_sdft_teacher_batch=counted
        try:return super().run(config)
        finally:v.build_sdft_teacher_batch=original
