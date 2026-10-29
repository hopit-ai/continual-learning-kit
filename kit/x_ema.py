"""P ordering assertion before any update: distinct student/teacher, post weights."""
from pathlib import Path
from kit.s_common import require, atomic


def assert_teacher_origin(student, teacher, expected, student_path, teacher_path):
    import torch
    require(Path(student_path).resolve() == Path(expected).resolve() == Path(teacher_path).resolve(),
            'P student and EMA must both initialise from post-replacement checkpoint')
    require(teacher is not student, 'P requires a separate fresh EMA copy')
    left, right = dict(student.named_parameters()), dict(teacher.named_parameters())
    require(left and left.keys() == right.keys(), 'P EMA parameter inventory differs')
    dtypes = {}
    for name, value in left.items():
        other = right[name]
        value = value.to_local() if hasattr(value,'to_local') else value
        other = other.to_local() if hasattr(other,'to_local') else other
        dtypes[name] = dict(student=str(value.dtype),teacher=str(other.dtype))
        other = other.to(device=value.device)
        require(value.shape == other.shape and torch.equal(value.detach().to(other.dtype),other.detach()),
                'P EMA did not copy post-replacement student: '+name)
    return dict(status='PASS',student_checkpoint=str(expected),teacher_checkpoint=str(expected),
                dtypes=dtypes,matched_parameter_tensors=len(left),before_first_update=True)


def install_init_guard(cls):
    import functools
    import os
    if getattr(cls,'_x_p_ema_guard',False): return
    original = cls.init_model
    @functools.wraps(original)
    def guarded(self,*args,**kwargs):
        expected = os.environ['X_P_POST_MODEL']
        model_path = self.config.model.path
        ref = self.config.ref.get('model',None)
        teacher_path = ref.get('path',model_path) if ref is not None else model_path
        require(Path(expected).resolve() == Path(model_path).resolve() == Path(teacher_path).resolve(),
                'P model/teacher path must name post-replacement weights before loading')
        result = original(self,*args,**kwargs)
        if self._is_actor:
            require(self.actor.teacher_module is self.ref_module_fsdp, 'P teaching module must be the fresh EMA reference')
            receipt = assert_teacher_origin(self.actor_module_fsdp,self.actor.teacher_module,expected,model_path,teacher_path)
            atomic(Path(os.environ['X_P_EMA_RECEIPTS'])/f'rank{self.rank}.json',receipt)
        return result
    cls.init_model = guarded
    cls._x_p_ema_guard = True
