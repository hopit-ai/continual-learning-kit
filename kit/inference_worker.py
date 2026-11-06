"""vLLM 0.18 worker with one bounded recovery for a neighbour releasing memory.

This uses vLLM's worker_cls API so it also runs in spawned engine workers.
No kernels, sampling parameters, model inputs or score reductions are changed.
"""
from vllm.utils.mem_utils import MemorySnapshot
from vllm.v1.worker.gpu_worker import Worker

from kit.inference_lifecycle import profile_with_retry


class MemoryProfileWorker(Worker):
    def determine_available_memory(self):
        return profile_with_retry(self, super().determine_available_memory, MemorySnapshot)
