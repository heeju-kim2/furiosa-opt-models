"""The worker of a vLLM-Omni stage the engine runs: each call goes to the stage's model runner,
which its subclass names, and which holds the engine model and the stage connector."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, ClassVar, Final, Protocol

import torch
from typing_extensions import override
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase
from vllm_omni.worker.mixins import OmniWorkerMixin

import furiosa_opt_engine as engine
from furiosa_opt_vllm.cache import LAYOUTS
from furiosa_opt_vllm.decoder import PES

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.tasks import SupportedTask
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
    from vllm.v1.outputs import ModelRunnerOutput


# Threads of a stage's torch pool: a mel spectrogram's FFTs gain nothing past a few.
TORCH_THREADS: Final = 4


class Runner(Protocol):
    def __init__(self, vllm_config: VllmConfig) -> None: ...
    def load_model(self) -> None: ...
    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]: ...
    def available_memory(self) -> int: ...
    def initialize_kv_cache(self, config: KVCacheConfig) -> None: ...
    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput: ...
    def shutdown_omni_connectors(self) -> None: ...


def device() -> engine.Device:
    """The engine device of the stage's first chip, as its `devices` names it."""
    return engine.Device(
        engine.Placement([current_platform.device_id_to_physical_device_id(0)], PES)
    )


class Worker(OmniWorkerMixin, WorkerBase):
    """A stage worker: its runner executes every step and returns the step's output itself, so
    vLLM collects no separate sampling."""

    model_runner_cls: ClassVar[type[Runner]]

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        self.model_runner: Runner | None = None

    @override
    def init_device(self) -> None:
        set_random_seed(self.model_config.seed)
        # The host's torch work is small, a clip's mel spectrogram and a row's draw, beside the
        # engine's polling of the device: a pool of every core spins against that polling and
        # stalls a clip's mel by up to hundreds of milliseconds on a loaded host.
        if "OMP_NUM_THREADS" not in os.environ:
            torch.set_num_threads(min(TORCH_THREADS, torch.get_num_threads()))
        self.model_runner = self.model_runner_cls(self.vllm_config)

    @override
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        if load_dummy_weights:
            raise ValueError("the backend loads checkpoint parameters")
        self._runner().load_model()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return ("generate",)

    @override
    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self._runner().get_kv_cache_spec()

    @override
    def get_supported_kv_cache_layouts(self) -> list[str]:
        return list(LAYOUTS)

    def determine_available_memory(self) -> int:
        return self._runner().available_memory()

    def update_max_model_len(self, max_model_len: int) -> None:
        self.model_config.max_model_len = max_model_len

    @override
    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        self._runner().initialize_kv_cache(kv_cache_config)

    @override
    def compile_or_warm_up_model(self) -> CompilationTimes:
        set_random_seed(self.model_config.seed)
        return CompilationTimes(language_model=0.0, encoder=0.0)

    @override
    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        return self._runner().execute_model(scheduler_output)

    @override
    def sample_tokens(self, grammar_output: GrammarOutput | None) -> ModelRunnerOutput:
        raise RuntimeError("an engine stage returns each step's output from execute_model")

    @override
    def get_cache_block_size_bytes(self) -> int:
        return sum(spec.page_size_bytes for spec in self.get_kv_cache_spec().values())

    def check_health(self) -> None:
        pass

    @override
    def shutdown(self) -> None:
        if self.model_runner is not None:
            self.model_runner.shutdown_omni_connectors()
            self.model_runner = None

    def _runner(self) -> Runner:
        if self.model_runner is None:
            raise RuntimeError("init_device must run before any other worker call")
        return self.model_runner
