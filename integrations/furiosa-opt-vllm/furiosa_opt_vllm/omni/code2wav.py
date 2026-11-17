"""vLLM-Omni's code2wav stage on the device: each request's codec frames, which the talker stage
sends as its prompt, decoded into waveform samples by the engine's speech decoder, which the
engine's registry loads as the `Qwen3TTSCode2Wav` architecture.

- The stage's prompt is the frames' codes codebook-major, sixteen codebooks of the request's
  frames, the reference's context frames first; their samples are left out, as vLLM-Omni's
  `Qwen3TTSCode2Wav` leaves them out.
- A request completes in the step that schedules its whole prompt, so the stage holds no cache.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Final

import numpy as np
import torch
from vllm_omni.distributed.omni_connectors.model_runner.omni_connector_runtime import (
    needs_omni_connector,
)
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin

import furiosa_opt_engine as engine
from furiosa_opt_vllm import checkpoint
from furiosa_opt_vllm.omni import stage

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec

ARCHITECTURE: Final = "Qwen3TTSCode2Wav"


class Runner(OmniConnectorModelRunnerMixin):
    def __init__(self, vllm_config: VllmConfig) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        # req_id -> its prompt, the frames' codes codebook-major, once the payload arrives.
        self.requests: dict[str, list[int]] = {}
        self.model_intermediate_buffer: dict[str, dict[str, Any]] = {}
        self.vocoder: engine.Vocoder | None = None
        if needs_omni_connector(self.model_config):
            self.init_omni_connectors(model_config=self.model_config)

    def load_model(self) -> None:
        location = checkpoint.located(self.model_config, self.vllm_config.load_config.download_dir)
        options = engine.LoadOptions().architecture(ARCHITECTURE)
        vocoder = options.load(engine.Checkpoint(location.directory), stage.device())
        if not isinstance(vocoder, engine.Vocoder):
            raise TypeError(
                f"{ARCHITECTURE} loads a {type(vocoder).__name__}, not a speech decoder"
            )
        self.vocoder = vocoder

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return {}

    def available_memory(self) -> int:
        return 0

    def initialize_kv_cache(self, config: KVCacheConfig) -> None:
        if config.kv_cache_groups:
            raise ValueError("the speech decoder holds no cache")

    def execute_model(self, scheduler_output: SchedulerOutput) -> OmniModelRunnerOutput:
        vocoder = self.vocoder
        if vocoder is None:
            raise RuntimeError("load_model must precede execution")
        groups, entries, samples = vocoder.groups, vocoder.entries, vocoder.samples
        for handle in getattr(scheduler_output, "pending_input_registrations", ()):
            self.register_chunk_recv(handle)
        self.recv_full_payload_inputs(scheduler_output)
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.cleanup_finished_request(req_id)
        for new in scheduler_output.scheduled_new_reqs:
            self.requests[new.req_id] = list(new.prompt_token_ids or ())
        cached = scheduler_output.scheduled_cached_reqs
        for req_id, prompt in (getattr(cached, "prompt_token_ids", None) or {}).items():
            if prompt:
                self.requests[req_id] = list(prompt)
        req_ids = list(scheduler_output.num_scheduled_tokens)
        if not req_ids:
            return self.attach_omni_connector_output(empty())
        audio: list[dict[str, object]] = []
        for req_id in req_ids:
            prompt = self.requests.get(req_id, [])
            if not prompt or len(prompt) % groups:
                raise ValueError(
                    f"{req_id}: the prompt holds {len(prompt)} codes, not frames of {groups}"
                )
            codes = np.asarray(prompt, dtype=np.int64).reshape(groups, -1).T
            # Rows of a frame with an entry outside the codebooks are vLLM-Omni's padding.
            codes = codes[((codes >= 0) & (codes < entries)).all(axis=1)]
            payload = self.get_local_stage_payload(req_id) or {}
            meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
            context = _integer(meta.get("left_context_size", 0))
            wave = vocoder.decode(codes.tolist()) if len(codes) else np.zeros(0, np.float32)
            audio.append(
                {
                    "model_outputs": torch.from_numpy(wave[context * samples :].copy()),
                    "sr": torch.tensor(vocoder.rate, dtype=torch.int32),
                }
            )
        output = OmniModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: at for at, req_id in enumerate(req_ids)},
            sampled_token_ids=[[] for _ in req_ids],
            multimodal_outputs=audio,
            inter_stage_outputs=audio,
        )
        output.omni_connector_output = self.get_omni_connector_output()
        return output


def empty() -> OmniModelRunnerOutput:
    return OmniModelRunnerOutput(req_ids=[], req_id_to_index={})


def _integer(value: object) -> int:
    match value:
        case list() | tuple():
            return _integer(value[0]) if value else 0
        case torch.Tensor():
            return int(value.reshape(-1)[0].item()) if value.numel() else 0
        case None:
            return 0
        case _:
            return int(value)  # type: ignore[call-overload]


class Worker(stage.Worker):
    model_runner_cls: ClassVar[type[stage.Runner]] = Runner  # type: ignore[assignment]
