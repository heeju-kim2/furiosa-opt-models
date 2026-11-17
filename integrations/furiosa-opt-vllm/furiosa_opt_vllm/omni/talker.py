"""vLLM-Omni's Qwen3-TTS talker stage on the device: the engine's speaker, the talker, its codec
head and its code predictor, returns each scheduled request's scores of its next frame's first
entry, which the host samples and vLLM schedules on; the speaker's predictor then draws the
frame's other fifteen, and the frames reach the code2wav stage as the stage connector's full
payload.

- A request's prompt rows are built on the device, as vLLM-Omni's prompt builder lays them out for
  a Base voice clone: the text's rows through the talker's text table and projection, each over
  its codec table row, the reference clip's speaker embedding among them, which the engine's
  speaker encoder computes. The host tokenizes the text, and decodes, resamples and takes the mel
  spectrogram of the clip.
- An in-context clone's prompt then reads the reference's codec frames, which the engine's speech
  encoder computes from the clip, each frame's sixteen codebook rows summed over the reference's
  and the request's text rows; the frames lead the code2wav payload as its context.
- A later row is the sum of the frame before's sixteen codebook rows, in FP32 and rounded once,
  plus the request's next text row, as Transformers sums them, also on the device.
- Every row is an override; the cache is the engine's, paged as vLLM's block tables say.
- The first entry is drawn as the request's sampling parameters say, at least two frames before the
  codec's end; the others as the stage's `subtalker_sampling_params` say, both from the request's
  seed when it gives one.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Final

import numpy as np
import torch
from transformers import AutoTokenizer
from vllm.multimodal.audio import AudioResampler
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm_omni.distributed.omni_connectors.model_runner.omni_connector_runtime import (
    needs_omni_connector,
)
from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
    REF_AUDIO_CACHE_KEY,
    build_assistant_text,
    build_instruct_text,
    build_ref_text,
    load_audio_to_np,
    mel_spectrogram,
)
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin

import furiosa_opt_engine as engine
from furiosa_opt_vllm import checkpoint
from furiosa_opt_vllm.omni import stage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec

ARCHITECTURE: Final = "Qwen3TTSTalkerForConditionalGeneration"
# The token of every row, all of which the caller's rows override: the codec pad.
PAD: Final = 2148
GROUPS: Final = 16
# The talker's rows: 2048 entries of each codebook and 1024 control rows; the compiled rows past
# them score lowest.
ROWS: Final = 3072
# Official qwen-tts generates at least two frames before the codec's end.
MIN_FRAMES: Final = 2
# Reference clips whose speaker embeddings the stage keeps, as vLLM-Omni's builder keeps them.
VOICES: Final = 1024
# The code predictor's draws when the stage names none, as vLLM-Omni's stage config sets them.
SUBTALKER: Final = {"do_sample": True, "temperature": 0.9, "top_k": 50, "top_p": 1.0}


@dataclass
class Sampler:
    """How a request draws its frames' first entries, from its own generator."""

    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    min_tokens: int
    eos: int
    generator: torch.Generator

    @classmethod
    def of(cls, params: SamplingParams | None, eos: int) -> Sampler:
        generator = torch.Generator()
        seed = getattr(params, "seed", None)
        generator.manual_seed(seed if seed is not None else torch.seed())
        if params is None:
            return cls(0.0, 0, 1.0, 1.0, MIN_FRAMES, eos, generator)
        return cls(
            temperature=float(params.temperature),
            top_k=max(int(params.top_k), 0),
            top_p=float(params.top_p),
            repetition_penalty=float(params.repetition_penalty),
            min_tokens=max(int(params.min_tokens), MIN_FRAMES),
            eos=eos,
            generator=generator,
        )

    def first(self, scores: torch.Tensor, history: Sequence[int]) -> int:
        """The first entry drawn from the talker head's FP32 `scores` after the frames whose first
        entries are `history`: the end held back before `min_tokens` frames, each entry drawn
        before scaled by the repetition penalty, then the temperature, top-k and top-p."""
        scores = scores[:ROWS].double()
        if len(history) < self.min_tokens:
            scores[self.eos] = float("-inf")
        if self.repetition_penalty != 1.0 and history:
            seen = torch.tensor(sorted(set(history)), dtype=torch.long)
            picked = scores[seen]
            scores[seen] = torch.where(
                picked > 0, picked / self.repetition_penalty, picked * self.repetition_penalty
            )
        if self.temperature <= 0:
            return int(torch.argmax(scores))
        scores = scores / self.temperature
        if 0 < self.top_k < len(scores):
            threshold = torch.topk(scores, self.top_k).values[-1]
            scores = scores.masked_fill(scores < threshold, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        if self.top_p < 1.0:
            ordered, order = torch.sort(probabilities, descending=True)
            outside = ordered.cumsum(0) - ordered >= self.top_p
            probabilities[order[outside]] = 0
        return int(torch.multinomial(probabilities, 1, generator=self.generator))

    def seed(self) -> int:
        """A seed of the code predictor's draws of a frame."""
        return int(torch.randint(0, 2**63 - 1, (1,), generator=self.generator))


@dataclass(frozen=True)
class Reference:
    """An in-context clone's reference: its text's token IDs past the role, and its codec frames."""

    ids: list[int]
    frames: list[list[int]]


@dataclass(frozen=True)
class Built:
    """A request's prompt rows on the device, the text rows its frames read, and the reference's
    codec frames an in-context clone decodes before its own."""

    prompt: engine.HbmTensor
    texts: list[engine.HbmTensor]
    reference: list[list[int]] | None


@dataclass
class State:
    """A request on the talker: its block table, its prompt rows and text rows on the device, its
    sampler, its first entries so far and the row its next position reads."""

    req_id: str
    blocks: list[int]
    prompt: engine.HbmTensor
    texts: list[engine.HbmTensor]
    sampler: Sampler
    computed: int = 0
    output_token_ids: list[int] = field(default_factory=list)
    next: engine.HbmTensor | None = None
    # The reference's codec frames the payload leads with, until the first frame carries them.
    reference: list[list[int]] | None = None

    @property
    def request_id(self) -> str:
        return self.req_id


class Runner(OmniConnectorModelRunnerMixin):
    def __init__(self, vllm_config: VllmConfig) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.requests: dict[str, State] = {}
        self.model_intermediate_buffer: dict[str, dict[str, Any]] = {}
        self.speaker: engine.qwen3_tts.Speaker | None = None
        self.cache: engine.decoder.Cache | None = None
        subtalker = getattr(self.model_config, "subtalker_sampling_params", None) or {}
        self.subtalker = SUBTALKER | dict(subtalker)
        if needs_omni_connector(self.model_config):
            self.init_omni_connectors(model_config=self.model_config)

    def load_model(self) -> None:
        location = checkpoint.located(self.model_config, self.vllm_config.load_config.download_dir)
        self.device = stage.device()
        options = engine.LoadOptions().architecture(ARCHITECTURE)
        speaker = options.load(engine.Checkpoint(location.directory), self.device)
        if not isinstance(speaker, engine.qwen3_tts.Speaker):
            raise TypeError(f"{ARCHITECTURE} loads a {type(speaker).__name__}, not the speaker")
        self.speaker = speaker
        self.prompts = Prompts(self.model_config, location.directory, speaker, self.device)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        cache = self._speaker().cache
        specs: dict[str, KVCacheSpec] = {}
        for group in cache.groups:
            pool = cache.pools[group.pool]
            spec = FullAttentionSpec(
                block_size=pool.block, num_kv_heads=8, head_size=128, dtype=torch.bfloat16
            )
            if spec.page_size_bytes != pool.bytes:
                raise ValueError(
                    f"the engine pages {pool.bytes} bytes a block, vLLM {spec.page_size_bytes}"
                )
            specs |= {f"talker.layers.{index}": spec for index in group.layers}
        return specs

    def available_memory(self) -> int:
        scheduler, config = self.vllm_config.scheduler_config, self.model_config
        spec = self.get_kv_cache_spec()
        block = next(iter(spec.values())).block_size
        # Every request at the context it may reach, and vLLM's null block.
        blocks = scheduler.max_num_seqs * -(-config.max_model_len // block) + 1
        return blocks * sum(layer.page_size_bytes for layer in spec.values())

    def initialize_kv_cache(self, config: KVCacheConfig) -> None:
        self.cache = engine.decoder.Cache(self._speaker().cache, config.num_blocks)

    def execute_model(self, scheduler_output: SchedulerOutput) -> OmniModelRunnerOutput:
        speaker, cache = self._speaker(), self.cache
        if cache is None:
            raise RuntimeError("the cache must be initialized before execution")
        for handle in getattr(scheduler_output, "pending_input_registrations", ()):
            self.register_chunk_recv(handle)
        self.recv_full_payload_inputs(scheduler_output)
        if self._pending_full_payload_send:
            pending = {
                req_id for req_id in self._pending_full_payload_send if req_id not in self.requests
            }
            self.flush_full_payload_outputs(set(scheduler_output.finished_req_ids) | pending)
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.cleanup_finished_request(req_id)
        for new in scheduler_output.scheduled_new_reqs:
            self.requests[new.req_id] = self._admit(new)
        cached = scheduler_output.scheduled_cached_reqs
        for at, req_id in enumerate(cached.req_ids):
            state = self.requests[req_id]
            blocks = cached.new_block_ids[at]
            if blocks is not None:
                resumed = req_id in getattr(cached, "resumed_req_ids", ())
                state.blocks = list(blocks[0]) if resumed else state.blocks + list(blocks[0])
        scheduled = dict(scheduler_output.num_scheduled_tokens)
        req_ids = list(scheduled)
        if not req_ids:
            return self.attach_omni_connector_output(
                OmniModelRunnerOutput(req_ids=[], req_id_to_index={})
            )
        frames: dict[str, list[int]] = {}
        # Calls of at most the speaker's rows, requests in schedule order, each row its position's.
        work = [
            (req_id, position)
            for req_id in req_ids
            for position in self._positions(req_id, scheduled[req_id])
        ]
        while work:
            call, work = work[: speaker.rows], work[speaker.rows :]
            requests, completing = [], []
            for req_id in dict.fromkeys(req_id for req_id, _ in call):
                state = self.requests[req_id]
                positions = [position for owner, position in call if owner == req_id]
                done = positions[-1] + 1 == state.computed + scheduled[req_id] and (
                    positions[-1] + 1 >= state.prompt.shape[0]
                )
                requests.append(
                    engine.decoder.Request(
                        [state.blocks],
                        positions[0],
                        self.prompts.pads(len(positions)),
                        engine.decoder.Keep.LAST if done else engine.decoder.Keep.NONE,
                        overrides=[(0, self._rows(state, positions))],
                    )
                )
                if done:
                    completing.append(req_id)
            scores = speaker.forward(requests, cache)
            if not completing:
                continue
            host = torch.from_numpy(scores.to_host().numpy())
            states = [self.requests[req_id] for req_id in completing]
            firsts = [
                state.sampler.first(row, state.output_token_ids)
                for state, row in zip(states, host, strict=True)
            ]
            drawn = speaker.frames(
                firsts,
                [state.sampler.seed() for state in states],
                **self._subtalker(),
            )
            for req_id, frame in zip(completing, drawn.tolist(), strict=True):
                frames[req_id] = frame
            self._next(states, drawn.tolist())
        for req_id in req_ids:
            self.requests[req_id].computed += scheduled[req_id]
        sampled, outputs = [], []
        for req_id in req_ids:
            state, frame = self.requests[req_id], frames.get(req_id)
            if frame is None:
                sampled.append([])
                outputs.append(None)
                continue
            state.output_token_ids.append(frame[0])
            rows = {"codes.audio": torch.tensor([frame], dtype=torch.long)}
            if state.reference is not None:
                # The reference's frames lead the payload, which code2wav decodes as context.
                rows["codes.ref"] = torch.tensor(state.reference, dtype=torch.long)
                rows["meta.ref_code_len"] = torch.tensor([len(state.reference)], dtype=torch.long)
                state.reference = None
            if self._should_accumulate_full_payload_output():
                self.accumulate_full_payload_output(req_id, rows, state)
            sampled.append([frame[0]])
            outputs.append(rows)
        output = OmniModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: at for at, req_id in enumerate(req_ids)},
            sampled_token_ids=sampled,
            multimodal_outputs=outputs,
            inter_stage_outputs=outputs,
        )
        output.omni_connector_output = self.get_omni_connector_output()
        return output

    def _admit(self, new: Any) -> State:
        info = deserialize_additional_information(getattr(new, "additional_information", None))
        built = self.prompts.build(info)
        prompt, texts = built.prompt, built.texts
        rows = len(new.prompt_token_ids)
        if prompt.shape[0] > rows:
            raise ValueError(f"{new.req_id}: {prompt.shape[0]} prompt rows for {rows} placeholders")
        if prompt.shape[0] < rows:
            prompt = self.prompts.padded(prompt, rows)
        return State(
            req_id=new.req_id,
            blocks=list(new.block_ids[0]),
            prompt=prompt,
            texts=texts,
            sampler=Sampler.of(getattr(new, "sampling_params", None), self.prompts.eos),
            computed=new.num_computed_tokens,
            reference=built.reference,
        )

    def _positions(self, req_id: str, count: int) -> range:
        state = self.requests[req_id]
        return range(state.computed, state.computed + count)

    def _rows(self, state: State, positions: list[int]) -> engine.HbmTensor:
        """The device rows of a request's `positions`: its prompt's, or the row after its last
        frame."""
        length = state.prompt.shape[0]
        if positions[-1] < length:
            return state.prompt[positions[0] : positions[-1] + 1]
        if len(positions) != 1 or positions[0] != length + len(state.output_token_ids) - 1:
            raise RuntimeError(f"{state.req_id}: positions {positions} past its prompt")
        if state.next is None:
            raise RuntimeError(f"{state.req_id}: a frame row before its first frame")
        return state.next

    def _next(self, states: list[State], frames: list[list[int]]) -> None:
        """Each request's row after its new frame: the frame's rows summed, then its next text
        row, or the text pad's past its text."""
        texts = []
        for state in states:
            step = len(state.output_token_ids)
            texts.append(state.texts[step] if step < len(state.texts) else self.prompts.pad)
        rows = self._speaker().frame_rows(frames, texts)
        for at, state in enumerate(states):
            state.next = rows[at : at + 1]

    def _subtalker(self) -> dict[str, float | int]:
        params = self.subtalker
        if not params.get("do_sample", True):
            return {"temperature": 0.0}
        return {
            "temperature": float(params.get("temperature", 0.9)),
            "top_k": max(int(params.get("top_k", 0)), 0),
            "top_p": float(params.get("top_p", 1.0)),
        }

    def _speaker(self) -> engine.qwen3_tts.Speaker:
        if self.speaker is None:
            raise RuntimeError("load_model must precede the speaker's use")
        return self.speaker


class Prompts:
    """A Base voice clone's talker prompt as vLLM-Omni's prompt builder lays it out, its rows built
    on the device: the role's text rows, then the text pad over the codec's think prefill and over
    the speaker's embedding, the text bos over the codec pad, then the text's rows.

    - Streaming, Base's default: the text's first row over the codec bos ends the prompt, and its
      later rows and the text eos follow one a frame.
    - Non-streaming: the text's rows and eos over the codec pad, then the text pad over the codec
      bos, end the prompt; the text pad follows every frame.
    """

    def __init__(
        self,
        model_config: Any,
        directory: str,
        speaker: engine.qwen3_tts.Speaker,
        device: engine.Device,
    ) -> None:
        config = model_config.hf_config
        self.config, self.talker = config, config.talker_config
        self.speaker, self.device = speaker, device
        self.eos = int(self.talker.codec_eos_token_id)
        self.audio = audio(config, directory, device)
        self.tokenizer = tokenizer(directory)
        specials = [config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]
        projected = speaker.text([int(token) for token in specials])
        self.bos, self.text_eos, self.pad = (projected[at : at + 1] for at in range(3))
        # Each reference clip's speaker embedding row by vLLM-Omni's artifact key: the API server
        # sends a clip it sent before as its key alone.
        self.voices: OrderedDict[str, engine.HbmTensor] = OrderedDict()
        # And each reference clip's codec frames, which an in-context clone reads.
        self.references: OrderedDict[str, list[list[int]]] = OrderedDict()

    def build(self, info: dict[str, Any]) -> Built:
        """The prompt rows of the request `info` describes, the text rows its frames read, and
        the reference clip's codec frames an in-context clone reads."""
        task = _first(info.get("task_type"), "Base")
        if task != "Base":
            raise NotImplementedError(f"the {task} task's talker is not served; Base is")
        prompt = info.get("voice_clone_prompt")
        prompt = _first(prompt, None) if isinstance(prompt, list) else prompt
        in_context = not _true(info.get("x_vector_only_mode"))
        if isinstance(prompt, dict) and isinstance(prompt.get("icl_mode"), bool):
            in_context = prompt["icl_mode"]
        text = _first(info.get("text"), "")
        ids = self.tokenizer(build_assistant_text(text))["input_ids"]
        reference = self._reference(info, prompt) if in_context else None
        instruct = _first(info.get("instruct"), "")
        heading = (
            self.tokenizer(build_instruct_text(instruct))["input_ids"]
            if isinstance(instruct, str) and instruct.strip()
            else []
        )
        body = ids[3:-5]
        said = reference.ids if reference is not None else []
        streaming = _streaming(info)
        projected = self.speaker.text(heading + ids[:3] + said + body)
        row = lambda at: projected[at : at + 1]  # noqa: E731
        heading_rows = [row(at) for at in range(len(heading))]
        role = [row(len(heading) + at) for at in range(3)]
        start = len(heading) + 3
        texts = [row(start + at) for at in range(len(said) + len(body))] + [self.text_eos]

        talker = self.talker
        language = _first(info.get("language"), "Auto")
        language_id = (
            talker.codec_language_id.get(language.lower())
            if isinstance(language, str) and language.lower() != "auto"
            else None
        )
        prefill = (
            [talker.codec_nothink_id, talker.codec_think_bos_id, talker.codec_think_eos_id]
            if language_id is None
            else [
                talker.codec_think_id,
                talker.codec_think_bos_id,
                int(language_id),
                talker.codec_think_eos_id,
            ]
        )
        codec = self.speaker.codec([int(id) for id in prefill] + [PAD, talker.codec_bos_id])
        entry = lambda at: codec[at : at + 1]  # noqa: E731
        codec_pad, codec_bos = entry(len(prefill)), entry(len(prefill) + 1)
        voice = self._voice(info, prompt)

        pairs: list[tuple[engine.HbmTensor | None, engine.HbmTensor | None]] = [
            (text, None) for text in heading_rows + role
        ]
        pairs += [(self.pad, entry(at)) for at in range(len(prefill))]
        pairs += [(self.pad, voice), (self.bos, codec_pad)]
        if reference is not None:
            rows, texts = self._in_context(
                reference.frames, texts, (codec_pad, codec_bos), streaming
            )
            pairs += [(row, None) for row in rows]
        elif streaming:
            pairs.append((texts.pop(0), codec_bos))
        else:
            pairs += [(text, codec_pad) for text in texts]
            pairs.append((self.pad, codec_bos))
            texts = []
        frames = reference.frames if reference is not None else None
        return Built(self.speaker.add(pairs), texts, frames)

    def _in_context(
        self,
        frames: list[list[int]],
        texts: list[engine.HbmTensor],
        codecs: tuple[engine.HbmTensor, engine.HbmTensor],
        streaming: bool,
    ) -> tuple[list[engine.HbmTensor], list[engine.HbmTensor]]:
        """The in-context rows of the reference's codec frames and the reference's and the request's
        text rows, and the text rows the frames read after them, as vLLM-Omni's
        `_generate_icl_prompt` lays them out: the codec bos, then each frame's sixteen rows summed,
        over the text rows or the text pad."""
        codec_pad, codec_bos = codecs
        count = len(frames) + 1
        if not streaming:
            rows = self._summed(frames, [self.pad] * len(frames))
            heading = self.speaker.add(
                [(text, codec_pad) for text in texts] + [(self.pad, codec_bos)]
            )
            return [heading[at : at + 1] for at in range(heading.shape[0])] + rows, []
        over = texts[:count] + [self.pad] * (count - len(texts))
        first = self.speaker.add([(over[0], codec_bos)])
        return [first, *self._summed(frames, over[1:])], texts[count:]

    def _summed(
        self, frames: list[list[int]], texts: list[engine.HbmTensor]
    ) -> list[engine.HbmTensor]:
        """Each frame's sixteen codebook rows summed, over its text row; a block of frames a
        call."""
        rows: list[engine.HbmTensor] = []
        for at in range(0, len(frames), 128):
            block = self.speaker.frame_rows(frames[at : at + 128], texts[at : at + 128])
            rows += [block[row : row + 1] for row in range(block.shape[0])]
        return rows

    def padded(self, prompt: engine.HbmTensor, rows: int) -> engine.HbmTensor:
        """`prompt` and text pad rows after it, `rows` in all."""
        pairs = [(prompt[at : at + 1], None) for at in range(prompt.shape[0])]
        pairs += [(self.pad, None)] * (rows - prompt.shape[0])
        return self.speaker.add(pairs)

    def pads(self, count: int) -> engine.HbmTensor:
        """`count` codec pad tokens, which the rows' overrides replace."""
        return upload(self.device, torch.full((count,), PAD, dtype=torch.int32))

    def _reference(self, info: dict[str, Any], prompt: object) -> Reference | None:
        """An in-context clone's reference text and codec frames: the prompt's own frames, else
        those of a clip sent before under the same artifact key, else the clip's, which the
        engine's speech encoder computes; none without a reference text, as vLLM-Omni's builder
        falls back to the speaker embedding alone."""
        prompt = prompt if isinstance(prompt, dict) else {}
        ids = _ids(info.get("ref_ids")) or _ids(prompt.get("ref_ids") or prompt.get("ref_id"))
        if ids is None:
            said = _first(info.get("ref_text"), None) or _first(prompt.get("ref_text"), None)
            if not isinstance(said, str) or not said.strip():
                return None
            ids = self.tokenizer(build_ref_text(said))["input_ids"]
        frames = _frames(prompt.get("ref_code"))
        key = _first(info.get(REF_AUDIO_CACHE_KEY), None)
        key = key if isinstance(key, str) and key else None
        if frames is None:
            clips = info.get("ref_audio")
            if isinstance(clips, list) and clips:
                frames = self.audio.codes(clips[0])
            elif key is not None and key in self.references:
                frames = self.references[key]
            else:
                raise ValueError("an in-context clone requires `ref_audio` or `ref_code`")
            if key is not None:
                self.references[key] = frames
                self.references.move_to_end(key)
                while len(self.references) > VOICES:
                    self.references.popitem(last=False)
        return Reference(ids[3:-2], frames)

    def _voice(self, info: dict[str, Any], prompt: object) -> engine.HbmTensor:
        """The speaker embedding row of the request: the prompt's own, else the reference clip's,
        or the row of a clip an earlier request sent under the same artifact key."""
        given = prompt.get("ref_spk_embedding") if isinstance(prompt, dict) else None
        if given is not None and len(given) > 0:
            embedding = torch.as_tensor(given, dtype=torch.float32).reshape(-1)
            return upload(self.device, embedding.to(torch.bfloat16).reshape(1, -1))
        key = _first(info.get(REF_AUDIO_CACHE_KEY), None)
        key = key if isinstance(key, str) and key else None
        clips = info.get("ref_audio")
        if isinstance(clips, list) and clips:
            voice = self.audio.embedding(clips[0])
            if key is not None:
                self.voices[key] = voice
                self.voices.move_to_end(key)
                while len(self.voices) > VOICES:
                    self.voices.popitem(last=False)
            return voice
        if key is not None and key in self.voices:
            self.voices.move_to_end(key)
            return self.voices[key]
        raise ValueError("Base requires `ref_audio`, or the artifact key of a clip sent before")


class Audio:
    """A reference clip's speaker embedding and codec frames on the device: the clip decoded and
    resampled to the encoders' rate on the host, as vLLM-Omni decodes and resamples it, and official
    qwen-tts's mel spectrogram of it, then the engine's speaker encoder and speech encoder."""

    def __init__(
        self,
        encoder: engine.qwen3_tts.SpeakerEncoder,
        speech: engine.qwen3_tts.SpeechEncoder,
        rate: int,
    ) -> None:
        self.encoder, self.speech, self.rate = encoder, speech, rate
        self.resamplers: dict[int, AudioResampler] = {}
        # The last clip and its samples: a request's codes and embedding read one clip.
        self.last: tuple[object, np.ndarray] | None = None

    def wave(self, clip: object) -> np.ndarray:
        """The samples of `clip` at the encoders' rate, decoded and resampled once for the clip
        both encoders read."""
        if self.last is not None and self.last[0] is clip:
            return self.last[1]
        wave, rate = _clip(clip)
        if rate != self.rate:
            resampler = self.resamplers.setdefault(rate, AudioResampler(target_sr=self.rate))
            wave = resampler.resample(wave, orig_sr=rate)
        wave = np.ascontiguousarray(wave, dtype=np.float32)
        self.last = (clip, wave)
        return wave

    def codes(self, clip: object) -> list[list[int]]:
        """The codec frames of `clip`, sixteen codes each, from the engine's speech encoder."""
        return self.speech.encode(self.wave(clip)).tolist()

    def embedding(self, clip: object) -> engine.HbmTensor:
        """The speaker embedding row of `clip`, `[1, 2048]` BF16 on the device."""
        mels = mel_spectrogram(
            torch.from_numpy(self.wave(clip)).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=self.rate,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)
        return self.encoder.embed(mels[0].contiguous().numpy())


def audio(config: Any, directory: str, device: engine.Device) -> Audio:
    """The checkpoint's speaker encoder and speech tokenizer encoder on the device."""
    encoder = engine.qwen3_tts.SpeakerEncoder(engine.Checkpoint(directory), device)
    speech = engine.qwen3_tts.SpeechEncoder(engine.Checkpoint(directory), device)
    rate = int(getattr(config.speaker_encoder_config, "sample_rate", 24000))
    return Audio(encoder, speech, rate)


def tokenizer(directory: str) -> Any:
    """The text tokenizer of the checkpoint, as vLLM-Omni's prompt builder loads it."""
    import transformers

    options: dict[str, Any] = {"trust_remote_code": True, "use_fast": True}
    if transformers.__version__ < "5":
        options["fix_mistral_regex"] = True
    loaded = AutoTokenizer.from_pretrained(directory, **options)
    loaded.padding_side = "left"
    return loaded


def _clip(value: object) -> tuple[np.ndarray, int]:
    """A reference clip's mono samples and their rate: a path, URL or base64 string decoded as
    vLLM-Omni decodes it, or the `[samples, rate]` pair its API server resolves a clip into, alone
    or in a list of one."""
    match value:
        case str():
            samples, rate = load_audio_to_np(value)
        case {"array": samples, "sampling_rate": rate} | {"wav": samples, "sr": rate}:
            pass
        case [samples, rate] | (samples, rate) if _rate(rate):
            pass
        case [inner] | (inner,):
            return _clip(inner)
        case _:
            raise ValueError("a reference clip is a path, URL or base64 string, or [samples, rate]")
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    wave = np.asarray(samples, dtype=np.float32)
    if wave.ndim == 2:
        # Channels, whichever axis holds them, averaged.
        wave = wave.mean(axis=int(np.argmin(wave.shape)))
    if wave.ndim != 1 or not wave.size:
        raise ValueError("a reference clip holds one channel's samples, or a few channels'")
    return wave, int(rate)


def _rate(value: object) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and value > 0


def _streaming(info: dict[str, Any]) -> bool:
    """Whether a Base request's text follows its prompt one row a frame, Base's default, or ends its
    prompt, as vLLM-Omni's builder reads `non_streaming_mode`."""
    value = _first(info.get("non_streaming_mode"), None)
    return not value if isinstance(value, bool) else True


def upload(device: engine.Device, tensor: torch.Tensor) -> engine.HbmTensor:
    return engine.HostTensor.from_dlpack(tensor.contiguous()).to_hbm(device)


def _first(value: object, default: Any) -> Any:
    match value:
        case list() | tuple():
            return value[0] if value else default
        case None:
            return default
        case _:
            return value


def _ids(value: object) -> list[int] | None:
    """Token IDs a request passes: a list, an array or a tensor, one row or a batch of one."""
    match value:
        case None:
            return None
        case torch.Tensor() | np.ndarray():
            return [int(token) for token in np.asarray(value).reshape(-1)] or None
        case list() | tuple() if value and isinstance(value[0], (list, tuple, np.ndarray)):
            return _ids(value[0])
        case list() | tuple():
            return [int(token) for token in value] or None
        case _:
            return None


def _frames(value: object) -> list[list[int]] | None:
    """Codec frames a request passes, `[frames, 16]`, or none."""
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], torch.Tensor):
        value = value[0]
    if value is None:
        return None
    frames = np.asarray(value.cpu() if isinstance(value, torch.Tensor) else value)
    if frames.ndim == 3:
        frames = frames[0]
    if frames.ndim != 2 or frames.shape[1] != GROUPS or not len(frames):
        return None
    return frames.astype(np.int64).tolist()


def _true(value: object) -> bool:
    match value:
        case list() | tuple():
            return bool(value) and _true(value[0])
        case _:
            return bool(value)


class Worker(stage.Worker):
    model_runner_cls: ClassVar[type[stage.Runner]] = Runner  # type: ignore[assignment]
