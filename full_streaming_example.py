"""Minimal full-streaming benchmark for this fork.

This script is intentionally local-first:
- defaults to a local model directory: ./Qwen3-TTS-12Hz-0.6B-Base
- defaults to a local reference wav: ./sample.wav

Example:
  python full_streaming_example.py \
    --text "Hello from the optimized streaming fork." \
    --language English \
    --model-dir ./Qwen3-TTS-12Hz-0.6B-Base \
    --ref-audio ./sample.wav \
    --out-wav ./output_full_streaming.wav
"""

from pathlib import Path
import argparse
import asyncio
import time
import typing as tp

import numpy as np
import soundfile as sf
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor

from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel


def split_text_for_stream(text: str, target_chars: int) -> list[str]:
    words = text.split(" ")
    chunks: list[str] = []
    current = ""
    for word in words:
        candidate = word if current == "" else f"{current} {word}"
        if current and len(candidate) > target_chars:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


class StreamingTextState:
    def __init__(self, tts_model: Qwen3TTSModel, holdback_tokens: int = 1):
        self.tts = tts_model
        self.talker = tts_model.model.talker
        self.config = tts_model.model.config
        self.holdback_tokens = holdback_tokens
        self.full_text = ""
        self.committed_trailing_ids: list[int] = []
        self.finalized = False
        self._empty_assistant_ids = self._assistant_ids("")[0].tolist()

    def _assistant_ids(self, text: str) -> torch.Tensor:
        return self.tts._tokenize_texts([self.tts._build_assistant_text(text)])[0]

    @staticmethod
    def _common_prefix_len(a: list[int], b: list[int]) -> int:
        n = min(len(a), len(b))
        idx = 0
        while idx < n and a[idx] == b[idx]:
            idx += 1
        return idx

    @staticmethod
    def _common_suffix_len(a: list[int], b: list[int], start_a: int, start_b: int) -> int:
        i = len(a) - 1
        j = len(b) - 1
        count = 0
        while i >= start_a and j >= start_b and a[i] == b[j]:
            count += 1
            i -= 1
            j -= 1
        return count

    def _extract_content_ids(self, full_ids: list[int]) -> list[int]:
        prefix = self._common_prefix_len(full_ids, self._empty_assistant_ids)
        suffix = self._common_suffix_len(full_ids, self._empty_assistant_ids, prefix, prefix)
        end = len(full_ids) - suffix if suffix > 0 else len(full_ids)
        if end < prefix:
            return []
        content = full_ids[prefix:end]
        if content:
            content = content[1:]
        return content

    def push_text(self, chunk: str, final_chunk: bool) -> tp.Optional[torch.Tensor]:
        if self.finalized:
            return None

        if self.full_text and chunk and not chunk.startswith((" ", ".", ",", "!", "?", ";", ":")):
            self.full_text += " "
        self.full_text += chunk
        all_ids = self._assistant_ids(self.full_text)[0].tolist()
        candidate_ids = self._extract_content_ids(all_ids)

        if final_chunk:
            stable_ids = candidate_ids + [self.config.tts_eos_token_id]
            self.finalized = True
        elif len(candidate_ids) <= self.holdback_tokens:
            stable_ids = []
        else:
            stable_ids = candidate_ids[:-self.holdback_tokens]

        common_prefix = self._common_prefix_len(self.committed_trailing_ids, stable_ids)
        if common_prefix < len(self.committed_trailing_ids):
            stable_ids = self.committed_trailing_ids

        new_ids = stable_ids[len(self.committed_trailing_ids):]
        if not new_ids:
            return None

        self.committed_trailing_ids.extend(new_ids)
        new_ids_tensor = torch.tensor([new_ids], device=self.talker.device, dtype=torch.long)
        return self.talker.text_projection(self.talker.get_text_embeddings()(new_ids_tensor))

    def current_input_ids(self) -> torch.Tensor:
        return self._assistant_ids(self.full_text)

    def trailing_text_hidden(self) -> torch.Tensor:
        if not self.committed_trailing_ids:
            hidden_size = self.talker.config.hidden_size
            return torch.empty((1, 0, hidden_size), device=self.talker.device, dtype=self.talker.dtype)
        ids_tensor = torch.tensor([self.committed_trailing_ids], device=self.talker.device, dtype=torch.long)
        return self.talker.text_projection(self.talker.get_text_embeddings()(ids_tensor))


def build_talker_inputs_xvector(
    tts_model: Qwen3TTSModel,
    input_id: torch.Tensor,
    voice_clone_prompt: tp.Any,
    language: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    talker = tts_model.model.talker
    cfg = tts_model.model.config

    voice_clone_spk_embeds = tts_model.model.generate_speaker_prompt(voice_clone_prompt)
    speaker_embed = voice_clone_spk_embeds[0]

    language_lower = language.lower()
    if language_lower == "auto":
        language_id = None
    else:
        language_id = cfg.talker_config.codec_language_id[language_lower]

    tts_bos_embed, _, tts_pad_embed = talker.text_projection(
        talker.get_text_embeddings()(
            torch.tensor(
                [[cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id]],
                device=talker.device,
                dtype=input_id.dtype,
            )
        )
    ).chunk(3, dim=1)

    if language_id is None:
        codec_prefill_list = [[
            cfg.talker_config.codec_nothink_id,
            cfg.talker_config.codec_think_bos_id,
            cfg.talker_config.codec_think_eos_id,
        ]]
    else:
        codec_prefill_list = [[
            cfg.talker_config.codec_think_id,
            cfg.talker_config.codec_think_bos_id,
            language_id,
            cfg.talker_config.codec_think_eos_id,
        ]]

    codec_input_embedding_0 = talker.get_input_embeddings()(
        torch.tensor(codec_prefill_list, device=talker.device, dtype=input_id.dtype)
    )
    codec_input_embedding_1 = talker.get_input_embeddings()(
        torch.tensor(
            [[cfg.talker_config.codec_pad_id, cfg.talker_config.codec_bos_id]],
            device=talker.device,
            dtype=input_id.dtype,
        )
    )
    codec_input_embedding = torch.cat(
        [codec_input_embedding_0, speaker_embed.view(1, 1, -1), codec_input_embedding_1],
        dim=1,
    )

    talker_input_embed_role = talker.text_projection(talker.get_text_embeddings()(input_id[:, :3]))
    talker_input_embed = torch.cat(
        (tts_pad_embed.expand(-1, codec_input_embedding.shape[1] - 2, -1), tts_bos_embed),
        dim=1,
    ) + codec_input_embedding[:, :-1]
    talker_input_embed = torch.cat((talker_input_embed_role, talker_input_embed), dim=1)

    first_text_token = input_id[:, 3:4]
    if first_text_token.shape[1] != 1:
        raise ValueError("Need at least one text token in the initial stream chunk.")
    talker_input_embed = torch.cat(
        [
            talker_input_embed,
            talker.text_projection(talker.get_text_embeddings()(first_text_token)) + codec_input_embedding[:, -1:],
        ],
        dim=1,
    )

    attention_mask = torch.ones((1, talker_input_embed.shape[1]), device=talker.device, dtype=torch.long)
    return talker_input_embed, attention_mask, tts_pad_embed


def sample_next_token(logits: torch.Tensor, do_sample: bool, top_k: int, top_p: float, temperature: float) -> torch.Tensor:
    if not do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)

    scores = logits / max(temperature, 1e-5)

    if top_k > 0:
        topk_vals, topk_idx = torch.topk(scores, k=min(top_k, scores.shape[-1]), dim=-1)
        masked = torch.full_like(scores, -float("inf"))
        masked.scatter_(1, topk_idx, topk_vals)
        scores = masked

    if top_p < 1.0:
        sorted_scores, sorted_idx = torch.sort(scores, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        sorted_scores = sorted_scores.masked_fill(remove, -float("inf"))
        scores = torch.full_like(scores, -float("inf"))
        scores.scatter_(1, sorted_idx, sorted_scores)

    probs = torch.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class AsyncTTSGenerator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.text_queue: asyncio.Queue[str] = asyncio.Queue()
        self.audio_queue: asyncio.Queue[tp.Optional[torch.Tensor]] = asyncio.Queue()
        self.generation_task: tp.Optional[asyncio.Task] = None
        self.finished = False
        self.profile: dict[str, float] = {}
        self.request_start_t: tp.Optional[float] = None

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this script.")
        self._configure_cuda_math()
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
        self.tts, self.config = self._build_tts(Path(args.model_dir))

        self.codec_eos_id = self.config.talker_config.codec_eos_token_id
        self.codebook_size = int(self.tts.model.speech_tokenizer.model.config.decoder_config.codebook_size)
        self.suppress_tokens = [
            i for i in range(self.codebook_size, self.config.talker_config.vocab_size) if i != self.codec_eos_id
        ]

        self._move_speech_tokenizer_to_cpu()
        prompt_items = self.tts.create_voice_clone_prompt(str(args.ref_audio), x_vector_only_mode=True)
        self.voice_clone_prompt = self.tts._prompt_items_to_voice_clone_prompt(prompt_items)
        self._move_speech_tokenizer_to_device(self.device)
        self._compile_code_predictor()
        self.sample_rate = args.sample_rate

    @staticmethod
    def _configure_cuda_math() -> None:
        torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True

    def _compile_code_predictor(self) -> None:
        inductor_config = getattr(torch, "_inductor", None)
        config_mod = getattr(inductor_config, "config", None)
        triton_cfg = getattr(config_mod, "triton", None)
        if triton_cfg is not None and hasattr(triton_cfg, "cudagraphs"):
            triton_cfg.cudagraphs = False
        code_predictor = self.tts.model.talker.code_predictor
        print(
            "[compile] code_predictor scope=model backend=inductor mode=default cudagraphs=off"
        )
        code_predictor.model = torch.compile(code_predictor.model, backend="inductor", mode="default")
        self._warmup_compiled_talker_step()

    def _warmup_compiled_talker_step(self) -> None:
        streamer = StreamingTextState(self.tts, holdback_tokens=self.args.token_holdback)
        streamer.push_text("Warm up.", final_chunk=True)
        input_ids = streamer.current_input_ids()
        talker_input_embeds, attention_mask, tts_pad_embed = build_talker_inputs_xvector(
            tts_model=self.tts,
            input_id=input_ids,
            voice_clone_prompt=self.voice_clone_prompt,
            language=self.args.language,
        )
        trailing_text_hidden = streamer.trailing_text_hidden()

        with torch.inference_mode():
            prefill = self.tts.model.talker(
                inputs_embeds=talker_input_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
            )

            input_token = torch.tensor(
                [[self.config.talker_config.codec_bos_id]],
                device=self.tts.model.device,
                dtype=attention_mask.dtype,
            )
            cache_position = torch.tensor([attention_mask.shape[1]], device=attention_mask.device, dtype=torch.long)
            step_attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype)],
                dim=1,
            )
            self.tts.model.talker(
                input_ids=input_token,
                attention_mask=step_attention_mask,
                past_key_values=prefill.past_key_values,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
                past_hidden=prefill.past_hidden,
                generation_step=prefill.generation_step,
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
                cache_position=cache_position,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _move_speech_tokenizer_to_device(self, device: torch.device) -> None:
        speech_tokenizer = getattr(self.tts.model, "speech_tokenizer", None)
        if speech_tokenizer is None:
            return
        if hasattr(speech_tokenizer, "device"):
            speech_tokenizer.device = device
        if hasattr(speech_tokenizer, "to"):
            speech_tokenizer.to(device)
        submodel = getattr(speech_tokenizer, "model", None)
        if submodel is not None and hasattr(submodel, "to"):
            submodel.to(device)
            if hasattr(submodel, "eval"):
                submodel.eval()

    def _move_speech_tokenizer_to_cpu(self) -> None:
        self._move_speech_tokenizer_to_device(torch.device("cpu"))

    def _build_tts(self, model_dir: Path) -> tuple[Qwen3TTSModel, tp.Any]:
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        if not Path(self.args.ref_audio).exists():
            raise FileNotFoundError(f"Reference audio not found: {self.args.ref_audio}")

        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
        AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

        load_kwargs = {
            "local_files_only": True,
            "use_safetensors": True,
            "dtype": self.dtype,
        }
        model = AutoModel.from_pretrained(model_dir, **load_kwargs)
        model = model.to(self.device)
        model.eval()

        processor = AutoProcessor.from_pretrained(
            model_dir,
            fix_mistral_regex=True,
            local_files_only=True,
        )
        tts = Qwen3TTSModel(model, processor, model.generate_config)
        return tts, tts.model.config

    async def start(self) -> None:
        if self.generation_task and not self.generation_task.done():
            return
        self.generation_task = asyncio.create_task(self._generation_loop())

    async def add_text(self, text: str) -> None:
        normalized = text.strip()
        if normalized:
            if self.request_start_t is None:
                self.request_start_t = time.perf_counter()
            await self.text_queue.put(normalized)

    async def finish(self) -> None:
        self.finished = True

    async def get_audio_chunk(self) -> tp.Optional[torch.Tensor]:
        return await self.audio_queue.get()

    async def _generation_loop(self) -> None:
        try:
            while self.text_queue.empty():
                if self.finished:
                    raise RuntimeError("Generation finished before any text chunk was received.")
                await asyncio.sleep(self.args.poll_interval)
            request_start_t = self.request_start_t if self.request_start_t is not None else time.perf_counter()
            first_chunk = self.text_queue.get_nowait()
            streamer = StreamingTextState(self.tts, holdback_tokens=self.args.token_holdback)
            prefill_start_t = time.perf_counter()
            first_codec_step_t: tp.Optional[float] = None
            first_decode_start_t: tp.Optional[float] = None
            first_decode_end_t: tp.Optional[float] = None
            first_audio_emit_t: tp.Optional[float] = None
            decode_total_s = 0.0
            frame_milestones = (1, 2, 4, 8, 12)
            frame_times: dict[int, float] = {}

            first_chunk_is_final = self.finished and self.text_queue.empty()
            streamer.push_text(first_chunk, final_chunk=first_chunk_is_final)

            input_ids = streamer.current_input_ids()
            talker_input_embeds, attention_mask, tts_pad_embed = build_talker_inputs_xvector(
                tts_model=self.tts,
                input_id=input_ids,
                voice_clone_prompt=self.voice_clone_prompt,
                language=self.args.language,
            )
            trailing_text_hidden = streamer.trailing_text_hidden()

            prefill = self._talker_call(
                inputs_embeds=talker_input_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
            )
            prefill_end_t = time.perf_counter()

            past_key_values = prefill.past_key_values
            past_hidden = prefill.past_hidden
            generation_step = prefill.generation_step
            input_token = torch.tensor(
                [[self.config.talker_config.codec_bos_id]],
                device=self.tts.model.device,
                dtype=attention_mask.dtype,
            )

            prefill_len = int(attention_mask.shape[1])
            full_seq_len = prefill_len + self.args.max_tokens
            attention_mask_full = torch.zeros((1, full_seq_len), device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask_full[:, :prefill_len] = 1
            cache_position = torch.empty((1,), device=attention_mask.device, dtype=torch.long)

            codec_buffer: list[torch.Tensor] = []
            emitted_chunk_count = 0
            finalized_text = first_chunk_is_final
            step_count = 0
            talker_step_s = 0.0

            for _ in range(self.args.max_tokens):
                while generation_step >= trailing_text_hidden.shape[1] and (not self.finished or not self.text_queue.empty()):
                    pushed = False
                    while not self.text_queue.empty():
                        chunk = self.text_queue.get_nowait()
                        is_final = self.finished and self.text_queue.empty()
                        new_hidden = streamer.push_text(chunk, final_chunk=is_final)
                        if new_hidden is not None:
                            trailing_text_hidden = torch.cat([trailing_text_hidden, new_hidden], dim=1)
                        pushed = True
                        finalized_text = finalized_text or is_final

                    if not pushed and self.finished and not finalized_text:
                        new_hidden = streamer.push_text("", final_chunk=True)
                        if new_hidden is not None:
                            trailing_text_hidden = torch.cat([trailing_text_hidden, new_hidden], dim=1)
                        finalized_text = True
                        pushed = True

                    if pushed:
                        continue
                    await asyncio.sleep(self.args.poll_interval)

                past_len = prefill_len + generation_step
                attention_mask_full[:, past_len] = 1
                step_attention_mask = attention_mask_full[:, : past_len + 1]
                cache_position.fill_(past_len)

                t_step0 = time.perf_counter()
                outputs = self._talker_call(
                    input_ids=input_token,
                    attention_mask=step_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=False,
                    return_dict=True,
                    past_hidden=past_hidden,
                    generation_step=generation_step,
                    trailing_text_hidden=trailing_text_hidden,
                    tts_pad_embed=tts_pad_embed,
                    cache_position=cache_position,
                )
                talker_step_s += time.perf_counter() - t_step0

                logits = outputs.logits[:, -1, :]
                if self.suppress_tokens:
                    logits = logits.clone()
                    logits[:, self.suppress_tokens] = -float("inf")
                input_token = sample_next_token(
                    logits,
                    do_sample=self.args.do_sample,
                    top_k=self.args.top_k,
                    top_p=self.args.top_p,
                    temperature=self.args.temperature,
                )

                if outputs.hidden_states is None or outputs.hidden_states[1] is None:
                    raise RuntimeError("Missing codec ids in talker output.")
                codec_ids = outputs.hidden_states[1].squeeze(0)
                if (codec_ids < 0).any() or (codec_ids >= self.codebook_size).any():
                    codec_ids = codec_ids.clamp(0, self.codebook_size - 1)
                codec_buffer.append(codec_ids)
                if first_codec_step_t is None:
                    first_codec_step_t = time.perf_counter()
                buffered_frames = len(codec_buffer)
                for milestone in frame_milestones:
                    if buffered_frames >= milestone and milestone not in frame_times:
                        frame_times[milestone] = time.perf_counter() - request_start_t

                past_key_values = outputs.past_key_values
                past_hidden = outputs.past_hidden
                generation_step = outputs.generation_step
                step_count += 1
                if input_token.item() == self.codec_eos_id:
                    break

                chunk_target_tokens = self.args.first_chunk_tokens if emitted_chunk_count == 0 else self.args.stream_chunk_tokens
                if len(codec_buffer) >= chunk_target_tokens:
                    decode_t0 = time.perf_counter()
                    if first_decode_start_t is None:
                        first_decode_start_t = decode_t0
                    await self._decode_and_emit(codec_buffer)
                    decode_t1 = time.perf_counter()
                    decode_total_s += decode_t1 - decode_t0
                    if first_decode_end_t is None:
                        first_decode_end_t = decode_t1
                    if first_audio_emit_t is None:
                        first_audio_emit_t = decode_t1
                    codec_buffer = []
                    emitted_chunk_count += 1
                    await asyncio.sleep(0)

            if codec_buffer:
                decode_t0 = time.perf_counter()
                if first_decode_start_t is None:
                    first_decode_start_t = decode_t0
                await self._decode_and_emit(codec_buffer)
                decode_t1 = time.perf_counter()
                decode_total_s += decode_t1 - decode_t0
                if first_decode_end_t is None:
                    first_decode_end_t = decode_t1
                if first_audio_emit_t is None:
                    first_audio_emit_t = decode_t1

            profile: dict[str, float] = {
                "steps": float(step_count),
                "talker_step_s": talker_step_s,
                "talker_tok_s": step_count / max(talker_step_s, 1e-6),
                "prefill_s": prefill_end_t - prefill_start_t,
                "prefill_from_request_s": prefill_end_t - request_start_t,
                "decode_total_s": decode_total_s,
            }
            if first_codec_step_t is not None:
                profile["first_codec_frame_s"] = first_codec_step_t - request_start_t
            if first_decode_start_t is not None:
                profile["first_decode_start_s"] = first_decode_start_t - request_start_t
            if first_decode_start_t is not None and first_decode_end_t is not None:
                profile["first_decode_s"] = first_decode_end_t - first_decode_start_t
            if first_audio_emit_t is not None:
                profile["first_audio_emit_s"] = first_audio_emit_t - request_start_t
            for milestone, dt in frame_times.items():
                profile[f"frames_{milestone}_s"] = dt

            self.profile = profile
            print(
                f"[profile] steps={step_count} prefill_s={profile['prefill_s']:.3f} "
                f"prefill_from_request_s={profile['prefill_from_request_s']:.3f} "
                f"talker_step_s={talker_step_s:.3f} talker_tok_s={profile['talker_tok_s']:.1f} "
                f"first_codec_frame_s={profile.get('first_codec_frame_s', float('nan')):.3f} "
                f"frames_4_s={profile.get('frames_4_s', float('nan')):.3f} "
                f"frames_8_s={profile.get('frames_8_s', float('nan')):.3f} "
                f"frames_12_s={profile.get('frames_12_s', float('nan')):.3f} "
                f"first_decode_start_s={profile.get('first_decode_start_s', float('nan')):.3f} "
                f"first_decode_s={profile.get('first_decode_s', float('nan')):.3f} "
                f"first_audio_emit_s={profile.get('first_audio_emit_s', float('nan')):.3f} "
                f"decode_total_s={profile['decode_total_s']:.3f}"
            )
        finally:
            await self.audio_queue.put(None)

    async def _decode_and_emit(self, codec_buffer: list[torch.Tensor]) -> None:
        codes_tensor = torch.stack(codec_buffer, dim=0).to(self.device)
        speech_tokenizer = self.tts.model.speech_tokenizer
        if speech_tokenizer is None:
            raise RuntimeError("Speech tokenizer is not loaded.")
        with torch.inference_mode():
            wavs, sample_rate = speech_tokenizer.decode([{"audio_codes": codes_tensor}])
        self.sample_rate = int(sample_rate)
        audio_np = np.asarray(wavs[0], dtype=np.float32)
        audio_torch = torch.from_numpy(audio_np)
        await self.audio_queue.put(audio_torch)
        await asyncio.sleep(0)

    def _talker_call(self, **kwargs):
        with torch.inference_mode():
            return self.tts.model.talker(**kwargs)


async def run_demo(args: argparse.Namespace) -> None:
    t_init0 = time.perf_counter()
    gen = AsyncTTSGenerator(args)
    t_init1 = time.perf_counter()
    await gen.start()

    chunks = split_text_for_stream(args.text, target_chars=args.text_chunk_target_chars)
    if not chunks:
        raise ValueError("Text is empty after chunking.")

    t0 = time.perf_counter()
    for chunk in chunks:
        await gen.add_text(chunk)
    await gen.finish()

    audio_chunks: list[np.ndarray] = []
    while True:
        audio = await gen.get_audio_chunk()
        if audio is None:
            break
        audio_chunks.append(audio.cpu().numpy())

    if not audio_chunks:
        print("No audio chunks produced.")
        return

    final_audio = np.concatenate(audio_chunks, axis=0)
    output_path = Path(args.out_wav)
    sf.write(str(output_path), final_audio, gen.sample_rate)
    generation_t = time.perf_counter() - t0
    init_t = t_init1 - t_init0
    audio_s = final_audio.shape[0] / max(gen.sample_rate, 1)
    rtf = generation_t / max(audio_s, 1e-6)
    print(
        f"Wrote {output_path} (samples={final_audio.shape[0]}, sr={gen.sample_rate}, chunks={len(audio_chunks)}, "
        f"generation_t={generation_t:.2f}s, init_t={init_t:.2f}s, rtf={rtf:.3f})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-streaming local benchmark for this Qwen3-TTS fork.")
    parser.add_argument("--model-dir", default="./Qwen3-TTS-12Hz-0.6B-Base", help="Local model directory.")
    parser.add_argument("--ref-audio", default="./sample.wav", help="Reference wav for voice cloning.")
    parser.add_argument("--text", default="Hello from the optimized Qwen3-TTS streaming fork.", help="Input text.")
    parser.add_argument("--language", default="English", help="Language name, e.g. English/German/Auto.")
    parser.add_argument("--out-wav", default="./output_full_streaming.wav", help="Output wav path.")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16", help="Compute dtype.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--first-chunk-tokens", type=int, default=12)
    parser.add_argument("--stream-chunk-tokens", type=int, default=24)
    parser.add_argument("--token-holdback", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--do-sample", action="store_true", default=True)
    parser.add_argument("--no-sample", dest="do_sample", action="store_false")
    parser.add_argument("--text-chunk-target-chars", type=int, default=26)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--poll-interval", type=float, default=0.005)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_demo(parse_args()))
