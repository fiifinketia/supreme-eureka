from __future__ import annotations

import asyncio
import os
import weakref
import torch
from dataclasses import dataclass, fields

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    tokenize,
    tts,
    utils,
)
import logging
from typing import Literal, Any
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# TTSModel = Literal["", "PlayDialog"]

NUM_CHANNELS = 1
logger = logging.getLogger("livekit.plugins.konkonsa")


@dataclass
class _Options:
    model:  str
    word_tokenizer: tokenize.WordTokenizer
    sample_rate: int


class CoquiTTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str = "s3://voice-cloning-zero-shot/d9ff78ba-d016-47f6-b0ef-dd630f59414e/female-cs/manifest.json",
        language: str = "en",
        sample_rate: int = 16000,
        model: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        word_tokenizer: tokenize.WordTokenizer = tokenize.basic.WordTokenizer(
            ignore_punctuation=False
        ),
        **kwargs,
    ) -> None:
        """
        Initialize the PlayAI TTS engine.

        Args:
            api_key (str): PlayAI API key.
            user_id (str): PlayAI user ID.
            voice (str): Voice manifest URL.
            model (TTSModel): TTS model, defaults to "Play3.0-mini".
            language (str): language, defaults to "english".
            sample_rate (int): sample rate (Hz), A number greater than or equal to 8000, and must be less than or equal to 48000
            word_tokenizer (tokenize.WordTokenizer): Tokenizer for processing text. Defaults to basic WordTokenizer.
            **kwargs: Additional options.
        """

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
            ),
            sample_rate=sample_rate,
            num_channels=1,
        )

        self._opts = _Options(
            model=model,
            word_tokenizer=word_tokenizer,
            sample_rate=16000
        )

        # model_storage_path = os.getenv['MODEL_STORAGE_PATH']

        config = XttsConfig()
        model_path = "/home/mlab/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/model.pth"
        config_path = "/home/mlab/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/config.json"
        vocab_path = "/home/mlab/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/vocab.json"
        config.load_json(config_path)

        self._model: Xtts = Xtts.init_from_config(config)
        self._model.load_checkpoint(config, vocab_path=vocab_path, checkpoint_path=model_path, eval=True, use_deepspeed=False)
        if torch.cuda.is_available():
            self._model.cuda()

        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._language = language

    def update_options(
        self,
        *,
        voice: str | None = None,
        model: str | None = None,
        language: str | None = None,
        **kwargs,
    ) -> None:
        """
        Update the TTS options.
        """
        updates = {}
        if voice is not None:
            updates["voice"] = voice
        if language is not None:
            updates["language"] = language
        tts_kwargs = {k: v for k, v in kwargs.items()}

        if model is not None:
            self._opts.model = model

        for stream in self._streams:
            if model is not None:
                stream._opts.model = model

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "ChunkedStream":
        return ChunkedStream(
            ctts=self,
            input_text=text,
            conn_options=conn_options,
            opts=self._opts,
        )

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "SynthesizeStream":
        stream = SynthesizeStream(
            ctts=self,
            conn_options=conn_options,
            opts=self._opts,
        )
        self._streams.add(stream)
        return stream


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        ctts: CoquiTTS,
        input_text: str,
        conn_options: APIConnectOptions,
        opts: _Options,
    ) -> None:
        super().__init__(tts=ctts, input_text=input_text, conn_options=conn_options)
        self._model = ctts._model
        self.language = ctts._language        
        self._opts = opts
        self._mp3_decoder = utils.codecs.Mp3StreamDecoder()

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        bstream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate, num_channels=NUM_CHANNELS
        )

        try:
            gpt_cond_latent, speaker_embedding = self._model.get_conditioning_latents(
                audio_path=["audio.mp3"]
            )
            for i,chunk in enumerate(self._model.inference_stream(
                self._input_text,
                self.language,
                gpt_cond_latent,
                speaker_embedding,
                enable_text_splitting=True
            )):
                for frame in bstream.write(chunk.numpy().tobytes()):
                    self._event_ch.send_nowait(
                        tts.SynthesizedAudio(
                            request_id=request_id,
                            frame=frame,
                        )
                    )
            for frame in bstream.flush():
                self._event_ch.send_nowait(
                    tts.SynthesizedAudio(request_id=request_id, frame=frame)
                )
        except Exception as e:
            raise Exception(e) from e


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        *,
        ctts: CoquiTTS,
        conn_options: APIConnectOptions,
        opts: _Options,
    ):
        super().__init__(tts=ctts, conn_options=conn_options)
        self._model = ctts._model
        self.language = ctts._language
        self._opts = opts
        self._segments_ch = utils.aio.Chan[tokenize.WordStream]()
        self._mp3_decoder = utils.codecs.Mp3StreamDecoder()

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        segment_id = utils.shortuuid()
        bstream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
        )
        last_frame: rtc.AudioFrame | None = None

        def _send_last_frame(*, segment_id: str, is_final: bool) -> None:
            nonlocal last_frame
            if last_frame is not None:
                self._event_ch.send_nowait(
                    tts.SynthesizedAudio(
                        request_id=request_id,
                        segment_id=segment_id,
                        frame=last_frame,
                        is_final=is_final,
                    )
                )
                last_frame = None

        input_task = asyncio.create_task(self._tokenize_input())

        try:
            text_stream = await self._create_text_stream()
            gpt_cond_latent, speaker_embedding = self._model.get_conditioning_latents(
                audio_path=["audio.mp3"]
            )
            async for text in text_stream:
                for chunk in self._model.inference_stream(
                    text,
                    self.language,
                    gpt_cond_latent,
                    speaker_embedding,
                    enable_text_splitting=True
                ):
                    for frame in bstream.write(chunk.numpy().tobytes()):
                        _send_last_frame(segment_id=segment_id, is_final=False)
                        last_frame = frame

            for frame in bstream.flush():
                _send_last_frame(segment_id=segment_id, is_final=False)
                last_frame = frame
            _send_last_frame(segment_id=segment_id, is_final=True)

        except Exception as e:
            raise Exception(e) from e
        finally:
            await utils.aio.gracefully_cancel(input_task)

    @utils.log_exceptions(logger=logger)
    async def _tokenize_input(self):
        # Converts incoming text into WordStreams and sends them into _segments_ch
        word_stream = None
        async for input_i in self._input_ch:
            if isinstance(input_i, str):
                if word_stream is None:
                    word_stream = self._opts.word_tokenizer.stream()
                    self._segments_ch.send_nowait(word_stream)
                word_stream.push_text(input_i)
            elif isinstance(input_i, self._FlushSentinel):
                if word_stream:
                    word_stream.end_input()
                word_stream = None
        self._segments_ch.close()

    @utils.log_exceptions(logger=logger)
    async def _create_text_stream(self):
        async def text_stream():
            async for word_stream in self._segments_ch:
                async for word in word_stream:
                    yield word.token

        return text_stream()
