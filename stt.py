from __future__ import annotations

import dataclasses
import os
import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    stt,
)
from livekit.agents.stt import SpeechEventType, STTCapabilities
from livekit.agents.utils import AudioBuffer


@dataclass
class _STTOptions:
    language: str
    task: str
    chunk_level: str
    version: str


class WizperSTT(stt.STT):

    def __init__(
        self,
        *,
        model: str = "openai/whisper-medium",
        language: Optional[str] = "en",
        task: Optional[str] = "transcribe",
        chunk_level: Optional[str] = "segment",
        version: Optional[str] = "3",
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            capabilities=STTCapabilities(streaming=False, interim_results=True)
        )
        self._opts = _STTOptions(
            language=language or "en",
            task=task or "transcribe",
            chunk_level=chunk_level or "segment",
            version=version or "3",
        )
        self._device = device
        self._torch_dtype = torch_dtype
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self._model.to(device)

        self._processor = AutoProcessor.from_pretrained(model)

    def update_options(self, *, language: Optional[str] = None) -> None:
        self._opts.language = language or self._opts.language

    def _sanitize_options(
        self,
        *,
        language: Optional[str] = None,
        task: Optional[str] = None,
        chunk_level: Optional[str] = None,
        version: Optional[str] = None,
    ) -> _STTOptions:
        config = dataclasses.replace(self._opts)
        config.language = language or config.language
        config.task = task or config.task
        config.chunk_level = chunk_level or config.chunk_level
        config.version = version or config.version
        return config

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        try:
            config = self._sanitize_options(language=language)
            text = await self._run_sst(buffer=buffer)
            return self._transcription_to_speech_event(text=text)
        except Exception as e:
            raise Exception(e) from e

    async def _run_sst(
        self,
        buffer: AudioBuffer,
    ):

        pipe = pipeline(
            "automatic-speech-recognition",
            model=self._model,
            tokenizer=self._processor.tokenizer,
            feature_extractor=self._processor.feature_extractor,
            torch_dtype=self._torch_dtype,
            device=self._device,
        )

        text  = pipe(np.frombuffer(buffer.to_wav_bytes(), dtype=np.int16))['text']
        # text = "This is a test speech recognition"
        print("stt %s" % text)

        return text

    def _transcription_to_speech_event(
        self, event_type=SpeechEventType.FINAL_TRANSCRIPT, text=None
    ) -> stt.SpeechEvent:
        return stt.SpeechEvent(
            type=event_type,
            alternatives=[stt.SpeechData(text=text, language=self._opts.language)],
        )

    async def aclose(self) -> None:
        pass
