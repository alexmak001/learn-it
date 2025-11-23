"""Speech-to-text powered by local Whisper small model."""
import os

import torch
import whisper


def _select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


_MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
_DEVICE = _select_device()
_model = whisper.load_model(_MODEL_NAME, device=_DEVICE)


def transcribe_audio(audio_path: str) -> str:
    result = _model.transcribe(audio_path)
    return result["text"].strip()
