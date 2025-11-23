"""Text-to-speech via the Hugging Face Inference API."""

from __future__ import annotations

from io import BytesIO
import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


load_dotenv()

_HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
if not _HF_TOKEN:
    raise RuntimeError("Missing HUGGINGFACE_HUB_TOKEN in environment/.env file.")

_MODEL_NAME = os.getenv("TTS_MODEL_NAME", "parler-tts/parler-tts-mini-v1")
_DEFAULT_SPEAKER = os.getenv("TTS_SPEAKER")
_DEFAULT_LANGUAGE = os.getenv("TTS_LANGUAGE")

_client = InferenceClient(token=_HF_TOKEN)


def _build_generation_kwargs(speaker: str | None, language: str | None) -> dict[str, str]:
    """Translate generic speaker/language hints to the HF API kwargs."""
    generation_kwargs: dict[str, str] = {}
    if speaker:
        generation_kwargs["voice"] = speaker
    elif _DEFAULT_SPEAKER:
        generation_kwargs["voice"] = _DEFAULT_SPEAKER

    if language:
        generation_kwargs["language"] = language
    elif _DEFAULT_LANGUAGE:
        generation_kwargs["language"] = _DEFAULT_LANGUAGE

    return generation_kwargs


def speak_text(
    text: str,
    speaker_wav: str | None = None,
    speaker: str | None = None,
    language: str | None = None,
) -> BytesIO:
    """Generate speech audio using Hugging Face-hosted models."""
    if speaker_wav:
        raise NotImplementedError("Voice cloning via speaker_wav is not supported with the HF Inference API.")

    generation_kwargs = _build_generation_kwargs(speaker, language)
    audio_bytes = _client.text_to_speech(text=text, model=_MODEL_NAME, **generation_kwargs)

    buffer = BytesIO(audio_bytes)
    buffer.seek(0)
    return buffer
