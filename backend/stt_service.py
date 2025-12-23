"""Speech-to-text powered by local Whisper small model."""
import logging
import os
from typing import Optional

from dotenv import load_dotenv
import torch

try:
    import whisper
except ImportError as exc:  # pragma: no cover - fail fast for missing dependency
    raise ImportError(
        "openai-whisper must be installed to run the speech-to-text service. "
        "Install it with 'pip install openai-whisper'."
    ) from exc


load_dotenv()


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # Metal backend tends to produce NaNs on Whisper due to half precision,
    # so only opt into it when explicitly requested.
    if torch.backends.mps.is_available() and os.getenv("WHISPER_ALLOW_MPS") == "1":
        return "mps"
    return "cpu"


_MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
_DEVICE = _select_device()
_model = whisper.load_model(_MODEL_NAME, device=_DEVICE)


def transcribe_audio(audio_path: str, logger: Optional[logging.Logger] = None) -> str:
    """Transcribe audio from disk and log the request lifecycle."""
    active_logger = logger or logging.getLogger(__name__)
    active_logger.info("Starting transcription for %s", audio_path)

    result = _model.transcribe(audio_path)
    text = result["text"].strip()

    active_logger.info("Transcription complete: %s", text)
    return text
