"""Text-to-speech via local XTTS-v2 (Coqui AI)."""
from io import BytesIO
import os

import soundfile as sf
import torch
from TTS.api import TTS


def _select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


_MODEL_NAME = os.getenv("TTS_MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2")
_DEVICE = _select_device()

# XTTS supports multiple reference voices; default to a lightweight multilingual voice.
_DEFAULT_SPEAKER = os.getenv("TTS_SPEAKER", "female-en-5")
_DEFAULT_LANGUAGE = os.getenv("TTS_LANGUAGE", "en")

_tts_model = TTS(model_name=_MODEL_NAME, progress_bar=False, gpu=False)
if _DEVICE != "cpu":
    _tts_model.to(_DEVICE)


def speak_text(text: str, speaker_wav: str | None = None, speaker: str | None = None, language: str | None = None) -> BytesIO:
    """Generate speech audio and return it as a WAV BytesIO buffer."""
    selected_language = language or _DEFAULT_LANGUAGE
    selected_speaker = speaker or _DEFAULT_SPEAKER

    audio = _tts_model.tts(
        text=text,
        speaker_wav=speaker_wav,
        speaker=selected_speaker,
        language=selected_language,
    )

    buffer = BytesIO()
    sf.write(buffer, audio, samplerate=_tts_model.synthesizer.output_sample_rate, format="WAV")
    buffer.seek(0)
    return buffer
