import logging
import os
from datetime import datetime
from typing import Tuple

import streamlit as st
from backend.ai_service import generate_dialogue
from backend.stt_service import transcribe_audio
from backend.tts_service import speak_text, stitch_mp3_chunks, voice_id_for


def _setup_logger(base_dir: str) -> Tuple[logging.Logger, str]:
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(base_dir, f"duo_mode_{timestamp}.log")

    logger = logging.getLogger("duo_mode_app")
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    logger.info("Logger initialized; writing to %s", log_path)
    return logger, log_path


st.title("🎭 Duo Mode Voice Tutor")

temp_dir = "logs"
audio_path = os.path.join(temp_dir, "recorded_audio.wav")
pause_between_lines_ms = 250
logger, log_path = _setup_logger(temp_dir)

st.caption("Record a question and let JOHN and CARTOON_DAD bring the lesson to life.")
st.caption(f"Session logs saved to `{log_path}`")

audio = st.audio_input("Upload or record your voice (topic request)")

if audio:
    st.write("Audio captured")
    st.audio(audio)
    logger.info("Audio clip received from user.")

    os.makedirs(temp_dir, exist_ok=True)
    with open(audio_path, "wb") as f:
        f.write(audio.getbuffer())
    logger.info("Audio saved to %s", audio_path)

    with st.status("Running Duo Mode...", expanded=True) as status:
        status.write("Transcribing your topic...")
        topic = transcribe_audio(audio_path, logger=logger)
        # topic = "decision trees and how they generate branches"
        st.write(f"Detected topic: **{topic}**")
        logger.info("Detected topic: %s", topic)

        status.update(label="Generating dialogue script...", state="running")
        dialogue = generate_dialogue(topic, logger=logger)
        status.write("Dialogue ready. Preview it below before synthesis.")

        st.subheader("Dialogue Script")
        for turn in dialogue:
            st.markdown(f"**{turn['speaker']}**: {turn['line']}")

        status.update(label="Synthesizing duo voices...", state="running")
        audio_chunks: list[bytes] = []
        for idx, turn in enumerate(dialogue, start=1):
            speaker_voice = voice_id_for(turn["speaker"])
            status.write(f"Generating line {idx} for {turn['speaker']}...")
            logger.info("Generating line %s for %s", idx, turn["speaker"])
            chunk = speak_text(turn["line"], voice_id=speaker_voice, logger=logger)
            audio_chunks.append(chunk)

        final_audio = stitch_mp3_chunks(
            audio_chunks,
            pause_ms=pause_between_lines_ms,
            logger=logger,
        )
        status.update(label="Duo Mode complete!", state="complete")
        logger.info("Dialogue audio stitched successfully.")

    st.audio(final_audio, format="audio/mp3")
    st.download_button(
        "Download Duo Dialogue",
        data=final_audio,
        file_name="duo-mode-dialogue.mp3",
        mime="audio/mpeg",
    )
