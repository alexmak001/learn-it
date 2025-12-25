# Shortform Studio (Duo Mode Shorts)

This Streamlit app turns a short topic into a Duo Mode dialogue (CARTOON_DAD + JOHN), generates TTS audio per line, and renders a vertical Shorts-style video with animated captions and character overlays.

<p align="center">
<img src="pics/example_vid.gif" alt="Shortform example" width="30%">
</p>

## How It Works

1. **Dialogue**: `backend/ai_service.py` produces a 3-turn JSON dialogue (Dad, John, Dad).
2. **TTS + Timing**: `backend/tts_service.py` generates MP3 chunks per line and computes line durations.
3. **Stitching**: Audio chunks are stitched into `temp/duo_audio.mp3`.
4. **Video Render**: `backend/shorts_renderer.py` composites:
   - A selected brainrot background segment.
   - Speaker images (left/right).
   - Karaoke-style 5-word caption chunks with bounce and speaker tags.
5. **Streamlit UI**: `app.py` wires the flow and lets you render/download the final MP4.

## Assets

Place these in `temp/`:

- Background videos: `temp/brainRotVideos/*.mp4`
- John image: `temp/john_character_cutout.png`
- Cartoon Dad image: `temp/cartoon_dad_transparent.png`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with:

```
OPENAI_API_KEY=...
ELEVENLABS_API_KEY=...
VOICE_ID_JOHN=...
VOICE_ID_CARTOON_DAD=...
```

Optional:

```
OPENAI_MODEL=gpt-5-nano-2025-08-07
ELEVENLABS_MODEL_ID=eleven_multilingual_v2
ELEVENLABS_OUTPUT_FORMAT=mp3_44100_128
```

## Run

```bash
streamlit run app.py
```

## Using the App

1. Load/generate your dialogue + audio (the sample in `app.py` currently uses a hardcoded example).
2. Choose a background from the dropdown (reads `temp/brainRotVideos/*.mp4`).
3. Click **Render Shorts Video**.
4. Download the MP4 from the UI.

Outputs:

- Stitched audio: `temp/duo_audio.mp3`
- Rendered video: `temp/output_short.mp4`

## Notes

- Captions are rendered with PIL to avoid clipping.
- A random segment of the chosen background video is used for each render.
- Captions bounce at the start of each 5-word chunk; characters slide in per line.
