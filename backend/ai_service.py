"""AI service powered by OpenAI GPT models."""
import os

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError


load_dotenv()

_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not _OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment/.env file.")

_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5-nano")
_client = OpenAI(api_key=_OPENAI_API_KEY)


def _call_openai(prompt: str) -> str:
    try:
        response = _client.responses.create(
            model=_MODEL_NAME,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a friendly AI tutor who explains topics clearly "
                        "and concisely for beginners."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=220
        )
    except OpenAIError as exc:  # pragma: no cover - depends on API availability
        raise RuntimeError(f"OpenAI API call failed: {exc}") from exc

    content = response.output[0].content[0].text if response.output else ""
    if not content:
        raise RuntimeError("OpenAI API returned an empty response.")
    return content.strip()


def generate_topic_explanation(topic: str) -> str:
    prompt = (
        f"Explain the topic '{topic}' in under 120 words using approachable language. "
        "Focus on clarity and practical intuition."
    )

    response = _call_openai(prompt)

    return response
