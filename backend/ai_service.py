"""Local LLM explanations using Llama 3.1 8B Instruct."""
import os

from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


load_dotenv()

_HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
if not _HF_TOKEN:
    raise RuntimeError("Missing HUGGINGFACE_HUB_TOKEN in environment/.env file.")


def _select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


_DEVICE = _select_device()
_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
_DTYPE = torch.float16 if _DEVICE != "cpu" else torch.float32

_tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME, token=_HF_TOKEN)
_model = AutoModelForCausalLM.from_pretrained(
    _MODEL_NAME, torch_dtype=_DTYPE, device_map="auto", token=_HF_TOKEN
)
_generation_pipeline = pipeline(
    "text-generation",
    model=_model,
    tokenizer=_tokenizer,
    device_map="auto",
)


def generate_topic_explanation(topic: str) -> str:
    prompt = (
        "You are a friendly AI tutor. Provide a concise, approachable "
        f"explanation of the topic '{topic}' for a beginner. Keep it under 120 words."
    )

    outputs = _generation_pipeline(
        prompt,
        max_new_tokens=180,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        return_full_text=False,
        eos_token_id=_tokenizer.eos_token_id,
    )
    return outputs[0]["generated_text"].strip()
