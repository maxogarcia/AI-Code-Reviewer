"""
Lazy-loaded inference engine: QLoRA fine-tuned Qwen2.5-Coder-7B-Instruct.

The model is loaded on first call to `generate()` so the FastAPI process
starts quickly and GPU memory is only allocated when needed.
"""

from __future__ import annotations

import threading
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
ADAPTER_DIR = str(Path(__file__).parent.parent / "outputs" / "qlora-adapter")

SYSTEM_PROMPT = (
    "You are an expert code reviewer. "
    "Analyze the submitted code for bugs, security issues, and style problems. "
    "Give clear, actionable feedback."
)


def _bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


class InferenceEngine:
    def __init__(self):
        self._model: PeftModel | None = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=_bnb_config(),
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base, ADAPTER_DIR)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model

    def _ensure_loaded(self) -> None:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._load()

    def generate(self, code: str, context: str = "", max_new_tokens: int = 512) -> str:
        self._ensure_loaded()

        user_content = code
        if context:
            user_content = (
                f"Here are similar code review examples for reference:\n\n{context}\n\n"
                f"---\n\nNow review the following code:\n\n{code}"
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        encoded = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # Newer transformers returns BatchEncoding instead of a raw tensor
        input_ids = (encoded if isinstance(encoded, torch.Tensor) else encoded["input_ids"]).to(self._model.device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# Module-level singleton — shared across all requests.
engine = InferenceEngine()
