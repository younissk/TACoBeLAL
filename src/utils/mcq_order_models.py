"""Reusable model interfaces for MCQ-ORDER benchmarks."""

from __future__ import annotations

import json
import os
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

TASK_ID_MCQ_ORDER = "MCQ-ORDER"
DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LLAMA_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


@dataclass(frozen=True)
class MCQOrderOption:
    label: str
    text: str
    option_type: str
    event_index: int | None


@dataclass(frozen=True)
class MCQOrderExample:
    example_id: str
    task_id: str
    audio_filename: str
    question: str
    options: tuple[MCQOrderOption, ...]
    answer_label: str
    answer_text: str
    raw: Mapping[str, Any]

    @classmethod
    def from_json(cls, payload: Mapping[str, Any], task_id: str = TASK_ID_MCQ_ORDER) -> "MCQOrderExample":
        options_payload = payload.get("options")
        if not isinstance(options_payload, list) or not options_payload:
            raise ValueError("Each example must contain a non-empty 'options' list.")

        options: list[MCQOrderOption] = []
        for option in options_payload:
            label = option.get("label")
            text = option.get("text")
            option_type = option.get("type")
            if not isinstance(label, str) or not label:
                raise ValueError("Option is missing string 'label'.")
            if not isinstance(text, str) or not text:
                raise ValueError("Option is missing string 'text'.")
            if not isinstance(option_type, str) or not option_type:
                raise ValueError("Option is missing string 'type'.")

            event_index = option.get("event_index")
            if event_index is not None and not isinstance(event_index, int):
                raise ValueError("Option 'event_index' must be an integer or null.")

            options.append(
                MCQOrderOption(
                    label=label,
                    text=text,
                    option_type=option_type,
                    event_index=event_index,
                )
            )

        answer_label = payload.get("answer_label")
        answer_text = payload.get("answer_text")
        if not isinstance(answer_label, str) or not answer_label:
            raise ValueError("Example is missing string 'answer_label'.")
        if not isinstance(answer_text, str) or not answer_text:
            raise ValueError("Example is missing string 'answer_text'.")

        if answer_label not in {option.label for option in options}:
            raise ValueError("Example answer label does not exist in options.")

        example_id = payload.get("id")
        audio_filename = payload.get("audio_filename")
        question = payload.get("question")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError("Example is missing string 'id'.")
        if not isinstance(audio_filename, str) or not audio_filename:
            raise ValueError("Example is missing string 'audio_filename'.")
        if not isinstance(question, str) or not question:
            raise ValueError("Example is missing string 'question'.")

        return cls(
            example_id=example_id,
            task_id=task_id,
            audio_filename=audio_filename,
            question=question,
            options=tuple(options),
            answer_label=answer_label,
            answer_text=answer_text,
            raw=payload,
        )

    def option_by_label(self, label: str) -> MCQOrderOption:
        for option in self.options:
            if option.label == label:
                return option
        raise KeyError(f"Unknown option label: {label}")


class MCQOrderModel(ABC):
    """Base class for models that answer MCQ-ORDER questions."""

    model_name: str = "base"

    def __init__(self, *, seed: int | None = None) -> None:
        self.seed = seed

    @abstractmethod
    def predict(self, example: MCQOrderExample) -> str:
        """Return predicted option label."""

    def run_metadata(self) -> dict[str, Any]:
        return {"seed": self.seed}


def _extract_label_from_text(text: str, valid_labels: set[str]) -> str | None:
    candidate = text.strip()
    if candidate in valid_labels:
        return candidate

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            label = parsed.get("label")
            if isinstance(label, str):
                label = label.strip()
                if label in valid_labels:
                    return label
    except json.JSONDecodeError:
        pass

    for match in re.finditer(r"\b([A-Z]{1,3})\b", candidate):
        label = match.group(1)
        if label in valid_labels:
            return label
    return None


def _format_mcq_prompt(example: MCQOrderExample) -> str:
    options_block = "\n".join(f"{option.label}. {option.text}" for option in example.options)
    valid_labels = ", ".join(option.label for option in example.options)
    return (
        "You are evaluating a text-only MCQ question for task MCQ-ORDER.\n"
        "Choose exactly one answer option label.\n"
        "Return only the label (e.g., A). No explanation.\n\n"
        f"Question:\n{example.question}\n\n"
        f"Options:\n{options_block}\n\n"
        f"Valid labels: {valid_labels}\n"
    )


class RandomMCQOrderModel(MCQOrderModel):
    """Random baseline for MCQ-ORDER."""

    model_name = "random"

    def __init__(self, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._rng = random.Random(seed)

    def predict(self, example: MCQOrderExample) -> str:
        labels = [option.label for option in example.options]
        if not labels:
            raise ValueError("Cannot predict without options.")
        return self._rng.choice(labels)


class OpenAILangChainMCQOrderModel(MCQOrderModel):
    """LLM-only baseline using LangChain with OpenAI chat models."""

    model_name = "llm-openai"

    def __init__(
        self,
        *,
        seed: int | None = None,
        openai_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        prediction_retries: int = 2,
    ) -> None:
        super().__init__(seed=seed)
        self.openai_model = openai_model
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.prediction_retries = prediction_retries

        try:
            from dotenv import load_dotenv
        except ImportError as exc:  # pragma: no cover - import tested through integration
            raise RuntimeError(
                "Missing dependency 'python-dotenv'. Install with: uv sync --extra llm"
            ) from exc
        load_dotenv()

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Put it in environment or .env before running llm-openai."
            )

        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - import tested through integration
            raise RuntimeError(
                "Missing dependency 'langchain-openai'. Install with: uv sync --extra llm"
            ) from exc

        self._client = ChatOpenAI(
            model=openai_model,
            temperature=temperature,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    @staticmethod
    def _extract_label(text: str, valid_labels: set[str]) -> str | None:
        return _extract_label_from_text(text, valid_labels)

    @staticmethod
    def _response_to_text(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if isinstance(part, str):
                    chunks.append(part)
                elif isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
            return " ".join(chunks).strip()
        return str(content)

    @staticmethod
    def _format_prompt(example: MCQOrderExample) -> str:
        return _format_mcq_prompt(example)

    def predict(self, example: MCQOrderExample) -> str:
        valid_labels = {option.label for option in example.options}
        prompt = self._format_prompt(example)

        last_text = ""
        for _ in range(self.prediction_retries + 1):
            response = self._client.invoke(prompt)
            text = self._response_to_text(response)
            last_text = text
            label = self._extract_label(text, valid_labels)
            if label is not None:
                return label
            prompt = (
                f"{self._format_prompt(example)}\n"
                f"Your previous answer was invalid: {text!r}. "
                "Return only one valid label."
            )

        raise ValueError(
            "LLM response did not contain a valid label after retries. "
            f"Last response: {last_text!r}"
        )

    def run_metadata(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "provider": "openai",
            "openai_model": self.openai_model,
            "temperature": self.temperature,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "prediction_retries": self.prediction_retries,
        }


class LocalTransformersMCQOrderModel(MCQOrderModel):
    """Text-only local Hugging Face CausalLM baseline."""

    model_name = "llm-local"

    def __init__(
        self,
        *,
        seed: int | None = None,
        model_id: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: int = 16,
        prediction_retries: int = 2,
        device_map: str = "auto",
        dtype: str = "float16",
        trust_remote_code: bool = False,
        hf_token: str | None = None,
        provider_label: str = "huggingface",
    ) -> None:
        super().__init__(seed=seed)
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.prediction_retries = prediction_retries
        self.device_map = device_map
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.hf_token = hf_token
        self.provider_label = provider_label

        self._torch: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._inference_device: Any | None = None

    @staticmethod
    def _resolve_torch_dtype(torch_module: Any, dtype_name: str) -> Any:
        key = dtype_name.strip().lower()
        if key == "auto":
            return "auto"
        if key in {"fp16", "float16", "half"}:
            return torch_module.float16
        if key in {"bf16", "bfloat16"}:
            return torch_module.bfloat16
        if key in {"fp32", "float32"}:
            return torch_module.float32
        raise ValueError(f"Unsupported dtype '{dtype_name}'. Use one of: auto, float16, bfloat16, float32.")

    @staticmethod
    def _format_prompt(example: MCQOrderExample) -> str:
        return _format_mcq_prompt(example)

    @staticmethod
    def _extract_label(text: str, valid_labels: set[str]) -> str | None:
        return _extract_label_from_text(text, valid_labels)

    def _ensure_model_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None and self._torch is not None:
            return

        try:
            from dotenv import load_dotenv
        except ImportError:
            load_dotenv = None

        if load_dotenv is not None:
            load_dotenv()

        token = self.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - import tested through integration
            raise RuntimeError(
                "Missing transformers stack. Install dependencies first (e.g., make install-llm)."
            ) from exc

        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        model_kwargs: dict[str, Any] = {
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": self._resolve_torch_dtype(torch, self.dtype),
        }
        if token:
            tokenizer_kwargs["token"] = token
            model_kwargs["token"] = token

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, **tokenizer_kwargs)
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        try:
            inference_device = next(model.parameters()).device
        except StopIteration:  # pragma: no cover - defensive
            inference_device = torch.device("cpu")

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._inference_device = inference_device

    def _generate_text(self, prompt: str) -> str:
        self._ensure_model_loaded()
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._inference_device is not None

        messages = [
            {
                "role": "system",
                "content": "Answer MCQ-ORDER questions by returning exactly one valid option label.",
            },
            {"role": "user", "content": prompt},
        ]
        if hasattr(self._tokenizer, "apply_chat_template"):
            rendered_prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered_prompt = (
                "System: Answer MCQ-ORDER questions by returning one valid option label.\n"
                f"User: {prompt}\nAssistant:"
            )

        tokenized = self._tokenizer(rendered_prompt, return_tensors="pt")
        model_inputs = {key: value.to(self._inference_device) for key, value in tokenized.items()}

        do_sample = self.temperature > 0.0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p
        if self._tokenizer.pad_token_id is not None:
            generation_kwargs["pad_token_id"] = self._tokenizer.pad_token_id
        if self._tokenizer.eos_token_id is not None:
            generation_kwargs["eos_token_id"] = self._tokenizer.eos_token_id

        with self._torch.inference_mode():
            generated = self._model.generate(**model_inputs, **generation_kwargs)

        prompt_len = model_inputs["input_ids"].shape[-1]
        completion_tokens = generated[0][prompt_len:]
        return self._tokenizer.decode(completion_tokens, skip_special_tokens=True).strip()

    def predict(self, example: MCQOrderExample) -> str:
        valid_labels = {option.label for option in example.options}
        prompt = self._format_prompt(example)

        last_text = ""
        for _ in range(self.prediction_retries + 1):
            text = self._generate_text(prompt)
            last_text = text
            label = self._extract_label(text, valid_labels)
            if label is not None:
                return label
            prompt = (
                f"{self._format_prompt(example)}\n"
                f"Your previous answer was invalid: {text!r}. Return only one valid label."
            )

        raise ValueError(
            "Local LLM response did not contain a valid label after retries. "
            f"Last response: {last_text!r}"
        )

    def run_metadata(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "provider": self.provider_label,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "prediction_retries": self.prediction_retries,
            "device_map": self.device_map,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
        }


class QwenTransformersMCQOrderModel(LocalTransformersMCQOrderModel):
    model_name = "llm-qwen"

    def __init__(
        self,
        *,
        seed: int | None = None,
        model_id: str = DEFAULT_QWEN_MODEL_ID,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: int = 16,
        prediction_retries: int = 2,
        device_map: str = "auto",
        dtype: str = "float16",
        trust_remote_code: bool = False,
        hf_token: str | None = None,
    ) -> None:
        super().__init__(
            seed=seed,
            model_id=model_id,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            prediction_retries=prediction_retries,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            hf_token=hf_token,
            provider_label="huggingface-qwen",
        )


class LlamaTransformersMCQOrderModel(LocalTransformersMCQOrderModel):
    model_name = "llm-llama"

    def __init__(
        self,
        *,
        seed: int | None = None,
        model_id: str = DEFAULT_LLAMA_MODEL_ID,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: int = 16,
        prediction_retries: int = 2,
        device_map: str = "auto",
        dtype: str = "float16",
        trust_remote_code: bool = False,
        hf_token: str | None = None,
    ) -> None:
        super().__init__(
            seed=seed,
            model_id=model_id,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            prediction_retries=prediction_retries,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            hf_token=hf_token,
            provider_label="huggingface-llama",
        )


def build_model(
    name: str,
    *,
    seed: int | None = None,
    openai_model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    timeout_seconds: float = 60.0,
    max_retries: int = 2,
    prediction_retries: int = 2,
    qwen_model_id: str = DEFAULT_QWEN_MODEL_ID,
    llama_model_id: str = DEFAULT_LLAMA_MODEL_ID,
    local_temperature: float = 0.0,
    local_top_p: float = 1.0,
    local_max_new_tokens: int = 16,
    local_device_map: str = "auto",
    local_dtype: str = "float16",
    local_trust_remote_code: bool = False,
    hf_token: str | None = None,
) -> MCQOrderModel:
    model_key = name.strip().lower()
    if model_key == "random":
        return RandomMCQOrderModel(seed=seed)
    if model_key in {"llm-openai", "openai", "llm"}:
        return OpenAILangChainMCQOrderModel(
            seed=seed,
            openai_model=openai_model,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            prediction_retries=prediction_retries,
        )
    if model_key in {"llm-qwen", "qwen"}:
        return QwenTransformersMCQOrderModel(
            seed=seed,
            model_id=qwen_model_id,
            temperature=local_temperature,
            top_p=local_top_p,
            max_new_tokens=local_max_new_tokens,
            prediction_retries=prediction_retries,
            device_map=local_device_map,
            dtype=local_dtype,
            trust_remote_code=local_trust_remote_code,
            hf_token=hf_token,
        )
    if model_key in {"llm-llama", "llama"}:
        return LlamaTransformersMCQOrderModel(
            seed=seed,
            model_id=llama_model_id,
            temperature=local_temperature,
            top_p=local_top_p,
            max_new_tokens=local_max_new_tokens,
            prediction_retries=prediction_retries,
            device_map=local_device_map,
            dtype=local_dtype,
            trust_remote_code=local_trust_remote_code,
            hf_token=hf_token,
        )
    raise ValueError(
        f"Unknown model '{name}'. Available models: random, llm-openai, llm-qwen, llm-llama"
    )
