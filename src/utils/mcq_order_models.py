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


def build_model(
    name: str,
    *,
    seed: int | None = None,
    openai_model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    timeout_seconds: float = 60.0,
    max_retries: int = 2,
    prediction_retries: int = 2,
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
    raise ValueError(f"Unknown model '{name}'. Available models: random, llm-openai")
