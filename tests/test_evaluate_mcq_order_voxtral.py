"""Tests for Voxtral MCQ-ORDER evaluator helpers."""

from __future__ import annotations

from pathlib import Path

from utils.evaluate_mcq_order_voxtral import build_conversation, build_user_prompt
from utils.mcq_order_models import MCQOrderExample


def _example(example_id: str) -> MCQOrderExample:
    payload: dict[str, object] = {
        "id": example_id,
        "audio_filename": "sample.mp3",
        "question": "What happens immediately after X?",
        "options": [
            {"label": "A", "text": "Option A", "type": "event", "event_index": 0},
            {"label": "B", "text": "Option B", "type": "event", "event_index": 1},
            {"label": "C", "text": "This is the last event, no immediate event after.", "type": "none"},
        ],
        "answer_label": "B",
        "answer_text": "Option B",
    }
    return MCQOrderExample.from_json(payload)


def test_build_user_prompt_contains_question_and_options() -> None:
    prompt = build_user_prompt(_example("ex-1"))
    assert "What happens immediately after X?" in prompt
    assert "A. Option A" in prompt
    assert "B. Option B" in prompt
    assert "Return only the option label" in prompt


def test_build_conversation_contains_audio_and_text_content() -> None:
    example = _example("ex-2")
    audio_path = Path("/tmp/audio.wav")
    conversation = build_conversation(example, audio_path)

    assert len(conversation) == 1
    assert conversation[0]["role"] == "user"
    user_content = conversation[0]["content"]
    assert isinstance(user_content, list)
    assert user_content[0]["type"] == "audio"
    assert user_content[0]["path"] == str(audio_path)
    assert user_content[1]["type"] == "text"
    assert "Return only the option label" in user_content[1]["text"]
