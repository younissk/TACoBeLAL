"""Tests for Qwen2-Audio MCQ-ORDER evaluator helpers."""

from __future__ import annotations

from utils.evaluate_mcq_order_qwen2_audio import (
    _extract_completion_ids,
    build_prompt,
    evaluate_qwen2_audio_outputs,
    parse_predicted_label,
)
from utils.mcq_order_models import MCQOrderExample


def _example(example_id: str, *, answer_label: str = "B") -> MCQOrderExample:
    payload: dict[str, object] = {
        "id": example_id,
        "audio_filename": "sample.mp3",
        "question": "What happens immediately after X?",
        "options": [
            {"label": "A", "text": "Option A", "type": "event", "event_index": 0},
            {"label": "B", "text": "Option B", "type": "event", "event_index": 1},
            {"label": "C", "text": "This is the last event, no immediate event after.", "type": "none"},
        ],
        "answer_label": answer_label,
        "answer_text": "Option B" if answer_label == "B" else "Option A",
    }
    return MCQOrderExample.from_json(payload)


def test_build_prompt_contains_audio_token_and_options() -> None:
    prompt = build_prompt(_example("ex-1"))
    assert "<|audio_bos|><|AUDIO|><|audio_eos|>" in prompt
    assert "What happens immediately after X?" in prompt
    assert "A. Option A" in prompt
    assert "B. Option B" in prompt
    assert "Return only the option label" in prompt


def test_parse_predicted_label() -> None:
    valid = {"A", "B", "C"}
    label_to_text = {"A": "Option A", "B": "Option B", "C": "None"}
    assert parse_predicted_label("A", valid_labels=valid, label_to_text=label_to_text)[0] == "A"
    assert parse_predicted_label('{"label":"B"}', valid_labels=valid, label_to_text=label_to_text)[0] == "B"
    assert parse_predicted_label("I think C.", valid_labels=valid, label_to_text=label_to_text)[0] == "C"
    assert parse_predicted_label("Option B", valid_labels=valid, label_to_text=label_to_text)[0] == "B"
    assert parse_predicted_label("", valid_labels=valid, label_to_text=label_to_text)[0] is None


def test_evaluate_qwen2_audio_outputs_handles_missing_and_invalid() -> None:
    ex1 = _example("ex-1")
    ex2 = _example("ex-2", answer_label="A")
    raw_outputs = [{"id": "ex-1", "pred": "not a valid answer"}]

    decisions, invalid_count, missing_count = evaluate_qwen2_audio_outputs(
        examples=[ex1, ex2],
        raw_outputs=raw_outputs,
        model_name="qwen2-audio-7b-instruct",
        model_base="Qwen/Qwen2-Audio-7B-Instruct",
    )

    assert len(decisions) == 2
    assert invalid_count == 1
    assert missing_count == 1

    first = decisions[0]
    second = decisions[1]
    assert first.predicted_label == "INVALID"
    assert first.parse_status == "invalid"
    assert first.is_correct is False
    assert second.predicted_label == "INVALID"
    assert second.parse_status == "missing"
    assert second.is_correct is False


def test_extract_completion_ids_prefers_input_ids_when_attention_sum_exceeds_generated() -> None:
    generated_row = [0, 1, 2, 3, 4, 5, 6]
    input_ids_row = [10, 11, 12, 13, 14]
    attention_mask_row = [1] * 9

    completion_ids = _extract_completion_ids(
        generated_row,
        input_ids_row=input_ids_row,
        attention_mask_row=attention_mask_row,
    )

    assert completion_ids == [5, 6]


def test_extract_completion_ids_falls_back_to_attention_sum_when_input_len_invalid() -> None:
    generated_row = [0, 1, 2, 3, 4, 5, 6]
    input_ids_row = [10, 11, 12, 13, 14, 15, 16, 17, 18]
    attention_mask_row = [1, 1, 1, 1, 1, 0, 0]

    completion_ids = _extract_completion_ids(
        generated_row,
        input_ids_row=input_ids_row,
        attention_mask_row=attention_mask_row,
    )

    assert completion_ids == [5, 6]


def test_extract_completion_ids_returns_empty_when_no_valid_start_exists() -> None:
    generated_row = [0, 1, 2]
    input_ids_row = [10, 11, 12, 13]
    attention_mask_row = [1, 1, 1, 1, 1]

    completion_ids = _extract_completion_ids(
        generated_row,
        input_ids_row=input_ids_row,
        attention_mask_row=attention_mask_row,
    )

    assert completion_ids == []
