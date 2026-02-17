"""Tests for Audio Flamingo MCQ-ORDER evaluator helpers."""

from __future__ import annotations

import json
from pathlib import Path

from utils.evaluate_mcq_order_audioflamingo import (
    build_prompt,
    evaluate_audioflamingo_outputs,
    parse_predicted_label,
    prepare_audioflamingo_input,
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


def test_build_prompt_contains_options() -> None:
    prompt = build_prompt(_example("ex-1"))
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


def test_prepare_audioflamingo_input_uses_unique_audio_links(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    (audio_root / "sample.mp3").write_bytes(b"dummy-audio")

    examples = [_example("ex-1"), _example("ex-2", answer_label="A")]
    input_json_path, mapping_json_path, mapping = prepare_audioflamingo_input(
        examples,
        audio_root=audio_root,
        work_dir=tmp_path / "work",
        use_audio=True,
    )

    payload = json.loads(input_json_path.read_text(encoding="utf-8"))
    assert len(payload["data"]) == 2
    names = [payload["data"][k]["name"] for k in sorted(payload["data"], key=int)]
    assert names[0] != names[1]
    assert len(mapping) == 2
    assert mapping_json_path.exists()

    for name in names:
        assert (tmp_path / "work" / "audio_links" / name).exists()


def test_evaluate_audioflamingo_outputs_handles_missing_and_invalid() -> None:
    ex = _example("ex-1")
    mapped_id = "/tmp/fake_audio_1.wav"
    mapping = {
        mapped_id: {
            "index": 0,
            "example_id": ex.example_id,
            "audio_filename": ex.audio_filename,
            "question": ex.question,
            "answer_label": ex.answer_label,
            "answer_text": ex.answer_text,
            "options": [
                {"label": opt.label, "text": opt.text, "type": opt.option_type}
                for opt in ex.options
            ],
        }
    }

    decisions, invalid_count, missing_count = evaluate_audioflamingo_outputs(
        mapping=mapping,
        raw_outputs=[{"id": mapped_id, "pred": "not a label"}],
        model_name="audio-flamingo-3",
        model_base="nvidia/audio-flamingo-3",
    )
    assert len(decisions) == 1
    assert invalid_count == 1
    assert missing_count == 0
    assert decisions[0].predicted_label == "INVALID"
    assert decisions[0].is_correct is False


def test_prepare_audioflamingo_input_without_audio_links_in_text_only_mode(tmp_path: Path) -> None:
    examples = [_example("ex-1"), _example("ex-2", answer_label="A")]
    input_json_path, mapping_json_path, mapping = prepare_audioflamingo_input(
        examples,
        audio_root=None,
        work_dir=tmp_path / "work",
        use_audio=False,
    )

    payload = json.loads(input_json_path.read_text(encoding="utf-8"))
    assert len(payload["data"]) == 2
    first_item = payload["data"]["0"]
    assert "name" not in first_item
    assert "prompt" in first_item
    assert len(mapping) == 2
    assert all(key.startswith("text::") for key in mapping)
    assert mapping_json_path.exists()
