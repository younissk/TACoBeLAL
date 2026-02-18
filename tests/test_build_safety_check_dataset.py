"""Tests for safety-check dataset generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import typer

from utils.build_safety_check_dataset import _normalize_event_text, main


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "text", "onset", "offset"])
        writer.writeheader()
        writer.writerows(rows)


def _load_examples(path: Path) -> list[dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_build_safety_dataset_balanced_and_schema(tmp_path: Path) -> None:
    csv_path = tmp_path / "strong.csv"
    output_path = tmp_path / "mcq_safety_presence.jsonl"
    _write_rows(
        csv_path,
        rows=[
            {"filename": "a.mp3", "text": "A dog barks.", "onset": "0.0", "offset": "1.0"},
            {"filename": "a.mp3", "text": "Wind blowing.", "onset": "1.0", "offset": "2.0"},
            {"filename": "b.mp3", "text": "A car horn.", "onset": "0.0", "offset": "1.0"},
            {"filename": "c.mp3", "text": "Rain falling.", "onset": "0.0", "offset": "1.0"},
            {"filename": "d.mp3", "text": "Bird chirping.", "onset": "0.0", "offset": "1.0"},
            {"filename": "e.mp3", "text": "People talking.", "onset": "0.0", "offset": "1.0"},
            {"filename": "f.mp3", "text": "Engine idling.", "onset": "0.0", "offset": "1.0"},
        ],
    )

    main(input_csv=csv_path, output_jsonl=output_path, num_audios=4, seed=11)
    examples = _load_examples(output_path)
    assert len(examples) == 4
    assert len({str(example["audio_filename"]) for example in examples}) == 4

    answer_types = [str(example["answer_type"]) for example in examples]
    assert answer_types.count("true") == 2
    assert answer_types.count("false") == 2

    by_audio: dict[str, set[str]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = (row.get("filename") or "").strip()
            text = (row.get("text") or "").strip()
            if not filename or not text:
                continue
            by_audio.setdefault(filename, set()).add(_normalize_event_text(text))

    for example in examples:
        options = example["options"]
        assert isinstance(options, list)
        assert len(options) == 2
        assert {str(option["text"]) for option in options} == {"True", "False"}
        assert str(example["answer_label"]) in {str(option["label"]) for option in options}

        answer_text_by_label = {str(option["label"]): str(option["text"]) for option in options}
        assert answer_text_by_label[str(example["answer_label"])] == str(example["answer_text"])

        question = str(example["question"])
        assert question.startswith('Is this event in the audio: "')
        assert question.endswith('"?')

        if str(example["answer_type"]) == "false":
            audio_filename = str(example["audio_filename"])
            query_event = str(example["query_event"])
            assert _normalize_event_text(query_event) not in by_audio[audio_filename]


def test_build_safety_dataset_is_deterministic(tmp_path: Path) -> None:
    csv_path = tmp_path / "strong.csv"
    out_a = tmp_path / "safety_a.jsonl"
    out_b = tmp_path / "safety_b.jsonl"
    _write_rows(
        csv_path,
        rows=[
            {"filename": "a.mp3", "text": "A dog barks.", "onset": "0.0", "offset": "1.0"},
            {"filename": "b.mp3", "text": "A car horn.", "onset": "0.0", "offset": "1.0"},
            {"filename": "c.mp3", "text": "Rain falling.", "onset": "0.0", "offset": "1.0"},
            {"filename": "d.mp3", "text": "Bird chirping.", "onset": "0.0", "offset": "1.0"},
        ],
    )

    main(input_csv=csv_path, output_jsonl=out_a, num_audios=4, seed=7)
    main(input_csv=csv_path, output_jsonl=out_b, num_audios=4, seed=7)

    assert out_a.read_text(encoding="utf-8") == out_b.read_text(encoding="utf-8")


def test_num_audios_must_be_even(tmp_path: Path) -> None:
    csv_path = tmp_path / "strong.csv"
    output_path = tmp_path / "safety.jsonl"
    _write_rows(
        csv_path,
        rows=[
            {"filename": "a.mp3", "text": "A dog barks.", "onset": "0.0", "offset": "1.0"},
            {"filename": "b.mp3", "text": "A car horn.", "onset": "0.0", "offset": "1.0"},
        ],
    )

    with pytest.raises(typer.BadParameter):
        main(input_csv=csv_path, output_jsonl=output_path, num_audios=3, seed=7)
