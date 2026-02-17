"""Tests for timeline MCQ dataset generation."""

from __future__ import annotations

import csv
from pathlib import Path

from utils.build_timeline_mcq_dataset import (
    LAST_EVENT_TEXT_DEFAULT,
    BuildStats,
    _drop_short_audios,
    _iter_examples,
    _load_filtered_events,
    _option_label,
)


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "text", "onset", "offset"])
        writer.writeheader()
        writer.writerows(rows)


def test_option_label() -> None:
    assert _option_label(0) == "A"
    assert _option_label(25) == "Z"
    assert _option_label(26) == "AA"
    assert _option_label(27) == "AB"
    assert _option_label(51) == "AZ"
    assert _option_label(52) == "BA"


def test_filters_and_mcq_answers(tmp_path: Path) -> None:
    csv_path = tmp_path / "strong.csv"
    _write_rows(
        csv_path,
        rows=[
            {"filename": "a.mp3", "text": "event a1", "onset": "0.0", "offset": "1.0"},
            {"filename": "a.mp3", "text": "event a1", "onset": "2.0", "offset": "3.0"},
            {"filename": "a.mp3", "text": "too short", "onset": "1.0", "offset": "1.2"},
            {"filename": "a.mp3", "text": "event a2", "onset": "4.0", "offset": "5.0"},
            {"filename": "b.mp3", "text": "event b1", "onset": "0.0", "offset": "1.0"},
        ],
    )

    stats = BuildStats()
    grouped_events = _load_filtered_events(csv_path, min_duration=0.5, stats=stats)
    assert stats.rows_total == 5
    assert stats.rows_short == 1
    assert stats.rows_kept == 4
    assert stats.rows_collapsed_by_caption_grouping == 1
    assert stats.events_after_caption_grouping == 3
    assert stats.audios_total == 2

    assert grouped_events["a.mp3"][0].captions == ("event a1",)
    assert grouped_events["a.mp3"][0].occurrence_count == 2
    grouped_events = _drop_short_audios(grouped_events, min_events_per_audio=2, stats=stats)
    assert sorted(grouped_events) == ["a.mp3"]
    assert stats.audios_removed_too_few_events == 1
    assert stats.audios_kept == 1

    examples = list(
        _iter_examples(
            grouped_events=grouped_events,
            seed=123,
            no_event_text=LAST_EVENT_TEXT_DEFAULT,
            onset_tie_epsilon=0.05,
            stats=stats,
        )
    )
    assert len(examples) == 1
    assert stats.questions_dropped_repeat_ambiguity == 1
    assert all(len(ex["options"]) == 2 for ex in examples)
    assert all(any(opt["type"] == "none" for opt in ex["options"]) for ex in examples)
    assert all(
        all(opt.get("event_index") != ex["base_event"]["event_index"] for opt in ex["options"] if opt["type"] == "event")
        for ex in examples
    )

    last = examples[0]
    assert last["base_event"]["event_index"] == 1
    assert last["answer_type"] == "none"
    assert last["answer_event_index"] is None
    assert last["answer_text"] == LAST_EVENT_TEXT_DEFAULT


def test_option_order_is_deterministic_with_seed(tmp_path: Path) -> None:
    csv_path = tmp_path / "strong.csv"
    _write_rows(
        csv_path,
        rows=[
            {"filename": "a.mp3", "text": "event a1", "onset": "0.0", "offset": "1.0"},
            {"filename": "a.mp3", "text": "event a2", "onset": "2.0", "offset": "3.0"},
        ],
    )

    stats_a = BuildStats()
    grouped_a = _drop_short_audios(
        _load_filtered_events(csv_path, min_duration=0.5, stats=stats_a),
        min_events_per_audio=2,
        stats=stats_a,
    )
    examples_a = list(
        _iter_examples(
            grouped_a,
            seed=9,
            no_event_text=LAST_EVENT_TEXT_DEFAULT,
            onset_tie_epsilon=0.05,
            stats=stats_a,
        )
    )

    stats_b = BuildStats()
    grouped_b = _drop_short_audios(
        _load_filtered_events(csv_path, min_duration=0.5, stats=stats_b),
        min_events_per_audio=2,
        stats=stats_b,
    )
    examples_b = list(
        _iter_examples(
            grouped_b,
            seed=9,
            no_event_text=LAST_EVENT_TEXT_DEFAULT,
            onset_tie_epsilon=0.05,
            stats=stats_b,
        )
    )

    assert examples_a == examples_b


def test_ambiguity_filters_drop_problematic_questions(tmp_path: Path) -> None:
    csv_path = tmp_path / "strong.csv"
    _write_rows(
        csv_path,
        rows=[
            {"filename": "tie.mp3", "text": "tie a", "onset": "0.00", "offset": "1.00"},
            {"filename": "tie.mp3", "text": "tie b", "onset": "0.03", "offset": "1.40"},
            {"filename": "tie.mp3", "text": "tie c", "onset": "3.00", "offset": "3.60"},
            {"filename": "tiebreak.mp3", "text": "tb a", "onset": "0.00", "offset": "0.60"},
            {"filename": "tiebreak.mp3", "text": "tb b", "onset": "1.00", "offset": "1.50"},
            {"filename": "tiebreak.mp3", "text": "tb c", "onset": "1.03", "offset": "1.60"},
            {"filename": "repeat.mp3", "text": "rep", "onset": "0.00", "offset": "0.60"},
            {"filename": "repeat.mp3", "text": "mid", "onset": "1.50", "offset": "2.10"},
            {"filename": "repeat.mp3", "text": "rep", "onset": "3.00", "offset": "3.70"},
            {"filename": "overlap.mp3", "text": "ov a", "onset": "0.00", "offset": "2.00"},
            {"filename": "overlap.mp3", "text": "ov b", "onset": "1.00", "offset": "1.50"},
            {"filename": "overlap.mp3", "text": "ov c", "onset": "3.00", "offset": "3.70"},
            {"filename": "clean.mp3", "text": "cl a", "onset": "0.00", "offset": "0.60"},
            {"filename": "clean.mp3", "text": "cl b", "onset": "1.00", "offset": "1.60"},
            {"filename": "clean.mp3", "text": "cl c", "onset": "2.00", "offset": "2.70"},
        ],
    )

    stats = BuildStats()
    grouped_events = _drop_short_audios(
        _load_filtered_events(csv_path, min_duration=0.5, stats=stats),
        min_events_per_audio=2,
        stats=stats,
    )
    examples = list(
        _iter_examples(
            grouped_events=grouped_events,
            seed=7,
            no_event_text=LAST_EVENT_TEXT_DEFAULT,
            onset_tie_epsilon=0.05,
            stats=stats,
        )
    )

    assert stats.questions_dropped_onset_tie_base == 4
    assert stats.questions_dropped_tiebreak_answer == 1
    assert stats.questions_dropped_repeat_ambiguity == 1
    assert stats.questions_dropped_overlap_ambiguity == 1
    assert len(examples) == 7

    kept_ids = {ex["id"] for ex in examples}
    assert "tiebreak.mp3__0" not in kept_ids
    assert "repeat.mp3__0" not in kept_ids
    assert "overlap.mp3__0" not in kept_ids
