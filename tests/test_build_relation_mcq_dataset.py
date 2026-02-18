"""Tests for MCQ-RELATION dataset generation."""

from __future__ import annotations

import csv
from pathlib import Path

from utils.build_relation_mcq_dataset import (
    REL_A_B_START_SAME_TIME,
    REL_A_OVERLAPS_B,
    REL_A_STARTS_AFTER_B,
    REL_A_STARTS_BEFORE_B,
    REL_CANNOT_BE_DETERMINED,
    BuildStats,
    _drop_short_audios,
    _iter_examples,
    _load_filtered_events,
    _option_label,
    _select_examples,
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


def test_relation_labels_and_prompt_format(tmp_path: Path) -> None:
    csv_path = tmp_path / "strong.csv"
    _write_rows(
        csv_path,
        rows=[
            {"filename": "rel.mp3", "text": "soft bell tone", "onset": "0.00", "offset": "1.00"},
            {"filename": "rel.mp3", "text": "drum pulse", "onset": "2.00", "offset": "3.00"},
            {"filename": "rel.mp3", "text": "humming pad", "onset": "2.03", "offset": "4.00"},
            {"filename": "rel.mp3", "text": "crowd murmur", "onset": "2.50", "offset": "4.50"},
            {"filename": "rel.mp3", "text": "engine revving", "onset": "5.00", "offset": "6.00"},
            {"filename": "rel.mp3", "text": "engine revving", "onset": "7.00", "offset": "8.00"},
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
            seed=11,
            same_start_epsilon=0.05,
            overlap_epsilon=0.0,
            include_start_same_time=True,
            stats=stats,
        )
    )

    assert len(examples) == 20
    assert stats.questions_candidate_pairs == 20
    assert stats.questions_relation_before > 0
    assert stats.questions_relation_after > 0
    assert stats.questions_relation_overlap > 0
    assert stats.questions_relation_same_start > 0
    assert stats.questions_relation_cannot_determine == 8

    by_id = {example["id"]: example for example in examples}
    assert by_id["rel.mp3__0__1"]["answer_relation"] == REL_A_STARTS_BEFORE_B
    assert by_id["rel.mp3__1__0"]["answer_relation"] == REL_A_STARTS_AFTER_B
    assert by_id["rel.mp3__1__2"]["answer_relation"] == REL_A_B_START_SAME_TIME
    assert by_id["rel.mp3__1__3"]["answer_relation"] == REL_A_OVERLAPS_B
    assert by_id["rel.mp3__4__0"]["answer_relation"] == REL_CANNOT_BE_DETERMINED

    sample = by_id["rel.mp3__1__3"]
    assert sample["question"].startswith("In this audio, what is the temporal relation between Event A and Event B?")
    assert "Event A:" in sample["question"]
    assert "Event B:" in sample["question"]
    assert "2.5" not in sample["question"]
    assert "3.0" not in sample["question"]
    assert "5.0" not in sample["question"]
    assert {option["text"] for option in sample["options"]} == {
        "A starts before B",
        "A starts after B",
        "A overlaps B",
        "A and B start at the same time",
        "cannot be determined",
    }


def test_near_ties_are_filtered_if_same_time_option_is_disabled(tmp_path: Path) -> None:
    csv_path = tmp_path / "strong.csv"
    _write_rows(
        csv_path,
        rows=[
            {"filename": "tie.mp3", "text": "first sound", "onset": "0.00", "offset": "1.00"},
            {"filename": "tie.mp3", "text": "second sound", "onset": "0.03", "offset": "1.20"},
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
            same_start_epsilon=0.05,
            overlap_epsilon=0.0,
            include_start_same_time=False,
            stats=stats,
        )
    )

    assert stats.questions_candidate_pairs == 2
    assert stats.questions_dropped_near_tie == 2
    assert examples == []


def test_output_is_deterministic_with_seed(tmp_path: Path) -> None:
    csv_path = tmp_path / "strong.csv"
    _write_rows(
        csv_path,
        rows=[
            {"filename": "a.mp3", "text": "first", "onset": "0.00", "offset": "1.00"},
            {"filename": "a.mp3", "text": "second", "onset": "1.50", "offset": "2.50"},
            {"filename": "a.mp3", "text": "third", "onset": "3.00", "offset": "4.00"},
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
            grouped_events=grouped_a,
            seed=19,
            same_start_epsilon=0.05,
            overlap_epsilon=0.0,
            include_start_same_time=True,
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
            grouped_events=grouped_b,
            seed=19,
            same_start_epsilon=0.05,
            overlap_epsilon=0.0,
            include_start_same_time=True,
            stats=stats_b,
        )
    )

    stats_c = BuildStats()
    grouped_c = _drop_short_audios(
        _load_filtered_events(csv_path, min_duration=0.5, stats=stats_c),
        min_events_per_audio=2,
        stats=stats_c,
    )
    examples_c = list(
        _iter_examples(
            grouped_events=grouped_c,
            seed=20,
            same_start_epsilon=0.05,
            overlap_epsilon=0.0,
            include_start_same_time=True,
            stats=stats_c,
        )
    )

    assert examples_a == examples_b
    assert examples_a != examples_c


def test_quality_caps_reduce_size_deterministically(tmp_path: Path) -> None:
    csv_path = tmp_path / "strong.csv"
    _write_rows(
        csv_path,
        rows=[
            {"filename": "a.mp3", "text": "a1", "onset": "0.00", "offset": "1.00"},
            {"filename": "a.mp3", "text": "a2", "onset": "2.00", "offset": "3.00"},
            {"filename": "a.mp3", "text": "a3", "onset": "4.00", "offset": "5.00"},
            {"filename": "b.mp3", "text": "b1", "onset": "0.00", "offset": "1.00"},
            {"filename": "b.mp3", "text": "b2", "onset": "1.20", "offset": "2.20"},
            {"filename": "b.mp3", "text": "b3", "onset": "2.40", "offset": "3.40"},
        ],
    )

    stats = BuildStats()
    grouped_events = _drop_short_audios(
        _load_filtered_events(csv_path, min_duration=0.5, stats=stats),
        min_events_per_audio=2,
        stats=stats,
    )
    candidates = list(
        _iter_examples(
            grouped_events=grouped_events,
            seed=5,
            same_start_epsilon=0.05,
            overlap_epsilon=0.0,
            include_start_same_time=True,
            stats=stats,
        )
    )
    assert len(candidates) == 12

    selected = _select_examples(
        candidates,
        max_pairs_per_audio=2,
        max_questions=6,
        balance_relations=True,
        stats=stats,
    )
    selected_again = _select_examples(
        candidates,
        max_pairs_per_audio=2,
        max_questions=6,
        balance_relations=True,
        stats=BuildStats(),
    )

    assert len(selected) == 4
    assert selected == selected_again
