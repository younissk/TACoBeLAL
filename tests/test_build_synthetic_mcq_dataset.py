"""Tests for synthetic MCQ benchmark dataset generation."""

from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf

from utils.build_synthetic_mcq_dataset import main


def _load_rows(path: Path) -> list[dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_build_synthetic_datasets_write_rows_and_audio(tmp_path: Path) -> None:
    main(
        benchmark="all",
        difficulty="easy",
        scenes_per_split=2,
        seed=7,
        sample_rate=16000,
        audio_root=tmp_path / "audio",
        dataset_root=tmp_path / "data",
        generator_version="synth-v1",
    )

    families = {
        "time": "MCQ-SYNTH-TIME",
        "pitch": "MCQ-SYNTH-PITCH",
        "loudness": "MCQ-SYNTH-LOUDNESS",
        "rhythm": "MCQ-SYNTH-RHYTHM",
    }
    for family, task_id in families.items():
        dataset_path = tmp_path / "data" / f"mcq_synth_{family}_easy.jsonl"
        rows = _load_rows(dataset_path)
        assert len(rows) == 2
        assert all(row["task_id"] == task_id for row in rows)
        assert all(row["benchmark_family"] == family for row in rows)
        assert all(row["difficulty"] == "easy" for row in rows)
        assert all("question_template" in row for row in rows)
        assert all("scene" in row for row in rows)

        for row in rows:
            options = row["options"]
            labels = [option["label"] for option in options]
            assert len(labels) == len(set(labels))
            assert row["answer_label"] in labels
            audio_path = tmp_path / "audio" / str(row["audio_filename"])
            assert audio_path.exists()
            samples, sample_rate = sf.read(audio_path)
            assert sample_rate == 16000
            assert samples.size > 0

    aggregate_rows = _load_rows(tmp_path / "data" / "mcq_synth_benchmark.jsonl")
    assert len(aggregate_rows) == 8
    assert all(row["task_id"] == "MCQ-SYNTH" for row in aggregate_rows)
    assert {row["benchmark_family"] for row in aggregate_rows} == set(families)
    assert {row["source_task_id"] for row in aggregate_rows} == set(families.values())


def test_build_synthetic_generation_is_deterministic(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    dataset_root = tmp_path / "data"

    main(
        benchmark="time",
        difficulty="easy",
        scenes_per_split=3,
        seed=11,
        sample_rate=16000,
        audio_root=audio_root,
        dataset_root=dataset_root,
        generator_version="synth-v1",
    )
    dataset_path = dataset_root / "mcq_synth_time_easy.jsonl"
    first_dataset = dataset_path.read_text(encoding="utf-8")
    first_audio = (audio_root / "synthetic" / "time" / "easy" / "synth_time_easy_000001.wav").read_bytes()

    main(
        benchmark="time",
        difficulty="easy",
        scenes_per_split=3,
        seed=11,
        sample_rate=16000,
        audio_root=audio_root,
        dataset_root=dataset_root,
        generator_version="synth-v1",
    )
    second_dataset = dataset_path.read_text(encoding="utf-8")
    second_audio = (audio_root / "synthetic" / "time" / "easy" / "synth_time_easy_000001.wav").read_bytes()

    assert first_dataset == second_dataset
    assert first_audio == second_audio


def test_synthetic_scene_invariants_hold(tmp_path: Path) -> None:
    main(
        benchmark="all",
        difficulty="hard",
        scenes_per_split=3,
        seed=5,
        sample_rate=16000,
        audio_root=tmp_path / "audio",
        dataset_root=tmp_path / "data",
        generator_version="synth-v1",
    )

    time_rows = _load_rows(tmp_path / "data" / "mcq_synth_time_hard.jsonl")
    for row in time_rows:
        events = row["scene"]["events"]
        onsets = [event["onset"] for event in events]
        durations = [event["duration"] for event in events]
        pitches = {event["pitch_hz"] for event in events}
        levels = {event["dbfs"] for event in events}
        assert len(onsets) == len(set(onsets))
        assert len(durations) == len(set(durations))
        assert len(pitches) == 1
        assert len(levels) == 1

    pitch_rows = _load_rows(tmp_path / "data" / "mcq_synth_pitch_hard.jsonl")
    for row in pitch_rows:
        events = row["scene"]["events"]
        onsets = {event["onset"] for event in events}
        durations = {event["duration"] for event in events}
        levels = {event["dbfs"] for event in events}
        pitches = [event["pitch_hz"] for event in events]
        assert onsets == {0.4, 1.2, 2.0}
        assert durations == {0.4}
        assert len(levels) == 1
        assert len(pitches) == len(set(pitches))

    loudness_rows = _load_rows(tmp_path / "data" / "mcq_synth_loudness_hard.jsonl")
    for row in loudness_rows:
        events = row["scene"]["events"]
        onsets = {event["onset"] for event in events}
        durations = {event["duration"] for event in events}
        pitches = {event["pitch_hz"] for event in events}
        levels = [event["dbfs"] for event in events]
        assert onsets == {0.4, 1.2, 2.0}
        assert durations == {0.4}
        assert len(pitches) == 1
        assert len(levels) == len(set(levels))

    rhythm_rows = _load_rows(tmp_path / "data" / "mcq_synth_rhythm_hard.jsonl")
    for row in rhythm_rows:
        bursts = row["scene"]["bursts"]
        assert len(bursts) == 2
        assert bursts[0]["count"] != bursts[1]["count"]
        assert bursts[0]["window_seconds"] == bursts[1]["window_seconds"]
        assert bursts[0]["pip_duration"] == bursts[1]["pip_duration"]

    aggregate_rows = _load_rows(tmp_path / "data" / "mcq_synth_benchmark.jsonl")
    assert len(aggregate_rows) == 12
    assert {row["difficulty"] for row in aggregate_rows} == {"hard"}
    assert {row["benchmark_family"] for row in aggregate_rows} == {"time", "pitch", "loudness", "rhythm"}
