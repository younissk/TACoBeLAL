"""Tests for synthetic MCQ benchmark dataset generation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import soundfile as sf

from utils.build_synthetic_mcq_dataset import main


def _load_rows(path: Path) -> list[dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _build_standalone(tmp_path: Path, benchmark: str, *, scenes_per_split: int = 100) -> list[dict[str, object]]:
    main(
        benchmark=benchmark,
        difficulty="easy",
        scenes_per_split=scenes_per_split,
        seed=7,
        sample_rate=16000,
        audio_root=tmp_path / "audio",
        dataset_root=tmp_path / "data",
        generator_version="synth-v1",
    )
    dataset_path = tmp_path / "data" / f"mcq_synth_{benchmark}_easy.jsonl"
    return _load_rows(dataset_path)


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


def test_build_pitch_order_trivial_writes_expected_rows_and_audio(tmp_path: Path) -> None:
    main(
        benchmark="pitch_order_trivial",
        difficulty="easy",
        scenes_per_split=500,
        seed=7,
        sample_rate=16000,
        audio_root=tmp_path / "audio",
        dataset_root=tmp_path / "data",
        generator_version="synth-v1",
    )

    dataset_path = tmp_path / "data" / "mcq_synth_pitch_order_trivial_easy.jsonl"
    rows = _load_rows(dataset_path)
    assert len(rows) == 500
    assert {row["task_id"] for row in rows} == {"MCQ-SYNTH-PITCH-ORDER-TRIVIAL"}
    assert {row["benchmark_family"] for row in rows} == {"pitch_order_trivial"}
    assert {row["difficulty"] for row in rows} == {"easy"}
    assert {row["question"] for row in rows} == {
        "You will hear a high beep and a low beep. Which one happened first?"
    }
    assert {row["question_template"] for row in rows} == {"high_vs_low_first"}

    high_first = 0
    low_first = 0
    for row in rows:
        assert len(row["options"]) == 2
        labels = [option["label"] for option in row["options"]]
        assert len(labels) == len(set(labels))
        assert row["answer_label"] in labels
        assert {option["text"] for option in row["options"]} == {"the high beep", "the low beep"}

        events = row["scene"]["events"]
        assert len(events) == 2
        assert {event["identity"] for event in events} == {"high", "low"}
        assert {event["waveform"] for event in events} == {"sine"}
        assert {event["duration"] for event in events} == {0.15}
        assert {event["dbfs"] for event in events} == {-18.0}
        assert {event["pitch_hz"] for event in events} == {220.0, 880.0}
        assert [event["onset"] for event in events] == [0.5, 2.0]
        first_event = min(events, key=lambda event: event["onset"])
        if first_event["identity"] == "high":
            high_first += 1
        else:
            low_first += 1
        assert row["answer_text"] == f"the {first_event['identity']} beep"

        audio_path = tmp_path / "audio" / str(row["audio_filename"])
        assert audio_path.exists()
        samples, sample_rate = sf.read(audio_path)
        assert sample_rate == 16000
        assert samples.size > 0

    assert high_first == 250
    assert low_first == 250
    assert not (tmp_path / "data" / "mcq_synth_benchmark.jsonl").exists()


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


def test_pitch_order_trivial_generation_is_deterministic(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    dataset_root = tmp_path / "data"

    main(
        benchmark="pitch_order_trivial",
        difficulty="easy",
        scenes_per_split=4,
        seed=11,
        sample_rate=16000,
        audio_root=audio_root,
        dataset_root=dataset_root,
        generator_version="synth-v1",
    )
    dataset_path = dataset_root / "mcq_synth_pitch_order_trivial_easy.jsonl"
    first_dataset = dataset_path.read_text(encoding="utf-8")
    first_audio = (
        audio_root / "synthetic" / "pitch_order_trivial" / "easy" / "synth_pitch_order_trivial_easy_000001.wav"
    ).read_bytes()

    main(
        benchmark="pitch_order_trivial",
        difficulty="easy",
        scenes_per_split=4,
        seed=11,
        sample_rate=16000,
        audio_root=audio_root,
        dataset_root=dataset_root,
        generator_version="synth-v1",
    )
    second_dataset = dataset_path.read_text(encoding="utf-8")
    second_audio = (
        audio_root / "synthetic" / "pitch_order_trivial" / "easy" / "synth_pitch_order_trivial_easy_000001.wav"
    ).read_bytes()

    assert first_dataset == second_dataset
    assert first_audio == second_audio


def test_temporal_trivial_standalone_families_write_expected_rows(tmp_path: Path) -> None:
    families = {
        "loudness_order_trivial": "MCQ-SYNTH-LOUDNESS-ORDER-TRIVIAL",
        "duration_order_trivial": "MCQ-SYNTH-DURATION-ORDER-TRIVIAL",
        "count_beeps_trivial": "MCQ-SYNTH-COUNT-BEEPS-TRIVIAL",
        "gap_trivial": "MCQ-SYNTH-GAP-TRIVIAL",
        "pattern_pitch_trivial": "MCQ-SYNTH-PATTERN-PITCH-TRIVIAL",
        "dog_car_order_trivial": "MCQ-SYNTH-DOG-CAR-ORDER-TRIVIAL",
    }

    for benchmark, task_id in families.items():
        rows = _build_standalone(tmp_path, benchmark)
        assert len(rows) == 100
        assert {row["task_id"] for row in rows} == {task_id}
        assert {row["benchmark_family"] for row in rows} == {benchmark}
        assert {row["difficulty"] for row in rows} == {"easy"}
        assert not (tmp_path / "data" / "mcq_synth_benchmark.jsonl").exists()

        for row in rows:
            audio_path = tmp_path / "audio" / str(row["audio_filename"])
            assert audio_path.exists()
            samples, sample_rate = sf.read(audio_path)
            assert sample_rate == 16000
            assert samples.size > 0


def test_loudness_order_trivial_invariants(tmp_path: Path) -> None:
    rows = _build_standalone(tmp_path, "loudness_order_trivial")
    answers = Counter()
    for row in rows:
        assert len(row["options"]) == 2
        assert {option["text"] for option in row["options"]} == {"the loud beep", "the quiet beep"}
        assert row["question"] == "You will hear a loud beep and a quiet beep. Which one happened first?"
        assert row["question_template"] == "loud_vs_quiet_first"
        events = row["scene"]["events"]
        assert len(events) == 2
        assert [event["onset"] for event in events] == [0.5, 2.0]
        assert {event["pitch_hz"] for event in events} == {440.0}
        assert {event["duration"] for event in events} == {0.15}
        assert {event["dbfs"] for event in events} == {-6.0, -24.0}
        first_event = min(events, key=lambda event: event["onset"])
        answers[first_event["identity"]] += 1
        assert row["answer_text"] == first_event["text"]
    assert answers == Counter({"loud": 50, "quiet": 50})


def test_duration_order_trivial_invariants(tmp_path: Path) -> None:
    rows = _build_standalone(tmp_path, "duration_order_trivial")
    answers = Counter()
    for row in rows:
        assert len(row["options"]) == 2
        assert {option["text"] for option in row["options"]} == {"the long beep", "the short beep"}
        assert row["question"] == "You will hear a long beep and a short beep. Which one happened first?"
        assert row["question_template"] == "long_vs_short_first"
        events = row["scene"]["events"]
        assert len(events) == 2
        assert [event["onset"] for event in events] == [0.5, 2.0]
        assert {event["pitch_hz"] for event in events} == {440.0}
        assert {event["dbfs"] for event in events} == {-18.0}
        assert {event["duration"] for event in events} == {1.0, 0.1}
        first_event = min(events, key=lambda event: event["onset"])
        answers[first_event["identity"]] += 1
        assert row["answer_text"] == first_event["text"]
    assert answers == Counter({"long": 50, "short": 50})


def test_count_beeps_trivial_invariants(tmp_path: Path) -> None:
    rows = _build_standalone(tmp_path, "count_beeps_trivial")
    counts = Counter()
    for row in rows:
        assert len(row["options"]) == 3
        assert {option["text"] for option in row["options"]} == {"one beep", "two beeps", "three beeps"}
        assert row["question"] == "How many beeps did you hear?"
        assert row["question_template"] == "count_beeps"
        events = row["scene"]["events"]
        observed_count = len(events)
        counts[observed_count] += 1
        assert row["scene"]["count"] == observed_count
        assert row["scene"]["gap_seconds"] == 1.0
        assert {event["pitch_hz"] for event in events} == {440.0}
        assert {event["duration"] for event in events} == {0.15}
        assert {event["dbfs"] for event in events} == {-18.0}
        assert row["answer_text"] == {1: "one beep", 2: "two beeps", 3: "three beeps"}[observed_count]
    assert counts == Counter({1: 34, 2: 33, 3: 33})


def test_gap_trivial_invariants(tmp_path: Path) -> None:
    rows = _build_standalone(tmp_path, "gap_trivial")
    gaps = Counter()
    for row in rows:
        assert len(row["options"]) == 2
        assert {option["text"] for option in row["options"]} == {"a short pause", "a long pause"}
        assert row["question"] == "Was the pause between the two beeps short or long?"
        assert row["question_template"] == "short_vs_long_pause"
        events = row["scene"]["events"]
        assert len(events) == 2
        assert {event["pitch_hz"] for event in events} == {440.0}
        assert {event["duration"] for event in events} == {0.15}
        assert {event["dbfs"] for event in events} == {-18.0}
        gap_seconds = round(events[1]["onset"] - events[0]["offset"], 3)
        gaps[gap_seconds] += 1
        assert row["scene"]["gap_seconds"] == gap_seconds
        assert row["answer_text"] == {0.2: "a short pause", 2.0: "a long pause"}[gap_seconds]
    assert gaps == Counter({0.2: 50, 2.0: 50})


def test_pattern_pitch_trivial_invariants(tmp_path: Path) -> None:
    rows = _build_standalone(tmp_path, "pattern_pitch_trivial")
    patterns = Counter()
    for row in rows:
        assert len(row["options"]) == 2
        assert {option["text"] for option in row["options"]} == {"high-low-high", "low-high-low"}
        assert row["question"] == "Which pattern did you hear?"
        assert row["question_template"] == "pitch_pattern"
        events = row["scene"]["events"]
        assert [event["onset"] for event in events] == [0.5, 1.5, 2.5]
        assert {event["duration"] for event in events} == {0.15}
        assert {event["dbfs"] for event in events} == {-18.0}
        pattern = tuple(event["identity"] for event in events)
        patterns[pattern] += 1
        assert row["scene"]["pattern"] == list(pattern)
        assert row["answer_text"] == "-".join(pattern)
    assert patterns == Counter({("high", "low", "high"): 50, ("low", "high", "low"): 50})


def test_dog_car_order_trivial_invariants(tmp_path: Path) -> None:
    rows = _build_standalone(tmp_path, "dog_car_order_trivial")
    answers = Counter()
    for row in rows:
        assert len(row["options"]) == 2
        assert {option["text"] for option in row["options"]} == {"the dog sound", "the car sound"}
        assert row["question"] == "You will hear a dog sound and a car sound. Which one happened first?"
        assert row["question_template"] == "dog_vs_car_first"
        events = row["scene"]["events"]
        assert len(events) == 2
        assert {event["identity"] for event in events} == {"dog", "car"}
        assert {event["waveform"] for event in events} == {"asset"}
        assert {event["asset_filename"] for event in events} == {"dog_bark.wav", "car_horn.wav"}
        answers[events[0]["identity"]] += 1
        assert round(events[1]["onset"] - events[0]["offset"], 3) == 1.5
        assert row["scene"]["gap_seconds"] == 1.5
        assert row["answer_text"] == events[0]["text"]
    assert answers == Counter({"dog": 50, "car": 50})


def test_new_trivial_families_generation_is_deterministic(tmp_path: Path) -> None:
    benchmarks = [
        "loudness_order_trivial",
        "duration_order_trivial",
        "count_beeps_trivial",
        "gap_trivial",
        "pattern_pitch_trivial",
        "dog_car_order_trivial",
    ]
    for benchmark in benchmarks:
        audio_root = tmp_path / benchmark / "audio"
        dataset_root = tmp_path / benchmark / "data"
        main(
            benchmark=benchmark,
            difficulty="easy",
            scenes_per_split=4,
            seed=11,
            sample_rate=16000,
            audio_root=audio_root,
            dataset_root=dataset_root,
            generator_version="synth-v1",
        )
        dataset_path = dataset_root / f"mcq_synth_{benchmark}_easy.jsonl"
        first_dataset = dataset_path.read_text(encoding="utf-8")
        first_audio = sorted((audio_root / "synthetic" / benchmark / "easy").glob("*.wav"))[0].read_bytes()

        main(
            benchmark=benchmark,
            difficulty="easy",
            scenes_per_split=4,
            seed=11,
            sample_rate=16000,
            audio_root=audio_root,
            dataset_root=dataset_root,
            generator_version="synth-v1",
        )
        second_dataset = dataset_path.read_text(encoding="utf-8")
        second_audio = sorted((audio_root / "synthetic" / benchmark / "easy").glob("*.wav"))[0].read_bytes()

        assert first_dataset == second_dataset
        assert first_audio == second_audio
