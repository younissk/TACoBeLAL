"""Build deterministic synthetic MCQ benchmark datasets and audio."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import typer
from rich.console import Console
from rich.table import Table

try:
    from .synthetic_audio import midi_to_hz, render_timeline, write_wav
except ImportError:  # pragma: no cover - enables direct script execution
    from synthetic_audio import midi_to_hz, render_timeline, write_wav

console = Console()

BENCHMARKS = ("time", "pitch", "loudness", "rhythm")
DIFFICULTIES = ("easy", "medium", "hard")
AGGREGATE_TASK_ID = "MCQ-SYNTH"
AGGREGATE_TASK_NAME = "synthetic_mcq_benchmark"
AGGREGATE_DATASET_NAME = "mcq_synth_benchmark.jsonl"

TASK_IDS = {
    "time": "MCQ-SYNTH-TIME",
    "pitch": "MCQ-SYNTH-PITCH",
    "loudness": "MCQ-SYNTH-LOUDNESS",
    "rhythm": "MCQ-SYNTH-RHYTHM",
}

TASK_NAMES = {
    "time": "synthetic_time_mcq",
    "pitch": "synthetic_pitch_mcq",
    "loudness": "synthetic_loudness_mcq",
    "rhythm": "synthetic_rhythm_mcq",
}

TIME_WAVEFORMS = [
    ("sine", "the sine tone", "sine"),
    ("triangle", "the triangle tone", "triangle"),
    ("square", "the square tone", "square"),
]

POSITION_OPTIONS = [
    ("first", "the first sound"),
    ("second", "the second sound"),
    ("third", "the third sound"),
]

RHYTHM_OPTIONS = [
    ("first_burst", "the first burst"),
    ("second_burst", "the second burst"),
]

TIME_TEMPLATE_IDS = ("starts_first", "starts_last", "longest_duration", "shortest_duration")
PITCH_TEMPLATE_IDS = ("highest_pitch", "lowest_pitch")
LOUDNESS_TEMPLATE_IDS = ("loudest", "quietest")


@dataclass(frozen=True)
class SplitSummary:
    benchmark: str
    difficulty: str
    scenes_written: int
    dataset_path: Path
    audio_dir: Path
    question_templates: Counter[str]


def _option_label(index: int) -> str:
    label = ""
    value = index
    while True:
        value, remainder = divmod(value, 26)
        label = chr(ord("A") + remainder) + label
        if value == 0:
            return label
        value -= 1


def _stable_seed(*parts: object) -> int:
    joined = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _rng_for(seed: int, benchmark: str, difficulty: str, scene_index: int) -> random.Random:
    return random.Random(_stable_seed(seed, benchmark, difficulty, scene_index))


def _deterministic_shuffle_options(
    options: list[dict[str, Any]],
    *,
    seed: int,
    example_id: str,
) -> list[dict[str, Any]]:
    keyed: list[tuple[str, dict[str, Any]]] = []
    for idx, option in enumerate(options):
        option_key = str(option.get("key", option.get("text", idx)))
        digest = hashlib.sha256(f"{seed}:{example_id}:{idx}:{option_key}".encode("utf-8")).hexdigest()
        keyed.append((digest, option))
    keyed.sort(key=lambda item: item[0])
    shuffled = [dict(item[1]) for item in keyed]
    for idx, option in enumerate(shuffled):
        option["label"] = _option_label(idx)
    return shuffled


def _sample_from_grid(
    rng: random.Random,
    *,
    values: list[float],
    count: int,
    min_gap: float,
) -> list[float]:
    if count > len(values):
        raise ValueError("Cannot sample more unique values than the grid contains.")

    for _ in range(10_000):
        sampled = sorted(rng.sample(values, count))
        if all(abs(sampled[idx] - sampled[idx - 1]) >= min_gap for idx in range(1, len(sampled))):
            return sampled
    raise ValueError(
        f"Unable to sample {count} values from grid of size {len(values)} with minimum gap {min_gap}."
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _time_scene(
    *,
    rng: random.Random,
    scene_index: int,
    difficulty: str,
    sample_rate: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gap_rules = {
        "easy": {"min_gap": 0.50, "min_duration_diff": 0.25},
        "medium": {"min_gap": 0.30, "min_duration_diff": 0.15},
        "hard": {"min_gap": 0.15, "min_duration_diff": 0.08},
    }[difficulty]
    duration_grid = [round(value, 2) for value in [0.28, 0.40, 0.52, 0.64, 0.76, 0.88, 1.00, 1.12]]
    gap_grid = [round(value, 2) for value in [0.18, 0.25, 0.32, 0.40, 0.48, 0.56, 0.64, 0.72, 0.80]]

    durations = _sample_from_grid(
        rng,
        values=duration_grid,
        count=3,
        min_gap=gap_rules["min_duration_diff"],
    )
    gaps = _sample_from_grid(rng, values=gap_grid, count=2, min_gap=0.0)
    gaps = [max(gap_rules["min_gap"], gap) for gap in gaps]

    onsets = [0.40]
    onsets.append(round(onsets[0] + durations[0] + gaps[0], 3))
    onsets.append(round(onsets[1] + durations[1] + gaps[1], 3))

    identities = list(TIME_WAVEFORMS)
    rng.shuffle(identities)
    events: list[dict[str, Any]] = []
    render_events: list[dict[str, float | str]] = []
    for idx, (onset, duration) in enumerate(zip(onsets, durations)):
        key, text, waveform = identities[idx]
        offset = round(onset + duration, 3)
        event = {
            "key": key,
            "text": text,
            "waveform": waveform,
            "onset": onset,
            "duration": duration,
            "offset": offset,
            "pitch_hz": 440.0,
            "dbfs": -18.0,
        }
        events.append(event)
        render_events.append(
            {
                "waveform": waveform,
                "pitch_hz": 440.0,
                "dbfs": -18.0,
                "onset": onset,
                "duration": duration,
            }
        )

    template = TIME_TEMPLATE_IDS[scene_index % len(TIME_TEMPLATE_IDS)]
    if template == "starts_first":
        question = "Which sound starts first?"
        answer_key = min(events, key=lambda event: event["onset"])["key"]
    elif template == "starts_last":
        question = "Which sound starts last?"
        answer_key = max(events, key=lambda event: event["onset"])["key"]
    elif template == "longest_duration":
        question = "Which sound lasts the longest?"
        answer_key = max(events, key=lambda event: event["duration"])["key"]
    else:
        question = "Which sound lasts the shortest time?"
        answer_key = min(events, key=lambda event: event["duration"])["key"]

    total_duration = round(max(event["offset"] for event in events) + 0.40, 3)
    waveform = render_timeline(
        events=render_events,
        total_duration_seconds=total_duration,
        sample_rate=sample_rate,
    )
    scene = {
        "duration_seconds": total_duration,
        "events": sorted(events, key=lambda event: (event["onset"], event["key"])),
    }
    return [
        {"key": key, "text": text, "type": "event"} for key, text, _ in TIME_WAVEFORMS
    ], {
        "question": question,
        "question_template": template,
        "answer_key": answer_key,
        "scene": scene,
        "waveform": waveform,
    }


def _pitch_scene(
    *,
    rng: random.Random,
    scene_index: int,
    difficulty: str,
    sample_rate: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    min_gap = {"easy": 7, "medium": 4, "hard": 2}[difficulty]
    midi_notes = _sample_from_grid(
        rng,
        values=[float(note) for note in range(48, 85)],
        count=3,
        min_gap=float(min_gap),
    )
    rng.shuffle(midi_notes)

    onsets = [0.40, 1.20, 2.00]
    duration = 0.40
    events: list[dict[str, Any]] = []
    render_events: list[dict[str, float | str]] = []
    for idx, (key, text) in enumerate(POSITION_OPTIONS):
        pitch_hz = round(midi_to_hz(int(midi_notes[idx])), 6)
        offset = round(onsets[idx] + duration, 3)
        event = {
            "key": key,
            "text": text,
            "waveform": "sine",
            "onset": onsets[idx],
            "duration": duration,
            "offset": offset,
            "pitch_hz": pitch_hz,
            "midi_note": int(midi_notes[idx]),
            "dbfs": -18.0,
        }
        events.append(event)
        render_events.append(
            {
                "waveform": "sine",
                "pitch_hz": pitch_hz,
                "dbfs": -18.0,
                "onset": onsets[idx],
                "duration": duration,
            }
        )

    template = PITCH_TEMPLATE_IDS[scene_index % len(PITCH_TEMPLATE_IDS)]
    if template == "highest_pitch":
        question = "Which sound has the highest pitch?"
        answer_key = max(events, key=lambda event: event["pitch_hz"])["key"]
    else:
        question = "Which sound has the lowest pitch?"
        answer_key = min(events, key=lambda event: event["pitch_hz"])["key"]

    total_duration = round(max(event["offset"] for event in events) + 0.40, 3)
    waveform = render_timeline(
        events=render_events,
        total_duration_seconds=total_duration,
        sample_rate=sample_rate,
    )
    scene = {
        "duration_seconds": total_duration,
        "events": events,
    }
    return [
        {"key": key, "text": text, "type": "event"} for key, text in POSITION_OPTIONS
    ], {
        "question": question,
        "question_template": template,
        "answer_key": answer_key,
        "scene": scene,
        "waveform": waveform,
    }


def _loudness_scene(
    *,
    rng: random.Random,
    scene_index: int,
    difficulty: str,
    sample_rate: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    min_gap = {"easy": 8.0, "medium": 5.0, "hard": 3.0}[difficulty]
    levels = _sample_from_grid(
        rng,
        values=[float(level) for level in range(-30, -7)],
        count=3,
        min_gap=min_gap,
    )
    rng.shuffle(levels)

    onsets = [0.40, 1.20, 2.00]
    duration = 0.40
    events: list[dict[str, Any]] = []
    render_events: list[dict[str, float | str]] = []
    for idx, (key, text) in enumerate(POSITION_OPTIONS):
        offset = round(onsets[idx] + duration, 3)
        event = {
            "key": key,
            "text": text,
            "waveform": "sine",
            "onset": onsets[idx],
            "duration": duration,
            "offset": offset,
            "pitch_hz": 440.0,
            "dbfs": float(levels[idx]),
        }
        events.append(event)
        render_events.append(
            {
                "waveform": "sine",
                "pitch_hz": 440.0,
                "dbfs": float(levels[idx]),
                "onset": onsets[idx],
                "duration": duration,
            }
        )

    template = LOUDNESS_TEMPLATE_IDS[scene_index % len(LOUDNESS_TEMPLATE_IDS)]
    if template == "loudest":
        question = "Which sound is the loudest?"
        answer_key = max(events, key=lambda event: event["dbfs"])["key"]
    else:
        question = "Which sound is the quietest?"
        answer_key = min(events, key=lambda event: event["dbfs"])["key"]

    total_duration = round(max(event["offset"] for event in events) + 0.40, 3)
    waveform = render_timeline(
        events=render_events,
        total_duration_seconds=total_duration,
        sample_rate=sample_rate,
    )
    scene = {
        "duration_seconds": total_duration,
        "events": events,
    }
    return [
        {"key": key, "text": text, "type": "event"} for key, text in POSITION_OPTIONS
    ], {
        "question": question,
        "question_template": template,
        "answer_key": answer_key,
        "scene": scene,
        "waveform": waveform,
    }


def _rhythm_scene(
    *,
    rng: random.Random,
    scene_index: int,
    difficulty: str,
    sample_rate: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    del scene_index
    min_gap = {"easy": 3, "medium": 2, "hard": 1}[difficulty]
    while True:
        first_count = rng.randint(3, 9)
        second_count = rng.randint(3, 9)
        if first_count != second_count and abs(first_count - second_count) >= min_gap:
            break

    burst_starts = [0.40, 2.00]
    burst_window = 0.90
    pip_duration = 0.05
    pitch_hz = 1000.0
    dbfs = -18.0

    bursts: list[dict[str, Any]] = []
    render_events: list[dict[str, float | str]] = []
    for idx, ((key, text), count) in enumerate(zip(RHYTHM_OPTIONS, [first_count, second_count])):
        interval = 0.0 if count == 1 else (burst_window - pip_duration) / (count - 1)
        pip_onsets = [round(burst_starts[idx] + interval * pip_index, 4) for pip_index in range(count)]
        bursts.append(
            {
                "key": key,
                "text": text,
                "start": burst_starts[idx],
                "count": count,
                "window_seconds": burst_window,
                "pip_duration": pip_duration,
                "pitch_hz": pitch_hz,
                "dbfs": dbfs,
                "pip_onsets": pip_onsets,
            }
        )
        for pip_onset in pip_onsets:
            render_events.append(
                {
                    "waveform": "sine",
                    "pitch_hz": pitch_hz,
                    "dbfs": dbfs,
                    "onset": pip_onset,
                    "duration": pip_duration,
                }
            )

    question = "Which burst contains more beeps?"
    answer_key = "first_burst" if first_count > second_count else "second_burst"
    total_duration = round(burst_starts[-1] + burst_window + 0.40, 3)
    waveform = render_timeline(
        events=render_events,
        total_duration_seconds=total_duration,
        sample_rate=sample_rate,
    )
    scene = {
        "duration_seconds": total_duration,
        "bursts": bursts,
    }
    return [
        {"key": key, "text": text, "type": "event"} for key, text in RHYTHM_OPTIONS
    ], {
        "question": question,
        "question_template": "more_beeps",
        "answer_key": answer_key,
        "scene": scene,
        "waveform": waveform,
    }


def _build_split(
    *,
    benchmark: str,
    difficulty: str,
    scenes_per_split: int,
    seed: int,
    sample_rate: int,
    audio_root: Path,
    dataset_root: Path,
    generator_version: str,
) -> SplitSummary:
    dataset_path = dataset_root / f"mcq_synth_{benchmark}_{difficulty}.jsonl"
    audio_dir = audio_root / "synthetic" / benchmark / difficulty
    manifest_path = dataset_root / "synthetic" / "manifests" / f"{benchmark}_{difficulty}_summary.json"

    if audio_dir.exists():
        shutil.rmtree(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    template_counts: Counter[str] = Counter()
    scene_builder = {
        "time": _time_scene,
        "pitch": _pitch_scene,
        "loudness": _loudness_scene,
        "rhythm": _rhythm_scene,
    }[benchmark]

    for scene_index in range(scenes_per_split):
        rng = _rng_for(seed, benchmark, difficulty, scene_index)
        example_id = f"synth_{benchmark}_{difficulty}_{scene_index + 1:06d}"
        audio_relative = Path("synthetic") / benchmark / difficulty / f"{example_id}.wav"
        audio_path = audio_root / audio_relative

        option_defs, payload = scene_builder(
            rng=rng,
            scene_index=scene_index,
            difficulty=difficulty,
            sample_rate=sample_rate,
        )
        options = _deterministic_shuffle_options(option_defs, seed=seed, example_id=example_id)

        answer_label = None
        answer_text = None
        for option in options:
            if option["key"] == payload["answer_key"]:
                answer_label = option["label"]
                answer_text = option["text"]
                break
        if answer_label is None or answer_text is None:
            raise RuntimeError(f"Failed to resolve answer option for '{example_id}'.")

        write_wav(audio_path, samples=payload["waveform"], sample_rate=sample_rate)
        template_counts[payload["question_template"]] += 1
        rows.append(
            {
                "id": example_id,
                "task": TASK_NAMES[benchmark],
                "task_id": TASK_IDS[benchmark],
                "audio_filename": audio_relative.as_posix(),
                "question": payload["question"],
                "options": options,
                "answer_label": answer_label,
                "answer_text": answer_text,
                "benchmark_family": benchmark,
                "difficulty": difficulty,
                "generator_version": generator_version,
                "question_template": payload["question_template"],
                "scene": payload["scene"],
            }
        )

    written = _write_jsonl(dataset_path, rows)
    manifest_payload = {
        "benchmark": benchmark,
        "difficulty": difficulty,
        "task": TASK_NAMES[benchmark],
        "task_id": TASK_IDS[benchmark],
        "scenes_written": written,
        "seed": seed,
        "sample_rate": sample_rate,
        "dataset_path": str(dataset_path),
        "audio_dir": str(audio_dir),
        "question_templates": dict(sorted(template_counts.items())),
        "generator_version": generator_version,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return SplitSummary(
        benchmark=benchmark,
        difficulty=difficulty,
        scenes_written=written,
        dataset_path=dataset_path,
        audio_dir=audio_dir,
        question_templates=template_counts,
    )


def _parse_axis(value: str, *, allowed: tuple[str, ...], axis_name: str) -> list[str]:
    normalized = value.strip().lower()
    if normalized == "all":
        return list(allowed)
    if normalized not in allowed:
        allowed_text = ", ".join((*allowed, "all"))
        raise typer.BadParameter(f"Unsupported {axis_name} '{value}'. Choose one of: {allowed_text}")
    return [normalized]


def _print_summary(summaries: list[SplitSummary], *, seed: int, sample_rate: int, generator_version: str) -> None:
    table = Table(title="Synthetic MCQ build summary", show_header=True, header_style="bold")
    table.add_column("Benchmark", style="cyan")
    table.add_column("Difficulty", style="cyan")
    table.add_column("Scenes", style="green")
    table.add_column("Dataset", style="green")
    table.add_column("Audio Dir", style="green")

    for summary in summaries:
        table.add_row(
            summary.benchmark,
            summary.difficulty,
            str(summary.scenes_written),
            str(summary.dataset_path),
            str(summary.audio_dir),
        )
    console.print(table)
    console.print(
        f"[bold]Generator version:[/bold] {generator_version}  "
        f"[bold]Seed:[/bold] {seed}  "
        f"[bold]Sample rate:[/bold] {sample_rate} Hz"
    )


def _build_aggregate_dataset(
    *,
    summaries: list[SplitSummary],
    dataset_root: Path,
    seed: int,
    sample_rate: int,
    generator_version: str,
) -> Path:
    aggregate_path = dataset_root / AGGREGATE_DATASET_NAME
    manifest_path = dataset_root / "synthetic" / "manifests" / "benchmark_summary.json"

    rows: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    template_counts: Counter[str] = Counter()

    for summary in summaries:
        for row in _load_jsonl(summary.dataset_path):
            family = str(row["benchmark_family"])
            diff = str(row["difficulty"])
            source_task_id = str(row["task_id"])
            source_task = str(row["task"])

            aggregate_row = dict(row)
            aggregate_row["task"] = AGGREGATE_TASK_NAME
            aggregate_row["task_id"] = AGGREGATE_TASK_ID
            aggregate_row["source_task"] = source_task
            aggregate_row["source_task_id"] = source_task_id
            rows.append(aggregate_row)

            family_counts[family] += 1
            difficulty_counts[diff] += 1
            template_counts[str(row["question_template"])] += 1

    rows.sort(key=lambda row: str(row["id"]))
    _write_jsonl(aggregate_path, rows)

    manifest_payload = {
        "task": AGGREGATE_TASK_NAME,
        "task_id": AGGREGATE_TASK_ID,
        "questions_written": len(rows),
        "source_splits": [str(summary.dataset_path) for summary in summaries],
        "benchmark_family_counts": dict(sorted(family_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "question_templates": dict(sorted(template_counts.items())),
        "seed": seed,
        "sample_rate": sample_rate,
        "dataset_path": str(aggregate_path),
        "generator_version": generator_version,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    console.print(
        f"[bold]Aggregate synthetic benchmark:[/bold] {aggregate_path} "
        f"([bold]{len(rows)}[/bold] questions)"
    )
    return aggregate_path


def main(
    benchmark: str = typer.Option(
        "all",
        "--benchmark",
        help="Benchmark family to build (time|pitch|loudness|rhythm|all).",
    ),
    difficulty: str = typer.Option(
        "all",
        "--difficulty",
        help="Difficulty split to build (easy|medium|hard|all).",
    ),
    scenes_per_split: int = typer.Option(
        500,
        "--scenes-per-split",
        help="Number of audio/question pairs per benchmark+difficulty split.",
        min=1,
    ),
    seed: int = typer.Option(
        7,
        "--seed",
        help="Top-level deterministic seed.",
    ),
    sample_rate: int = typer.Option(
        16000,
        "--sample-rate",
        help="Audio sample rate in Hz.",
        min=8000,
    ),
    audio_root: Path = typer.Option(
        Path("data/audio"),
        "--audio-root",
        path_type=Path,
        help="Root directory where rendered synthetic audio will be written.",
    ),
    dataset_root: Path = typer.Option(
        Path("data"),
        "--dataset-root",
        path_type=Path,
        help="Root directory where JSONL datasets and manifests will be written.",
    ),
    generator_version: str = typer.Option(
        "synth-v1",
        "--generator-version",
        help="Version string saved into each row for reproducibility.",
    ),
) -> None:
    selected_benchmarks = _parse_axis(benchmark, allowed=BENCHMARKS, axis_name="benchmark")
    selected_difficulties = _parse_axis(difficulty, allowed=DIFFICULTIES, axis_name="difficulty")

    summaries: list[SplitSummary] = []
    for benchmark_key in selected_benchmarks:
        for difficulty_key in selected_difficulties:
            summaries.append(
                _build_split(
                    benchmark=benchmark_key,
                    difficulty=difficulty_key,
                    scenes_per_split=scenes_per_split,
                    seed=seed,
                    sample_rate=sample_rate,
                    audio_root=audio_root,
                    dataset_root=dataset_root,
                    generator_version=generator_version,
                )
            )

    if len(summaries) > 1:
        _build_aggregate_dataset(
            summaries=summaries,
            dataset_root=dataset_root,
            seed=seed,
            sample_rate=sample_rate,
            generator_version=generator_version,
        )

    _print_summary(summaries, seed=seed, sample_rate=sample_rate, generator_version=generator_version)


if __name__ == "__main__":
    typer.run(main)
