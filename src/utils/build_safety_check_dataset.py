"""Build an easy binary safety-check MCQ dataset from strong TACoBeLAL annotations."""

from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

console = Console()

TASK_ID_MCQ_SAFETY = "MCQ-SAFETY"
QUESTION_TEMPLATE = 'Is this event in the audio: "{event}"?'


@dataclass
class BuildStats:
    rows_total: int = 0
    rows_missing_fields: int = 0
    rows_kept: int = 0
    audios_total: int = 0
    unique_audio_events: int = 0
    audios_eligible: int = 0
    audios_selected: int = 0
    examples_written: int = 0
    examples_true: int = 0
    examples_false: int = 0


def _normalize_event_text(text: str) -> str:
    normalized = text.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.rstrip(" .,!?:;")
    return normalized


def _option_label(index: int) -> str:
    """Map 0 -> A, 25 -> Z, 26 -> AA, ..."""
    if index < 0:
        raise ValueError("Option index must be non-negative.")

    label = ""
    value = index
    while True:
        value, remainder = divmod(value, 26)
        label = chr(ord("A") + remainder) + label
        if value == 0:
            return label
        value -= 1


def _load_audio_events(input_csv: Path, stats: BuildStats) -> dict[str, dict[str, str]]:
    """Load normalized unique event text per audio; keep first display variant."""
    by_audio: dict[str, dict[str, str]] = {}

    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats.rows_total += 1
            filename = (row.get("filename") or "").strip()
            text = (row.get("text") or "").strip()
            if not filename or not text:
                stats.rows_missing_fields += 1
                continue

            normalized = _normalize_event_text(text)
            if not normalized:
                stats.rows_missing_fields += 1
                continue

            stats.rows_kept += 1
            events = by_audio.setdefault(filename, {})
            events.setdefault(normalized, text)

    stats.audios_total = len(by_audio)
    stats.unique_audio_events = sum(len(events) for events in by_audio.values())
    return by_audio


def _build_negative_pool(by_audio: dict[str, dict[str, str]]) -> list[tuple[str, str, str]]:
    """Build deterministic pool of (normalized_text, display_text, source_audio)."""
    pool: list[tuple[str, str, str]] = []
    for audio_filename in sorted(by_audio):
        events = by_audio[audio_filename]
        for normalized_text in sorted(events):
            pool.append((normalized_text, events[normalized_text], audio_filename))
    return pool


def _sample_negative_event(
    *,
    target_audio: str,
    target_events: set[str],
    negative_pool: list[tuple[str, str, str]],
    rng: random.Random,
) -> tuple[str, str, str]:
    if not negative_pool:
        raise ValueError("Cannot sample negatives from an empty pool.")

    max_attempts = min(512, max(32, len(negative_pool)))
    for _ in range(max_attempts):
        normalized_text, display_text, source_audio = negative_pool[rng.randrange(len(negative_pool))]
        if source_audio != target_audio and normalized_text not in target_events:
            return normalized_text, display_text, source_audio

    start = rng.randrange(len(negative_pool))
    for offset in range(len(negative_pool)):
        normalized_text, display_text, source_audio = negative_pool[(start + offset) % len(negative_pool)]
        if source_audio != target_audio and normalized_text not in target_events:
            return normalized_text, display_text, source_audio

    raise RuntimeError(f"Failed to sample a valid negative event for audio '{target_audio}'.")


def _build_options(*, rng: random.Random) -> list[dict[str, Any]]:
    options = [
        {"type": "true", "text": "True"},
        {"type": "false", "text": "False"},
    ]
    rng.shuffle(options)

    labeled_options: list[dict[str, Any]] = []
    for i, option in enumerate(options):
        labeled = dict(option)
        labeled["label"] = _option_label(i)
        labeled_options.append(labeled)
    return labeled_options


def _build_example(
    *,
    audio_filename: str,
    query_event: str,
    query_event_normalized: str,
    query_source: str,
    query_source_audio: str,
    answer_type: str,
    rng: random.Random,
) -> dict[str, Any]:
    options = _build_options(rng=rng)

    answer_label = None
    answer_text = None
    for option in options:
        if option["type"] == answer_type:
            answer_label = option["label"]
            answer_text = option["text"]
            break

    if answer_label is None or answer_text is None:
        raise RuntimeError("Failed to locate answer option.")

    return {
        "id": f"{audio_filename}__safety_presence",
        "task": "event_presence_binary",
        "task_id": TASK_ID_MCQ_SAFETY,
        "audio_filename": audio_filename,
        "question": QUESTION_TEMPLATE.format(event=query_event),
        "query_event": query_event,
        "query_event_normalized": query_event_normalized,
        "query_source": query_source,
        "query_source_audio": query_source_audio,
        "options": options,
        "answer_label": answer_label,
        "answer_type": answer_type,
        "answer_text": answer_text,
    }


def _write_jsonl(examples: list[dict[str, Any]], output_jsonl: Path) -> int:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    return len(examples)


def _print_summary(
    *,
    stats: BuildStats,
    input_csv: Path,
    output_jsonl: Path,
    num_audios: int,
    seed: int,
) -> None:
    table = Table(title="Safety-check MCQ build summary", show_header=True, header_style="bold")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Task ID", TASK_ID_MCQ_SAFETY)
    table.add_row("Input CSV", str(input_csv))
    table.add_row("Output JSONL", str(output_jsonl))
    table.add_row("Num audios requested", str(num_audios))
    table.add_row("Deterministic seed", str(seed))
    table.add_row("Rows read", str(stats.rows_total))
    table.add_row("Rows dropped (missing fields)", str(stats.rows_missing_fields))
    table.add_row("Rows kept", str(stats.rows_kept))
    table.add_row("Audios with events", str(stats.audios_total))
    table.add_row("Unique audio-events", str(stats.unique_audio_events))
    table.add_row("Eligible audios", str(stats.audios_eligible))
    table.add_row("Selected audios", str(stats.audios_selected))
    table.add_row("Examples written", str(stats.examples_written))
    table.add_row("Ground-truth True", str(stats.examples_true))
    table.add_row("Ground-truth False", str(stats.examples_false))
    console.print(table)


def main(
    input_csv: Path = typer.Option(
        Path("data/annotations_strong.csv"),
        "--input",
        "-i",
        path_type=Path,
        help="Strong annotations CSV (filename,text,onset,offset).",
        exists=True,
        dir_okay=False,
    ),
    output_jsonl: Path = typer.Option(
        Path("data/mcq_safety_presence_100.jsonl"),
        "--output",
        "-o",
        path_type=Path,
        help="Destination JSONL file.",
    ),
    num_audios: int = typer.Option(
        100,
        "--num-audios",
        help="Number of audios/examples to include (must be even).",
        min=2,
    ),
    seed: int = typer.Option(
        7,
        "--seed",
        help="Deterministic seed for audio selection and option ordering.",
    ),
) -> None:
    if num_audios % 2 != 0:
        raise typer.BadParameter("--num-audios must be even to guarantee a 50/50 true-false split.")

    stats = BuildStats()
    rng = random.Random(seed)

    by_audio = _load_audio_events(input_csv=input_csv, stats=stats)
    all_normalized_events = set()
    for events in by_audio.values():
        all_normalized_events.update(events.keys())

    eligible_audios = [
        audio_filename
        for audio_filename, events in by_audio.items()
        if events and len(events) < len(all_normalized_events)
    ]
    stats.audios_eligible = len(eligible_audios)

    if len(eligible_audios) < num_audios:
        raise ValueError(
            f"Requested {num_audios} audios, but only {len(eligible_audios)} eligible audios are available."
        )

    eligible_audios = sorted(eligible_audios)
    rng.shuffle(eligible_audios)
    selected_audios = eligible_audios[:num_audios]
    stats.audios_selected = len(selected_audios)

    answer_types = (["true"] * (num_audios // 2)) + (["false"] * (num_audios // 2))
    rng.shuffle(answer_types)

    negative_pool = _build_negative_pool(by_audio)
    examples: list[dict[str, Any]] = []

    for audio_filename, answer_type in zip(selected_audios, answer_types):
        audio_events = by_audio[audio_filename]
        audio_event_items = sorted(audio_events.items())
        audio_event_set = set(audio_events.keys())

        if answer_type == "true":
            query_event_normalized, query_event = rng.choice(audio_event_items)
            query_source = "positive"
            query_source_audio = audio_filename
            stats.examples_true += 1
        else:
            query_event_normalized, query_event, query_source_audio = _sample_negative_event(
                target_audio=audio_filename,
                target_events=audio_event_set,
                negative_pool=negative_pool,
                rng=rng,
            )
            query_source = "negative"
            stats.examples_false += 1

        examples.append(
            _build_example(
                audio_filename=audio_filename,
                query_event=query_event,
                query_event_normalized=query_event_normalized,
                query_source=query_source,
                query_source_audio=query_source_audio,
                answer_type=answer_type,
                rng=rng,
            )
        )

    stats.examples_written = _write_jsonl(examples=examples, output_jsonl=output_jsonl)
    _print_summary(
        stats=stats,
        input_csv=input_csv,
        output_jsonl=output_jsonl,
        num_audios=num_audios,
        seed=seed,
    )


if __name__ == "__main__":
    typer.run(main)
