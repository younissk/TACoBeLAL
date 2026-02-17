"""Build timeline MCQ dataset from strong TACoBeLAL annotations."""

from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import typer
from rich.console import Console
from rich.table import Table

console = Console()

LAST_EVENT_TEXT_DEFAULT = "This is the last event, no immediate event after."


@dataclass(frozen=True)
class Event:
    index: int
    text: str
    onset: float
    offset: float
    duration: float
    captions: tuple[str, ...]
    occurrence_count: int
    last_offset: float


@dataclass
class BuildStats:
    rows_total: int = 0
    rows_invalid_time: int = 0
    rows_short: int = 0
    rows_kept: int = 0
    rows_collapsed_by_caption_grouping: int = 0
    events_after_caption_grouping: int = 0
    audios_total: int = 0
    audios_removed_too_few_events: int = 0
    audios_kept: int = 0
    questions_dropped_onset_tie_base: int = 0
    questions_dropped_tiebreak_answer: int = 0
    questions_dropped_repeat_ambiguity: int = 0
    questions_dropped_overlap_ambiguity: int = 0
    questions_written: int = 0


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


def _normalize_caption(text: str) -> str:
    normalized = text.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.rstrip(" .,!?:;")
    return normalized


def _load_filtered_events(input_csv: Path, min_duration: float, stats: BuildStats) -> Dict[str, List[Event]]:
    """Load strong events and keep only rows with duration >= min_duration.

    Events are grouped by normalized caption per audio and represented by their
    first occurrence. This avoids duplicate answer choices with identical text.
    """
    grouped_raw: Dict[str, Dict[str, Dict[str, Any]]] = {}

    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader):
            stats.rows_total += 1

            filename = (row.get("filename") or "").strip()
            text = (row.get("text") or "").strip()
            if not filename or not text:
                continue

            try:
                onset = float(row["onset"])
                offset = float(row["offset"])
            except (KeyError, TypeError, ValueError):
                stats.rows_invalid_time += 1
                continue

            duration = offset - onset
            if duration <= 0:
                stats.rows_invalid_time += 1
                continue
            if duration < min_duration:
                stats.rows_short += 1
                continue

            stats.rows_kept += 1
            caption_groups = grouped_raw.setdefault(filename, {})
            normalized_caption = _normalize_caption(text)
            group = caption_groups.get(normalized_caption)
            if group is None:
                caption_groups[normalized_caption] = {
                    "first_onset": onset,
                    "first_offset": offset,
                    "last_offset": offset,
                    "first_row": row_index,
                    "caption_variants": [text],
                    "intervals": {(onset, offset)},
                    "occurrence_count": 1,
                }
                continue

            if text not in group["caption_variants"]:
                group["caption_variants"].append(text)

            interval = (onset, offset)
            if interval not in group["intervals"]:
                group["intervals"].add(interval)
                group["occurrence_count"] += 1

            if onset < group["first_onset"] or (
                onset == group["first_onset"] and row_index < group["first_row"]
            ):
                group["first_onset"] = onset
                group["first_offset"] = offset
                group["first_row"] = row_index

            if offset > group["last_offset"]:
                group["last_offset"] = offset

    stats.audios_total = len(grouped_raw)

    grouped_events: Dict[str, List[Event]] = {}
    for filename, caption_groups in grouped_raw.items():
        raw_events = sorted(
            caption_groups.values(),
            key=lambda item: (item["first_onset"], item["first_offset"], item["first_row"]),
        )
        events: List[Event] = []
        for i, item in enumerate(raw_events):
            unique_captions = list(dict.fromkeys(c.strip() for c in item["caption_variants"] if c.strip()))
            chosen_caption = unique_captions[0]
            events.append(
                Event(
                    index=i,
                    text=chosen_caption,
                    onset=item["first_onset"],
                    offset=item["first_offset"],
                    duration=item["first_offset"] - item["first_onset"],
                    captions=tuple(unique_captions),
                    occurrence_count=item["occurrence_count"],
                    last_offset=item["last_offset"],
                )
            )
            stats.events_after_caption_grouping += 1
        grouped_events[filename] = events

    stats.rows_collapsed_by_caption_grouping = stats.rows_kept - stats.events_after_caption_grouping
    return grouped_events


def _drop_short_audios(
    grouped_events: Dict[str, List[Event]], min_events_per_audio: int, stats: BuildStats
) -> Dict[str, List[Event]]:
    """Keep audios that have at least min_events_per_audio events."""
    filtered: Dict[str, List[Event]] = {}
    for filename, events in grouped_events.items():
        if len(events) < min_events_per_audio:
            stats.audios_removed_too_few_events += 1
            continue
        filtered[filename] = events

    stats.audios_kept = len(filtered)
    return filtered


def _format_event(event: Event) -> Dict[str, Any]:
    return {
        "event_index": event.index,
        "text": event.text,
        "captions": list(event.captions),
        "onset": event.onset,
        "offset": event.offset,
        "duration": event.duration,
        "occurrence_count": event.occurrence_count,
        "last_offset": event.last_offset,
    }


def _build_options(
    events: List[Event],
    base_event_index: int,
    no_event_text: str,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    options: List[Dict[str, Any]] = []
    for event in events:
        if event.index == base_event_index:
            continue
        option: Dict[str, Any] = _format_event(event)
        option["type"] = "event"
        options.append(option)

    options.append({"type": "none", "text": no_event_text})
    rng.shuffle(options)

    labeled_options: List[Dict[str, Any]] = []
    for i, option in enumerate(options):
        labeled = dict(option)
        labeled["label"] = _option_label(i)
        labeled_options.append(labeled)
    return labeled_options


def _build_example(
    audio_filename: str,
    events: List[Event],
    base_event_index: int,
    no_event_text: str,
    rng: random.Random,
) -> Dict[str, Any]:
    base_event = events[base_event_index]
    next_event_index = base_event_index + 1 if base_event_index + 1 < len(events) else None

    options = _build_options(
        events=events,
        base_event_index=base_event_index,
        no_event_text=no_event_text,
        rng=rng,
    )

    answer_label = None
    answer_type = None
    answer_text = None
    for option in options:
        if next_event_index is None:
            if option["type"] == "none":
                answer_label = option["label"]
                answer_type = "none"
                answer_text = option["text"]
                break
        elif option["type"] == "event" and option["event_index"] == next_event_index:
            answer_label = option["label"]
            answer_type = "event"
            answer_text = option["text"]
            break

    if answer_label is None or answer_type is None or answer_text is None:
        raise RuntimeError("Failed to locate answer option.")

    return {
        "id": f"{audio_filename}__{base_event_index}",
        "task": "event_timeline_next",
        "audio_filename": audio_filename,
        "question": f'What happens immediately after this event first appears: "{base_event.text}"?',
        "base_event": _format_event(base_event),
        "options": options,
        "answer_label": answer_label,
        "answer_type": answer_type,
        "answer_text": answer_text,
        "answer_event_index": next_event_index,
    }


def _has_onset_tie(events: List[Event], target_event_index: int, epsilon: float) -> bool:
    target = events[target_event_index]
    for event in events:
        if event.index == target_event_index:
            continue
        if abs(event.onset - target.onset) <= epsilon:
            return True
    return False


def _answer_relies_on_tiebreak(
    events: List[Event],
    *,
    base_event_index: int,
    next_event_index: int,
    epsilon: float,
) -> bool:
    base_event = events[base_event_index]
    next_event = events[next_event_index]
    for event in events:
        if event.index == next_event_index:
            continue
        if event.onset <= base_event.onset + epsilon:
            continue
        if abs(event.onset - next_event.onset) <= epsilon:
            return True
    return False


def _iter_examples(
    grouped_events: Dict[str, List[Event]],
    seed: int,
    no_event_text: str,
    onset_tie_epsilon: float,
    stats: BuildStats,
) -> Iterator[Dict[str, Any]]:
    rng = random.Random(seed)
    for audio_filename in sorted(grouped_events):
        events = grouped_events[audio_filename]
        for base_event_index in range(len(events)):
            base_event = events[base_event_index]
            if _has_onset_tie(events, base_event_index, onset_tie_epsilon):
                stats.questions_dropped_onset_tie_base += 1
                continue
            if base_event.occurrence_count > 1:
                stats.questions_dropped_repeat_ambiguity += 1
                continue

            next_event_index = base_event_index + 1 if base_event_index + 1 < len(events) else None
            if next_event_index is not None:
                next_event = events[next_event_index]
                if _answer_relies_on_tiebreak(
                    events,
                    base_event_index=base_event_index,
                    next_event_index=next_event_index,
                    epsilon=onset_tie_epsilon,
                ):
                    stats.questions_dropped_tiebreak_answer += 1
                    continue
                if next_event.occurrence_count > 1:
                    stats.questions_dropped_repeat_ambiguity += 1
                    continue
                if next_event.onset < base_event.offset + onset_tie_epsilon:
                    stats.questions_dropped_overlap_ambiguity += 1
                    continue

            yield _build_example(
                audio_filename=audio_filename,
                events=events,
                base_event_index=base_event_index,
                no_event_text=no_event_text,
                rng=rng,
            )


def _write_jsonl(examples: Iterable[Dict[str, Any]], output_jsonl: Path) -> int:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1
    return count


def _print_summary(
    *,
    stats: BuildStats,
    input_csv: Path,
    output_jsonl: Path,
    min_duration: float,
    min_events_per_audio: int,
    onset_tie_epsilon: float,
    seed: int,
) -> None:
    table = Table(title="Timeline MCQ build summary", show_header=True, header_style="bold")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Input CSV", str(input_csv))
    table.add_row("Output JSONL", str(output_jsonl))
    table.add_row("Min event duration (s)", str(min_duration))
    table.add_row("Min events per audio", str(min_events_per_audio))
    table.add_row("Onset tie epsilon (s)", str(onset_tie_epsilon))
    table.add_row("Random seed", str(seed))
    table.add_row("Rows read", str(stats.rows_total))
    table.add_row("Rows dropped (invalid time)", str(stats.rows_invalid_time))
    table.add_row("Rows dropped (too short)", str(stats.rows_short))
    table.add_row("Rows kept", str(stats.rows_kept))
    table.add_row("Rows collapsed by caption grouping", str(stats.rows_collapsed_by_caption_grouping))
    table.add_row("Events after caption grouping", str(stats.events_after_caption_grouping))
    table.add_row("Audios after row filter", str(stats.audios_total))
    table.add_row("Audios dropped (< min events)", str(stats.audios_removed_too_few_events))
    table.add_row("Audios kept", str(stats.audios_kept))
    table.add_row("Questions dropped (base onset tie)", str(stats.questions_dropped_onset_tie_base))
    table.add_row("Questions dropped (answer tie-break)", str(stats.questions_dropped_tiebreak_answer))
    table.add_row("Questions dropped (repeated-event ambiguity)", str(stats.questions_dropped_repeat_ambiguity))
    table.add_row("Questions dropped (overlap ambiguity)", str(stats.questions_dropped_overlap_ambiguity))
    table.add_row("Questions written", str(stats.questions_written))
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
        Path("data/mcq_event_timeline_strong.jsonl"),
        "--output",
        "-o",
        path_type=Path,
        help="Destination JSONL file.",
    ),
    min_duration: float = typer.Option(
        0.5,
        "--min-duration",
        help="Drop events shorter than this duration in seconds.",
        min=0.0,
    ),
    min_events_per_audio: int = typer.Option(
        2,
        "--min-events-per-audio",
        help="Drop audios with fewer events than this after filtering.",
        min=1,
    ),
    onset_tie_epsilon: float = typer.Option(
        0.05,
        "--onset-tie-epsilon",
        help="Drop questions with onset ties within this epsilon (seconds).",
        min=0.0,
    ),
    seed: int = typer.Option(
        7,
        "--seed",
        help="Seed used to randomize option order deterministically.",
    ),
    no_event_text: str = typer.Option(
        LAST_EVENT_TEXT_DEFAULT,
        "--no-event-text",
        help="Mandatory option text for the no-immediate-next-event case.",
    ),
) -> None:
    stats = BuildStats()

    grouped_events = _load_filtered_events(
        input_csv=input_csv,
        min_duration=min_duration,
        stats=stats,
    )
    grouped_events = _drop_short_audios(
        grouped_events=grouped_events,
        min_events_per_audio=min_events_per_audio,
        stats=stats,
    )

    stats.questions_written = _write_jsonl(
        examples=_iter_examples(
            grouped_events=grouped_events,
            seed=seed,
            no_event_text=no_event_text,
            onset_tie_epsilon=onset_tie_epsilon,
            stats=stats,
        ),
        output_jsonl=output_jsonl,
    )

    _print_summary(
        stats=stats,
        input_csv=input_csv,
        output_jsonl=output_jsonl,
        min_duration=min_duration,
        min_events_per_audio=min_events_per_audio,
        onset_tie_epsilon=onset_tie_epsilon,
        seed=seed,
    )


if __name__ == "__main__":
    typer.run(main)
