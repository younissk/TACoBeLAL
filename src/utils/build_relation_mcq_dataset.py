"""Build MCQ-RELATION dataset from strong TACoBeLAL annotations."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import typer
from rich.console import Console
from rich.table import Table

console = Console()

TASK_ID_MCQ_RELATION = "MCQ-RELATION"

REL_A_STARTS_BEFORE_B = "a_starts_before_b"
REL_A_STARTS_AFTER_B = "a_starts_after_b"
REL_A_OVERLAPS_B = "a_overlaps_b"
REL_A_B_START_SAME_TIME = "a_b_start_same_time"
REL_CANNOT_BE_DETERMINED = "cannot_be_determined"

RELATION_OPTION_TEXT = {
    REL_A_STARTS_BEFORE_B: "A starts before B",
    REL_A_STARTS_AFTER_B: "A starts after B",
    REL_A_OVERLAPS_B: "A overlaps B",
    REL_A_B_START_SAME_TIME: "A and B start at the same time",
    REL_CANNOT_BE_DETERMINED: "cannot be determined",
}


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
    questions_candidate_pairs: int = 0
    questions_dropped_near_tie: int = 0
    questions_relation_before: int = 0
    questions_relation_after: int = 0
    questions_relation_overlap: int = 0
    questions_relation_same_start: int = 0
    questions_relation_cannot_determine: int = 0
    questions_dropped_per_audio_cap: int = 0
    questions_dropped_global_cap: int = 0
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
    """Load strong events and keep rows with duration >= min_duration.

    Events are grouped by normalized caption per audio and represented by their
    first occurrence. This avoids duplicate choices with identical text.
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


def _deterministic_shuffle_options(options: list[dict[str, Any]], *, seed: int, example_id: str) -> list[dict[str, Any]]:
    """Return a deterministic pseudo-random ordering keyed by seed+example_id."""
    keyed: list[tuple[str, dict[str, Any]]] = []
    for idx, option in enumerate(options):
        digest = hashlib.sha256(
            f"{seed}:{example_id}:{idx}:{option['relation']}".encode("utf-8")
        ).hexdigest()
        keyed.append((digest, option))
    keyed.sort(key=lambda item: item[0])
    return [dict(item[1]) for item in keyed]


def _base_relation_options(include_start_same_time: bool) -> list[dict[str, Any]]:
    keys = [
        REL_A_STARTS_BEFORE_B,
        REL_A_STARTS_AFTER_B,
        REL_A_OVERLAPS_B,
        REL_CANNOT_BE_DETERMINED,
    ]
    if include_start_same_time:
        keys.insert(3, REL_A_B_START_SAME_TIME)

    return [
        {
            "type": "relation",
            "relation": key,
            "text": RELATION_OPTION_TEXT[key],
        }
        for key in keys
    ]


def _build_options(example_id: str, *, seed: int, include_start_same_time: bool) -> list[dict[str, Any]]:
    options = _base_relation_options(include_start_same_time=include_start_same_time)
    options = _deterministic_shuffle_options(options, seed=seed, example_id=example_id)

    labeled_options: list[dict[str, Any]] = []
    for i, option in enumerate(options):
        labeled = dict(option)
        labeled["label"] = _option_label(i)
        labeled_options.append(labeled)
    return labeled_options


def _classify_relation(
    event_a: Event,
    event_b: Event,
    *,
    same_start_epsilon: float,
    overlap_epsilon: float,
    include_start_same_time: bool,
) -> str | None:
    if event_a.occurrence_count > 1 or event_b.occurrence_count > 1:
        return REL_CANNOT_BE_DETERMINED

    start_gap = event_a.onset - event_b.onset
    if abs(start_gap) <= same_start_epsilon:
        if include_start_same_time:
            return REL_A_B_START_SAME_TIME
        return None

    overlap_amount = min(event_a.offset, event_b.offset) - max(event_a.onset, event_b.onset)
    if overlap_amount > overlap_epsilon:
        return REL_A_OVERLAPS_B

    if start_gap < 0:
        return REL_A_STARTS_BEFORE_B
    return REL_A_STARTS_AFTER_B


def _relation_quality_score(
    event_a: Event,
    event_b: Event,
    *,
    relation: str,
    same_start_epsilon: float,
) -> float:
    start_gap = event_a.onset - event_b.onset
    overlap_amount = min(event_a.offset, event_b.offset) - max(event_a.onset, event_b.onset)

    if relation == REL_CANNOT_BE_DETERMINED:
        return 0.0
    if relation == REL_A_B_START_SAME_TIME:
        return max(0.0, same_start_epsilon - abs(start_gap))
    if relation == REL_A_OVERLAPS_B:
        return max(0.0, overlap_amount)

    temporal_separation = max(event_a.onset, event_b.onset) - min(event_a.offset, event_b.offset)
    return abs(start_gap) + max(0.0, temporal_separation)


def _question_text(event_a_text: str, event_b_text: str) -> str:
    return (
        "In this audio, what is the temporal relation between Event A and Event B?\n"
        f'Event A: "{event_a_text}"\n'
        f'Event B: "{event_b_text}"'
    )


def _update_relation_stats(stats: BuildStats, relation: str) -> None:
    if relation == REL_A_STARTS_BEFORE_B:
        stats.questions_relation_before += 1
    elif relation == REL_A_STARTS_AFTER_B:
        stats.questions_relation_after += 1
    elif relation == REL_A_OVERLAPS_B:
        stats.questions_relation_overlap += 1
    elif relation == REL_A_B_START_SAME_TIME:
        stats.questions_relation_same_start += 1
    elif relation == REL_CANNOT_BE_DETERMINED:
        stats.questions_relation_cannot_determine += 1


def _build_example(
    audio_filename: str,
    *,
    event_a: Event,
    event_b: Event,
    answer_relation: str,
    quality_score: float,
    seed: int,
    include_start_same_time: bool,
) -> Dict[str, Any]:
    example_id = f"{audio_filename}__{event_a.index}__{event_b.index}"
    options = _build_options(
        example_id=example_id,
        seed=seed,
        include_start_same_time=include_start_same_time,
    )

    answer_label = None
    answer_text = None
    for option in options:
        if option["relation"] == answer_relation:
            answer_label = option["label"]
            answer_text = option["text"]
            break

    if answer_label is None or answer_text is None:
        raise RuntimeError("Failed to locate answer option.")

    return {
        "id": example_id,
        "task": "event_timeline_relation",
        "task_id": TASK_ID_MCQ_RELATION,
        "audio_filename": audio_filename,
        "question": _question_text(event_a.text, event_b.text),
        "event_a": _format_event(event_a),
        "event_b": _format_event(event_b),
        "options": options,
        "answer_label": answer_label,
        "answer_type": "relation",
        "answer_relation": answer_relation,
        "answer_text": answer_text,
        "_quality_score": quality_score,
    }


def _iter_examples(
    grouped_events: Dict[str, List[Event]],
    *,
    seed: int,
    same_start_epsilon: float,
    overlap_epsilon: float,
    include_start_same_time: bool,
    stats: BuildStats,
) -> Iterator[Dict[str, Any]]:
    for audio_filename in sorted(grouped_events):
        events = grouped_events[audio_filename]
        for a_index in range(len(events)):
            for b_index in range(len(events)):
                if a_index == b_index:
                    continue

                stats.questions_candidate_pairs += 1
                event_a = events[a_index]
                event_b = events[b_index]
                relation = _classify_relation(
                    event_a=event_a,
                    event_b=event_b,
                    same_start_epsilon=same_start_epsilon,
                    overlap_epsilon=overlap_epsilon,
                    include_start_same_time=include_start_same_time,
                )
                if relation is None:
                    stats.questions_dropped_near_tie += 1
                    continue

                _update_relation_stats(stats, relation)
                quality_score = _relation_quality_score(
                    event_a=event_a,
                    event_b=event_b,
                    relation=relation,
                    same_start_epsilon=same_start_epsilon,
                )
                yield _build_example(
                    audio_filename=audio_filename,
                    event_a=event_a,
                    event_b=event_b,
                    answer_relation=relation,
                    quality_score=quality_score,
                    seed=seed,
                    include_start_same_time=include_start_same_time,
                )


def _rank_examples(examples: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    return sorted(
        examples,
        key=lambda ex: (
            -float(ex.get("_quality_score", 0.0)),
            str(ex["id"]),
        ),
    )


def _select_examples(
    examples: list[Dict[str, Any]],
    *,
    max_pairs_per_audio: int | None,
    max_questions: int | None,
    balance_relations: bool,
    stats: BuildStats,
) -> list[Dict[str, Any]]:
    selected = list(examples)

    if max_pairs_per_audio is not None:
        by_audio: Dict[str, list[Dict[str, Any]]] = {}
        for ex in selected:
            by_audio.setdefault(str(ex["audio_filename"]), []).append(ex)

        reduced: list[Dict[str, Any]] = []
        for audio_filename in sorted(by_audio):
            ranked = _rank_examples(by_audio[audio_filename])
            reduced.extend(ranked[:max_pairs_per_audio])
            stats.questions_dropped_per_audio_cap += max(0, len(ranked) - max_pairs_per_audio)
        selected = reduced

    if max_questions is None or len(selected) <= max_questions:
        return sorted(selected, key=lambda ex: str(ex["id"]))

    if not balance_relations:
        ranked = _rank_examples(selected)
        kept = ranked[:max_questions]
        stats.questions_dropped_global_cap += len(ranked) - len(kept)
        return sorted(kept, key=lambda ex: str(ex["id"]))

    relation_order = [
        REL_A_STARTS_BEFORE_B,
        REL_A_STARTS_AFTER_B,
        REL_A_OVERLAPS_B,
        REL_A_B_START_SAME_TIME,
        REL_CANNOT_BE_DETERMINED,
    ]
    by_relation: Dict[str, list[Dict[str, Any]]] = {relation: [] for relation in relation_order}
    for ex in selected:
        relation = str(ex["answer_relation"])
        if relation in by_relation:
            by_relation[relation].append(ex)

    for relation in relation_order:
        by_relation[relation] = _rank_examples(by_relation[relation])

    target_per_relation = max_questions // len(relation_order)
    kept_ids: set[str] = set()
    kept: list[Dict[str, Any]] = []

    for relation in relation_order:
        for ex in by_relation[relation][:target_per_relation]:
            ex_id = str(ex["id"])
            if ex_id in kept_ids:
                continue
            kept_ids.add(ex_id)
            kept.append(ex)

    remaining_slots = max_questions - len(kept)
    if remaining_slots > 0:
        leftovers: list[Dict[str, Any]] = []
        for relation in relation_order:
            leftovers.extend([ex for ex in by_relation[relation] if str(ex["id"]) not in kept_ids])
        leftovers = _rank_examples(leftovers)
        for ex in leftovers[:remaining_slots]:
            ex_id = str(ex["id"])
            if ex_id in kept_ids:
                continue
            kept_ids.add(ex_id)
            kept.append(ex)

    stats.questions_dropped_global_cap += max(0, len(selected) - len(kept))
    return sorted(kept, key=lambda ex: str(ex["id"]))


def _write_jsonl(examples: Iterable[Dict[str, Any]], output_jsonl: Path) -> int:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for example in examples:
            serializable = {k: v for k, v in example.items() if not k.startswith("_")}
            f.write(json.dumps(serializable, ensure_ascii=False) + "\n")
            count += 1
    return count


def _print_summary(
    *,
    stats: BuildStats,
    input_csv: Path,
    output_jsonl: Path,
    min_duration: float,
    min_events_per_audio: int,
    same_start_epsilon: float,
    overlap_epsilon: float,
    seed: int,
    include_start_same_time: bool,
    max_pairs_per_audio: int | None,
    max_questions: int | None,
    balance_relations: bool,
) -> None:
    table = Table(title="MCQ-RELATION build summary", show_header=True, header_style="bold")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Task ID", TASK_ID_MCQ_RELATION)
    table.add_row("Input CSV", str(input_csv))
    table.add_row("Output JSONL", str(output_jsonl))
    table.add_row("Min event duration (s)", str(min_duration))
    table.add_row("Min events per audio", str(min_events_per_audio))
    table.add_row("Same-start epsilon (s)", str(same_start_epsilon))
    table.add_row("Overlap epsilon (s)", str(overlap_epsilon))
    table.add_row("Include 'start at same time' option", str(include_start_same_time))
    table.add_row("Deterministic seed", str(seed))
    table.add_row("Max pairs per audio", "None" if max_pairs_per_audio is None else str(max_pairs_per_audio))
    table.add_row("Max questions", "None" if max_questions is None else str(max_questions))
    table.add_row("Balanced relation sampling", str(balance_relations))
    table.add_row("Rows read", str(stats.rows_total))
    table.add_row("Rows dropped (invalid time)", str(stats.rows_invalid_time))
    table.add_row("Rows dropped (too short)", str(stats.rows_short))
    table.add_row("Rows kept", str(stats.rows_kept))
    table.add_row("Rows collapsed by caption grouping", str(stats.rows_collapsed_by_caption_grouping))
    table.add_row("Events after caption grouping", str(stats.events_after_caption_grouping))
    table.add_row("Audios after row filter", str(stats.audios_total))
    table.add_row("Audios dropped (< min events)", str(stats.audios_removed_too_few_events))
    table.add_row("Audios kept", str(stats.audios_kept))
    table.add_row("Candidate ordered event pairs", str(stats.questions_candidate_pairs))
    table.add_row("Pairs dropped (near-start ties without same-time option)", str(stats.questions_dropped_near_tie))
    table.add_row("Answers: A starts before B", str(stats.questions_relation_before))
    table.add_row("Answers: A starts after B", str(stats.questions_relation_after))
    table.add_row("Answers: A overlaps B", str(stats.questions_relation_overlap))
    table.add_row("Answers: A and B start at same time", str(stats.questions_relation_same_start))
    table.add_row("Answers: cannot be determined", str(stats.questions_relation_cannot_determine))
    table.add_row("Questions dropped (per-audio cap)", str(stats.questions_dropped_per_audio_cap))
    table.add_row("Questions dropped (global cap)", str(stats.questions_dropped_global_cap))
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
        Path("data/mcq_relation_timeline_strong.jsonl"),
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
    same_start_epsilon: float = typer.Option(
        0.05,
        "--same-start-epsilon",
        help="Onset tolerance (seconds) for 'A and B start at the same time'.",
        min=0.0,
    ),
    overlap_epsilon: float = typer.Option(
        0.0,
        "--overlap-epsilon",
        help="Minimum overlap (seconds) required to label 'A overlaps B'.",
        min=0.0,
    ),
    seed: int = typer.Option(
        7,
        "--seed",
        help="Deterministic seed for option ordering.",
    ),
    max_pairs_per_audio: int | None = typer.Option(
        4,
        "--max-pairs-per-audio",
        help="Keep at most this many highest-quality pairs per audio (set to 0 for no cap).",
        min=0,
    ),
    max_questions: int | None = typer.Option(
        12000,
        "--max-questions",
        help="Global cap for selected highest-quality questions (set to 0 for no cap).",
        min=0,
    ),
    balance_relations: bool = typer.Option(
        True,
        "--balance-relations/--no-balance-relations",
        help="When applying --max-questions, balance picks across relation types first.",
    ),
    include_start_same_time: bool = typer.Option(
        True,
        "--include-start-same-time/--exclude-start-same-time",
        help=(
            "Include the 'A and B start at the same time' option. "
            "If disabled, near-start ties are filtered out."
        ),
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

    if max_pairs_per_audio == 0:
        max_pairs_per_audio = None
    if max_questions == 0:
        max_questions = None

    candidates = list(
        _iter_examples(
            grouped_events=grouped_events,
            seed=seed,
            same_start_epsilon=same_start_epsilon,
            overlap_epsilon=overlap_epsilon,
            include_start_same_time=include_start_same_time,
            stats=stats,
        )
    )
    selected = _select_examples(
        candidates,
        max_pairs_per_audio=max_pairs_per_audio,
        max_questions=max_questions,
        balance_relations=balance_relations,
        stats=stats,
    )

    stats.questions_written = _write_jsonl(
        examples=selected,
        output_jsonl=output_jsonl,
    )

    _print_summary(
        stats=stats,
        input_csv=input_csv,
        output_jsonl=output_jsonl,
        min_duration=min_duration,
        min_events_per_audio=min_events_per_audio,
        same_start_epsilon=same_start_epsilon,
        overlap_epsilon=overlap_epsilon,
        seed=seed,
        include_start_same_time=include_start_same_time,
        max_pairs_per_audio=max_pairs_per_audio,
        max_questions=max_questions,
        balance_relations=balance_relations,
    )


if __name__ == "__main__":
    typer.run(main)
