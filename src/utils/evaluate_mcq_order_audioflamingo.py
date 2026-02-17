"""Evaluate MCQ-ORDER using Audio Flamingo 3 (audio-capable LALM)."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

try:
    from .evaluate_mcq_order import load_examples
    from .mcq_order_models import MCQOrderExample, TASK_ID_MCQ_ORDER
    from .wandb_tracker import WandbTracker
except ImportError:  # pragma: no cover - enables direct script execution
    from evaluate_mcq_order import load_examples
    from mcq_order_models import MCQOrderExample, TASK_ID_MCQ_ORDER
    from wandb_tracker import WandbTracker

console = Console()
BASE_MODEL_NAME = "audio-flamingo-3"


@dataclass(frozen=True)
class Decision:
    task_id: str
    model_name: str
    model_base: str
    example_id: str
    audio_filename: str
    question: str
    predicted_label: str
    predicted_text: str
    answer_label: str
    answer_text: str
    answer_type: str
    is_correct: bool
    parse_status: str
    raw_prediction: str
    n_options: int
    predicted_type: str

    def to_json(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "model_name": self.model_name,
            "model_base": self.model_base,
            "example_id": self.example_id,
            "audio_filename": self.audio_filename,
            "question": self.question,
            "predicted_label": self.predicted_label,
            "predicted_text": self.predicted_text,
            "answer_label": self.answer_label,
            "answer_text": self.answer_text,
            "answer_type": self.answer_type,
            "is_correct": self.is_correct,
            "parse_status": self.parse_status,
            "raw_prediction": self.raw_prediction,
            "n_options": self.n_options,
            "predicted_type": self.predicted_type,
        }


@dataclass(frozen=True)
class RunMetrics:
    run_id: str
    task_id: str
    model_name: str
    model_base: str
    dataset_path: str
    audio_root: str
    audioflamingo_repo: str
    examples: int
    correct: int
    accuracy: float
    parse_invalid: int
    missing_predictions: int
    elapsed_seconds: float
    average_latency_ms: float
    started_at_utc: str
    finished_at_utc: str
    decisions_path: str
    raw_outputs_path: str

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "model_name": self.model_name,
            "model_base": self.model_base,
            "dataset_path": self.dataset_path,
            "audio_root": self.audio_root,
            "audioflamingo_repo": self.audioflamingo_repo,
            "examples": self.examples,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 6),
            "parse_invalid": self.parse_invalid,
            "missing_predictions": self.missing_predictions,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "average_latency_ms": round(self.average_latency_ms, 6),
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "decisions_path": self.decisions_path,
            "raw_outputs_path": self.raw_outputs_path,
        }


def _safe_name(value: str, max_len: int = 120) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not sanitized:
        sanitized = "item"
    return sanitized[:max_len]


def _resolve_model_name(*, use_audio: bool = True) -> str:
    return BASE_MODEL_NAME if use_audio else f"{BASE_MODEL_NAME}-no-audio"


def build_prompt(example: MCQOrderExample) -> str:
    options_text = "\n".join(f"{option.label}. {option.text}" for option in example.options)
    labels_text = ", ".join(option.label for option in example.options)
    return (
        f"{example.question}\n\n"
        "Choose exactly one option.\n"
        f"{options_text}\n\n"
        f"Return only the option label from: {labels_text}."
    )


def _normalize_prediction_text(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.rstrip(" .,!?:;")
    return normalized


def parse_predicted_label(
    prediction: str,
    *,
    valid_labels: set[str],
    label_to_text: dict[str, str],
) -> tuple[str | None, str]:
    candidate = prediction.strip()
    if not candidate:
        return None, "empty"
    if candidate in valid_labels:
        return candidate, "exact-label"

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            label = parsed.get("label")
            if isinstance(label, str) and label.strip() in valid_labels:
                return label.strip(), "json-label"
    except json.JSONDecodeError:
        pass

    for match in re.finditer(r"\b([A-Z]{1,3})\b", candidate):
        label = match.group(1)
        if label in valid_labels:
            return label, "regex-label"

    normalized = _normalize_prediction_text(candidate)
    for label, text in label_to_text.items():
        if _normalize_prediction_text(text) == normalized:
            return label, "text-match"

    return None, "invalid"


def _create_audio_link(source: Path, link_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    try:
        link_path.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source.resolve(), link_path)


def _normalize_output_id_path(value: str) -> str:
    return str(Path(value).absolute())


def _normalize_output_id(value: str) -> str:
    if value.startswith("text::"):
        return value
    return _normalize_output_id_path(value)


def prepare_audioflamingo_input(
    examples: list[MCQOrderExample],
    *,
    audio_root: Path | None,
    work_dir: Path,
    use_audio: bool,
) -> tuple[Path, Path, dict[str, Any]]:
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    links_dir = work_dir / "audio_links"
    links_dir.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {}
    mapping: dict[str, Any] = {}
    missing_audio: list[str] = []

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("Preparing AudioFlamingo input", total=len(examples))
        for idx, example in enumerate(examples):
            key = str(idx)
            if use_audio:
                if audio_root is None:
                    raise ValueError("audio_root is required when use_audio=True.")
                source_audio = (audio_root / example.audio_filename).resolve()
                if not source_audio.exists():
                    missing_audio.append(str(source_audio))
                    progress.advance(task, 1)
                    continue

                extension = source_audio.suffix or ".wav"
                link_name = f"{idx:06d}_{_safe_name(example.example_id)}{extension}"
                link_path = links_dir / link_name
                _create_audio_link(source_audio, link_path)

                data[key] = {
                    "name": link_name,
                    "prompt": build_prompt(example),
                    "output": example.answer_label,
                }
                mapped_output_id = _normalize_output_id_path(str(link_path))
            else:
                mapped_output_id = f"text::{example.example_id}"
                data[key] = {
                    "prompt": build_prompt(example),
                    "output": example.answer_label,
                }

            mapping[mapped_output_id] = {
                "index": idx,
                "fallback_output_id": key,
                "example_id": example.example_id,
                "audio_filename": example.audio_filename,
                "question": example.question,
                "answer_label": example.answer_label,
                "answer_text": example.answer_text,
                "options": [
                    {"label": option.label, "text": option.text, "type": option.option_type}
                    for option in example.options
                ],
            }
            progress.advance(task, 1)

    if missing_audio:
        preview = "\n".join(missing_audio[:5])
        raise FileNotFoundError(
            f"Missing {len(missing_audio)} audio files under '{audio_root}'. First examples:\n{preview}"
        )

    af_payload = {
        "dataset_path": str(audio_root) if audio_root is not None else str(work_dir),
        "split": "test",
        "split_path": str(links_dir),
        "flamingo_task": TASK_ID_MCQ_ORDER,
        "data": data,
    }

    input_json_path = work_dir / "audioflamingo_input.json"
    mapping_json_path = work_dir / "audioflamingo_mapping.json"
    with open(input_json_path, "w", encoding="utf-8") as f:
        json.dump(af_payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    with open(mapping_json_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return input_json_path, mapping_json_path, mapping


def _run_subprocess(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    console.print(f"[cyan]$ {' '.join(command)}[/cyan]")
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def run_audioflamingo_inference(
    *,
    audioflamingo_repo: Path,
    model_base: str,
    input_json_path: Path,
    output_dir: Path,
    num_gpus: int,
    batch_size: int,
    max_new_tokens: int,
    think_mode: bool,
) -> float:
    if not (audioflamingo_repo / ".git").exists():
        raise FileNotFoundError(
            f"Audio Flamingo repo not found at '{audioflamingo_repo}'. "
            "Run setup first (make download-audioflamingo)."
        )

    input_json_path = input_json_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    generation_config = json.dumps({"max_new_tokens": max_new_tokens})
    # Use module invocation so that the Audio Flamingo repo root stays on sys.path
    # and the local `llava` package can be imported correctly.
    command = [
        "torchrun",
        f"--nproc-per-node={num_gpus}",
        "-m",
        "llava.cli.infer_batch",
        "--model-base",
        model_base,
        "--json-path",
        str(input_json_path),
        "--output-dir",
        str(output_dir),
        "--batch-size",
        str(batch_size),
        "--generation-config",
        generation_config,
        "--overwrite",
    ]
    if think_mode:
        command.extend(["--think-mode", "true"])

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    started = time.perf_counter()
    with console.status("[bold green]Running Audio Flamingo inference...[/bold green]"):
        _run_subprocess(command, cwd=audioflamingo_repo, env=env)
    return time.perf_counter() - started


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def evaluate_audioflamingo_outputs(
    *,
    mapping: dict[str, Any],
    raw_outputs: list[dict[str, Any]],
    model_name: str,
    model_base: str,
    tracker: WandbTracker | None = None,
) -> tuple[list[Decision], int, int]:
    outputs_by_id: dict[str, dict[str, Any]] = {}
    for row in raw_outputs:
        output_id = row.get("id")
        if isinstance(output_id, str):
            outputs_by_id[_normalize_output_id(output_id)] = row

    decisions: list[Decision] = []
    invalid_count = 0
    missing_count = 0

    ordered_items = sorted(mapping.items(), key=lambda item: item[1]["index"])
    for mapped_id, metadata in ordered_items:
        options = metadata["options"]
        valid_labels = {option["label"] for option in options}
        label_to_text = {option["label"]: option["text"] for option in options}
        label_to_type = {option["label"]: option["type"] for option in options}

        raw_prediction = ""
        parse_status = "missing"
        predicted_label: str | None = None
        row = outputs_by_id.get(mapped_id)
        if row is None:
            fallback_output_id = metadata.get("fallback_output_id")
            if isinstance(fallback_output_id, str):
                row = outputs_by_id.get(fallback_output_id)
        if row is None:
            row_index = metadata.get("index")
            if isinstance(row_index, int) and 0 <= row_index < len(raw_outputs):
                row = raw_outputs[row_index]
        if row is None:
            missing_count += 1
        else:
            raw_prediction = str(row.get("pred", ""))
            predicted_label, parse_status = parse_predicted_label(
                raw_prediction,
                valid_labels=valid_labels,
                label_to_text=label_to_text,
            )
            if predicted_label is None:
                invalid_count += 1

        final_label = predicted_label if predicted_label is not None else "INVALID"
        predicted_text = label_to_text.get(final_label, "")
        predicted_type = label_to_type.get(final_label, "invalid")
        is_correct = final_label == metadata["answer_label"]
        answer_type = label_to_type.get(metadata["answer_label"], "unknown")

        decisions.append(
            Decision(
                task_id=TASK_ID_MCQ_ORDER,
                model_name=model_name,
                model_base=model_base,
                example_id=metadata["example_id"],
                audio_filename=metadata["audio_filename"],
                question=metadata["question"],
                predicted_label=final_label,
                predicted_text=predicted_text,
                answer_label=metadata["answer_label"],
                answer_text=metadata["answer_text"],
                answer_type=answer_type,
                is_correct=is_correct,
                parse_status=parse_status,
                raw_prediction=raw_prediction,
                n_options=len(options),
                predicted_type=predicted_type,
            )
        )
        if tracker is not None and tracker.active:
            index = len(decisions)
            correct_so_far = sum(1 for d in decisions if d.is_correct)
            tracker.log_live(
                index=index,
                total=len(mapping),
                accuracy_so_far=correct_so_far / index,
                latency_ms=0.0,
                is_correct=decisions[-1].is_correct,
                correct_so_far=correct_so_far,
                force=(index == 1 or index == len(mapping)),
            )

    return decisions, invalid_count, missing_count


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        if probability > 0:
            entropy -= probability * math.log2(probability)
    return entropy


def _build_analysis_payload(decisions: list[Decision]) -> dict[str, Any]:
    total = len(decisions)
    correct = sum(1 for decision in decisions if decision.is_correct)
    answer_counts: Counter[str] = Counter(decision.answer_label for decision in decisions)
    prediction_counts: Counter[str] = Counter(decision.predicted_label for decision in decisions)
    correct_by_answer: Counter[str] = Counter(
        decision.answer_label for decision in decisions if decision.is_correct
    )
    parse_status_counts: Counter[str] = Counter(decision.parse_status for decision in decisions)
    by_n_options: defaultdict[int, dict[str, int]] = defaultdict(lambda: {"examples": 0, "correct": 0})
    by_answer_type: defaultdict[str, dict[str, int]] = defaultdict(lambda: {"examples": 0, "correct": 0})

    for decision in decisions:
        by_n_options[decision.n_options]["examples"] += 1
        by_n_options[decision.n_options]["correct"] += int(decision.is_correct)
        by_answer_type[decision.answer_type]["examples"] += 1
        by_answer_type[decision.answer_type]["correct"] += int(decision.is_correct)

    labels = sorted(set(answer_counts) | set(prediction_counts))
    answer_distribution = [
        {
            "label": label,
            "count": answer_counts[label],
            "rate": _safe_divide(answer_counts[label], total),
        }
        for label in labels
    ]
    prediction_distribution = [
        {
            "label": label,
            "count": prediction_counts[label],
            "rate": _safe_divide(prediction_counts[label], total),
        }
        for label in labels
    ]
    by_answer_label = [
        {
            "label": label,
            "support": answer_counts[label],
            "correct": correct_by_answer[label],
            "accuracy": _safe_divide(correct_by_answer[label], answer_counts[label]),
            "predicted_count": prediction_counts[label],
            "prediction_rate": _safe_divide(prediction_counts[label], total),
        }
        for label in labels
    ]
    by_option_count = [
        {
            "n_options": n_options,
            "examples": stats["examples"],
            "correct": stats["correct"],
            "accuracy": _safe_divide(stats["correct"], stats["examples"]),
        }
        for n_options, stats in sorted(by_n_options.items())
    ]
    by_type = [
        {
            "answer_type": answer_type,
            "examples": stats["examples"],
            "correct": stats["correct"],
            "accuracy": _safe_divide(stats["correct"], stats["examples"]),
        }
        for answer_type, stats in sorted(by_answer_type.items())
    ]
    parse_status = [
        {
            "parse_status": status,
            "count": count,
            "rate": _safe_divide(count, total),
        }
        for status, count in sorted(parse_status_counts.items())
    ]

    return {
        "examples": total,
        "correct": correct,
        "accuracy": _safe_divide(correct, total),
        "labels": labels,
        "answer_distribution": answer_distribution,
        "prediction_distribution": prediction_distribution,
        "by_answer_label": by_answer_label,
        "by_n_options": by_option_count,
        "by_answer_type": by_type,
        "parse_status": parse_status,
        "answer_entropy": _entropy(answer_counts),
        "prediction_entropy": _entropy(prediction_counts),
    }


def _log_wandb_analysis(
    *,
    tracker: WandbTracker,
    analysis: dict[str, Any],
    decisions: list[Decision],
) -> None:
    if not tracker.active:
        return

    summary_metrics: dict[str, Any] = {
        "analysis/labels": len(analysis["labels"]),
        "analysis/answer_entropy": analysis["answer_entropy"],
        "analysis/prediction_entropy": analysis["prediction_entropy"],
    }
    for row in analysis["by_answer_label"]:
        label = row["label"]
        summary_metrics[f"analysis/answer_label_accuracy/{label}"] = row["accuracy"]
        summary_metrics[f"analysis/answer_label_support/{label}"] = row["support"]
        summary_metrics[f"analysis/prediction_rate/{label}"] = row["prediction_rate"]
    for row in analysis["by_n_options"]:
        n_options = row["n_options"]
        summary_metrics[f"analysis/n_options_accuracy/{n_options}"] = row["accuracy"]
        summary_metrics[f"analysis/n_options_examples/{n_options}"] = row["examples"]
    for row in analysis["by_answer_type"]:
        answer_type = row["answer_type"]
        summary_metrics[f"analysis/answer_type_accuracy/{answer_type}"] = row["accuracy"]
        summary_metrics[f"analysis/answer_type_examples/{answer_type}"] = row["examples"]
    for row in analysis["parse_status"]:
        status = row["parse_status"]
        summary_metrics[f"analysis/parse_status_rate/{status}"] = row["rate"]
        summary_metrics[f"analysis/parse_status_count/{status}"] = row["count"]
    tracker.log(summary_metrics)

    tracker.log_table(
        key="analysis/by_answer_label",
        columns=["label", "support", "correct", "accuracy", "predicted_count", "prediction_rate"],
        rows=[
            [
                row["label"],
                row["support"],
                row["correct"],
                row["accuracy"],
                row["predicted_count"],
                row["prediction_rate"],
            ]
            for row in analysis["by_answer_label"]
        ],
    )
    tracker.log_table(
        key="analysis/by_n_options",
        columns=["n_options", "examples", "correct", "accuracy"],
        rows=[[row["n_options"], row["examples"], row["correct"], row["accuracy"]] for row in analysis["by_n_options"]],
    )
    tracker.log_table(
        key="analysis/by_answer_type",
        columns=["answer_type", "examples", "correct", "accuracy"],
        rows=[
            [row["answer_type"], row["examples"], row["correct"], row["accuracy"]]
            for row in analysis["by_answer_type"]
        ],
    )
    tracker.log_table(
        key="analysis/parse_status",
        columns=["parse_status", "count", "rate"],
        rows=[[row["parse_status"], row["count"], row["rate"]] for row in analysis["parse_status"]],
    )
    tracker.log_table(
        key="analysis/prediction_distribution",
        columns=["label", "count", "rate"],
        rows=[
            [row["label"], row["count"], row["rate"]]
            for row in analysis["prediction_distribution"]
        ],
    )
    tracker.log_table(
        key="analysis/answer_distribution",
        columns=["label", "count", "rate"],
        rows=[[row["label"], row["count"], row["rate"]] for row in analysis["answer_distribution"]],
    )
    tracker.log_confusion_matrix(
        key="analysis/confusion_matrix",
        y_true=[decision.answer_label for decision in decisions],
        y_pred=[decision.predicted_label for decision in decisions],
        class_names=analysis["labels"],
    )


def _write_decisions(path: Path, decisions: Iterable[Decision]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for decision in decisions:
            f.write(json.dumps(decision.to_json(), ensure_ascii=False) + "\n")


def _write_results_table(path: Path, metrics: RunMetrics) -> None:
    content = (
        "| Task ID | Model | Model Base | Examples | Correct | Accuracy | Parse Invalid | Missing | "
        "Elapsed (s) | Avg latency (ms) |\n"
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        f"| {metrics.task_id} | {metrics.model_name} | {metrics.model_base} | {metrics.examples} | "
        f"{metrics.correct} | {metrics.accuracy:.4f} | {metrics.parse_invalid} | {metrics.missing_predictions} | "
        f"{metrics.elapsed_seconds:.4f} | {metrics.average_latency_ms:.4f} |\n"
    )
    path.write_text(content, encoding="utf-8")


def _append_runs_csv(path: Path, metrics: RunMetrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    has_file = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "task_id",
                "model_name",
                "examples",
                "correct",
                "accuracy",
                "elapsed_seconds",
                "average_latency_ms",
                "started_at_utc",
                "finished_at_utc",
                "dataset_path",
                "decisions_path",
            ],
        )
        if not has_file:
            writer.writeheader()
        writer.writerow(
            {
                "run_id": metrics.run_id,
                "task_id": metrics.task_id,
                "model_name": metrics.model_name,
                "examples": metrics.examples,
                "correct": metrics.correct,
                "accuracy": round(metrics.accuracy, 6),
                "elapsed_seconds": round(metrics.elapsed_seconds, 6),
                "average_latency_ms": round(metrics.average_latency_ms, 6),
                "started_at_utc": metrics.started_at_utc,
                "finished_at_utc": metrics.finished_at_utc,
                "dataset_path": metrics.dataset_path,
                "decisions_path": metrics.decisions_path,
            }
        )


def _print_summary(metrics: RunMetrics, run_dir: Path) -> None:
    table = Table(title="MCQ-ORDER Audio Flamingo summary", show_header=True, header_style="bold")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Task ID", metrics.task_id)
    table.add_row("Model", metrics.model_name)
    table.add_row("Model base", metrics.model_base)
    table.add_row("Examples", str(metrics.examples))
    table.add_row("Correct", str(metrics.correct))
    table.add_row("Accuracy", f"{metrics.accuracy:.4f}")
    table.add_row("Parse invalid", str(metrics.parse_invalid))
    table.add_row("Missing predictions", str(metrics.missing_predictions))
    table.add_row("Elapsed (s)", f"{metrics.elapsed_seconds:.4f}")
    table.add_row("Avg latency (ms)", f"{metrics.average_latency_ms:.4f}")
    table.add_row("Run directory", str(run_dir))
    console.print(table)


def _create_run_dir(results_root: Path, run_id: str, *, model_name: str) -> Path:
    base = results_root / "mcq-order" / model_name
    candidate = base / run_id
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    suffix = 1
    while True:
        nxt = base / f"{run_id}_{suffix:02d}"
        if not nxt.exists():
            nxt.mkdir(parents=True, exist_ok=False)
            return nxt
        suffix += 1


def main(
    dataset: Path = typer.Option(
        Path("data/mcq_event_timeline_strong.jsonl"),
        "--dataset",
        "-d",
        path_type=Path,
        exists=True,
        dir_okay=False,
        help="MCQ-ORDER dataset JSONL.",
    ),
    audio_root: Path = typer.Option(
        Path("data/audio"),
        "--audio-root",
        path_type=Path,
        help="Directory containing extracted audio files.",
    ),
    audioflamingo_repo: Path = typer.Option(
        Path("external/audio-flamingo"),
        "--audioflamingo-repo",
        path_type=Path,
        help="Path to cloned Audio Flamingo repository.",
    ),
    model_base: str = typer.Option(
        "nvidia/audio-flamingo-3",
        "--model-base",
        help="Audio Flamingo model base (HF id or local path).",
    ),
    results_root: Path = typer.Option(
        Path("results"),
        "--results-root",
        "-o",
        path_type=Path,
        help="Root directory for evaluation outputs.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Number of examples to evaluate (omit for full dataset).",
    ),
    num_gpus: int = typer.Option(
        1,
        "--num-gpus",
        help="GPUs to use with torchrun. A40 default is 1.",
        min=1,
    ),
    batch_size: int = typer.Option(
        2,
        "--batch-size",
        help="Audio Flamingo batch size. A40-safe default is 2.",
        min=1,
    ),
    max_new_tokens: int = typer.Option(
        16,
        "--max-new-tokens",
        help="Generation max tokens for MCQ label output.",
        min=1,
    ),
    think_mode: bool = typer.Option(
        False,
        "--think-mode",
        help="Enable AF3 think mode (higher compute, slower).",
    ),
    use_audio: bool = typer.Option(
        True,
        "--use-audio/--disable-audio",
        help="Use model audio input. Disable for text-only probing with the same wrapper.",
    ),
    keep_workdir: bool = typer.Option(
        True,
        "--keep-workdir/--cleanup-workdir",
        help="Keep generated input JSON and audio symlink workspace.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Prepare inputs only; skip model inference and evaluation.",
    ),
    wandb: bool = typer.Option(
        True,
        "--wandb/--no-wandb",
        help="Enable live and final logging to Weights & Biases.",
    ),
    wandb_project: str = typer.Option(
        "tacobelal",
        "--wandb-project",
        help="W&B project name.",
    ),
    wandb_entity: str | None = typer.Option(
        None,
        "--wandb-entity",
        help="W&B entity/team (optional).",
    ),
    wandb_run_name: str | None = typer.Option(
        None,
        "--wandb-run-name",
        help="W&B run name (optional).",
    ),
    wandb_log_every: int = typer.Option(
        50,
        "--wandb-log-every",
        help="Log every N evaluated examples to W&B.",
        min=1,
    ),
) -> None:
    if limit is not None and limit < 1:
        raise typer.BadParameter("--limit must be >= 1 when provided.")

    examples = load_examples(dataset, limit=limit)
    if not examples:
        raise typer.BadParameter("No examples found to evaluate.")

    model_name = _resolve_model_name(use_audio=use_audio)
    tags = ["mcq-order", model_name]
    if not use_audio:
        tags.append("no-audio")
    tracker = WandbTracker(
        enabled=wandb,
        project=wandb_project,
        entity=wandb_entity,
        run_name=wandb_run_name,
        log_every=wandb_log_every,
        config={
            "task_id": TASK_ID_MCQ_ORDER,
            "dataset": str(dataset),
            "audio_root": str(audio_root),
            "audioflamingo_repo": str(audioflamingo_repo),
            "model_base": model_base,
            "limit": limit,
            "num_gpus": num_gpus,
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "think_mode": think_mode,
            "use_audio": use_audio,
        },
        tags=tags,
    )

    started_at = datetime.now(UTC)
    run_id = started_at.strftime("%Y%m%d_%H%M%S")
    run_dir = _create_run_dir(results_root, run_id, model_name=model_name)
    work_dir = run_dir / "workdir"
    af_output_dir = run_dir / "audioflamingo_outputs"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        console.print(f"[bold]Preparing {len(examples)} examples for Audio Flamingo...[/bold]")
        input_json_path, mapping_json_path, mapping = prepare_audioflamingo_input(
            examples,
            audio_root=audio_root if use_audio else None,
            work_dir=work_dir,
            use_audio=use_audio,
        )
        tracker.log({"stage/prepare_complete": 1, "stage/examples_prepared": len(mapping)})
        console.print(f"[green]Prepared input JSON:[/green] {input_json_path}")
        console.print(f"[green]Prepared mapping JSON:[/green] {mapping_json_path}")

        if dry_run:
            console.print("[yellow]Dry run enabled; skipping inference and evaluation.[/yellow]")
            tracker.log({"stage/dry_run": 1})
            return

        tracker.log({"stage/inference_started": 1})
        inference_elapsed = run_audioflamingo_inference(
            audioflamingo_repo=audioflamingo_repo,
            model_base=model_base,
            input_json_path=input_json_path,
            output_dir=af_output_dir,
            num_gpus=num_gpus,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            think_mode=think_mode,
        )
        tracker.log({"stage/inference_complete": 1, "stage/inference_elapsed_seconds": inference_elapsed})

        raw_outputs_path = af_output_dir / "outputs.jsonl"
        if not raw_outputs_path.exists():
            raise FileNotFoundError(f"Audio Flamingo output file not found: {raw_outputs_path}")

        raw_outputs = _load_jsonl(raw_outputs_path)
        raw_copy_path = run_dir / "raw_model_outputs.jsonl"
        shutil.copy2(raw_outputs_path, raw_copy_path)

        decisions, invalid_count, missing_count = evaluate_audioflamingo_outputs(
            mapping=mapping,
            raw_outputs=raw_outputs,
            model_name=model_name,
            model_base=model_base,
            tracker=tracker,
        )

        finished_at = datetime.now(UTC)
        correct = sum(decision.is_correct for decision in decisions)
        total = len(decisions)
        accuracy = correct / total if total else 0.0
        average_latency_ms = (inference_elapsed * 1000.0 / total) if total else 0.0

        decisions_path = run_dir / "decisions.jsonl"
        _write_decisions(decisions_path, decisions)

        metrics = RunMetrics(
            run_id=run_dir.name,
            task_id=TASK_ID_MCQ_ORDER,
            model_name=model_name,
            model_base=model_base,
            dataset_path=str(dataset),
            audio_root=str(audio_root),
            audioflamingo_repo=str(audioflamingo_repo),
            examples=total,
            correct=correct,
            accuracy=accuracy,
            parse_invalid=invalid_count,
            missing_predictions=missing_count,
            elapsed_seconds=inference_elapsed,
            average_latency_ms=average_latency_ms,
            started_at_utc=started_at.isoformat(),
            finished_at_utc=finished_at.isoformat(),
            decisions_path=str(decisions_path),
            raw_outputs_path=str(raw_copy_path),
        )

        _write_json(run_dir / "metrics.json", metrics.to_json())
        _write_results_table(run_dir / "results_table.md", metrics)
        _append_runs_csv(results_root / "mcq-order" / "runs.csv", metrics)
        analysis_payload = _build_analysis_payload(decisions)
        analysis_path = run_dir / "analysis.json"
        _write_json(analysis_path, analysis_payload)

        config_payload = {
            "task_id": TASK_ID_MCQ_ORDER,
            "dataset": str(dataset),
            "audio_root": str(audio_root),
            "audioflamingo_repo": str(audioflamingo_repo),
            "model_base": model_base,
            "limit": limit,
            "num_gpus": num_gpus,
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "think_mode": think_mode,
            "use_audio": use_audio,
        }
        _write_json(run_dir / "run_config.json", config_payload)

        tracker.update_summary(
            {
                "task_id": metrics.task_id,
                "model_name": metrics.model_name,
                "model_base": metrics.model_base,
                "examples": metrics.examples,
                "correct": metrics.correct,
                "accuracy": metrics.accuracy,
                "parse_invalid": metrics.parse_invalid,
                "missing_predictions": metrics.missing_predictions,
                "elapsed_seconds": metrics.elapsed_seconds,
                "average_latency_ms": metrics.average_latency_ms,
                "run_id": metrics.run_id,
                "dataset_path": metrics.dataset_path,
                "analysis/answer_entropy": analysis_payload["answer_entropy"],
                "analysis/prediction_entropy": analysis_payload["prediction_entropy"],
            }
        )
        _log_wandb_analysis(tracker=tracker, analysis=analysis_payload, decisions=decisions)
        tracker.log_artifact(
            name=f"{metrics.task_id.lower()}-{metrics.model_name}-{metrics.run_id}",
            artifact_type="evaluation",
            files=[
                run_dir / "run_config.json",
                decisions_path,
                run_dir / "metrics.json",
                run_dir / "results_table.md",
                analysis_path,
                raw_copy_path,
            ],
        )

        if not keep_workdir:
            shutil.rmtree(work_dir, ignore_errors=True)

        _print_summary(metrics, run_dir)
    finally:
        tracker.finish()


if __name__ == "__main__":
    typer.run(main)
