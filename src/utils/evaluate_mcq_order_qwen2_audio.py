"""Evaluate MCQ-ORDER using Qwen2-Audio models (audio-capable LALMs)."""

from __future__ import annotations

import csv
import json
import math
import os
import re
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
DEFAULT_MODEL_BASE = "Qwen/Qwen2-Audio-7B-Instruct"


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


def _resolve_model_name(model_base: str) -> str:
    return _safe_name(model_base.strip().split("/")[-1].lower())


def build_prompt(example: MCQOrderExample) -> str:
    options_text = "\n".join(f"{option.label}. {option.text}" for option in example.options)
    labels_text = ", ".join(option.label for option in example.options)
    return (
        "<|audio_bos|><|AUDIO|><|audio_eos|>\n"
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


def _validate_audio_files(examples: list[MCQOrderExample], *, audio_root: Path) -> None:
    missing_audio: list[str] = []
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("Validating local audio files", total=len(examples))
        for example in examples:
            source_audio = (audio_root / example.audio_filename).resolve()
            if not source_audio.exists():
                missing_audio.append(str(source_audio))
            progress.advance(task, 1)

    if missing_audio:
        preview = "\n".join(missing_audio[:5])
        raise FileNotFoundError(
            f"Missing {len(missing_audio)} audio files under '{audio_root}'. First examples:\n{preview}"
        )


def _resolve_torch_dtype(torch_module: Any, dtype_name: str) -> Any:
    key = dtype_name.strip().lower()
    if key == "auto":
        return "auto"
    if key in {"fp16", "float16", "half"}:
        return torch_module.float16
    if key in {"bf16", "bfloat16"}:
        return torch_module.bfloat16
    if key in {"fp32", "float32"}:
        return torch_module.float32
    raise ValueError(f"Unsupported dtype '{dtype_name}'. Use one of: auto, float16, bfloat16, float32.")


def _resolve_input_device(model: Any, torch_module: Any) -> Any:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for value in hf_device_map.values():
            if isinstance(value, int):
                return torch_module.device(f"cuda:{value}")
            if isinstance(value, str) and value not in {"cpu", "disk"}:
                if value == "cuda":
                    return torch_module.device("cuda:0")
                return torch_module.device(value)

    model_device = getattr(model, "device", None)
    if model_device is not None:
        return model_device

    return torch_module.device("cpu")


def _load_audio_file(path: Path, *, sampling_rate: int) -> Any:
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - import tested through integration
        raise RuntimeError("Missing dependency 'librosa'. Install dependencies first.") from exc

    audio, _ = librosa.load(str(path), sr=sampling_rate, mono=True)
    return audio


def run_qwen2_audio_inference(
    *,
    examples: list[MCQOrderExample],
    audio_root: Path,
    model_base: str,
    batch_size: int,
    max_new_tokens: int,
    device_map: str,
    dtype: str,
    trust_remote_code: bool,
    hf_token: str | None,
) -> tuple[list[dict[str, Any]], float]:
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None

    if load_dotenv is not None:
        load_dotenv()

    token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    try:
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    except ImportError as exc:  # pragma: no cover - import tested through integration
        raise RuntimeError("Missing transformers stack. Install dependencies first.") from exc

    processor_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    model_kwargs: dict[str, Any] = {
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        "torch_dtype": _resolve_torch_dtype(torch, dtype),
    }
    if token:
        processor_kwargs["token"] = token
        model_kwargs["token"] = token

    with console.status("[bold green]Loading Qwen2-Audio model and processor...[/bold green]"):
        processor = AutoProcessor.from_pretrained(model_base, **processor_kwargs)
        model = Qwen2AudioForConditionalGeneration.from_pretrained(model_base, **model_kwargs)
        model.eval()

    sampling_rate = int(getattr(processor.feature_extractor, "sampling_rate", 16000))
    input_device = _resolve_input_device(model, torch)
    raw_outputs: list[dict[str, Any]] = []

    started = time.perf_counter()
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("Running Qwen2-Audio inference", total=len(examples))
        for offset in range(0, len(examples), batch_size):
            batch_examples = examples[offset : offset + batch_size]
            batch_prompts: list[str] = []
            batch_audio_arrays: list[Any] = []

            for example in batch_examples:
                source_audio = (audio_root / example.audio_filename).resolve()
                if not source_audio.exists():
                    raise FileNotFoundError(f"Audio file not found: {source_audio}")
                batch_audio_arrays.append(_load_audio_file(source_audio, sampling_rate=sampling_rate))
                batch_prompts.append(build_prompt(example))

            batch_inputs = processor(
                text=batch_prompts,
                audios=batch_audio_arrays,
                sampling_rate=sampling_rate,
                padding=True,
                return_tensors="pt",
            )
            batch_inputs = {key: value.to(input_device) for key, value in batch_inputs.items()}

            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
            }
            pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
            eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
            if pad_token_id is not None:
                generation_kwargs["pad_token_id"] = pad_token_id
            if eos_token_id is not None:
                generation_kwargs["eos_token_id"] = eos_token_id

            with torch.inference_mode():
                generated_ids = model.generate(**batch_inputs, **generation_kwargs)

            attention_mask = batch_inputs.get("attention_mask")
            for index, example in enumerate(batch_examples):
                if attention_mask is not None:
                    prompt_len = int(attention_mask[index].sum().item())
                else:
                    prompt_len = int(batch_inputs["input_ids"][index].shape[-1])

                completion_ids = generated_ids[index][prompt_len:].detach().cpu().tolist()
                predicted_text = processor.decode(
                    completion_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
                raw_outputs.append(
                    {
                        "id": example.example_id,
                        "question": example.question,
                        "gt_answer": example.answer_label,
                        "pred": predicted_text,
                    }
                )
                progress.advance(task, 1)

    elapsed = time.perf_counter() - started
    return raw_outputs, elapsed


def evaluate_qwen2_audio_outputs(
    *,
    examples: list[MCQOrderExample],
    raw_outputs: list[dict[str, Any]],
    model_name: str,
    model_base: str,
    tracker: WandbTracker | None = None,
) -> tuple[list[Decision], int, int]:
    outputs_by_id: dict[str, dict[str, Any]] = {}
    for row in raw_outputs:
        output_id = row.get("id")
        if isinstance(output_id, str):
            outputs_by_id[output_id] = row

    decisions: list[Decision] = []
    invalid_count = 0
    missing_count = 0

    for example in examples:
        valid_labels = {option.label for option in example.options}
        label_to_text = {option.label: option.text for option in example.options}
        label_to_type = {option.label: option.option_type for option in example.options}

        raw_prediction = ""
        parse_status = "missing"
        predicted_label: str | None = None
        row = outputs_by_id.get(example.example_id)
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
        is_correct = final_label == example.answer_label
        answer_type = label_to_type.get(example.answer_label, "unknown")

        decisions.append(
            Decision(
                task_id=TASK_ID_MCQ_ORDER,
                model_name=model_name,
                model_base=model_base,
                example_id=example.example_id,
                audio_filename=example.audio_filename,
                question=example.question,
                predicted_label=final_label,
                predicted_text=predicted_text,
                answer_label=example.answer_label,
                answer_text=example.answer_text,
                answer_type=answer_type,
                is_correct=is_correct,
                parse_status=parse_status,
                raw_prediction=raw_prediction,
                n_options=len(example.options),
                predicted_type=predicted_type,
            )
        )
        if tracker is not None and tracker.active:
            index = len(decisions)
            correct_so_far = sum(1 for d in decisions if d.is_correct)
            tracker.log_live(
                index=index,
                total=len(examples),
                accuracy_so_far=correct_so_far / index,
                latency_ms=0.0,
                is_correct=decisions[-1].is_correct,
                correct_so_far=correct_so_far,
                force=(index == 1 or index == len(examples)),
            )

    return decisions, invalid_count, missing_count


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    table = Table(title="MCQ-ORDER Qwen2-Audio summary", show_header=True, header_style="bold")
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


def _create_run_dir(results_root: Path, *, model_name: str, run_id: str) -> Path:
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
    model_base: str = typer.Option(
        DEFAULT_MODEL_BASE,
        "--model-base",
        help="Qwen2-Audio model id (HF id or local path).",
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
    batch_size: int = typer.Option(
        2,
        "--batch-size",
        help="Inference batch size. A40-safe default is 2.",
        min=1,
    ),
    max_new_tokens: int = typer.Option(
        16,
        "--max-new-tokens",
        help="Generation max tokens for MCQ label output.",
        min=1,
    ),
    device_map: str = typer.Option(
        "auto",
        "--device-map",
        help="Transformers device_map for model loading.",
    ),
    dtype: str = typer.Option(
        "float16",
        "--dtype",
        help="Precision for model loading (auto|float16|bfloat16|float32).",
    ),
    trust_remote_code: bool = typer.Option(
        False,
        "--trust-remote-code",
        help="Enable trust_remote_code for model and processor.",
    ),
    hf_token: str | None = typer.Option(
        None,
        "--hf-token",
        help="Optional Hugging Face token (otherwise use HF_TOKEN/HUGGINGFACE_HUB_TOKEN).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate dataset/audio setup only; skip model inference and evaluation.",
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

    _validate_audio_files(examples, audio_root=audio_root)

    model_name = _resolve_model_name(model_base)
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
            "model_name": model_name,
            "model_base": model_base,
            "limit": limit,
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "device_map": device_map,
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
        },
        tags=["mcq-order", "qwen2-audio", model_name],
    )

    started_at = datetime.now(UTC)
    run_id = started_at.strftime("%Y%m%d_%H%M%S")
    run_dir = _create_run_dir(results_root, model_name=model_name, run_id=run_id)

    try:
        config_payload = {
            "task_id": TASK_ID_MCQ_ORDER,
            "dataset": str(dataset),
            "audio_root": str(audio_root),
            "model_name": model_name,
            "model_base": model_base,
            "limit": limit,
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "device_map": device_map,
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
        }
        _write_json(run_dir / "run_config.json", config_payload)

        if dry_run:
            console.print("[yellow]Dry run enabled; inference and evaluation skipped.[/yellow]")
            tracker.log({"stage/dry_run": 1, "stage/examples_prepared": len(examples)})
            return

        tracker.log({"stage/inference_started": 1})
        raw_outputs, inference_elapsed = run_qwen2_audio_inference(
            examples=examples,
            audio_root=audio_root,
            model_base=model_base,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            hf_token=hf_token,
        )
        tracker.log({"stage/inference_complete": 1, "stage/inference_elapsed_seconds": inference_elapsed})

        raw_outputs_path = run_dir / "raw_model_outputs.jsonl"
        _write_jsonl(raw_outputs_path, raw_outputs)

        decisions, invalid_count, missing_count = evaluate_qwen2_audio_outputs(
            examples=examples,
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
            raw_outputs_path=str(raw_outputs_path),
        )

        _write_json(run_dir / "metrics.json", metrics.to_json())
        _write_results_table(run_dir / "results_table.md", metrics)
        _append_runs_csv(results_root / "mcq-order" / "runs.csv", metrics)
        analysis_payload = _build_analysis_payload(decisions)
        analysis_path = run_dir / "analysis.json"
        _write_json(analysis_path, analysis_payload)

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
                raw_outputs_path,
            ],
        )

        _print_summary(metrics, run_dir)
    finally:
        tracker.finish()


if __name__ == "__main__":
    typer.run(main)
