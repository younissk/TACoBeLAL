"""Run MCQ-ORDER models and store decisions + summary artifacts."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import typer
from rich.console import Console
from rich.table import Table

try:
    from .mcq_order_models import TASK_ID_MCQ_ORDER, MCQOrderExample, MCQOrderModel, build_model
except ImportError:  # pragma: no cover - enables direct script execution
    from mcq_order_models import TASK_ID_MCQ_ORDER, MCQOrderExample, MCQOrderModel, build_model

console = Console()


@dataclass(frozen=True)
class Decision:
    task_id: str
    model_name: str
    example_id: str
    audio_filename: str
    question: str
    predicted_label: str
    predicted_text: str
    answer_label: str
    answer_text: str
    is_correct: bool
    latency_ms: float
    n_options: int

    def to_json(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "model_name": self.model_name,
            "example_id": self.example_id,
            "audio_filename": self.audio_filename,
            "question": self.question,
            "predicted_label": self.predicted_label,
            "predicted_text": self.predicted_text,
            "answer_label": self.answer_label,
            "answer_text": self.answer_text,
            "is_correct": self.is_correct,
            "latency_ms": round(self.latency_ms, 6),
            "n_options": self.n_options,
        }


@dataclass(frozen=True)
class RunMetrics:
    task_id: str
    model_name: str
    model_metadata: dict[str, Any]
    examples: int
    correct: int
    accuracy: float
    elapsed_seconds: float
    average_latency_ms: float
    run_id: str
    started_at_utc: str
    finished_at_utc: str
    dataset_path: str
    decisions_path: str

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "model_name": self.model_name,
            "model_metadata": self.model_metadata,
            "dataset_path": self.dataset_path,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "examples": self.examples,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 6),
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "average_latency_ms": round(self.average_latency_ms, 6),
            "decisions_path": self.decisions_path,
        }


def load_examples(dataset_jsonl: Path, limit: int | None = None) -> list[MCQOrderExample]:
    examples: list[MCQOrderExample] = []
    with open(dataset_jsonl, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            try:
                example = MCQOrderExample.from_json(payload, task_id=TASK_ID_MCQ_ORDER)
            except ValueError as exc:
                raise ValueError(f"Invalid example at line {line_number}: {exc}") from exc
            examples.append(example)
            if limit is not None and len(examples) >= limit:
                break
    return examples


def evaluate(model: MCQOrderModel, examples: Iterable[MCQOrderExample]) -> tuple[list[Decision], float]:
    decisions: list[Decision] = []
    started = time.perf_counter()

    for example in examples:
        decision_started = time.perf_counter()
        predicted_label = model.predict(example)
        latency_ms = (time.perf_counter() - decision_started) * 1000.0

        try:
            predicted_option = example.option_by_label(predicted_label)
        except KeyError as exc:
            raise ValueError(
                f"Model '{model.model_name}' predicted invalid label '{predicted_label}' "
                f"for example '{example.example_id}'."
            ) from exc

        is_correct = predicted_label == example.answer_label
        decisions.append(
            Decision(
                task_id=TASK_ID_MCQ_ORDER,
                model_name=model.model_name,
                example_id=example.example_id,
                audio_filename=example.audio_filename,
                question=example.question,
                predicted_label=predicted_label,
                predicted_text=predicted_option.text,
                answer_label=example.answer_label,
                answer_text=example.answer_text,
                is_correct=is_correct,
                latency_ms=latency_ms,
                n_options=len(example.options),
            )
        )

    elapsed_seconds = time.perf_counter() - started
    return decisions, elapsed_seconds


def _write_decisions(path: Path, decisions: Iterable[Decision]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for decision in decisions:
            f.write(json.dumps(decision.to_json(), ensure_ascii=False) + "\n")


def _write_metrics(path: Path, metrics: RunMetrics) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics.to_json(), f, ensure_ascii=False, indent=2)
        f.write("\n")


def _build_results_table(metrics: RunMetrics) -> str:
    return (
        "| Task ID | Model | Examples | Correct | Accuracy | Elapsed (s) | Avg latency (ms) |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |\n"
        f"| {metrics.task_id} | {metrics.model_name} | {metrics.examples} | {metrics.correct} "
        f"| {metrics.accuracy:.4f} | {metrics.elapsed_seconds:.4f} | {metrics.average_latency_ms:.4f} |\n"
    )


def _write_results_table(path: Path, metrics: RunMetrics) -> None:
    path.write_text(_build_results_table(metrics), encoding="utf-8")


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
    table = Table(title="MCQ-ORDER evaluation summary", show_header=True, header_style="bold")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Task ID", metrics.task_id)
    table.add_row("Model", metrics.model_name)
    openai_model = metrics.model_metadata.get("openai_model")
    if isinstance(openai_model, str) and openai_model:
        table.add_row("OpenAI model", openai_model)
    table.add_row("Examples", str(metrics.examples))
    table.add_row("Correct", str(metrics.correct))
    table.add_row("Accuracy", f"{metrics.accuracy:.4f}")
    table.add_row("Elapsed (s)", f"{metrics.elapsed_seconds:.4f}")
    table.add_row("Avg latency (ms)", f"{metrics.average_latency_ms:.4f}")
    table.add_row("Run directory", str(run_dir))
    console.print(table)


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
    model: str = typer.Option(
        "random",
        "--model",
        "-m",
        help="Model name to run (available: random, llm-openai).",
    ),
    seed: int = typer.Option(
        7,
        "--seed",
        help="Random seed used by stochastic models.",
    ),
    openai_model: str = typer.Option(
        "gpt-4o-mini",
        "--openai-model",
        help="OpenAI model name used when --model llm-openai.",
    ),
    temperature: float = typer.Option(
        0.0,
        "--temperature",
        help="Sampling temperature used by llm-openai.",
    ),
    timeout_seconds: float = typer.Option(
        60.0,
        "--timeout-seconds",
        help="Request timeout used by llm-openai.",
    ),
    max_retries: int = typer.Option(
        2,
        "--max-retries",
        help="OpenAI transport retries used by llm-openai.",
        min=0,
    ),
    prediction_retries: int = typer.Option(
        2,
        "--prediction-retries",
        help="Extra retries when model response label is invalid.",
        min=0,
    ),
    results_root: Path = typer.Option(
        Path("results"),
        "--results-root",
        "-o",
        path_type=Path,
        help="Root directory for benchmark outputs.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Optional max number of examples to evaluate.",
        min=1,
    ),
) -> None:
    model_instance = build_model(
        model,
        seed=seed,
        openai_model=openai_model,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        prediction_retries=prediction_retries,
    )
    examples = load_examples(dataset, limit=limit)
    if not examples:
        raise typer.BadParameter("No examples found in dataset.")

    started_at = datetime.now(UTC)
    decisions, elapsed_seconds = evaluate(model=model_instance, examples=examples)
    finished_at = datetime.now(UTC)

    correct = sum(decision.is_correct for decision in decisions)
    average_latency_ms = sum(decision.latency_ms for decision in decisions) / len(decisions)
    run_id = started_at.strftime("%Y%m%d_%H%M%S")
    run_dir = results_root / "mcq-order" / model_instance.model_name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    decisions_path = run_dir / "decisions.jsonl"
    _write_decisions(decisions_path, decisions)

    metrics = RunMetrics(
        task_id=TASK_ID_MCQ_ORDER,
        model_name=model_instance.model_name,
        model_metadata=model_instance.run_metadata(),
        examples=len(decisions),
        correct=correct,
        accuracy=correct / len(decisions),
        elapsed_seconds=elapsed_seconds,
        average_latency_ms=average_latency_ms,
        run_id=run_id,
        started_at_utc=started_at.isoformat(),
        finished_at_utc=finished_at.isoformat(),
        dataset_path=str(dataset),
        decisions_path=str(decisions_path),
    )

    _write_metrics(run_dir / "metrics.json", metrics)
    _write_results_table(run_dir / "results_table.md", metrics)
    _append_runs_csv(results_root / "mcq-order" / "runs.csv", metrics)

    _print_summary(metrics, run_dir)


if __name__ == "__main__":
    typer.run(main)
