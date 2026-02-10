"""Run MCQ-ORDER models and store decisions + summary artifacts."""

from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import typer
from rich.console import Console
from rich.table import Table

try:
    from .mcq_order_models import TASK_ID_MCQ_ORDER, MCQOrderExample, MCQOrderModel, build_model
    from .wandb_tracker import WandbTracker
except ImportError:  # pragma: no cover - enables direct script execution
    from mcq_order_models import TASK_ID_MCQ_ORDER, MCQOrderExample, MCQOrderModel, build_model
    from wandb_tracker import WandbTracker

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
    answer_type: str
    is_correct: bool
    latency_ms: float
    n_options: int
    predicted_type: str

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
            "answer_type": self.answer_type,
            "is_correct": self.is_correct,
            "latency_ms": round(self.latency_ms, 6),
            "n_options": self.n_options,
            "predicted_type": self.predicted_type,
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
    return evaluate_with_callback(model=model, examples=examples, callback=None)


def evaluate_with_callback(
    model: MCQOrderModel,
    examples: Iterable[MCQOrderExample],
    callback: Callable[[int, Decision, int], None] | None,
) -> tuple[list[Decision], float]:
    decisions: list[Decision] = []
    started = time.perf_counter()
    correct_so_far = 0

    for index, example in enumerate(examples, start=1):
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
        answer_option = example.option_by_label(example.answer_label)

        is_correct = predicted_label == example.answer_label
        if is_correct:
            correct_so_far += 1
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
                answer_type=answer_option.option_type,
                is_correct=is_correct,
                latency_ms=latency_ms,
                n_options=len(example.options),
                predicted_type=predicted_option.option_type,
            )
        )
        if callback is not None:
            callback(index, decisions[-1], correct_so_far)

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


def _write_analysis(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
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
    local_model = metrics.model_metadata.get("model_id")
    if isinstance(local_model, str) and local_model:
        table.add_row("HF model", local_model)
    table.add_row("Examples", str(metrics.examples))
    table.add_row("Correct", str(metrics.correct))
    table.add_row("Accuracy", f"{metrics.accuracy:.4f}")
    table.add_row("Elapsed (s)", f"{metrics.elapsed_seconds:.4f}")
    table.add_row("Avg latency (ms)", f"{metrics.average_latency_ms:.4f}")
    table.add_row("Run directory", str(run_dir))
    console.print(table)


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


def _build_analysis_payload(
    *,
    task_id: str,
    model_name: str,
    decisions: list[Decision],
) -> dict[str, Any]:
    total = len(decisions)
    correct = sum(1 for decision in decisions if decision.is_correct)
    answer_counts: Counter[str] = Counter(decision.answer_label for decision in decisions)
    prediction_counts: Counter[str] = Counter(decision.predicted_label for decision in decisions)
    correct_by_answer: Counter[str] = Counter(
        decision.answer_label for decision in decisions if decision.is_correct
    )
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

    return {
        "task_id": task_id,
        "model_name": model_name,
        "examples": total,
        "correct": correct,
        "accuracy": _safe_divide(correct, total),
        "labels": labels,
        "answer_distribution": answer_distribution,
        "prediction_distribution": prediction_distribution,
        "by_answer_label": by_answer_label,
        "by_n_options": by_option_count,
        "by_answer_type": by_type,
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
        help="Model name to run (available: random, llm-openai, llm-qwen, llm-llama).",
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
    qwen_model_id: str = typer.Option(
        "Qwen/Qwen2.5-7B-Instruct",
        "--qwen-model-id",
        help="Hugging Face model id used by --model llm-qwen.",
    ),
    llama_model_id: str = typer.Option(
        "meta-llama/Llama-3.1-8B-Instruct",
        "--llama-model-id",
        help="Hugging Face model id used by --model llm-llama.",
    ),
    local_temperature: float = typer.Option(
        0.0,
        "--local-temperature",
        help="Sampling temperature for local Hugging Face models (qwen/llama).",
    ),
    local_top_p: float = typer.Option(
        1.0,
        "--local-top-p",
        help="Top-p for local Hugging Face models when local-temperature > 0.",
    ),
    local_max_new_tokens: int = typer.Option(
        16,
        "--local-max-new-tokens",
        help="Max generation length for local Hugging Face models.",
        min=1,
    ),
    local_device_map: str = typer.Option(
        "auto",
        "--local-device-map",
        help="Transformers device_map for local Hugging Face models.",
    ),
    local_dtype: str = typer.Option(
        "float16",
        "--local-dtype",
        help="Precision for local Hugging Face models (auto|float16|bfloat16|float32).",
    ),
    local_trust_remote_code: bool = typer.Option(
        False,
        "--local-trust-remote-code",
        help="Enable trust_remote_code for local Hugging Face models.",
    ),
    hf_token: str | None = typer.Option(
        None,
        "--hf-token",
        help="Optional Hugging Face token (otherwise use HF_TOKEN/HUGGINGFACE_HUB_TOKEN).",
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
    model_instance = build_model(
        model,
        seed=seed,
        openai_model=openai_model,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        prediction_retries=prediction_retries,
        qwen_model_id=qwen_model_id,
        llama_model_id=llama_model_id,
        local_temperature=local_temperature,
        local_top_p=local_top_p,
        local_max_new_tokens=local_max_new_tokens,
        local_device_map=local_device_map,
        local_dtype=local_dtype,
        local_trust_remote_code=local_trust_remote_code,
        hf_token=hf_token,
    )
    examples = load_examples(dataset, limit=limit)
    if not examples:
        raise typer.BadParameter("No examples found in dataset.")

    tracker = WandbTracker(
        enabled=wandb,
        project=wandb_project,
        entity=wandb_entity,
        run_name=wandb_run_name,
        log_every=wandb_log_every,
        config={
            "task_id": TASK_ID_MCQ_ORDER,
            "dataset": str(dataset),
            "model": model_instance.model_name,
            "model_config": model_instance.run_metadata(),
            "limit": limit,
            "openai_model": openai_model,
            "qwen_model_id": qwen_model_id,
            "llama_model_id": llama_model_id,
            "local_temperature": local_temperature,
            "local_top_p": local_top_p,
            "local_max_new_tokens": local_max_new_tokens,
            "local_device_map": local_device_map,
            "local_dtype": local_dtype,
            "local_trust_remote_code": local_trust_remote_code,
        },
        tags=["mcq-order", model_instance.model_name],
    )

    try:
        started_at = datetime.now(UTC)
        total_examples = len(examples)

        def _on_decision(index: int, decision: Decision, correct_so_far: int) -> None:
            tracker.log_live(
                index=index,
                total=total_examples,
                accuracy_so_far=correct_so_far / index,
                latency_ms=decision.latency_ms,
                is_correct=decision.is_correct,
                correct_so_far=correct_so_far,
                force=(index == 1 or index == total_examples),
            )

        decisions, elapsed_seconds = evaluate_with_callback(
            model=model_instance,
            examples=examples,
            callback=_on_decision if tracker.active else None,
        )
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
        analysis_payload = _build_analysis_payload(
            task_id=TASK_ID_MCQ_ORDER,
            model_name=model_instance.model_name,
            decisions=decisions,
        )
        analysis_path = run_dir / "analysis.json"
        _write_analysis(analysis_path, analysis_payload)

        tracker.update_summary(
            {
                "task_id": metrics.task_id,
                "model_name": metrics.model_name,
                "examples": metrics.examples,
                "correct": metrics.correct,
                "accuracy": metrics.accuracy,
                "elapsed_seconds": metrics.elapsed_seconds,
                "average_latency_ms": metrics.average_latency_ms,
                "dataset_path": metrics.dataset_path,
                "run_id": metrics.run_id,
                "analysis/answer_entropy": analysis_payload["answer_entropy"],
                "analysis/prediction_entropy": analysis_payload["prediction_entropy"],
            }
        )
        _log_wandb_analysis(tracker=tracker, analysis=analysis_payload, decisions=decisions)
        tracker.log_artifact(
            name=f"{metrics.task_id.lower()}-{metrics.model_name}-{metrics.run_id}",
            artifact_type="evaluation",
            files=[
                decisions_path,
                run_dir / "metrics.json",
                run_dir / "results_table.md",
                analysis_path,
            ],
        )

        _print_summary(metrics, run_dir)
    finally:
        tracker.finish()


if __name__ == "__main__":
    typer.run(main)
