"""Evaluate MCQ-ORDER using Voxtral models (audio-capable LALMs)."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

try:
    from .evaluate_mcq_order import load_examples
    from .evaluate_mcq_order_qwen2_audio import (
        RunMetrics,
        _append_runs_csv,
        _build_analysis_payload,
        _create_run_dir,
        _log_wandb_analysis,
        _resolve_input_device,
        _resolve_torch_dtype,
        _safe_name,
        _validate_audio_files,
        _write_decisions,
        _write_json,
        _write_jsonl,
        _write_results_table,
        evaluate_qwen2_audio_outputs,
    )
    from .mcq_order_models import MCQOrderExample, TASK_ID_MCQ_ORDER
    from .wandb_tracker import WandbTracker
except ImportError:  # pragma: no cover - enables direct script execution
    from evaluate_mcq_order import load_examples
    from evaluate_mcq_order_qwen2_audio import (
        RunMetrics,
        _append_runs_csv,
        _build_analysis_payload,
        _create_run_dir,
        _log_wandb_analysis,
        _resolve_input_device,
        _resolve_torch_dtype,
        _safe_name,
        _validate_audio_files,
        _write_decisions,
        _write_json,
        _write_jsonl,
        _write_results_table,
        evaluate_qwen2_audio_outputs,
    )
    from mcq_order_models import MCQOrderExample, TASK_ID_MCQ_ORDER
    from wandb_tracker import WandbTracker

console = Console()

DEFAULT_MODEL_BASE = "mistralai/Voxtral-Mini-3B-2507"
TRANSFORMERS_MIN_VERSION_HINT = "4.57.0"


def _resolve_model_name(model_base: str) -> str:
    return _safe_name(model_base.strip().split("/")[-1].lower())


def build_user_prompt(example: MCQOrderExample) -> str:
    options_text = "\n".join(f"{option.label}. {option.text}" for option in example.options)
    labels_text = ", ".join(option.label for option in example.options)
    return (
        f"{example.question}\n\n"
        "Choose exactly one option.\n"
        f"{options_text}\n\n"
        f"Return only the option label from: {labels_text}."
    )


def build_conversation(example: MCQOrderExample, audio_path: Path) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "path": str(audio_path)},
                {"type": "text", "text": build_user_prompt(example)},
            ],
        },
    ]


def run_voxtral_inference(
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
    attn_implementation: str | None,
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
        import transformers
    except ImportError as exc:  # pragma: no cover - import tested through integration
        raise RuntimeError("Missing transformers stack. Install dependencies first.") from exc

    if not hasattr(transformers, "VoxtralProcessor") or not hasattr(
        transformers, "VoxtralForConditionalGeneration"
    ):
        raise RuntimeError(
            "Voxtral is not available in this transformers version "
            f"({transformers.__version__}). Use transformers>={TRANSFORMERS_MIN_VERSION_HINT} "
            "(or v5+)."
        )

    processor_cls = transformers.VoxtralProcessor
    model_cls = transformers.VoxtralForConditionalGeneration

    processor_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    model_kwargs: dict[str, Any] = {
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        "torch_dtype": _resolve_torch_dtype(torch, dtype),
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    if token:
        processor_kwargs["token"] = token
        model_kwargs["token"] = token

    with console.status("[bold green]Loading Voxtral model and processor...[/bold green]"):
        processor = processor_cls.from_pretrained(model_base, **processor_kwargs)
        model = model_cls.from_pretrained(model_base, **model_kwargs)
        model.eval()

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
        task = progress.add_task("Running Voxtral inference", total=len(examples))
        for offset in range(0, len(examples), batch_size):
            batch_examples = examples[offset : offset + batch_size]
            conversations: list[list[dict[str, Any]]] = []

            for example in batch_examples:
                source_audio = (audio_root / example.audio_filename).resolve()
                if not source_audio.exists():
                    raise FileNotFoundError(f"Audio file not found: {source_audio}")
                conversations.append(build_conversation(example, source_audio))

            batch_inputs = processor.apply_chat_template(
                conversations,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            batch_inputs = {
                key: value.to(input_device) if hasattr(value, "to") else value
                for key, value in batch_inputs.items()
            }

            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
            }
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


def _print_summary(metrics: RunMetrics, run_dir: Path) -> None:
    table = Table(title="MCQ-ORDER Voxtral summary", show_header=True, header_style="bold")
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
        help="Voxtral model id (HF id or local path).",
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
    attn_implementation: str | None = typer.Option(
        None,
        "--attn-implementation",
        help="Optional attention backend (e.g., flash_attention_2, sdpa).",
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
            "attn_implementation": attn_implementation,
        },
        tags=["mcq-order", "voxtral", model_name],
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
            "attn_implementation": attn_implementation,
        }
        _write_json(run_dir / "run_config.json", config_payload)

        if dry_run:
            console.print("[yellow]Dry run enabled; inference and evaluation skipped.[/yellow]")
            tracker.log({"stage/dry_run": 1, "stage/examples_prepared": len(examples)})
            return

        tracker.log({"stage/inference_started": 1})
        raw_outputs, inference_elapsed = run_voxtral_inference(
            examples=examples,
            audio_root=audio_root,
            model_base=model_base,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            hf_token=hf_token,
            attn_implementation=attn_implementation,
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
