"""Unified benchmark launcher for local and SLURM workflows."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import typer
from rich.console import Console
from rich.table import Table

console = Console()

TRANSFORMERS_NEW_API = "transformers>=4.57.0"


@dataclass(frozen=True)
class TaskSpec:
    key: str
    dataset_default: Path
    build_target: str


@dataclass(frozen=True)
class ModelSpec:
    key: str
    evaluator_script: Path
    needs_audio_root: bool
    supports_audio_toggle: bool
    uses_text_evaluator: bool
    requires_new_transformers: bool
    needs_audioflamingo_repo: bool


TASKS: dict[str, TaskSpec] = {
    "mcq-order": TaskSpec(
        key="mcq-order",
        dataset_default=Path("data/mcq_event_timeline_strong.jsonl"),
        build_target="build-mcq-dataset",
    ),
    "mcq-relation": TaskSpec(
        key="mcq-relation",
        dataset_default=Path("data/mcq_relation_timeline_strong.jsonl"),
        build_target="build-mcq-relation-dataset",
    ),
    "mcq-safety": TaskSpec(
        key="mcq-safety",
        dataset_default=Path("data/mcq_safety_presence_100.jsonl"),
        build_target="build-mcq-safety-dataset",
    ),
}

MODELS: dict[str, ModelSpec] = {
    "random": ModelSpec(
        key="random",
        evaluator_script=Path("src/utils/evaluate_mcq_order.py"),
        needs_audio_root=False,
        supports_audio_toggle=False,
        uses_text_evaluator=True,
        requires_new_transformers=False,
        needs_audioflamingo_repo=False,
    ),
    "llm-openai": ModelSpec(
        key="llm-openai",
        evaluator_script=Path("src/utils/evaluate_mcq_order.py"),
        needs_audio_root=False,
        supports_audio_toggle=False,
        uses_text_evaluator=True,
        requires_new_transformers=False,
        needs_audioflamingo_repo=False,
    ),
    "llm-qwen": ModelSpec(
        key="llm-qwen",
        evaluator_script=Path("src/utils/evaluate_mcq_order.py"),
        needs_audio_root=False,
        supports_audio_toggle=False,
        uses_text_evaluator=True,
        requires_new_transformers=False,
        needs_audioflamingo_repo=False,
    ),
    "llm-llama": ModelSpec(
        key="llm-llama",
        evaluator_script=Path("src/utils/evaluate_mcq_order.py"),
        needs_audio_root=False,
        supports_audio_toggle=False,
        uses_text_evaluator=True,
        requires_new_transformers=False,
        needs_audioflamingo_repo=False,
    ),
    "qwen2-audio": ModelSpec(
        key="qwen2-audio",
        evaluator_script=Path("src/utils/evaluate_mcq_order_qwen2_audio.py"),
        needs_audio_root=True,
        supports_audio_toggle=True,
        uses_text_evaluator=False,
        requires_new_transformers=False,
        needs_audioflamingo_repo=False,
    ),
    "qwen2-5-omni": ModelSpec(
        key="qwen2-5-omni",
        evaluator_script=Path("src/utils/evaluate_mcq_order_qwen2_5_omni.py"),
        needs_audio_root=True,
        supports_audio_toggle=True,
        uses_text_evaluator=False,
        requires_new_transformers=True,
        needs_audioflamingo_repo=False,
    ),
    "voxtral": ModelSpec(
        key="voxtral",
        evaluator_script=Path("src/utils/evaluate_mcq_order_voxtral.py"),
        needs_audio_root=True,
        supports_audio_toggle=True,
        uses_text_evaluator=False,
        requires_new_transformers=True,
        needs_audioflamingo_repo=False,
    ),
    "audioflamingo": ModelSpec(
        key="audioflamingo",
        evaluator_script=Path("src/utils/evaluate_mcq_order_audioflamingo.py"),
        needs_audio_root=True,
        supports_audio_toggle=True,
        uses_text_evaluator=False,
        requires_new_transformers=False,
        needs_audioflamingo_repo=True,
    ),
}


def _normalize_task(value: str) -> str:
    key = value.strip().lower()
    if key not in TASKS:
        allowed = ", ".join(sorted(TASKS))
        raise typer.BadParameter(f"Unsupported task '{value}'. Choose one of: {allowed}")
    return key


def _normalize_model(value: str) -> str:
    aliases = {
        "openai": "llm-openai",
        "qwen": "llm-qwen",
        "llama": "llm-llama",
        "qwen2_audio": "qwen2-audio",
        "qwen2-5": "qwen2-5-omni",
        "af3": "audioflamingo",
    }
    key = aliases.get(value.strip().lower(), value.strip().lower())
    if key not in MODELS:
        allowed = ", ".join(sorted(MODELS))
        raise typer.BadParameter(f"Unsupported model '{value}'. Choose one of: {allowed}")
    return key


def _command_to_str(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _build_setup_targets(
    *,
    model: ModelSpec,
    task: TaskSpec,
    prepare_data: bool,
    install_deps: bool,
    use_audio: bool,
    dataset_is_default: bool,
) -> list[str]:
    targets: list[str] = []
    if install_deps:
        targets.append("install-tracking")
        if model.uses_text_evaluator and model.key != "random":
            targets.append("install-llm")

    if prepare_data:
        targets.append("download-dataset")
        if dataset_is_default:
            targets.append(task.build_target)
        if model.needs_audio_root and use_audio:
            targets.append("extract-audio")
        if model.needs_audioflamingo_repo:
            targets.append("download-audioflamingo")

    seen: set[str] = set()
    unique: list[str] = []
    for target in targets:
        if target in seen:
            continue
        unique.append(target)
        seen.add(target)
    return unique


def _build_eval_command(
    *,
    model: ModelSpec,
    dataset: Path,
    results_root: Path,
    samples: int | None,
    seed: int,
    use_audio: bool,
    audio_root: Path,
    audioflamingo_repo: Path,
    openai_model: str,
    qwen_model_id: str,
    llama_model_id: str,
    qwen2_audio_model_id: str,
    qwen2_5_omni_model_id: str,
    voxtral_model_id: str,
    audioflamingo_model_base: str,
    batch_size: int,
    max_new_tokens: int,
    num_gpus: int,
    attn_implementation: str | None,
    local_dtype: str,
    local_device_map: str,
    local_temperature: float,
    local_top_p: float,
    local_max_new_tokens: int,
    wandb: bool,
    wandb_project: str,
    wandb_entity: str | None,
    wandb_run_name: str | None,
    wandb_log_every: int,
    hf_token: str | None,
) -> list[str]:
    command: list[str] = ["uv", "run"]
    if model.requires_new_transformers:
        command += ["--with", TRANSFORMERS_NEW_API]
    command += ["python", str(model.evaluator_script), "--dataset", str(dataset), "--results-root", str(results_root)]

    if samples is not None:
        command += ["--limit", str(samples)]
    if hf_token:
        command += ["--hf-token", hf_token]
    if wandb:
        command += ["--wandb", "--wandb-project", wandb_project, "--wandb-log-every", str(wandb_log_every)]
        if wandb_entity:
            command += ["--wandb-entity", wandb_entity]
        if wandb_run_name:
            command += ["--wandb-run-name", wandb_run_name]
    else:
        command.append("--no-wandb")

    if model.uses_text_evaluator:
        command += ["--model", model.key, "--seed", str(seed)]
        if model.key == "llm-openai":
            command += ["--openai-model", openai_model, "--temperature", "0"]
        if model.key == "llm-qwen":
            command += [
                "--qwen-model-id",
                qwen_model_id,
                "--local-dtype",
                local_dtype,
                "--local-device-map",
                local_device_map,
                "--local-temperature",
                str(local_temperature),
                "--local-top-p",
                str(local_top_p),
                "--local-max-new-tokens",
                str(local_max_new_tokens),
            ]
        if model.key == "llm-llama":
            command += [
                "--llama-model-id",
                llama_model_id,
                "--local-dtype",
                local_dtype,
                "--local-device-map",
                local_device_map,
                "--local-temperature",
                str(local_temperature),
                "--local-top-p",
                str(local_top_p),
                "--local-max-new-tokens",
                str(local_max_new_tokens),
            ]
        return command

    if model.needs_audio_root:
        command += ["--audio-root", str(audio_root)]
    if model.needs_audioflamingo_repo:
        command += ["--audioflamingo-repo", str(audioflamingo_repo), "--num-gpus", str(num_gpus)]

    command += ["--batch-size", str(batch_size), "--max-new-tokens", str(max_new_tokens)]
    if model.key in {"qwen2-audio", "qwen2-5-omni", "voxtral"}:
        command += ["--dtype", local_dtype, "--device-map", local_device_map]
    if attn_implementation and model.key in {"qwen2-5-omni", "voxtral"}:
        command += ["--attn-implementation", attn_implementation]
    if model.key == "qwen2-audio":
        command += ["--model-base", qwen2_audio_model_id]
    elif model.key == "qwen2-5-omni":
        command += ["--model-base", qwen2_5_omni_model_id]
    elif model.key == "voxtral":
        command += ["--model-base", voxtral_model_id]
    elif model.key == "audioflamingo":
        command += ["--model-base", audioflamingo_model_base]

    if model.supports_audio_toggle and not use_audio:
        command += ["--disable-audio"]
    return command


def _run(command: Sequence[str], *, cwd: Path) -> None:
    console.print(f"[cyan]$ {_command_to_str(command)}[/cyan]")
    subprocess.run(command, cwd=cwd, check=True)


def _print_plan(
    *,
    task: TaskSpec,
    model: ModelSpec,
    dataset: Path,
    results_root: Path,
    samples: int | None,
    setup_targets: list[str],
) -> None:
    table = Table(title="Benchmark run plan", show_header=True, header_style="bold")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Task", task.key)
    table.add_row("Model", model.key)
    table.add_row("Dataset", str(dataset))
    table.add_row("Results root", str(results_root))
    table.add_row("Samples", "full dataset" if samples is None else str(samples))
    table.add_row("Setup targets", ", ".join(setup_targets) if setup_targets else "(none)")
    console.print(table)


def main(
    task: str = typer.Option(
        "mcq-order",
        "--task",
        help="Benchmark task id (mcq-order|mcq-relation|mcq-safety).",
    ),
    model: str = typer.Option(
        "random",
        "--model",
        "-m",
        help=(
            "Model backend "
            "(random|llm-openai|llm-qwen|llm-llama|qwen2-audio|qwen2-5-omni|voxtral|audioflamingo)."
        ),
    ),
    samples: int = typer.Option(
        100,
        "--samples",
        help="Number of examples to evaluate (0 means full dataset).",
        min=0,
    ),
    dataset: Path | None = typer.Option(
        None,
        "--dataset",
        path_type=Path,
        help="Override dataset JSONL path. Defaults by task.",
    ),
    results_root: Path = typer.Option(
        Path("results"),
        "--results-root",
        path_type=Path,
        help="Root directory for benchmark outputs.",
    ),
    seed: int = typer.Option(
        7,
        "--seed",
        help="Random seed for stochastic text models.",
    ),
    use_audio: bool = typer.Option(
        True,
        "--use-audio/--disable-audio",
        help="Use audio input for audio-capable wrappers.",
    ),
    audio_root: Path = typer.Option(
        Path("data/audio"),
        "--audio-root",
        path_type=Path,
        help="Root directory for extracted audio files.",
    ),
    audioflamingo_repo: Path = typer.Option(
        Path("external/audio-flamingo"),
        "--audioflamingo-repo",
        path_type=Path,
        help="Path to cloned Audio Flamingo repo.",
    ),
    openai_model: str = typer.Option(
        "gpt-4o-mini",
        "--openai-model",
        help="OpenAI model for llm-openai backend.",
    ),
    qwen_model_id: str = typer.Option(
        "Qwen/Qwen2.5-7B-Instruct",
        "--qwen-model-id",
        help="HF model id for llm-qwen backend.",
    ),
    llama_model_id: str = typer.Option(
        "meta-llama/Llama-3.1-8B-Instruct",
        "--llama-model-id",
        help="HF model id for llm-llama backend.",
    ),
    qwen2_audio_model_id: str = typer.Option(
        "Qwen/Qwen2-Audio-7B-Instruct",
        "--qwen2-audio-model-id",
        help="HF model id for qwen2-audio backend.",
    ),
    qwen2_5_omni_model_id: str = typer.Option(
        "Qwen/Qwen2.5-Omni-7B",
        "--qwen2-5-omni-model-id",
        help="HF model id for qwen2-5-omni backend.",
    ),
    voxtral_model_id: str = typer.Option(
        "mistralai/Voxtral-Mini-3B-2507",
        "--voxtral-model-id",
        help="HF model id for voxtral backend.",
    ),
    audioflamingo_model_base: str = typer.Option(
        "nvidia/audio-flamingo-3",
        "--audioflamingo-model-base",
        help="Model base for audioflamingo backend.",
    ),
    batch_size: int = typer.Option(
        2,
        "--batch-size",
        help="Batch size for audio-capable evaluators.",
        min=1,
    ),
    max_new_tokens: int = typer.Option(
        16,
        "--max-new-tokens",
        help="Generation max tokens for MCQ label output.",
        min=1,
    ),
    num_gpus: int = typer.Option(
        1,
        "--num-gpus",
        help="GPU count for Audio Flamingo torchrun path.",
        min=1,
    ),
    attn_implementation: str | None = typer.Option(
        None,
        "--attn-implementation",
        help="Optional attention backend for qwen2-5-omni/voxtral.",
    ),
    local_dtype: str = typer.Option(
        "float16",
        "--dtype",
        help="dtype for local/audio wrappers (auto|float16|bfloat16|float32).",
    ),
    local_device_map: str = typer.Option(
        "auto",
        "--device-map",
        help="device_map for local/audio wrappers.",
    ),
    local_temperature: float = typer.Option(
        0.0,
        "--local-temperature",
        help="Temperature for llm-qwen/llm-llama.",
    ),
    local_top_p: float = typer.Option(
        1.0,
        "--local-top-p",
        help="Top-p for llm-qwen/llm-llama.",
    ),
    local_max_new_tokens: int = typer.Option(
        16,
        "--local-max-new-tokens",
        help="Generation max tokens for llm-qwen/llm-llama.",
        min=1,
    ),
    hf_token: str | None = typer.Option(
        None,
        "--hf-token",
        help="Optional HF token override.",
    ),
    prepare_data: bool = typer.Option(
        True,
        "--prepare-data/--no-prepare-data",
        help="Run data preparation steps before evaluation.",
    ),
    install_deps: bool = typer.Option(
        False,
        "--install-deps/--no-install-deps",
        help="Run model-appropriate `make install-*` targets before evaluation.",
    ),
    wandb: bool = typer.Option(
        True,
        "--wandb/--no-wandb",
        help="Enable Weights & Biases logging.",
    ),
    wandb_project: str = typer.Option(
        "tacobelal",
        "--wandb-project",
        help="W&B project name.",
    ),
    wandb_entity: str | None = typer.Option(
        None,
        "--wandb-entity",
        help="W&B entity/team.",
    ),
    wandb_run_name: str | None = typer.Option(
        None,
        "--wandb-run-name",
        help="W&B run name.",
    ),
    wandb_log_every: int = typer.Option(
        50,
        "--wandb-log-every",
        help="Log every N evaluated examples.",
        min=1,
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print planned commands without executing.",
    ),
) -> None:
    task_key = _normalize_task(task)
    model_key = _normalize_model(model)
    task_spec = TASKS[task_key]
    model_spec = MODELS[model_key]

    if dataset is None:
        dataset = task_spec.dataset_default
        dataset_is_default = True
    else:
        dataset_is_default = dataset.resolve() == task_spec.dataset_default.resolve()
    selected_samples = None if samples == 0 else samples
    effective_results_root = results_root if task_spec.key == "mcq-order" else results_root / task_spec.key

    setup_targets = _build_setup_targets(
        model=model_spec,
        task=task_spec,
        prepare_data=prepare_data,
        install_deps=install_deps,
        use_audio=use_audio,
        dataset_is_default=dataset_is_default,
    )
    eval_command = _build_eval_command(
        model=model_spec,
        dataset=dataset,
        results_root=effective_results_root,
        samples=selected_samples,
        seed=seed,
        use_audio=use_audio,
        audio_root=audio_root,
        audioflamingo_repo=audioflamingo_repo,
        openai_model=openai_model,
        qwen_model_id=qwen_model_id,
        llama_model_id=llama_model_id,
        qwen2_audio_model_id=qwen2_audio_model_id,
        qwen2_5_omni_model_id=qwen2_5_omni_model_id,
        voxtral_model_id=voxtral_model_id,
        audioflamingo_model_base=audioflamingo_model_base,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        num_gpus=num_gpus,
        attn_implementation=attn_implementation,
        local_dtype=local_dtype,
        local_device_map=local_device_map,
        local_temperature=local_temperature,
        local_top_p=local_top_p,
        local_max_new_tokens=local_max_new_tokens,
        wandb=wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_name=wandb_run_name,
        wandb_log_every=wandb_log_every,
        hf_token=hf_token,
    )

    _print_plan(
        task=task_spec,
        model=model_spec,
        dataset=dataset,
        results_root=effective_results_root,
        samples=selected_samples,
        setup_targets=setup_targets,
    )
    console.print(f"[green]Eval command:[/green] {_command_to_str(eval_command)}")

    if dry_run:
        return

    cwd = Path.cwd()
    for target in setup_targets:
        _run(["make", target], cwd=cwd)
    _run(eval_command, cwd=cwd)


if __name__ == "__main__":
    typer.run(main)
