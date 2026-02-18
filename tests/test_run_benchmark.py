"""Tests for unified benchmark launcher command planning."""

from __future__ import annotations

from pathlib import Path

from utils.run_benchmark import (
    MODELS,
    TASKS,
    _build_eval_command,
    _build_setup_targets,
    _normalize_model,
    _normalize_task,
)


def test_normalize_aliases() -> None:
    assert _normalize_task("mcq-order") == "mcq-order"
    assert _normalize_task("mcq-relation") == "mcq-relation"
    assert _normalize_task("mcq-safety") == "mcq-safety"
    assert _normalize_model("qwen") == "llm-qwen"
    assert _normalize_model("af3") == "audioflamingo"


def test_setup_targets_for_audioflamingo_default_dataset() -> None:
    targets = _build_setup_targets(
        model=MODELS["audioflamingo"],
        task=TASKS["mcq-order"],
        prepare_data=True,
        install_deps=True,
        use_audio=True,
        dataset_is_default=True,
    )
    assert targets == [
        "install-tracking",
        "download-dataset",
        "build-mcq-dataset",
        "extract-audio",
        "download-audioflamingo",
    ]


def test_setup_targets_for_safety_default_dataset() -> None:
    targets = _build_setup_targets(
        model=MODELS["random"],
        task=TASKS["mcq-safety"],
        prepare_data=True,
        install_deps=False,
        use_audio=True,
        dataset_is_default=True,
    )
    assert targets == ["download-dataset", "build-mcq-safety-dataset"]


def test_build_eval_command_random_relation() -> None:
    command = _build_eval_command(
        model=MODELS["random"],
        dataset=Path("data/mcq_relation_timeline_strong.jsonl"),
        results_root=Path("results"),
        samples=250,
        seed=9,
        use_audio=True,
        audio_root=Path("data/audio"),
        audioflamingo_repo=Path("external/audio-flamingo"),
        openai_model="gpt-4o-mini",
        qwen_model_id="Qwen/Qwen2.5-7B-Instruct",
        llama_model_id="meta-llama/Llama-3.1-8B-Instruct",
        qwen2_audio_model_id="Qwen/Qwen2-Audio-7B-Instruct",
        qwen2_5_omni_model_id="Qwen/Qwen2.5-Omni-7B",
        voxtral_model_id="mistralai/Voxtral-Mini-3B-2507",
        audioflamingo_model_base="nvidia/audio-flamingo-3",
        batch_size=2,
        max_new_tokens=16,
        num_gpus=1,
        attn_implementation=None,
        local_dtype="float16",
        local_device_map="auto",
        local_temperature=0.0,
        local_top_p=1.0,
        local_max_new_tokens=16,
        wandb=False,
        wandb_project="tacobelal",
        wandb_entity=None,
        wandb_run_name=None,
        wandb_log_every=50,
        hf_token=None,
    )
    assert command[:3] == ["uv", "run", "python"]
    assert "src/utils/evaluate_mcq_order.py" in command
    assert "--dataset" in command and "data/mcq_relation_timeline_strong.jsonl" in command
    assert "--model" in command and "random" in command
    assert "--limit" in command and "250" in command
    assert "--no-wandb" in command


def test_build_eval_command_omni_no_audio_uses_transformers_override() -> None:
    command = _build_eval_command(
        model=MODELS["qwen2-5-omni"],
        dataset=Path("data/mcq_event_timeline_strong.jsonl"),
        results_root=Path("results"),
        samples=None,
        seed=7,
        use_audio=False,
        audio_root=Path("data/audio"),
        audioflamingo_repo=Path("external/audio-flamingo"),
        openai_model="gpt-4o-mini",
        qwen_model_id="Qwen/Qwen2.5-7B-Instruct",
        llama_model_id="meta-llama/Llama-3.1-8B-Instruct",
        qwen2_audio_model_id="Qwen/Qwen2-Audio-7B-Instruct",
        qwen2_5_omni_model_id="Qwen/Qwen2.5-Omni-7B",
        voxtral_model_id="mistralai/Voxtral-Mini-3B-2507",
        audioflamingo_model_base="nvidia/audio-flamingo-3",
        batch_size=1,
        max_new_tokens=16,
        num_gpus=1,
        attn_implementation="flash_attention_2",
        local_dtype="float16",
        local_device_map="auto",
        local_temperature=0.0,
        local_top_p=1.0,
        local_max_new_tokens=16,
        wandb=True,
        wandb_project="tacobelal",
        wandb_entity="team",
        wandb_run_name="omni_no_audio",
        wandb_log_every=25,
        hf_token="token",
    )
    assert command[:4] == ["uv", "run", "--with", "transformers>=4.57.0"]
    assert "src/utils/evaluate_mcq_order_qwen2_5_omni.py" in command
    assert "--disable-audio" in command
    assert "--attn-implementation" in command and "flash_attention_2" in command
    assert "--wandb" in command
    assert "--hf-token" in command and "token" in command
