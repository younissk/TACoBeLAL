"""Tests for MCQ-ORDER model baseline and evaluator."""

from __future__ import annotations

import json
from pathlib import Path

from utils.evaluate_mcq_order import load_examples, main
from utils.mcq_order_models import (
    MCQOrderExample,
    OpenAILangChainMCQOrderModel,
    RandomMCQOrderModel,
    TASK_ID_MCQ_ORDER,
)


def _example_payload(*, example_id: str, answer_label: str = "B") -> dict[str, object]:
    return {
        "id": example_id,
        "audio_filename": "a.mp3",
        "question": "What happens immediately after X?",
        "options": [
            {"label": "A", "text": "Option A", "type": "event", "event_index": 0},
            {"label": "B", "text": "Option B", "type": "event", "event_index": 1},
            {"label": "C", "text": "This is the last event, no immediate event after.", "type": "none"},
        ],
        "answer_label": answer_label,
        "answer_text": "Option B" if answer_label == "B" else "Option A",
    }


def test_random_model_is_deterministic() -> None:
    example = MCQOrderExample.from_json(_example_payload(example_id="ex-1"))

    model_a = RandomMCQOrderModel(seed=123)
    model_b = RandomMCQOrderModel(seed=123)

    outputs_a = [model_a.predict(example) for _ in range(8)]
    outputs_b = [model_b.predict(example) for _ in range(8)]

    assert outputs_a == outputs_b
    assert all(label in {"A", "B", "C"} for label in outputs_a)


def test_openai_output_label_extraction() -> None:
    valid_labels = {"A", "B", "C"}
    assert OpenAILangChainMCQOrderModel._extract_label("A", valid_labels) == "A"
    assert OpenAILangChainMCQOrderModel._extract_label('{"label":"B"}', valid_labels) == "B"
    assert OpenAILangChainMCQOrderModel._extract_label("I choose C.", valid_labels) == "C"
    assert OpenAILangChainMCQOrderModel._extract_label("Option Z", valid_labels) is None


def test_load_examples_reads_valid_jsonl(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    rows = [
        _example_payload(example_id="ex-1", answer_label="B"),
        _example_payload(example_id="ex-2", answer_label="A"),
    ]
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    examples = load_examples(dataset)
    assert len(examples) == 2
    assert examples[0].task_id == TASK_ID_MCQ_ORDER
    assert examples[0].example_id == "ex-1"


def test_main_writes_results_artifacts(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    rows = [
        _example_payload(example_id="ex-1", answer_label="B"),
        _example_payload(example_id="ex-2", answer_label="A"),
    ]
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    results_root = tmp_path / "results"
    main(
        dataset=dataset,
        model="random",
        seed=7,
        results_root=results_root,
        limit=None,
    )

    run_dirs = sorted((results_root / "mcq-order" / "random").glob("*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    decisions_path = run_dir / "decisions.jsonl"
    metrics_path = run_dir / "metrics.json"
    table_path = run_dir / "results_table.md"
    runs_csv_path = results_root / "mcq-order" / "runs.csv"

    assert decisions_path.exists()
    assert metrics_path.exists()
    assert table_path.exists()
    assert runs_csv_path.exists()

    with open(decisions_path, "r", encoding="utf-8") as f:
        decisions = [json.loads(line) for line in f if line.strip()]
    assert len(decisions) == 2
    assert all(row["task_id"] == TASK_ID_MCQ_ORDER for row in decisions)
    assert all("is_correct" in row for row in decisions)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["task_id"] == TASK_ID_MCQ_ORDER
    assert metrics["model_name"] == "random"
    assert metrics["examples"] == 2
    assert "model_metadata" in metrics

    table_content = table_path.read_text(encoding="utf-8")
    assert "MCQ-ORDER" in table_content
    assert "random" in table_content
