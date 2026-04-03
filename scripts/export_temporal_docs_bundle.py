"""Build a tiny GH Pages bundle for trivial temporal benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


BENCHMARKS = [
    {
        "key": "mcq-synth-pitch-order-trivial",
        "label": "Pitch Order",
        "dataset": "data/mcq_synth_pitch_order_trivial_easy.jsonl",
        "runs": "results/mcq-synth-pitch-order-trivial/mcq-order/runs.csv",
    },
    {
        "key": "mcq-synth-loudness-order-trivial",
        "label": "Loudness Order",
        "dataset": "data/mcq_synth_loudness_order_trivial_easy.jsonl",
        "runs": "results/mcq-synth-loudness-order-trivial/mcq-order/runs.csv",
    },
    {
        "key": "mcq-synth-duration-order-trivial",
        "label": "Duration Order",
        "dataset": "data/mcq_synth_duration_order_trivial_easy.jsonl",
        "runs": "results/mcq-synth-duration-order-trivial/mcq-order/runs.csv",
    },
    {
        "key": "mcq-synth-count-beeps-trivial",
        "label": "Count Beeps",
        "dataset": "data/mcq_synth_count_beeps_trivial_easy.jsonl",
        "runs": "results/mcq-synth-count-beeps-trivial/mcq-order/runs.csv",
    },
    {
        "key": "mcq-synth-gap-trivial",
        "label": "Gap Length",
        "dataset": "data/mcq_synth_gap_trivial_easy.jsonl",
        "runs": "results/mcq-synth-gap-trivial/mcq-order/runs.csv",
    },
    {
        "key": "mcq-synth-pattern-pitch-trivial",
        "label": "Pitch Pattern",
        "dataset": "data/mcq_synth_pattern_pitch_trivial_easy.jsonl",
        "runs": "results/mcq-synth-pattern-pitch-trivial/mcq-order/runs.csv",
    },
    {
        "key": "mcq-synth-dog-car-order-trivial",
        "label": "Dog vs Car",
        "dataset": "data/mcq_synth_dog_car_order_trivial_easy.jsonl",
        "runs": "results/mcq-synth-dog-car-order-trivial/mcq-order/runs.csv",
    },
]

MODEL_KEYS = [
    "random",
    "llm-qwen",
    "qwen2-audio-7b-instruct",
    "audio-flamingo-3",
]


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_latest_runs(path: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            model = row["model_name"]
            if model not in MODEL_KEYS:
                continue
            if model not in latest or latest[model]["started_at_utc"] < row["started_at_utc"]:
                latest[model] = row
    return latest


def load_decisions(decisions_path: Path) -> dict[str, dict]:
    decisions: dict[str, dict] = {}
    with decisions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            decisions[row["example_id"]] = row
    return decisions


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_audio(source_root: Path, target_root: Path, audio_rel: str) -> None:
    source = source_root / audio_rel
    target = target_root / audio_rel
    ensure_parent(target)
    shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=5, help="Examples per benchmark.")
    parser.add_argument("--output", type=Path, default=Path("docs/data/temporal_bundle.json"))
    parser.add_argument("--audio-target", type=Path, default=Path("docs/audio"))
    args = parser.parse_args()

    bundle: dict[str, list] = {"benchmarks": []}
    source_audio_root = Path("data/audio")

    for bench in BENCHMARKS:
        dataset = load_jsonl(Path(bench["dataset"]))
        runs = load_latest_runs(Path(bench["runs"]))
        decisions_by_model = {
            model: load_decisions(Path(runs[model]["decisions_path"]))
            for model in runs
            if Path(runs[model]["decisions_path"]).exists()
        }

        examples: list[dict] = []
        for row in dataset[: args.examples]:
            audio_rel = row["audio_filename"]
            copy_audio(source_audio_root, args.audio_target, audio_rel)
            model_answers = {}
            for model in MODEL_KEYS:
                decision = decisions_by_model.get(model, {}).get(row["id"])
                if decision:
                    model_answers[model] = {
                        "predicted_text": decision.get("predicted_text"),
                        "is_correct": decision.get("is_correct"),
                    }
            examples.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "answer_text": row["answer_text"],
                    "audio_filename": audio_rel,
                    "model_answers": model_answers,
                }
            )

        bundle["benchmarks"].append(
            {
                "key": bench["key"],
                "label": bench["label"],
                "examples": examples,
            }
        )

    ensure_parent(args.output)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(bundle, handle, indent=2)


if __name__ == "__main__":
    main()
