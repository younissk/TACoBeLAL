"""Build a human-in-the-loop debug bundle for MCQ underperformance analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import typer


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


@dataclass(frozen=True)
class PairSummary:
    pair_name: str
    audio_model: str
    no_audio_model: str
    examples: int
    audio_accuracy: float
    no_audio_accuracy: float
    prediction_agreement: float
    both_correct: int
    audio_only_correct: int
    no_audio_only_correct: int
    both_wrong: int


def load_dataset_index(dataset_path: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(dataset_path):
        example_id = row.get("id")
        if isinstance(example_id, str):
            index[example_id] = row
    return index


def load_run_registry(results_root: Path) -> pd.DataFrame:
    runs_csv = results_root / "runs.csv"
    if not runs_csv.exists():
        raise FileNotFoundError(f"Run registry missing: {runs_csv}")

    runs = pd.read_csv(runs_csv)
    if runs.empty:
        raise ValueError("Run registry is empty.")

    rows: list[dict[str, Any]] = []
    for row in runs.to_dict(orient="records"):
        decisions_path = Path(str(row["decisions_path"]))
        run_dir = decisions_path.parent
        metrics_path = run_dir / "metrics.json"
        run_config_path = run_dir / "run_config.json"
        analysis_path = run_dir / "analysis.json"

        metrics = _read_json(metrics_path) if metrics_path.exists() else {}
        run_config = _read_json(run_config_path) if run_config_path.exists() else {}
        analysis = _read_json(analysis_path) if analysis_path.exists() else {}

        model_name = str(row.get("model_name", "unknown"))
        use_audio_cfg = run_config.get("use_audio")
        use_audio = bool(use_audio_cfg) if isinstance(use_audio_cfg, bool) else not model_name.endswith("-no-audio")

        rows.append(
            {
                "run_id": str(row.get("run_id", run_dir.name)),
                "task_id": str(row.get("task_id", "")),
                "model_name": model_name,
                "examples": int(row.get("examples", 0)),
                "correct": int(row.get("correct", 0)),
                "accuracy": float(row.get("accuracy", 0.0)),
                "elapsed_seconds": float(row.get("elapsed_seconds", 0.0)),
                "average_latency_ms": float(row.get("average_latency_ms", 0.0)),
                "started_at_utc": str(row.get("started_at_utc", "")),
                "finished_at_utc": str(row.get("finished_at_utc", "")),
                "dataset_path": str(row.get("dataset_path", "")),
                "decisions_path": str(decisions_path),
                "run_dir": str(run_dir),
                "metrics_path": str(metrics_path),
                "run_config_path": str(run_config_path),
                "analysis_path": str(analysis_path),
                "parse_invalid": int(metrics.get("parse_invalid", 0) or 0),
                "missing_predictions": int(metrics.get("missing_predictions", 0) or 0),
                "prediction_entropy": float(analysis.get("prediction_entropy", 0.0) or 0.0),
                "answer_entropy": float(analysis.get("answer_entropy", 0.0) or 0.0),
                "use_audio": use_audio,
            }
        )

    run_df = pd.DataFrame(rows)
    run_df["started_at_utc"] = pd.to_datetime(run_df["started_at_utc"], utc=True, errors="coerce")
    run_df["finished_at_utc"] = pd.to_datetime(run_df["finished_at_utc"], utc=True, errors="coerce")
    run_df = run_df.sort_values(["model_name", "finished_at_utc", "run_id"], ascending=[True, False, False])
    return run_df.reset_index(drop=True)


def select_latest_runs(run_df: pd.DataFrame, *, task_id: str = "MCQ-ORDER") -> pd.DataFrame:
    scoped = run_df[run_df["task_id"] == task_id].copy()
    if scoped.empty:
        return scoped
    latest = scoped.sort_values(["model_name", "finished_at_utc"], ascending=[True, False]).groupby(
        "model_name", as_index=False
    ).head(1)
    return latest.sort_values(["model_name"]).reset_index(drop=True)


def load_decisions_for_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    all_rows: list[dict[str, Any]] = []
    for row in run_df.to_dict(orient="records"):
        decisions_path = Path(str(row["decisions_path"]))
        for decision in _read_jsonl(decisions_path):
            decision["run_id"] = row["run_id"]
            decision["model_name"] = row["model_name"]
            decision["use_audio"] = row["use_audio"]
            if "parse_status" not in decision:
                decision["parse_status"] = "n/a"
            all_rows.append(decision)
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    if "is_correct" in df.columns:
        df["is_correct"] = df["is_correct"].astype(bool)
    if "n_options" in df.columns:
        df["n_options"] = pd.to_numeric(df["n_options"], errors="coerce").fillna(0).astype(int)
    return df


def summarize_pair(decisions_df: pd.DataFrame, *, audio_model: str, no_audio_model: str) -> PairSummary | None:
    left = decisions_df[decisions_df["model_name"] == audio_model].copy()
    right = decisions_df[decisions_df["model_name"] == no_audio_model].copy()
    if left.empty or right.empty:
        return None

    merged = left.merge(
        right,
        on="example_id",
        suffixes=("_audio", "_noaudio"),
    )
    if merged.empty:
        return None

    both_correct = int((merged["is_correct_audio"] & merged["is_correct_noaudio"]).sum())
    audio_only_correct = int((merged["is_correct_audio"] & ~merged["is_correct_noaudio"]).sum())
    no_audio_only_correct = int((~merged["is_correct_audio"] & merged["is_correct_noaudio"]).sum())
    both_wrong = int((~merged["is_correct_audio"] & ~merged["is_correct_noaudio"]).sum())

    total = len(merged)
    return PairSummary(
        pair_name=f"{audio_model}__vs__{no_audio_model}",
        audio_model=audio_model,
        no_audio_model=no_audio_model,
        examples=total,
        audio_accuracy=_safe_rate(int(merged["is_correct_audio"].sum()), total),
        no_audio_accuracy=_safe_rate(int(merged["is_correct_noaudio"].sum()), total),
        prediction_agreement=_safe_rate(
            int((merged["predicted_label_audio"] == merged["predicted_label_noaudio"]).sum()), total
        ),
        both_correct=both_correct,
        audio_only_correct=audio_only_correct,
        no_audio_only_correct=no_audio_only_correct,
        both_wrong=both_wrong,
    )


def _plot_accuracy_by_model(latest_runs: pd.DataFrame, out_path: Path) -> None:
    plot_df = latest_runs.sort_values("accuracy", ascending=False)
    colors = ["#2E7D32" if use_audio else "#546E7A" for use_audio in plot_df["use_audio"]]

    plt.figure(figsize=(11, 5))
    plt.bar(plot_df["model_name"], plot_df["accuracy"], color=colors)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Latest Run Accuracy by Model")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_invalid_missing_rates(latest_runs: pd.DataFrame, out_path: Path) -> None:
    plot_df = latest_runs.copy()
    plot_df["invalid_rate"] = [
        _safe_rate(i, n) for i, n in zip(plot_df["parse_invalid"].tolist(), plot_df["examples"].tolist())
    ]
    plot_df["missing_rate"] = [
        _safe_rate(i, n) for i, n in zip(plot_df["missing_predictions"].tolist(), plot_df["examples"].tolist())
    ]
    plot_df = plot_df.sort_values("invalid_rate", ascending=False)

    x = list(range(len(plot_df)))
    plt.figure(figsize=(11, 5))
    plt.bar(x, plot_df["invalid_rate"], label="parse_invalid_rate", color="#C62828")
    plt.bar(x, plot_df["missing_rate"], bottom=plot_df["invalid_rate"], label="missing_rate", color="#EF6C00")
    plt.xticks(x, plot_df["model_name"], rotation=30, ha="right")
    plt.ylabel("Rate")
    plt.ylim(0, 1)
    plt.title("Invalid + Missing Prediction Rates")
    plt.legend()
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_accuracy_by_option_count(decisions_df: pd.DataFrame, out_path: Path) -> None:
    if decisions_df.empty:
        return
    grouped = (
        decisions_df.groupby(["model_name", "n_options"], as_index=False)["is_correct"]
        .mean()
        .rename(columns={"is_correct": "accuracy"})
    )
    pivot = grouped.pivot(index="model_name", columns="n_options", values="accuracy").fillna(0.0)

    plt.figure(figsize=(12, max(4, 0.5 * len(pivot.index))))
    plt.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(0.4, float(pivot.values.max())))
    plt.colorbar(label="Accuracy")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.xlabel("Number of options")
    plt.ylabel("Model")
    plt.title("Accuracy by Option Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_audio_ablation_summaries(pair_rows: list[PairSummary], out_path: Path) -> None:
    if not pair_rows:
        return
    labels = [row.pair_name for row in pair_rows]
    audio_acc = [row.audio_accuracy for row in pair_rows]
    noaudio_acc = [row.no_audio_accuracy for row in pair_rows]

    x = list(range(len(labels)))
    width = 0.35
    plt.figure(figsize=(12, 5))
    plt.bar([i - width / 2 for i in x], audio_acc, width=width, label="audio")
    plt.bar([i + width / 2 for i in x], noaudio_acc, width=width, label="no-audio")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Audio vs No-Audio Accuracy (Matched Example IDs)")
    plt.legend()
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_prediction_bias(decisions_df: pd.DataFrame, out_path: Path) -> None:
    if decisions_df.empty:
        return

    labels = sorted(set(decisions_df["answer_label"].unique()) | set(decisions_df["predicted_label"].unique()))
    rows: list[list[float]] = []
    models = sorted(decisions_df["model_name"].unique())
    for model_name in models:
        subset = decisions_df[decisions_df["model_name"] == model_name]
        total = len(subset)
        answer_counts = subset["answer_label"].value_counts().to_dict()
        pred_counts = subset["predicted_label"].value_counts().to_dict()
        row: list[float] = []
        for label in labels:
            pred_rate = _safe_rate(int(pred_counts.get(label, 0)), total)
            answer_rate = _safe_rate(int(answer_counts.get(label, 0)), total)
            row.append(pred_rate - answer_rate)
        rows.append(row)

    plt.figure(figsize=(max(8, 0.5 * len(labels)), max(4, 0.5 * len(models))))
    plt.imshow(rows, aspect="auto", cmap="coolwarm", vmin=-0.25, vmax=0.25)
    plt.colorbar(label="prediction_rate - answer_rate")
    plt.yticks(range(len(models)), models)
    plt.xticks(range(len(labels)), labels)
    plt.xlabel("Label")
    plt.ylabel("Model")
    plt.title("Prediction Bias Heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_accuracy_by_answer_type(decisions_df: pd.DataFrame, out_path: Path) -> None:
    if decisions_df.empty or "answer_type" not in decisions_df.columns:
        return

    grouped = (
        decisions_df.groupby(["model_name", "answer_type"], as_index=False)["is_correct"]
        .mean()
        .rename(columns={"is_correct": "accuracy"})
    )
    if grouped.empty:
        return

    pivot = grouped.pivot(index="model_name", columns="answer_type", values="accuracy").fillna(0.0)
    ordered_cols = [col for col in ("event", "none") if col in pivot.columns] + [
        col for col in pivot.columns if col not in {"event", "none"}
    ]
    pivot = pivot[ordered_cols]

    plt.figure(figsize=(8, max(3.5, 0.5 * len(pivot.index))))
    plt.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(0.4, float(pivot.values.max())))
    plt.colorbar(label="Accuracy")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.xlabel("Answer type")
    plt.ylabel("Model")
    plt.title("Accuracy by Answer Type")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _build_model_baselines(decisions_df: pd.DataFrame) -> pd.DataFrame:
    if decisions_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for model_name, subset in decisions_df.groupby("model_name"):
        total = len(subset)
        accuracy = float(subset["is_correct"].mean()) if total else 0.0
        expected_random = float((1.0 / subset["n_options"]).mean()) if total else 0.0

        answer_counts = subset["answer_label"].value_counts()
        majority_label = str(answer_counts.index[0]) if not answer_counts.empty else ""
        majority_label_accuracy = _safe_rate(int(answer_counts.iloc[0]) if not answer_counts.empty else 0, total)

        always_none_accuracy = float((subset["answer_type"] == "none").mean()) if "answer_type" in subset else 0.0

        by_n = subset.groupby("n_options")["answer_label"].value_counts().rename("count").reset_index()
        majority_per_n_correct = int(by_n.groupby("n_options")["count"].max().sum()) if not by_n.empty else 0
        majority_per_n_accuracy = _safe_rate(majority_per_n_correct, total)

        true_none_rate = float((subset["answer_type"] == "none").mean()) if "answer_type" in subset else 0.0
        pred_none_rate = float((subset["predicted_type"] == "none").mean()) if "predicted_type" in subset else 0.0

        none_subset = subset[subset["answer_type"] == "none"] if "answer_type" in subset else subset.iloc[0:0]
        event_subset = subset[subset["answer_type"] == "event"] if "answer_type" in subset else subset.iloc[0:0]
        none_accuracy = float(none_subset["is_correct"].mean()) if not none_subset.empty else 0.0
        event_accuracy = float(event_subset["is_correct"].mean()) if not event_subset.empty else 0.0

        rows.append(
            {
                "model_name": model_name,
                "examples": total,
                "accuracy": accuracy,
                "expected_random": expected_random,
                "majority_label": majority_label,
                "majority_label_accuracy": majority_label_accuracy,
                "majority_per_n_accuracy": majority_per_n_accuracy,
                "always_none_accuracy": always_none_accuracy,
                "true_none_rate": true_none_rate,
                "pred_none_rate": pred_none_rate,
                "none_accuracy": none_accuracy,
                "event_accuracy": event_accuracy,
            }
        )

    return pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)


def _plot_model_vs_baselines(model_baselines_df: pd.DataFrame, out_path: Path) -> None:
    if model_baselines_df.empty:
        return

    plot_df = model_baselines_df.sort_values("accuracy", ascending=False)
    x = list(range(len(plot_df)))
    width = 0.16

    plt.figure(figsize=(12, 5.5))
    plt.bar([i - 1.5 * width for i in x], plot_df["accuracy"], width=width, label="model_accuracy", color="#2E7D32")
    plt.bar([i - 0.5 * width for i in x], plot_df["expected_random"], width=width, label="expected_random", color="#1565C0")
    plt.bar(
        [i + 0.5 * width for i in x],
        plot_df["majority_label_accuracy"],
        width=width,
        label="majority_label",
        color="#6A1B9A",
    )
    plt.bar(
        [i + 1.5 * width for i in x],
        plot_df["always_none_accuracy"],
        width=width,
        label="always_none",
        color="#EF6C00",
    )
    plt.xticks(x, plot_df["model_name"], rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy vs Simple Baselines")
    plt.legend()
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_none_rate_calibration(model_baselines_df: pd.DataFrame, out_path: Path) -> None:
    if model_baselines_df.empty:
        return

    plot_df = model_baselines_df.sort_values("model_name")
    x = list(range(len(plot_df)))
    width = 0.35

    plt.figure(figsize=(11, 5))
    plt.bar([i - width / 2 for i in x], plot_df["true_none_rate"], width=width, label="true_none_rate", color="#455A64")
    plt.bar([i + width / 2 for i in x], plot_df["pred_none_rate"], width=width, label="pred_none_rate", color="#C62828")
    plt.xticks(x, plot_df["model_name"], rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Rate")
    plt.title("None-Option Calibration (Ground Truth vs Predicted)")
    plt.legend()
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _build_example_slices(
    decisions_df: pd.DataFrame,
    dataset_index: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    if decisions_df.empty:
        return pd.DataFrame()

    example_slice_rows: list[dict[str, Any]] = []
    unique_ids = {str(value) for value in decisions_df["example_id"].tolist()}
    for example_id in unique_ids:
        dataset_row = dataset_index.get(example_id)
        if not isinstance(dataset_row, dict):
            continue
        options = dataset_row.get("options")
        if not isinstance(options, list):
            continue

        base_event = dataset_row.get("base_event")
        base_onset = None
        if isinstance(base_event, dict):
            onset_value = base_event.get("onset")
            if isinstance(onset_value, (int, float)):
                base_onset = float(onset_value)

        none_label = None
        none_position = None
        concurrent_at_base_onset = False
        for idx, option in enumerate(options):
            if not isinstance(option, dict):
                continue
            option_type = option.get("type")
            if option_type == "none":
                label_value = option.get("label")
                if isinstance(label_value, str):
                    none_label = label_value
                    none_position = idx
            if base_onset is not None and option_type == "event":
                onset = option.get("onset")
                offset = option.get("offset")
                if isinstance(onset, (int, float)) and isinstance(offset, (int, float)):
                    if float(onset) < base_onset < float(offset):
                        concurrent_at_base_onset = True

        example_slice_rows.append(
            {
                "example_id": example_id,
                "answer_type_dataset": str(dataset_row.get("answer_type", "")),
                "n_options_dataset": len(options),
                "none_label": none_label or "",
                "none_position": -1 if none_position is None else int(none_position),
                "concurrent_at_base_onset": bool(concurrent_at_base_onset),
            }
        )

    if not example_slice_rows:
        return pd.DataFrame()

    slice_df = pd.DataFrame(example_slice_rows)
    merged = decisions_df.merge(slice_df, on="example_id", how="left")
    return merged


def _build_answer_type_diagnostics(decisions_with_slices: pd.DataFrame) -> pd.DataFrame:
    if decisions_with_slices.empty or "answer_type" not in decisions_with_slices.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    grouped = decisions_with_slices.groupby(["model_name", "answer_type"], as_index=False)
    for row in grouped:
        (model_name, answer_type), subset = row
        total = len(subset)
        rows.append(
            {
                "model_name": model_name,
                "answer_type": answer_type,
                "examples": total,
                "accuracy": float(subset["is_correct"].mean()) if total else 0.0,
                "pred_none_rate": float((subset["predicted_type"] == "none").mean())
                if "predicted_type" in subset
                else 0.0,
                "pred_event_rate": float((subset["predicted_type"] == "event").mean())
                if "predicted_type" in subset
                else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["model_name", "answer_type"]).reset_index(drop=True)


def _build_audioflamingo_id_integrity(latest_runs: pd.DataFrame) -> pd.DataFrame:
    if latest_runs.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for run in latest_runs.to_dict(orient="records"):
        model_name = str(run.get("model_name", ""))
        if not model_name.startswith("audio-flamingo-3"):
            continue

        run_dir = Path(str(run.get("run_dir", "")))
        mapping_path = run_dir / "workdir" / "audioflamingo_mapping.json"
        raw_outputs_path = run_dir / "raw_model_outputs.jsonl"
        fallback_outputs_path = run_dir / "audioflamingo_outputs" / "outputs.jsonl"
        if not raw_outputs_path.exists() and fallback_outputs_path.exists():
            raw_outputs_path = fallback_outputs_path

        if not mapping_path.exists() or not raw_outputs_path.exists():
            rows.append(
                {
                    "model_name": model_name,
                    "run_id": str(run.get("run_id", "")),
                    "mapping_found": mapping_path.exists(),
                    "raw_outputs_found": raw_outputs_path.exists(),
                    "mapping_size": 0,
                    "outputs_size": 0,
                    "direct_id_match_rate": 0.0,
                    "basename_match_rate": 0.0,
                }
            )
            continue

        mapping_payload = _read_json(mapping_path)
        mapping_keys = list(mapping_payload.keys())
        mapping_key_set = set(mapping_keys)
        mapping_basename_set = {Path(key).name for key in mapping_keys}

        output_rows = _read_jsonl(raw_outputs_path)
        output_ids = [str(row.get("id", "")) for row in output_rows]
        outputs_total = len(output_ids)
        if outputs_total == 0:
            direct_rate = 0.0
            basename_rate = 0.0
        else:
            direct_matches = sum(1 for output_id in output_ids if output_id in mapping_key_set)
            basename_matches = sum(1 for output_id in output_ids if Path(output_id).name in mapping_basename_set)
            direct_rate = _safe_rate(direct_matches, outputs_total)
            basename_rate = _safe_rate(basename_matches, outputs_total)

        rows.append(
            {
                "model_name": model_name,
                "run_id": str(run.get("run_id", "")),
                "mapping_found": True,
                "raw_outputs_found": True,
                "mapping_size": len(mapping_keys),
                "outputs_size": outputs_total,
                "direct_id_match_rate": direct_rate,
                "basename_match_rate": basename_rate,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["model_name"]).reset_index(drop=True)


def build_review_queue(
    decisions_df: pd.DataFrame,
    dataset_index: dict[str, dict[str, Any]],
    pair_rows: list[PairSummary],
) -> pd.DataFrame:
    queue_rows: list[dict[str, Any]] = []
    for pair in pair_rows:
        left = decisions_df[decisions_df["model_name"] == pair.audio_model].copy()
        right = decisions_df[decisions_df["model_name"] == pair.no_audio_model].copy()
        merged = left.merge(right, on="example_id", suffixes=("_audio", "_noaudio"))
        if merged.empty:
            continue

        for row in merged.to_dict(orient="records"):
            audio_correct = bool(row["is_correct_audio"])
            noaudio_correct = bool(row["is_correct_noaudio"])
            if audio_correct and noaudio_correct:
                category = "both_correct"
                priority = 0
            elif audio_correct and not noaudio_correct:
                category = "audio_only_correct"
                priority = 3
            elif not audio_correct and noaudio_correct:
                category = "noaudio_only_correct"
                priority = 3
            else:
                labels_equal = row["predicted_label_audio"] == row["predicted_label_noaudio"]
                category = "both_wrong_same_pred" if labels_equal else "both_wrong_diff_pred"
                priority = 2 if not labels_equal else 1

            dataset_row = dataset_index.get(str(row["example_id"]), {})
            options = dataset_row.get("options")
            options_text = ""
            if isinstance(options, list):
                parts: list[str] = []
                for option in options:
                    if isinstance(option, dict):
                        label = str(option.get("label", "")).strip()
                        text = str(option.get("text", "")).strip()
                        if label and text:
                            parts.append(f"{label}. {text}")
                options_text = " | ".join(parts)

            queue_rows.append(
                {
                    "pair_name": pair.pair_name,
                    "priority": priority,
                    "category": category,
                    "example_id": row["example_id"],
                    "audio_filename": row.get("audio_filename_audio", ""),
                    "question": row.get("question_audio", ""),
                    "options_text": options_text,
                    "answer_label": row.get("answer_label_audio", ""),
                    "audio_prediction": row.get("predicted_label_audio", ""),
                    "noaudio_prediction": row.get("predicted_label_noaudio", ""),
                    "audio_parse_status": row.get("parse_status_audio", ""),
                    "noaudio_parse_status": row.get("parse_status_noaudio", ""),
                    "audio_raw_prediction": row.get("raw_prediction_audio", ""),
                    "noaudio_raw_prediction": row.get("raw_prediction_noaudio", ""),
                    "n_options": row.get("n_options_audio", 0),
                    "audio_is_correct": audio_correct,
                    "noaudio_is_correct": noaudio_correct,
                }
            )

    if not queue_rows:
        return pd.DataFrame()
    queue_df = pd.DataFrame(queue_rows)
    queue_df = queue_df.sort_values(
        ["priority", "pair_name", "category", "n_options"],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)
    return queue_df


def build_debug_bundle(
    *,
    results_root: Path,
    dataset_path: Path,
    output_dir: Path,
    top_k_review: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    run_df = load_run_registry(results_root)
    latest_runs = select_latest_runs(run_df)
    decisions_df = load_decisions_for_runs(latest_runs)
    dataset_index = load_dataset_index(dataset_path)

    pair_rows: list[PairSummary] = []
    latest_models = set(latest_runs["model_name"].tolist())
    for model_name in sorted(latest_models):
        if model_name.endswith("-no-audio"):
            base_model = model_name[: -len("-no-audio")]
            if base_model in latest_models:
                pair = summarize_pair(decisions_df, audio_model=base_model, no_audio_model=model_name)
                if pair is not None:
                    pair_rows.append(pair)

    run_df.to_csv(output_dir / "run_registry_enriched.csv", index=False)
    latest_runs.to_csv(output_dir / "latest_runs.csv", index=False)
    if not decisions_df.empty:
        decisions_df.to_csv(output_dir / "latest_decisions_long.csv", index=False)

    pair_df = pd.DataFrame([row.__dict__ for row in pair_rows])
    if not pair_df.empty:
        pair_df.to_csv(output_dir / "audio_ablation_pairs.csv", index=False)

    decisions_with_slices = _build_example_slices(decisions_df, dataset_index)
    if not decisions_with_slices.empty:
        decisions_with_slices.to_csv(output_dir / "latest_decisions_with_slices.csv", index=False)

    model_baselines_df = _build_model_baselines(decisions_df)
    if not model_baselines_df.empty:
        model_baselines_df.to_csv(output_dir / "model_baselines.csv", index=False)

    answer_type_diag_df = _build_answer_type_diagnostics(decisions_with_slices)
    if not answer_type_diag_df.empty:
        answer_type_diag_df.to_csv(output_dir / "answer_type_diagnostics.csv", index=False)

    audioflamingo_id_integrity_df = _build_audioflamingo_id_integrity(latest_runs)
    if not audioflamingo_id_integrity_df.empty:
        audioflamingo_id_integrity_df.to_csv(output_dir / "audioflamingo_id_integrity.csv", index=False)

    review_df = build_review_queue(decisions_df, dataset_index, pair_rows)
    if not review_df.empty:
        review_df.to_csv(output_dir / "human_review_queue.csv", index=False)
        review_df.head(top_k_review).to_csv(output_dir / "human_review_queue_topk.csv", index=False)

    _plot_accuracy_by_model(latest_runs, output_dir / "plot_accuracy_by_model.png")
    _plot_invalid_missing_rates(latest_runs, output_dir / "plot_invalid_missing_rates.png")
    _plot_accuracy_by_option_count(decisions_df, output_dir / "plot_accuracy_by_option_count.png")
    _plot_prediction_bias(decisions_df, output_dir / "plot_prediction_bias.png")
    _plot_accuracy_by_answer_type(decisions_df, output_dir / "plot_accuracy_by_answer_type.png")
    _plot_model_vs_baselines(model_baselines_df, output_dir / "plot_model_vs_baselines.png")
    _plot_none_rate_calibration(model_baselines_df, output_dir / "plot_none_calibration.png")
    _plot_audio_ablation_summaries(pair_rows, output_dir / "plot_audio_ablation_accuracy.png")

    random_runs = latest_runs[latest_runs["model_name"] == "random"]
    random_acc = float(random_runs["accuracy"].iloc[0]) if not random_runs.empty else None

    summary = {
        "results_root": str(results_root),
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "latest_runs_count": int(len(latest_runs)),
        "latest_models": sorted(latest_models),
        "random_accuracy": random_acc,
        "audio_ablation_pairs": [row.__dict__ for row in pair_rows],
        "review_queue_rows": int(len(review_df)),
        "models_below_expected_random": (
            model_baselines_df[model_baselines_df["accuracy"] < model_baselines_df["expected_random"]]["model_name"]
            .astype(str)
            .tolist()
            if not model_baselines_df.empty
            else []
        ),
    }
    with open(output_dir / "debug_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return summary


def main(
    results_root: Path = typer.Option(
        Path("results/mcq-order"),
        "--results-root",
        path_type=Path,
        help="Path to mcq-order results root (contains runs.csv and run folders).",
    ),
    dataset_path: Path = typer.Option(
        Path("data/mcq_event_timeline_strong.jsonl"),
        "--dataset",
        path_type=Path,
        exists=True,
        dir_okay=False,
        help="Dataset JSONL used for MCQ-ORDER runs.",
    ),
    output_dir: Path = typer.Option(
        Path("results/mcq-order/debug_bundle"),
        "--output-dir",
        path_type=Path,
        help="Destination directory for debug CSVs/plots.",
    ),
    top_k_review: int = typer.Option(
        200,
        "--top-k-review",
        min=1,
        help="Number of highest-priority rows to emit as human_review_queue_topk.csv.",
    ),
) -> None:
    summary = build_debug_bundle(
        results_root=results_root,
        dataset_path=dataset_path,
        output_dir=output_dir,
        top_k_review=top_k_review,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    typer.run(main)
