"""Manual Good/Bad reviewer for MCQ-ORDER examples with audio playback."""

from __future__ import annotations

import csv
import html
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import typer

app = typer.Typer(add_completion=False)

CUSTOM_CSS = """
.app-shell { max-width: 1200px; margin: 0 auto; }
.card {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 12px;
}
.stats-line {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
}
.stat-pill {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  padding: 8px 10px;
  text-align: center;
}
.stat-label { font-size: 12px; color: #475569; }
.stat-value { font-size: 18px; font-weight: 600; color: #0f172a; }
.compact-title h1, .compact-title h2, .compact-title h3, .compact-title p { margin: 0; }
.question-title { font-weight: 700; margin-bottom: 8px; font-size: 20px; }
.question-text { margin-bottom: 10px; font-size: 24px; line-height: 1.35; }
.options-list { display: flex; flex-direction: column; gap: 8px; }
.option-row {
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  padding: 8px 10px;
  background: #f8fafc;
  font-size: 20px;
  line-height: 1.35;
}
.option-correct {
  border-color: #16a34a;
  background: #f0fdf4;
  color: #166534;
  font-weight: 700;
}
.pill-wrap { margin-left: 8px; display: inline-flex; gap: 6px; flex-wrap: wrap; vertical-align: middle; }
.model-pill {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
  line-height: 1.2;
}
.pill-af3 { background: #ffedd5; color: #c2410c; border: 1px solid #fdba74; }
.pill-qwen { background: #f3e8ff; color: #7e22ce; border: 1px solid #d8b4fe; }
.pill-other { background: #f1f5f9; color: #334155; border: 1px solid #cbd5e1; }
.timeline-wrap { display: flex; flex-direction: column; gap: 10px; }
.timeline-title { font-weight: 700; font-size: 16px; }
.timeline-row { display: grid; grid-template-columns: 170px 1fr; gap: 10px; align-items: center; }
.timeline-meta { font-size: 12px; color: #334155; line-height: 1.25; }
.timeline-track { position: relative; height: 18px; border-radius: 999px; background: #e2e8f0; overflow: hidden; }
.timeline-seg {
  position: absolute;
  top: 2px;
  height: 14px;
  border-radius: 999px;
  background: #60a5fa;
}
.timeline-seg-base { background: #16a34a; }
.timeline-label { font-size: 12px; margin-top: 2px; color: #0f172a; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
"""

CUSTOM_JS = """
() => {
  if (window.__mcqReviewerHotkeysBound) return;
  window.__mcqReviewerHotkeysBound = true;
  document.addEventListener("keydown", (event) => {
    if (event.defaultPrevented) return;
    const active = document.activeElement;
    if (active) {
      const tag = (active.tagName || "").toLowerCase();
      const isTyping = tag === "input" || tag === "textarea" || tag === "select" || active.isContentEditable;
      if (isTyping) return;
    }
    const key = (event.key || "").toLowerCase();
    let target = null;
    if (key === "g") {
      target = document.querySelector("#good-btn button") || document.querySelector("button#good-btn");
    } else if (key === "b") {
      target = document.querySelector("#bad-btn button") || document.querySelector("button#bad-btn");
    }
    if (!target) return;
    event.preventDefault();
    target.click();
  });
}
"""

EXCLUDED_MODEL_NAMES = {"random"}
EXCLUDED_MODEL_SUFFIXES = ("-no-audio",)


@dataclass(frozen=True)
class ReviewItem:
    """One MCQ example enriched for UI rendering."""

    example_id: str
    audio_filename: str
    audio_path: Path
    n_options: int
    payload: dict[str, Any]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _write_jsonl_line(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _normalize_path_string(path_like: str) -> str:
    return str(Path(path_like).as_posix())


def _resolve_maybe_relative(path_like: str, base_dir: Path) -> Path:
    candidate = Path(path_like)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve(strict=False)


def _same_dataset_path(dataset_path: Path, row_dataset: str, base_dir: Path) -> bool:
    target = dataset_path.resolve(strict=False)
    row_resolved = _resolve_maybe_relative(row_dataset, base_dir).resolve(strict=False)
    if row_resolved == target:
        return True
    return _normalize_path_string(row_dataset) == _normalize_path_string(str(dataset_path))


def _parse_timestamp(value: str, fallback: str) -> datetime:
    if value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    try:
        return datetime.strptime(fallback, "%Y%m%d_%H%M%S")
    except ValueError:
        return datetime.min


def _load_latest_runs_by_model(
    runs_csv: Path,
    *,
    dataset_path: Path,
    task_id: str = "MCQ-ORDER",
) -> dict[str, dict[str, str]]:
    if not runs_csv.exists():
        return {}

    repo_root = Path.cwd()
    latest: dict[str, tuple[datetime, dict[str, str]]] = {}
    with runs_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("task_id") or "").strip() != task_id:
                continue
            row_dataset = (row.get("dataset_path") or "").strip()
            if not row_dataset:
                continue
            if not _same_dataset_path(dataset_path=dataset_path, row_dataset=row_dataset, base_dir=repo_root):
                continue
            model_name = (row.get("model_name") or "").strip()
            run_id = (row.get("run_id") or "").strip()
            if not model_name or not run_id:
                continue
            model_key = model_name.lower()
            if model_key in EXCLUDED_MODEL_NAMES:
                continue
            if any(model_key.endswith(suffix) for suffix in EXCLUDED_MODEL_SUFFIXES):
                continue
            run_time = _parse_timestamp(row.get("finished_at_utc") or "", run_id)
            prev = latest.get(model_name)
            if prev is None or run_time >= prev[0]:
                latest[model_name] = (run_time, row)
    return {model: row for model, (_, row) in latest.items()}


def _load_model_outcomes_by_example(
    runs_csv: Path,
    *,
    dataset_path: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    latest_runs = _load_latest_runs_by_model(runs_csv=runs_csv, dataset_path=dataset_path)
    if not latest_runs:
        return {}

    repo_root = Path.cwd()
    outcomes: dict[str, dict[str, dict[str, Any]]] = {}
    for model_name, run_row in latest_runs.items():
        decisions_ref = (run_row.get("decisions_path") or "").strip()
        if not decisions_ref:
            continue
        decisions_path = _resolve_maybe_relative(decisions_ref, repo_root)
        if not decisions_path.exists():
            continue
        with decisions_path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                row = json.loads(text)
                example_id = str(row.get("example_id", "")).strip()
                if not example_id:
                    continue
                bucket = outcomes.setdefault(example_id, {})
                bucket[model_name] = {
                    "is_correct": bool(row.get("is_correct", False)),
                    "predicted_label": row.get("predicted_label"),
                    "predicted_text": row.get("predicted_text"),
                }
    return outcomes


def _load_existing_labels(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    labels: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            example_id = str(row.get("id", "")).strip()
            label = str(row.get("label", "")).strip().lower()
            if example_id and label in {"good", "bad"}:
                labels[example_id] = label
    return labels


class _SemanticOptionPruner:
    """Deduplicate near-duplicate event options using sentence embeddings."""

    def __init__(self, model_id: str, batch_size: int = 64) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id)
        self._model.eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        self._batch_size = max(1, batch_size)
        self._cache: dict[str, Any] = {}

    def _encode_batch(self, texts: list[str]) -> list[Any]:
        torch = self._torch
        tokenized = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokenized = {key: value.to(self._device) for key, value in tokenized.items()}
        with torch.no_grad():
            outputs = self._model(**tokenized)
            hidden = outputs.last_hidden_state
            mask = tokenized["attention_mask"].unsqueeze(-1)
            summed = (hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1)
            pooled = summed / denom
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return [row.detach().cpu() for row in pooled]

    def encode(self, texts: list[str]) -> list[Any]:
        missing = [text for text in texts if text not in self._cache]
        for start in range(0, len(missing), self._batch_size):
            batch = missing[start : start + self._batch_size]
            vectors = self._encode_batch(batch)
            for text, vec in zip(batch, vectors):
                self._cache[text] = vec
        return [self._cache[text] for text in texts]

    def cosine(self, left: Any, right: Any) -> float:
        return float((left * right).sum().item())


def _prune_similar_options_in_row(
    row: dict[str, Any],
    *,
    pruner: _SemanticOptionPruner,
    similarity_threshold: float,
) -> dict[str, Any]:
    options = row.get("options")
    if not isinstance(options, list) or len(options) < 2:
        return row

    answer_label = str(row.get("answer_label", "")).strip()
    answer_type = str(row.get("answer_type", "")).strip()

    event_positions: list[int] = []
    event_options: list[dict[str, Any]] = []
    for idx, option in enumerate(options):
        if not isinstance(option, dict):
            continue
        if str(option.get("type", "")).strip() != "event":
            continue
        event_positions.append(idx)
        event_options.append(option)

    if len(event_options) < 2:
        return row

    texts = [str(option.get("text", "")) for option in event_options]
    embeddings = pruner.encode(texts)
    n = len(event_options)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if pruner.cosine(embeddings[i], embeddings[j]) >= similarity_threshold:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for idx in range(n):
        clusters.setdefault(find(idx), []).append(idx)

    keep_event_positions: set[int] = set()
    for members in clusters.values():
        protected_idx: int | None = None
        for idx in members:
            if str(event_options[idx].get("label", "")).strip() == answer_label and answer_type == "event":
                protected_idx = idx
                break
        chosen_idx = protected_idx if protected_idx is not None else min(members)
        keep_event_positions.add(event_positions[chosen_idx])

    keep_none_position: int | None = None
    for idx, option in enumerate(options):
        if not isinstance(option, dict):
            continue
        if str(option.get("type", "")).strip() != "none":
            continue
        if answer_type == "none" and str(option.get("label", "")).strip() == answer_label:
            keep_none_position = idx
            break
        if keep_none_position is None:
            keep_none_position = idx

    kept_options: list[dict[str, Any]] = []
    for idx, option in enumerate(options):
        if not isinstance(option, dict):
            continue
        option_type = str(option.get("type", "")).strip()
        if option_type == "event":
            if idx in keep_event_positions:
                kept_options.append(option)
        elif option_type == "none":
            if keep_none_position is not None and idx == keep_none_position:
                kept_options.append(option)
        else:
            kept_options.append(option)

    kept_labels = {str(option.get("label", "")).strip() for option in kept_options if isinstance(option, dict)}
    if answer_label and answer_label not in kept_labels:
        return row

    updated = dict(row)
    updated["options"] = kept_options
    return updated


def _dedupe_dataset_options(
    dataset_rows: list[dict[str, Any]],
    *,
    semantic_dedupe: bool,
    model_id: str,
    similarity_threshold: float,
    batch_size: int,
) -> list[dict[str, Any]]:
    if not semantic_dedupe:
        return dataset_rows
    pruner = _SemanticOptionPruner(model_id=model_id, batch_size=batch_size)
    return [
        _prune_similar_options_in_row(
            row,
            pruner=pruner,
            similarity_threshold=similarity_threshold,
        )
        for row in dataset_rows
    ]


def _build_filtered_items(
    dataset_rows: list[dict[str, Any]],
    *,
    audio_root: Path,
    min_options: int,
    max_options: int,
) -> list[ReviewItem]:
    items: list[ReviewItem] = []
    for row in dataset_rows:
        example_id = str(row.get("id", "")).strip()
        audio_filename = str(row.get("audio_filename", "")).strip()
        options = row.get("options", [])
        if not isinstance(options, list):
            continue
        n_options = len(options)
        if n_options < min_options or n_options > max_options:
            continue
        if not example_id or not audio_filename:
            continue
        items.append(
            ReviewItem(
                example_id=example_id,
                audio_filename=audio_filename,
                audio_path=audio_root / audio_filename,
                n_options=n_options,
                payload=row,
            )
        )
    return items


def _count_labels(labels_by_id: dict[str, str], item_ids: set[str]) -> tuple[int, int]:
    good = 0
    bad = 0
    for item_id, label in labels_by_id.items():
        if item_id not in item_ids:
            continue
        if label == "good":
            good += 1
        elif label == "bad":
            bad += 1
    return good, bad


def _find_next_unlabeled(items: list[ReviewItem], labels_by_id: dict[str, str], start: int) -> int:
    if not items:
        return 0
    total = len(items)
    for offset in range(total):
        idx = (start + offset) % total
        if items[idx].example_id not in labels_by_id:
            return idx
    return start % total


def _model_pill(model_name: str) -> tuple[str, str]:
    key = model_name.lower()
    if "audio-flamingo" in key:
        return "AF3", "pill-af3"
    if "qwen" in key:
        return "QWEN", "pill-qwen"
    short = model_name.upper()
    if len(short) > 10:
        short = short[:10]
    return short, "pill-other"


def _build_option_model_predictions(model_data: dict[str, dict[str, Any]]) -> dict[str, list[tuple[str, str]]]:
    by_label: dict[str, list[tuple[str, str]]] = {}
    for model_name, details in model_data.items():
        predicted_label = str(details.get("predicted_label", "")).strip()
        if not predicted_label:
            continue
        by_label.setdefault(predicted_label, []).append(_model_pill(model_name))
    for label in by_label:
        by_label[label].sort(key=lambda x: x[0])
    return by_label


def _format_question_html(item: ReviewItem, outcomes_by_example: dict[str, dict[str, dict[str, Any]]]) -> str:
    row = item.payload
    question = html.escape(str(row.get("question", "")))
    options = row.get("options", [])
    answer_label = str(row.get("answer_label", "")).strip()
    model_data = outcomes_by_example.get(item.example_id, {})
    option_model_predictions = _build_option_model_predictions(model_data)

    option_rows: list[str] = []
    for option in options:
        label = html.escape(str(option.get("label", "?")))
        text = html.escape(str(option.get("text", "")))
        option_type = html.escape(str(option.get("type", "event")))
        is_correct = label == answer_label
        row_class = "option-row option-correct" if is_correct else "option-row"

        pills = option_model_predictions.get(label, [])
        pills_html = ""
        if pills:
            joined = "".join(
                f'<span class="model-pill {pill_class}">{html.escape(pill_text)}</span>'
                for pill_text, pill_class in pills
            )
            pills_html = f'<span class="pill-wrap">{joined}</span>'

        option_rows.append(
            f'<div class="{row_class}"><strong>{label}</strong> ({option_type}): {text}{pills_html}</div>'
        )

    return (
        '<div class="question-title">Question</div>'
        f'<div class="question-text">{question}</div>'
        '<div class="question-title">Options</div>'
        f'<div class="options-list">{"".join(option_rows)}</div>'
    )


def _build_option_deletion_choices(item: ReviewItem) -> tuple[list[tuple[str, str]], dict[str, dict[str, Any]]]:
    row = item.payload
    options = row.get("options", [])
    answer_label = str(row.get("answer_label", "")).strip()
    answer_type = str(row.get("answer_type", "")).strip()
    choices: list[tuple[str, str]] = []
    by_key: dict[str, dict[str, Any]] = {}
    for option in options if isinstance(options, list) else []:
        if not isinstance(option, dict):
            continue
        if str(option.get("type", "")).strip() != "event":
            continue
        label = str(option.get("label", "")).strip()
        if answer_type == "event" and label == answer_label:
            # Keep gold event immutable so question validity is preserved.
            continue
        key = f"opt::{label}"
        text = str(option.get("text", ""))
        short = text if len(text) <= 96 else f"{text[:93]}..."
        choice_label = f"{label} | {short}"
        choices.append((choice_label, key))
        by_key[key] = option
    return choices, by_key


def _question_with_deleted_options(item: ReviewItem, deleted_option_keys: set[str]) -> ReviewItem:
    row = item.payload
    options = row.get("options", [])
    if not isinstance(options, list) or not deleted_option_keys:
        return item
    kept_options: list[dict[str, Any]] = []
    for option in options:
        if not isinstance(option, dict):
            continue
        label = str(option.get("label", "")).strip()
        key = f"opt::{label}"
        if key in deleted_option_keys and str(option.get("type", "")).strip() == "event":
            continue
        kept_options.append(option)
    updated = dict(row)
    updated["options"] = kept_options
    return ReviewItem(
        example_id=item.example_id,
        audio_filename=item.audio_filename,
        audio_path=item.audio_path,
        n_options=len(kept_options),
        payload=updated,
    )


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _collect_timeline_events(item: ReviewItem, deleted_option_keys: set[str] | None = None) -> list[dict[str, Any]]:
    if deleted_option_keys is None:
        deleted_option_keys = set()
    row = item.payload
    base_event = row.get("base_event") if isinstance(row.get("base_event"), dict) else {}
    options = row.get("options", [])
    events: list[dict[str, Any]] = []

    if isinstance(base_event, dict):
        events.append(
            {
                "key": f"base::{base_event.get('event_index')}",
                "label": "Base",
                "text": str(base_event.get("text", "")),
                "onset": _to_float(base_event.get("onset")),
                "offset": _to_float(base_event.get("offset")),
                "event_index": base_event.get("event_index"),
                "is_base": True,
            }
        )

    if isinstance(options, list):
        for option in options:
            if not isinstance(option, dict):
                continue
            if str(option.get("type", "")).strip() != "event":
                continue
            key = f"opt::{str(option.get('label', '')).strip()}"
            if key in deleted_option_keys:
                continue
            events.append(
                {
                    "key": f"opt::{option.get('event_index')}::{option.get('label')}",
                    "label": str(option.get("label", "")),
                    "text": str(option.get("text", "")),
                    "onset": _to_float(option.get("onset")),
                    "offset": _to_float(option.get("offset")),
                    "event_index": option.get("event_index"),
                    "is_base": False,
                }
            )

    dedup: dict[str, dict[str, Any]] = {}
    for event in events:
        key = str(event.get("key"))
        if key not in dedup:
            dedup[key] = event
    ordered = sorted(
        dedup.values(),
        key=lambda e: (
            float("inf") if e.get("onset") is None else e.get("onset"),
            float("inf") if e.get("offset") is None else e.get("offset"),
        ),
    )
    return ordered


def _format_timeline_html(item: ReviewItem, deleted_option_keys: set[str] | None = None) -> str:
    ordered = _collect_timeline_events(item, deleted_option_keys=deleted_option_keys)
    valid = [e for e in ordered if e.get("onset") is not None and e.get("offset") is not None]
    if not valid:
        return '<div class="timeline-wrap"><div class="timeline-title">Event Timeline</div><div>No timing info available.</div></div>'

    min_onset = min(e["onset"] for e in valid)
    max_offset = max(e["offset"] for e in valid)
    span = max(max_offset - min_onset, 1e-6)

    rows: list[str] = []
    for event in valid:
        onset = float(event["onset"])
        offset = float(event["offset"])
        width = max((offset - onset) / span * 100.0, 1.5)
        left = max((onset - min_onset) / span * 100.0, 0.0)
        seg_class = "timeline-seg timeline-seg-base" if event.get("is_base") else "timeline-seg"
        label = html.escape(str(event.get("label", "")))
        text = html.escape(str(event.get("text", "")))
        meta = f"{label} | {onset:.2f}s to {offset:.2f}s"
        rows.append(
            '<div class="timeline-row">'
            f'<div class="timeline-meta">{meta}</div>'
            '<div>'
            f'<div class="timeline-track"><div class="{seg_class}" style="left:{left:.2f}%;width:{width:.2f}%"></div></div>'
            f'<div class="timeline-label">{text}</div>'
            "</div>"
            "</div>"
        )

    return (
        '<div class="timeline-wrap">'
        f'<div class="timeline-title">Event Timeline ({min_onset:.2f}s to {max_offset:.2f}s)</div>'
        + "".join(rows)
        + "</div>"
    )
@app.command()
def main(
    dataset: Path = typer.Option(
        Path("data/mcq_event_timeline_strong.jsonl"),
        "--dataset",
        help="MCQ-ORDER dataset JSONL file.",
    ),
    audio_root: Path = typer.Option(
        Path("data/audio"),
        "--audio-root",
        help="Directory containing audio files referenced by the dataset.",
    ),
    runs_csv: Path = typer.Option(
        Path("results/mcq-order/runs.csv"),
        "--runs-csv",
        help="Runs registry CSV used to find latest model decisions per model.",
    ),
    labels_output: Path = typer.Option(
        Path("results/mcq-order/review/manual_good_bad_labels.jsonl"),
        "--labels-output",
        help="Append-only JSONL where Good/Bad labels are persisted in real time.",
    ),
    min_options: int = typer.Option(
        4,
        "--min-options",
        help="Minimum number of options per question to include in review.",
    ),
    max_options: int = typer.Option(
        6,
        "--max-options",
        help="Maximum number of options per question to include in review.",
    ),
    semantic_dedupe: bool = typer.Option(
        False,
        "--semantic-dedupe/--no-semantic-dedupe",
        help="Deduplicate semantically similar event options before option-count filtering.",
    ),
    similarity_model_id: str = typer.Option(
        "sentence-transformers/all-MiniLM-L6-v2",
        "--similarity-model-id",
        help="Embedding model id used for option similarity.",
    ),
    similarity_threshold: float = typer.Option(
        0.88,
        "--similarity-threshold",
        help="Cosine threshold above which two option texts are treated as near-duplicates.",
    ),
    similarity_batch_size: int = typer.Option(
        64,
        "--similarity-batch-size",
        help="Batch size for embedding option texts.",
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Host for local Gradio app."),
    port: int = typer.Option(7860, "--port", help="Port for local Gradio app."),
    share: bool = typer.Option(False, "--share", help="Expose Gradio share link."),
    inbrowser: bool = typer.Option(True, "--inbrowser", help="Open UI in browser on launch."),
) -> None:
    if min_options < 1:
        raise typer.BadParameter("--min-options must be >= 1")
    if max_options < min_options:
        raise typer.BadParameter("--max-options must be >= --min-options")

    dataset_rows = _read_jsonl(dataset)
    dataset_rows = _dedupe_dataset_options(
        dataset_rows,
        semantic_dedupe=semantic_dedupe,
        model_id=similarity_model_id,
        similarity_threshold=similarity_threshold,
        batch_size=similarity_batch_size,
    )
    items = _build_filtered_items(
        dataset_rows,
        audio_root=audio_root,
        min_options=min_options,
        max_options=max_options,
    )
    if not items:
        raise typer.BadParameter(
            f"No examples after filter. Try adjusting --min-options/--max-options. "
            f"Current filter: [{min_options}, {max_options}]"
        )

    outcomes_by_example = _load_model_outcomes_by_example(runs_csv=runs_csv, dataset_path=dataset)
    labels_by_id = _load_existing_labels(labels_output)
    deletions_by_id: dict[str, set[str]] = {}
    item_ids = {item.example_id for item in items}
    start_idx = _find_next_unlabeled(items=items, labels_by_id=labels_by_id, start=0)

    def render(position: int) -> tuple[int, str, str | None, str, str, gr.CheckboxGroup, str]:
        idx = position % len(items)
        item = items[idx]
        deleted_keys = deletions_by_id.get(item.example_id, set())
        view_item = _question_with_deleted_options(item, deleted_option_keys=deleted_keys)
        good_count, bad_count = _count_labels(labels_by_id, item_ids)
        reviewed = good_count + bad_count
        total = len(items)
        progress_md = (
            '<div class="stats-line">'
            f'<div class="stat-pill"><div class="stat-label">Reviewed</div><div class="stat-value">{reviewed}/{total}</div></div>'
            f'<div class="stat-pill"><div class="stat-label">Good</div><div class="stat-value">{good_count}</div></div>'
            f'<div class="stat-pill"><div class="stat-label">Bad</div><div class="stat-value">{bad_count}</div></div>'
            f'<div class="stat-pill"><div class="stat-label">Position</div><div class="stat-value">{idx + 1}/{total}</div></div>'
            "</div>"
        )
        audio_value = str(item.audio_path) if item.audio_path.exists() else None
        question_md = _format_question_html(view_item, outcomes_by_example)
        timeline_md = _format_timeline_html(item, deleted_option_keys=deleted_keys)
        choices, by_key = _build_option_deletion_choices(item)
        valid = set(by_key.keys())
        selected = sorted(key for key in deleted_keys if key in valid)
        deletion_hint = f"Marked for deletion: `{len(selected)}` event option(s)."
        return (
            idx,
            progress_md,
            audio_value,
            question_md,
            timeline_md,
            gr.update(choices=choices, value=selected),
            deletion_hint,
        )

    def go_prev(position: int) -> tuple[int, str, str | None, str, str, gr.CheckboxGroup, str]:
        return render(position - 1)

    def go_next(position: int) -> tuple[int, str, str | None, str, str, gr.CheckboxGroup, str]:
        return render(position + 1)

    def go_next_unlabeled(position: int) -> tuple[int, str, str | None, str, str, gr.CheckboxGroup, str]:
        idx = _find_next_unlabeled(items=items, labels_by_id=labels_by_id, start=position + 1)
        return render(idx)

    def update_deletions(position: int, selected_keys: list[str] | None) -> tuple[int, str, str | None, str, str, gr.CheckboxGroup, str]:
        idx = position % len(items)
        item = items[idx]
        _, by_key = _build_option_deletion_choices(item)
        valid = set(by_key.keys())
        selected = {key for key in (selected_keys or []) if key in valid}
        if selected:
            deletions_by_id[item.example_id] = selected
        else:
            deletions_by_id.pop(item.example_id, None)
        return render(idx)

    def clear_deletions(position: int) -> tuple[int, str, str | None, str, str, gr.CheckboxGroup, str]:
        idx = position % len(items)
        item = items[idx]
        deletions_by_id.pop(item.example_id, None)
        return render(idx)

    def set_label(position: int, label: str, selected_keys: list[str] | None) -> tuple[int, str, str | None, str, str, gr.CheckboxGroup, str]:
        idx = position % len(items)
        item = items[idx]
        _, by_key = _build_option_deletion_choices(item)
        valid = set(by_key.keys())
        selected = {key for key in (selected_keys or []) if key in valid}
        if selected:
            deletions_by_id[item.example_id] = selected
        else:
            deletions_by_id.pop(item.example_id, None)

        deleted_options = []
        for key in sorted(selected):
            option = by_key[key]
            deleted_options.append(
                {
                    "key": key,
                    "label": option.get("label"),
                    "text": option.get("text"),
                    "onset": option.get("onset"),
                    "offset": option.get("offset"),
                    "event_index": option.get("event_index"),
                }
            )
        labels_by_id[item.example_id] = label
        _write_jsonl_line(
            labels_output,
            {
                "id": item.example_id,
                "label": label,
                "audio_filename": item.audio_filename,
                "deleted_option_keys": sorted(selected),
                "deleted_options": deleted_options,
                "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            },
        )
        next_idx = _find_next_unlabeled(items=items, labels_by_id=labels_by_id, start=idx + 1)
        return render(next_idx)

    with gr.Blocks(
        title="MCQ-ORDER Manual Good/Bad Review",
        css=CUSTOM_CSS,
        head=f"<script>{CUSTOM_JS}</script>",
        theme=gr.themes.Soft(spacing_size="sm", radius_size="md"),
    ) as demo:
        gr.Markdown("# MCQ-ORDER Manual Review", elem_classes=["compact-title", "app-shell"])
        gr.Markdown(
            f"Filter active: option count in [{min_options}, {max_options}] | "
            f"labels saved to `{labels_output}` in real time. Shortcuts: `g` = Good, `b` = Bad.",
            elem_classes=["app-shell"],
        )

        position_state = gr.State(start_idx)
        with gr.Column(elem_classes=["app-shell"]):
            progress = gr.HTML(elem_classes=["card", "compact-title"])
            with gr.Column(elem_classes=["card"]):
                audio = gr.Audio(label="Audio", type="filepath", interactive=False, autoplay=False)
            timeline = gr.HTML(elem_classes=["card"])
            deletion_picker = gr.CheckboxGroup(
                label="Delete event options before saving",
                choices=[],
                value=[],
            )
            deletion_hint = gr.Markdown("Marked for deletion: `0` event option(s).")
            clear_deletions_btn = gr.Button("Clear deletions")
            with gr.Row():
                good_btn = gr.Button("Good [G]", variant="primary", elem_id="good-btn")
                bad_btn = gr.Button("Bad [B]", variant="stop", elem_id="bad-btn")
            with gr.Row():
                prev_btn = gr.Button("Prev")
                next_btn = gr.Button("Next")
                next_unlabeled_btn = gr.Button("Next Unlabeled")
            question = gr.HTML(elem_classes=["card"])

        outputs = [position_state, progress, audio, question, timeline, deletion_picker, deletion_hint]

        demo.load(fn=lambda: render(start_idx), outputs=outputs)
        prev_btn.click(fn=go_prev, inputs=position_state, outputs=outputs)
        next_btn.click(fn=go_next, inputs=position_state, outputs=outputs)
        next_unlabeled_btn.click(fn=go_next_unlabeled, inputs=position_state, outputs=outputs)
        deletion_picker.change(fn=update_deletions, inputs=[position_state, deletion_picker], outputs=outputs)
        clear_deletions_btn.click(fn=clear_deletions, inputs=position_state, outputs=outputs)
        good_btn.click(fn=lambda pos, sel: set_label(pos, "good", sel), inputs=[position_state, deletion_picker], outputs=outputs)
        bad_btn.click(fn=lambda pos, sel: set_label(pos, "bad", sel), inputs=[position_state, deletion_picker], outputs=outputs)

    demo.queue(default_concurrency_limit=1).launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=inbrowser,
        show_error=True,
    )


if __name__ == "__main__":
    app()
