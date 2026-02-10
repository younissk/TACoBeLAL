"""Optional Weights & Biases tracking utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass
class WandbTracker:
    enabled: bool
    project: str = "tacobelal"
    entity: str | None = None
    run_name: str | None = None
    log_every: int = 50
    config: dict[str, Any] | None = None
    tags: list[str] | None = None

    def __post_init__(self) -> None:
        self.log_every = max(1, self.log_every)
        self._wandb: Any | None = None
        self._run: Any | None = None

        if not self.enabled:
            return

        try:
            import wandb  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised in integration
            raise RuntimeError(
                "W&B logging was requested but 'wandb' is not installed. "
                "Install with: uv sync --extra tracking"
            ) from exc

        init_kwargs: dict[str, Any] = {
            "project": self.project,
            "config": self.config or {},
        }
        if self.entity:
            init_kwargs["entity"] = self.entity
        if self.run_name:
            init_kwargs["name"] = self.run_name
        if self.tags:
            init_kwargs["tags"] = self.tags

        self._wandb = wandb
        self._run = wandb.init(**init_kwargs)

    @property
    def active(self) -> bool:
        return self.enabled and self._run is not None and self._wandb is not None

    def log(self, metrics: dict[str, Any], *, step: int | None = None, commit: bool = True) -> None:
        if not self.active:
            return
        self._wandb.log(metrics, step=step, commit=commit)

    def log_live(
        self,
        *,
        index: int,
        total: int,
        accuracy_so_far: float,
        latency_ms: float,
        is_correct: bool,
        correct_so_far: int | None = None,
        force: bool = False,
    ) -> None:
        if not self.active:
            return
        if not force and index % self.log_every != 0:
            return
        if correct_so_far is None:
            correct_so_far = int(round(accuracy_so_far * index))
        incorrect_so_far = max(0, index - correct_so_far)
        self._wandb.log(
            {
                "live/example_index": index,
                "live/examples_total": total,
                "live/progress": index / total if total else 0.0,
                "live/accuracy_so_far": accuracy_so_far,
                "live/correct_so_far": correct_so_far,
                "live/incorrect_so_far": incorrect_so_far,
                "live/latency_ms": latency_ms,
                "live/is_correct": int(is_correct),
            },
            step=index,
        )

    def update_summary(self, summary: dict[str, Any]) -> None:
        if not self.active:
            return
        for key, value in summary.items():
            self._run.summary[key] = value

    def log_artifact(self, *, name: str, artifact_type: str, files: Iterable[Path]) -> None:
        if not self.active:
            return
        artifact = self._wandb.Artifact(name=name, type=artifact_type)
        for file_path in files:
            path = Path(file_path)
            if path.exists():
                artifact.add_file(str(path))
        self._run.log_artifact(artifact)

    def log_table(self, *, key: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        if not self.active:
            return
        table = self._wandb.Table(columns=list(columns), data=[list(row) for row in rows])
        self._wandb.log({key: table})

    def log_confusion_matrix(
        self,
        *,
        key: str,
        y_true: Sequence[str],
        y_pred: Sequence[str],
        class_names: Sequence[str],
    ) -> None:
        if not self.active:
            return
        if not y_true or not y_pred:
            return
        confusion_plot = self._wandb.plot.confusion_matrix(
            y_true=list(y_true),
            preds=list(y_pred),
            class_names=list(class_names),
        )
        self._wandb.log({key: confusion_plot})

    def finish(self) -> None:
        if not self.active:
            return
        self._run.finish()
