"""Optional Weights & Biases tracking utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


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
        force: bool = False,
    ) -> None:
        if not self.active:
            return
        if not force and index % self.log_every != 0:
            return
        self._wandb.log(
            {
                "live/example_index": index,
                "live/examples_total": total,
                "live/progress": index / total if total else 0.0,
                "live/accuracy_so_far": accuracy_so_far,
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

    def finish(self) -> None:
        if not self.active:
            return
        self._run.finish()
