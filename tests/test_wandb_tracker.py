"""Tests for optional W&B tracker utility."""

from __future__ import annotations

from pathlib import Path

from utils.wandb_tracker import WandbTracker


def test_tracker_disabled_is_noop(tmp_path: Path) -> None:
    tracker = WandbTracker(enabled=False)
    assert tracker.active is False

    tracker.log({"metric": 1.0})
    tracker.log_live(
        index=1,
        total=10,
        accuracy_so_far=0.5,
        latency_ms=12.3,
        is_correct=True,
        correct_so_far=1,
        force=True,
    )
    tracker.update_summary({"accuracy": 0.5})
    tracker.log_table(
        key="analysis/table",
        columns=["label", "score"],
        rows=[["A", 1.0]],
    )
    tracker.log_confusion_matrix(
        key="analysis/confusion_matrix",
        y_true=["A"],
        y_pred=["B"],
        class_names=["A", "B"],
    )
    tracker.log_artifact(
        name="dummy",
        artifact_type="evaluation",
        files=[tmp_path / "does_not_exist.json"],
    )
    tracker.finish()
