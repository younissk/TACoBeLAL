"""Clone or update Audio Flamingo repository for cluster evaluation setup."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def _run(cmd: Sequence[str], cwd: Path | None = None) -> str:
    console.print(f"[cyan]$ {' '.join(cmd)}[/cyan]")
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def main(
    repo_url: str = typer.Option(
        "https://github.com/NVIDIA/audio-flamingo.git",
        "--repo-url",
        help="Audio Flamingo git repository URL.",
    ),
    branch: str = typer.Option(
        "audio_flamingo_3",
        "--branch",
        help="Branch to checkout.",
    ),
    destination: Path = typer.Option(
        Path("external/audio-flamingo"),
        "--destination",
        "-d",
        path_type=Path,
        help="Destination directory for the repo.",
    ),
    update_existing: bool = typer.Option(
        True,
        "--update-existing/--no-update-existing",
        help="If destination exists, update to latest origin/<branch>.",
    ),
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if (destination / ".git").exists():
        if update_existing:
            console.print(f"[yellow]Updating existing repository at {destination}[/yellow]")
            _run(["git", "fetch", "--all", "--tags"], cwd=destination)
            _run(["git", "checkout", branch], cwd=destination)
            _run(["git", "pull", "--ff-only", "origin", branch], cwd=destination)
        else:
            console.print(f"[yellow]Repository already exists at {destination}; skipping update.[/yellow]")
    else:
        if destination.exists() and any(destination.iterdir()):
            raise typer.BadParameter(
                f"Destination '{destination}' exists and is not a git repo. "
                "Choose an empty destination."
            )
        console.print(f"[green]Cloning {repo_url} (branch: {branch}) into {destination}[/green]")
        _run(["git", "clone", "--depth", "1", "--branch", branch, repo_url, str(destination)])

    commit = _run(["git", "rev-parse", "HEAD"], cwd=destination)
    active_branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=destination)

    table = Table(title="Audio Flamingo setup", show_header=True, header_style="bold")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Repository", repo_url)
    table.add_row("Destination", str(destination))
    table.add_row("Branch", active_branch)
    table.add_row("Commit", commit)
    console.print(table)


if __name__ == "__main__":
    typer.run(main)
