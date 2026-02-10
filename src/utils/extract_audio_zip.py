"""Extract TACoBeLAL audio.zip with progress reporting."""

from __future__ import annotations

import zipfile
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()


def main(
    zip_path: Path = typer.Option(
        Path("data/audio.zip"),
        "--zip-path",
        "-z",
        path_type=Path,
        exists=True,
        dir_okay=False,
        help="Path to audio.zip.",
    ),
    output_dir: Path = typer.Option(
        Path("data/audio"),
        "--output-dir",
        "-o",
        path_type=Path,
        help="Directory where audio files will be extracted.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Re-extract even if target files already exist.",
    ),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        if not members:
            raise typer.BadParameter(f"No files found in archive: {zip_path}")

        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        )

        extracted = 0
        skipped = 0
        with progress:
            task = progress.add_task("Extracting audio files", total=len(members))
            for member in members:
                target = output_dir / member.filename
                if target.exists() and not overwrite:
                    skipped += 1
                    progress.advance(task, 1)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member, "r") as src, open(target, "wb") as dst:
                    dst.write(src.read())
                extracted += 1
                progress.advance(task, 1)

    table = Table(title="Audio extraction summary", show_header=True, header_style="bold")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Archive", str(zip_path))
    table.add_row("Output directory", str(output_dir))
    table.add_row("Files in archive", str(len(members)))
    table.add_row("Extracted files", str(extracted))
    table.add_row("Skipped existing", str(skipped))
    console.print(table)


if __name__ == "__main__":
    typer.run(main)
