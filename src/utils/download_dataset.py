"""Download TACOS dataset files from Zenodo using the bundled zenodo record JSON."""

import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

console = Console()

# Default zenodo record path: next to this script (permanent location).
_DEFAULT_ZENODO_PATH = Path(__file__).resolve().parent / "zenodo.json"


def _load_zenodo_record(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        console.print(f"[red]Error: zenodo JSON file not found at '{path}'.[/red]")
        raise typer.Exit(1)
    except json.JSONDecodeError as exc:
        console.print(f"[red]Error: failed to parse JSON at '{path}': {exc}[/red]")
        raise typer.Exit(1)


def _iter_file_entries(record: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    files = record.get("files") or {}
    entries = files.get("entries") or {}
    if not isinstance(entries, dict):
        console.print("[red]Error: unexpected structure for files.entries in zenodo JSON.[/red]")
        raise typer.Exit(1)
    for key, meta in entries.items():
        yield key, meta


def _should_skip_file(local_path: Path, size: int | None) -> bool:
    if not local_path.exists():
        return False
    if size is None:
        return False
    try:
        return local_path.stat().st_size == size
    except OSError:
        return False


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KiB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MiB"
    return f"{size_bytes / (1024 * 1024 * 1024):.1f} GiB"


def _download_file_with_progress(
    url: str,
    dest_path: Path,
    total_bytes: int | None,
    filename: str,
) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    total = total_bytes or 0
    chunk_size = 1024 * 1024  # 1 MiB

    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task(f"[cyan]{filename}", total=total)
        try:
            with urllib.request.urlopen(url) as response, open(dest_path, "wb") as out_file:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    progress.advance(task, len(chunk))
        except urllib.error.URLError as exc:
            console.print(f"[red]Error: failed to download '{url}': {exc}[/red]")
            raise typer.Exit(1)
        except OSError as exc:
            console.print(f"[red]Error: failed to write to '{dest_path}': {exc}[/red]")
            raise typer.Exit(1)


def _compute_md5(path: Path) -> str:
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _expected_md5(checksum_field: str | None) -> str | None:
    if not checksum_field:
        return None
    parts = checksum_field.split(":", 1)
    if len(parts) == 2 and parts[0].lower() == "md5":
        return parts[1]
    return None


def _print_dataset_stats(
    record: Dict[str, Any],
    output_dir: Path,
    downloaded_files: list[Tuple[str, int]],
) -> None:
    """Print rough dataset stats after download using Rich."""
    meta = record.get("metadata") or {}
    title = meta.get("title", "Dataset")
    if isinstance(title, dict):
        title = title.get("en", str(title))
    pids = record.get("pids") or {}
    doi = (pids.get("doi") or {}).get("identifier", "")
    record_id = record.get("id", "")

    total_size = sum(s for _, s in downloaded_files)
    file_count = len(downloaded_files)

    table = Table(title="Dataset summary", show_header=True, header_style="bold")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Title", title)
    table.add_row("Zenodo ID", str(record_id))
    table.add_row("DOI", doi)
    table.add_row("Output directory", str(output_dir))
    table.add_row("Total files", str(file_count))
    table.add_row("Total size", _format_size(total_size))
    console.print(table)

    if downloaded_files:
        file_table = Table(title="Files", show_header=True, header_style="bold")
        file_table.add_column("File", style="cyan")
        file_table.add_column("Size", justify="right", style="green")
        for name, size in sorted(downloaded_files, key=lambda x: -x[1]):
            file_table.add_row(name, _format_size(size))
        console.print(file_table)


def main(
    zenodo: Path = typer.Option(
        _DEFAULT_ZENODO_PATH,
        "--zenodo",
        "-z",
        path_type=Path,
        help="Path to the Zenodo record JSON file.",
        exists=False,
    ),
    output: Path = typer.Option(
        Path("data"),
        "--output",
        "-o",
        path_type=Path,
        help="Output directory for downloaded files.",
    ),
    skip_audio: bool = typer.Option(
        False,
        "--skip-audio",
        help="Skip downloading audio.zip (only fetch CSV files).",
    ),
    verify_checksums: bool = typer.Option(
        False,
        "--verify-checksums",
        help="Verify MD5 checksums after download.",
    ),
) -> None:
    """Download TACOS dataset files from Zenodo. Zenodo record is read from src/utils/zenodo.json by default."""
    record = _load_zenodo_record(zenodo)
    output.mkdir(parents=True, exist_ok=True)
    downloaded: list[Tuple[str, int]] = []

    for filename, meta in _iter_file_entries(record):
        if skip_audio and filename == "audio.zip":
            console.print(f"[dim]Skipping '{filename}' (--skip-audio).[/dim]")
            continue

        links = meta.get("links") or {}
        url = links.get("content")
        if not url:
            console.print(f"[yellow]Warning: no content URL for '{filename}', skipping.[/yellow]")
            continue

        size = meta.get("size")
        dest_path = output / filename

        if _should_skip_file(dest_path, size):
            console.print(f"[dim]Already present (size match): {dest_path}[/dim]")
        else:
            _download_file_with_progress(url, dest_path, size, filename)
            console.print(f"[green]Done:[/green] {dest_path}")
        if isinstance(size, (int, float)):
            downloaded.append((filename, int(size)))

        if verify_checksums:
            checksum_field = meta.get("checksum")
            expected = _expected_md5(checksum_field)
            if expected is None:
                console.print(
                    f"[yellow]Warning: no MD5 in JSON for '{filename}', skipping verification.[/yellow]"
                )
            else:
                actual = _compute_md5(dest_path)
                if actual != expected:
                    console.print(
                        f"[red]Checksum mismatch for '{filename}': expected={expected}, actual={actual}[/red]"
                    )
                    raise typer.Exit(1)
                console.print(f"[green]Checksum OK:[/green] {filename}")

    console.print()
    _print_dataset_stats(record, output, downloaded)


if __name__ == "__main__":
    typer.run(main)
