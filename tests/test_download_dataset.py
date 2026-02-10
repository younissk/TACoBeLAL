"""Tests for the Zenodo dataset download script. No network access."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer

from utils.download_dataset import (
    _compute_md5,
    _expected_md5,
    _format_size,
    _iter_file_entries,
    _load_zenodo_record,
    _should_skip_file,
)


class TestFormatSize:
    def test_bytes(self) -> None:
        assert _format_size(0) == "0 B"
        assert _format_size(100) == "100 B"

    def test_kibibytes(self) -> None:
        assert _format_size(1024) == "1.0 KiB"
        assert _format_size(1536) == "1.5 KiB"

    def test_mebibytes(self) -> None:
        assert _format_size(1024 * 1024) == "1.0 MiB"
        assert _format_size(4109451) == "3.9 MiB"

    def test_gibibytes(self) -> None:
        assert _format_size(1024 * 1024 * 1024) == "1.0 GiB"
        assert _format_size(1629958981) == "1.5 GiB"


class TestExpectedMd5:
    def test_valid_md5_field(self) -> None:
        assert _expected_md5("md5:882d5a5a28f59441f4c7b4ed17ebab05") == "882d5a5a28f59441f4c7b4ed17ebab05"
        assert _expected_md5("MD5:abc123") == "abc123"

    def test_empty_or_none(self) -> None:
        assert _expected_md5(None) is None
        assert _expected_md5("") is None

    def test_invalid_format(self) -> None:
        assert _expected_md5("sha256:abc") is None
        assert _expected_md5("nocolon") is None


class TestShouldSkipFile:
    def test_missing_file(self, tmp_path: Path) -> None:
        assert _should_skip_file(tmp_path / "missing", 100) is False

    def test_size_none(self, tmp_path: Path) -> None:
        (tmp_path / "f").write_bytes(b"x")
        assert _should_skip_file(tmp_path / "f", None) is False

    def test_size_matches(self, tmp_path: Path) -> None:
        p = tmp_path / "f"
        p.write_bytes(b"hello")
        assert _should_skip_file(p, 5) is True

    def test_size_mismatch(self, tmp_path: Path) -> None:
        p = tmp_path / "f"
        p.write_bytes(b"hello")
        assert _should_skip_file(p, 10) is False


class TestIterFileEntries:
    def test_valid_entries(self) -> None:
        record = {
            "files": {
                "entries": {
                    "a.csv": {"size": 100, "links": {"content": "https://example.com/a"}},
                    "b.zip": {"size": 200, "links": {"content": "https://example.com/b"}},
                }
            }
        }
        items = list(_iter_file_entries(record))
        assert len(items) == 2
        names = [k for k, _ in items]
        assert "a.csv" in names and "b.zip" in names

    def test_empty_entries(self) -> None:
        record = {"files": {"entries": {}}}
        assert list(_iter_file_entries(record)) == []

    def test_missing_files_key(self) -> None:
        record = {}
        assert list(_iter_file_entries(record)) == []

    def test_entries_not_dict_exits(self) -> None:
        record = {"files": {"entries": "not a dict"}}
        with pytest.raises(typer.Exit):
            list(_iter_file_entries(record))


class TestLoadZenodoRecord:
    def test_valid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "zenodo.json"
        path.write_text('{"files":{"entries":{},"count":0}}', encoding="utf-8")
        data = _load_zenodo_record(path)
        assert data["files"]["count"] == 0

    def test_missing_file_exits(self) -> None:
        with pytest.raises(typer.Exit):
            _load_zenodo_record(Path("/nonexistent/zenodo.json"))

    def test_invalid_json_exits(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("not json {", encoding="utf-8")
        with pytest.raises(typer.Exit):
            _load_zenodo_record(path)


class TestComputeMd5:
    def test_known_content(self, tmp_path: Path) -> None:
        p = tmp_path / "f"
        p.write_bytes(b"hello")
        assert _compute_md5(p) == "5d41402abc4b2a76b9719d911017c592"


class TestMainIntegration:
    """Run main() with mocked network; no real HTTP calls."""

    def test_main_downloads_with_mocked_urlopen(self, tmp_path: Path) -> None:
        zenodo_path = tmp_path / "zenodo.json"
        zenodo_path.write_text(
            json.dumps({
                "files": {
                    "entries": {
                        "small.csv": {
                            "size": 5,
                            "links": {"content": "https://zenodo.org/files/small.csv/content"},
                            "checksum": "md5:5d41402abc4b2a76b9719d911017c592",
                        },
                    }
                },
                "metadata": {"title": "Test Dataset"},
                "pids": {"doi": {"identifier": "10.1234/zenodo.1"}},
                "id": "1",
            }),
            encoding="utf-8",
        )
        output_dir = tmp_path / "data"
        fake_response = MagicMock()
        fake_response.read = MagicMock(side_effect=[b"hello", b""])
        fake_response.__enter__ = MagicMock(return_value=fake_response)
        fake_response.__exit__ = MagicMock(return_value=False)

        with patch("utils.download_dataset.urllib.request.urlopen", return_value=fake_response):
            from utils.download_dataset import main

            main(
                zenodo=zenodo_path,
                output=output_dir,
                skip_audio=False,
                verify_checksums=True,
            )

        assert (output_dir / "small.csv").exists()
        assert (output_dir / "small.csv").read_bytes() == b"hello"

    def test_main_skip_audio_skips_audio_zip(self, tmp_path: Path) -> None:
        zenodo_path = tmp_path / "zenodo.json"
        zenodo_path.write_text(
            json.dumps({
                "files": {
                    "entries": {
                        "audio.zip": {
                            "size": 100,
                            "links": {"content": "https://zenodo.org/files/audio.zip/content"},
                        },
                        "meta.csv": {
                            "size": 3,
                            "links": {"content": "https://zenodo.org/files/meta.csv/content"},
                        },
                    }
                },
                "metadata": {"title": "Test"},
                "pids": {},
                "id": "1",
            }),
            encoding="utf-8",
        )
        output_dir = tmp_path / "out"
        call_count = 0

        def mock_urlopen(url: str, **_: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.read = MagicMock(side_effect=[b"abc", b""])
            cm = MagicMock()
            cm.__enter__ = MagicMock(return_value=resp)
            cm.__exit__ = MagicMock(return_value=False)
            return cm

        with patch("utils.download_dataset.urllib.request.urlopen", side_effect=mock_urlopen):
            from utils.download_dataset import main

            main(
                zenodo=zenodo_path,
                output=output_dir,
                skip_audio=True,
                verify_checksums=False,
            )

        assert call_count == 1
        assert (output_dir / "meta.csv").exists()
        assert not (output_dir / "audio.zip").exists()
