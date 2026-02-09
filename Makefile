.PHONY: download-dataset test

download-dataset:
	uv run python src/utils/download_dataset.py --output data

test:
	uv sync --extra dev
	uv run pytest
