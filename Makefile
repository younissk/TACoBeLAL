.PHONY: download-dataset build-mcq-dataset eval-mcq-order-random eval-mcq-order-openai test

download-dataset:
	uv run python src/utils/download_dataset.py --output data

build-mcq-dataset:
	uv run python src/utils/build_timeline_mcq_dataset.py --input data/annotations_strong.csv --output data/mcq_event_timeline_strong.jsonl

eval-mcq-order-random:
	uv run python src/utils/evaluate_mcq_order.py --dataset data/mcq_event_timeline_strong.jsonl --model random --results-root results

eval-mcq-order-openai:
	uv sync --extra llm
	uv run python src/utils/evaluate_mcq_order.py --dataset data/mcq_event_timeline_strong.jsonl --model llm-openai --openai-model gpt-4o-mini --temperature 0 --results-root results

test:
	uv sync --extra dev
	uv run pytest
