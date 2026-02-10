.PHONY: \
	install-dev \
	install-llm \
	download-dataset \
	extract-audio \
	build-mcq-dataset \
	download-audioflamingo \
	setup-from-scratch \
	eval-mcq-order-random \
	eval-mcq-order-openai \
	eval-mcq-order-audioflamingo-smoke \
	eval-mcq-order-audioflamingo-full \
	test

DATA_DIR ?= data
RESULTS_DIR ?= results
MCQ_DATASET ?= $(DATA_DIR)/mcq_event_timeline_strong.jsonl
AUDIO_ROOT ?= $(DATA_DIR)/audio
AUDIO_ZIP ?= $(DATA_DIR)/audio.zip

AF_REPO_URL ?= https://github.com/NVIDIA/audio-flamingo.git
AF_BRANCH ?= audio_flamingo_3
AF_HOME ?= external/audio-flamingo
AF_MODEL_BASE ?= nvidia/audio-flamingo-3
AF_NUM_GPUS ?= 1
AF_BATCH_SIZE ?= 2
AF_MAX_NEW_TOKENS ?= 16
AF_SMOKE_LIMIT ?= 100

install-dev:
	uv sync --extra dev

install-llm:
	uv sync --extra llm

download-dataset:
	uv run python src/utils/download_dataset.py --output $(DATA_DIR)

extract-audio:
	uv run python src/utils/extract_audio_zip.py --zip-path $(AUDIO_ZIP) --output-dir $(AUDIO_ROOT)

build-mcq-dataset:
	uv run python src/utils/build_timeline_mcq_dataset.py --input $(DATA_DIR)/annotations_strong.csv --output $(MCQ_DATASET)

download-audioflamingo:
	uv run python src/utils/setup_audioflamingo.py --repo-url $(AF_REPO_URL) --branch $(AF_BRANCH) --destination $(AF_HOME)

setup-from-scratch: install-dev install-llm download-dataset extract-audio build-mcq-dataset download-audioflamingo

eval-mcq-order-random:
	uv run python src/utils/evaluate_mcq_order.py --dataset $(MCQ_DATASET) --model random --results-root $(RESULTS_DIR)

eval-mcq-order-openai:
	uv sync --extra llm
	uv run python src/utils/evaluate_mcq_order.py --dataset $(MCQ_DATASET) --model llm-openai --openai-model gpt-4o-mini --temperature 0 --results-root $(RESULTS_DIR)

eval-mcq-order-audioflamingo-smoke:
	uv run python src/utils/evaluate_mcq_order_audioflamingo.py \
		--dataset $(MCQ_DATASET) \
		--audio-root $(AUDIO_ROOT) \
		--audioflamingo-repo $(AF_HOME) \
		--model-base $(AF_MODEL_BASE) \
		--num-gpus $(AF_NUM_GPUS) \
		--batch-size $(AF_BATCH_SIZE) \
		--max-new-tokens $(AF_MAX_NEW_TOKENS) \
		--limit $(AF_SMOKE_LIMIT) \
		--results-root $(RESULTS_DIR)

eval-mcq-order-audioflamingo-full:
	uv run python src/utils/evaluate_mcq_order_audioflamingo.py \
		--dataset $(MCQ_DATASET) \
		--audio-root $(AUDIO_ROOT) \
		--audioflamingo-repo $(AF_HOME) \
		--model-base $(AF_MODEL_BASE) \
		--num-gpus $(AF_NUM_GPUS) \
		--batch-size $(AF_BATCH_SIZE) \
		--max-new-tokens $(AF_MAX_NEW_TOKENS) \
		--results-root $(RESULTS_DIR)

test:
	uv sync --extra dev
	uv run pytest
