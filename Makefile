.PHONY: \
	install-dev \
	install-llm \
	install-tracking \
	download-dataset \
	extract-audio \
	build-mcq-dataset \
	download-audioflamingo \
	setup-from-scratch \
	eval-mcq-order-random \
	eval-mcq-order-openai \
	eval-mcq-order-qwen \
	eval-mcq-order-llama \
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

QWEN_MODEL_ID ?= Qwen/Qwen2.5-7B-Instruct
LLAMA_MODEL_ID ?= meta-llama/Llama-3.1-8B-Instruct
LOCAL_DTYPE ?= float16
LOCAL_DEVICE_MAP ?= auto
LOCAL_MAX_NEW_TOKENS ?= 16
LOCAL_TEMPERATURE ?= 0.0
LOCAL_TOP_P ?= 1.0
LOCAL_LIMIT ?= 100

WANDB_PROJECT ?= tacobelal
WANDB_ENTITY ?=
WANDB_LOG_EVERY ?= 50
WANDB_RUN_NAME ?=

WAND_ARGS = --wandb --wandb-project $(WANDB_PROJECT) --wandb-log-every $(WANDB_LOG_EVERY)
ifneq ($(strip $(WANDB_ENTITY)),)
WAND_ARGS += --wandb-entity $(WANDB_ENTITY)
endif
ifneq ($(strip $(WANDB_RUN_NAME)),)
WAND_ARGS += --wandb-run-name $(WANDB_RUN_NAME)
endif

install-dev:
	uv sync --extra dev

install-llm:
	uv sync --extra llm

install-tracking:
	uv sync --extra tracking

download-dataset:
	uv run python src/utils/download_dataset.py --output $(DATA_DIR)

extract-audio:
	uv run python src/utils/extract_audio_zip.py --zip-path $(AUDIO_ZIP) --output-dir $(AUDIO_ROOT)

build-mcq-dataset:
	uv run python src/utils/build_timeline_mcq_dataset.py --input $(DATA_DIR)/annotations_strong.csv --output $(MCQ_DATASET)

download-audioflamingo:
	uv run python src/utils/setup_audioflamingo.py --repo-url $(AF_REPO_URL) --branch $(AF_BRANCH) --destination $(AF_HOME)

setup-from-scratch: install-dev install-llm install-tracking download-dataset extract-audio build-mcq-dataset download-audioflamingo

eval-mcq-order-random:
	uv sync --extra tracking
	uv run python src/utils/evaluate_mcq_order.py --dataset $(MCQ_DATASET) --model random --results-root $(RESULTS_DIR) $(WAND_ARGS)

eval-mcq-order-openai:
	uv sync --extra llm --extra tracking
	uv run python src/utils/evaluate_mcq_order.py --dataset $(MCQ_DATASET) --model llm-openai --openai-model gpt-4o-mini --temperature 0 --results-root $(RESULTS_DIR) $(WAND_ARGS)

eval-mcq-order-qwen:
	uv sync --extra llm --extra tracking
	uv run python src/utils/evaluate_mcq_order.py \
		--dataset $(MCQ_DATASET) \
		--model llm-qwen \
		--qwen-model-id $(QWEN_MODEL_ID) \
		--local-dtype $(LOCAL_DTYPE) \
		--local-device-map $(LOCAL_DEVICE_MAP) \
		--local-max-new-tokens $(LOCAL_MAX_NEW_TOKENS) \
		--local-temperature $(LOCAL_TEMPERATURE) \
		--local-top-p $(LOCAL_TOP_P) \
		--limit $(LOCAL_LIMIT) \
		--results-root $(RESULTS_DIR) \
		$(WAND_ARGS)

eval-mcq-order-llama:
	uv sync --extra llm --extra tracking
	uv run python src/utils/evaluate_mcq_order.py \
		--dataset $(MCQ_DATASET) \
		--model llm-llama \
		--llama-model-id $(LLAMA_MODEL_ID) \
		--local-dtype $(LOCAL_DTYPE) \
		--local-device-map $(LOCAL_DEVICE_MAP) \
		--local-max-new-tokens $(LOCAL_MAX_NEW_TOKENS) \
		--local-temperature $(LOCAL_TEMPERATURE) \
		--local-top-p $(LOCAL_TOP_P) \
		--limit $(LOCAL_LIMIT) \
		--results-root $(RESULTS_DIR) \
		$(WAND_ARGS)

eval-mcq-order-audioflamingo-smoke:
	uv sync --extra tracking
	uv run python src/utils/evaluate_mcq_order_audioflamingo.py \
		--dataset $(MCQ_DATASET) \
		--audio-root $(AUDIO_ROOT) \
		--audioflamingo-repo $(AF_HOME) \
		--model-base $(AF_MODEL_BASE) \
		--num-gpus $(AF_NUM_GPUS) \
		--batch-size $(AF_BATCH_SIZE) \
		--max-new-tokens $(AF_MAX_NEW_TOKENS) \
		--limit $(AF_SMOKE_LIMIT) \
		--results-root $(RESULTS_DIR) \
		$(WAND_ARGS)

eval-mcq-order-audioflamingo-full:
	uv sync --extra tracking
	uv run python src/utils/evaluate_mcq_order_audioflamingo.py \
		--dataset $(MCQ_DATASET) \
		--audio-root $(AUDIO_ROOT) \
		--audioflamingo-repo $(AF_HOME) \
		--model-base $(AF_MODEL_BASE) \
		--num-gpus $(AF_NUM_GPUS) \
		--batch-size $(AF_BATCH_SIZE) \
		--max-new-tokens $(AF_MAX_NEW_TOKENS) \
		--results-root $(RESULTS_DIR) \
		$(WAND_ARGS)

test:
	uv sync --extra dev
	uv run pytest
