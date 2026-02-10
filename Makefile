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
	eval-mcq-order-qwen2-audio-smoke \
	eval-mcq-order-qwen2-audio-full \
	eval-mcq-order-qwen2-5-omni-smoke \
	eval-mcq-order-qwen2-5-omni-full \
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

QWEN2_AUDIO_MODEL_ID ?= Qwen/Qwen2-Audio-7B-Instruct
QWEN2_AUDIO_BATCH_SIZE ?= 2
QWEN2_AUDIO_MAX_NEW_TOKENS ?= 16
QWEN2_AUDIO_DTYPE ?= float16
QWEN2_AUDIO_DEVICE_MAP ?= auto
QWEN2_AUDIO_SMOKE_LIMIT ?= 100

QWEN2_5_OMNI_MODEL_ID ?= Qwen/Qwen2.5-Omni-7B
QWEN2_5_OMNI_BATCH_SIZE ?= 1
QWEN2_5_OMNI_MAX_NEW_TOKENS ?= 16
QWEN2_5_OMNI_DTYPE ?= float16
QWEN2_5_OMNI_DEVICE_MAP ?= auto
QWEN2_5_OMNI_ATTN ?=
QWEN2_5_OMNI_SMOKE_LIMIT ?= 100
QWEN2_5_OMNI_TRANSFORMERS ?= transformers>=4.57.0
QWEN2_5_OMNI_ATTN_ARG := $(if $(QWEN2_5_OMNI_ATTN),--attn-implementation $(QWEN2_5_OMNI_ATTN),)

QWEN_MODEL_ID ?= Qwen/Qwen2.5-7B-Instruct
LLAMA_MODEL_ID ?= meta-llama/Llama-3.1-8B-Instruct
LOCAL_DTYPE ?= float16
LOCAL_DEVICE_MAP ?= auto
LOCAL_MAX_NEW_TOKENS ?= 16
LOCAL_TEMPERATURE ?= 0.0
LOCAL_TOP_P ?= 1.0
LOCAL_LIMIT ?= 100
LIMIT_ARG := $(if $(LOCAL_LIMIT),--limit $(LOCAL_LIMIT),)

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
		$(LIMIT_ARG) \
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
		$(LIMIT_ARG) \
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

eval-mcq-order-qwen2-audio-smoke:
	uv sync --extra tracking
	uv run python src/utils/evaluate_mcq_order_qwen2_audio.py \
		--dataset $(MCQ_DATASET) \
		--audio-root $(AUDIO_ROOT) \
		--model-base $(QWEN2_AUDIO_MODEL_ID) \
		--batch-size $(QWEN2_AUDIO_BATCH_SIZE) \
		--max-new-tokens $(QWEN2_AUDIO_MAX_NEW_TOKENS) \
		--dtype $(QWEN2_AUDIO_DTYPE) \
		--device-map $(QWEN2_AUDIO_DEVICE_MAP) \
		--limit $(QWEN2_AUDIO_SMOKE_LIMIT) \
		--results-root $(RESULTS_DIR) \
		$(WAND_ARGS)

eval-mcq-order-qwen2-audio-full:
	uv sync --extra tracking
	uv run python src/utils/evaluate_mcq_order_qwen2_audio.py \
		--dataset $(MCQ_DATASET) \
		--audio-root $(AUDIO_ROOT) \
		--model-base $(QWEN2_AUDIO_MODEL_ID) \
		--batch-size $(QWEN2_AUDIO_BATCH_SIZE) \
		--max-new-tokens $(QWEN2_AUDIO_MAX_NEW_TOKENS) \
		--dtype $(QWEN2_AUDIO_DTYPE) \
		--device-map $(QWEN2_AUDIO_DEVICE_MAP) \
		--results-root $(RESULTS_DIR) \
		$(WAND_ARGS)

eval-mcq-order-qwen2-5-omni-smoke:
	uv sync --extra tracking
	uv run --with "$(QWEN2_5_OMNI_TRANSFORMERS)" python src/utils/evaluate_mcq_order_qwen2_5_omni.py \
		--dataset $(MCQ_DATASET) \
		--audio-root $(AUDIO_ROOT) \
		--model-base $(QWEN2_5_OMNI_MODEL_ID) \
		--batch-size $(QWEN2_5_OMNI_BATCH_SIZE) \
		--max-new-tokens $(QWEN2_5_OMNI_MAX_NEW_TOKENS) \
		--dtype $(QWEN2_5_OMNI_DTYPE) \
		--device-map $(QWEN2_5_OMNI_DEVICE_MAP) \
		$(QWEN2_5_OMNI_ATTN_ARG) \
		--limit $(QWEN2_5_OMNI_SMOKE_LIMIT) \
		--results-root $(RESULTS_DIR) \
		$(WAND_ARGS)

eval-mcq-order-qwen2-5-omni-full:
	uv sync --extra tracking
	uv run --with "$(QWEN2_5_OMNI_TRANSFORMERS)" python src/utils/evaluate_mcq_order_qwen2_5_omni.py \
		--dataset $(MCQ_DATASET) \
		--audio-root $(AUDIO_ROOT) \
		--model-base $(QWEN2_5_OMNI_MODEL_ID) \
		--batch-size $(QWEN2_5_OMNI_BATCH_SIZE) \
		--max-new-tokens $(QWEN2_5_OMNI_MAX_NEW_TOKENS) \
		--dtype $(QWEN2_5_OMNI_DTYPE) \
		--device-map $(QWEN2_5_OMNI_DEVICE_MAP) \
		$(QWEN2_5_OMNI_ATTN_ARG) \
		--results-root $(RESULTS_DIR) \
		$(WAND_ARGS)

test:
	uv sync --extra dev
	uv run pytest
