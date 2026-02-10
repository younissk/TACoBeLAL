# TACoBeLAL

**T**emporally-aligned **A**udio **C**apti**O**ns **BE**nchmark for **L**arge **A**udio **L**anguage models 


TACoBeLAL is a benchmark testing Large Audio Language models (LALMs) on the task of temporally-aligned Audio Question Answering (AQA).


## Task A: MCQ

The following will be assessed through Multiple Choice Question (MCQ) tasks:

- Event ordering (Ask which event happened first)
- Duration comparison (Pick two regions with clearly different lengths. Ask which lasted longer)
- to be continued...

### Task ID

- `MCQ-ORDER`

### Build MCQ-ORDER dataset

```bash
make build-mcq-dataset
```

This builds `data/mcq_event_timeline_strong.jsonl` from strong annotations with:
- event filtering by minimum duration
- per-audio filtering by minimum event count
- randomized answer option ordering

### One-command setup from scratch

```bash
make setup-from-scratch
```

This target:
- installs local dev + llm extras (`uv sync --extra dev --extra llm`)
- downloads TACoBeLAL dataset into `data/`
- extracts `data/audio.zip` to `data/audio/`
- builds `data/mcq_event_timeline_strong.jsonl`
- clones Audio Flamingo 3 branch to `external/audio-flamingo`

Default paths used in later commands:
- dataset: `data/mcq_event_timeline_strong.jsonl`
- audio root: `data/audio`
- Audio Flamingo repo: `external/audio-flamingo`
- results root: `results`

### Run baseline model (Random)

```bash
make eval-mcq-order-random
```

This writes results to:

```text
results/
  mcq-order/
    runs.csv
    random/
      <run_id>/
        decisions.jsonl
        metrics.json
        results_table.md
```

Artifacts:
- `decisions.jsonl`: one line per decision, including predicted label and `is_correct`.
- `metrics.json`: run metadata + aggregate metrics (accuracy, elapsed time, latency).
- `results_table.md`: final table for the run.

### Run text-only LLM baseline (OpenAI via LangChain)

Create `.env` in repo root:

```bash
OPENAI_API_KEY=your_key_here
```

Run:

```bash
make eval-mcq-order-openai
```

Notes:
- This is intentionally text-only (`MCQ-ORDER` without audio input).
- It uses LangChain + OpenAI chat models and evaluates predictions with the same pipeline as `random`.
- Full-dataset LLM evaluation can be expensive; use `--limit` for a quick check first.
- You can also run directly and choose another OpenAI model:

```bash
uv sync --extra llm
uv run python src/utils/evaluate_mcq_order.py \
  --dataset data/mcq_event_timeline_strong.jsonl \
  --model llm-openai \
  --openai-model gpt-4o-mini \
  --temperature 0 \
  --limit 100 \
  --results-root results
```

### Run audio-capable LALM baseline (Audio Flamingo 3)

Smoke test (default 100 examples):

```bash
make eval-mcq-order-audioflamingo-smoke
```

Full run:

```bash
make eval-mcq-order-audioflamingo-full
```

Important:
- `download-audioflamingo` clones only code.
- model checkpoints are pulled by Audio Flamingo at inference time (first run), so run the Audio Flamingo evaluation targets on your compute cluster.

What this wrapper does:
- converts MCQ-ORDER JSONL into Audio Flamingo batch input format
- builds per-example audio symlinks so repeated audio files can still be evaluated per question
- runs AF3 batched inference with `torchrun`
- parses outputs back to option labels and evaluates correctness

Artifacts per run:

```text
results/
  mcq-order/
    audio-flamingo-3/
      <run_id>/
        run_config.json
        metrics.json
        results_table.md
        decisions.jsonl
        raw_model_outputs.jsonl
        workdir/
          audioflamingo_input.json
          audioflamingo_mapping.json
          audio_links/
```

A40 defaults (tuned for stability first):
- `--num-gpus 1`
- `--batch-size 2`
- `--max-new-tokens 16`
- `--think-mode false`

Override Make variables when needed:

```bash
make eval-mcq-order-audioflamingo-smoke AF_NUM_GPUS=1 AF_BATCH_SIZE=3 AF_SMOKE_LIMIT=200
```

SLURM template for cluster runs:
- `scripts/slurm/eval_mcq_order_audioflamingo_a40.slurm`

## Task B: Temporal grounding

- Give the model the audio plus a query caption, and ask it to output onset and offset times.

Why this is important: MCQ can also just be shallow guessing, so it should be compared to an LLM guessing the answer, without even looking at the audio.
