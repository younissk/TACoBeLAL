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
- `MCQ-RELATION`
- `MCQ-SAFETY`

### Build MCQ-ORDER dataset

```bash
make build-mcq-dataset
```

This builds `data/mcq_event_timeline_strong.jsonl` from strong annotations with:
- event filtering by minimum duration
- per-audio filtering by minimum event count
- randomized answer option ordering

### Manual Good/Bad review UI (human-in-the-loop)

Launch a local UI to review one MCQ at a time (audio + question/options):

```bash
make review-mcq-dataset
```

Default behavior:
- semantic dedupe is disabled by default (it can be re-enabled if needed)
- only shows examples with option count in `[4, 6]` (override with `MCQ_REVIEW_MIN_OPTIONS` and `MCQ_REVIEW_MAX_OPTIONS`)
- only two labels: `Good` and `Bad`
- optional manual event-option deletion per question before saving (`Good`/`Bad`) when dedupe misses edge cases
- keyboard shortcuts: `g` -> `Good`, `b` -> `Bad`
- top progress counters: reviewed / good / bad
- compact card-based UI (single-example workflow, audio/timeline/actions on top, question/options below)
- real-time persistence on every click to:
  - `results/mcq-order/review/manual_good_bad_labels.jsonl`
- loads latest per-model run from `results/mcq-order/runs.csv` and adds model pills next to options based on predicted labels (excluding `random` and `*-no-audio` models)

Example override:

```bash
make review-mcq-dataset \
  MCQ_REVIEW_MIN_OPTIONS=5 \
  MCQ_REVIEW_MAX_OPTIONS=6 \
  MCQ_REVIEW_SEMANTIC_DEDUPE=1 \
  MCQ_REVIEW_SIMILARITY_THRESHOLD=0.90 \
  MCQ_REVIEW_PORT=7861
```

#### Methodology: building the curated Good/Bad dataset

We create the curated split with a human-in-the-loop protocol on top of `MCQ-ORDER`:

1. Start from `data/mcq_event_timeline_strong.jsonl`.
2. (Optional) remove semantically near-duplicate `event` options within each question using sentence embeddings (default model `sentence-transformers/all-MiniLM-L6-v2`, cosine threshold `0.88`).
3. Pre-filter candidate questions by option count (default `[4, 6]`).
4. Review one question at a time in a local UI with:
   - original audio playback (`data/audio/<audio_filename>`)
   - question + options + gold answer
   - optional manual deletion of non-gold event options for that question
   - model prediction pills per option, loaded from `results/mcq-order/runs.csv` + each run's `decisions.jsonl`
5. Annotate each example with exactly one binary label:
   - `Good`
   - `Bad`
6. Persist each click immediately (append-only JSONL + flush/fsync) to avoid progress loss if the process/browser crashes.

Label log format (saved to `results/mcq-order/review/manual_good_bad_labels.jsonl`):
- `id` (example id, e.g. `100476.mp3__0`)
- `label` (`good` or `bad`)
- `audio_filename`
- `deleted_option_keys` / `deleted_options` (manual per-question option deletions, if any)
- `timestamp_utc`

Important reproducibility note:
- the label log is append-only, so if an example is re-labeled later, the **latest entry for that `id` is authoritative**.

Optional materialization of final curated files (latest label wins):

```bash
uv run python - <<'PY'
import json
from pathlib import Path

dataset_path = Path("data/mcq_event_timeline_strong.jsonl")
labels_path = Path("results/mcq-order/review/manual_good_bad_labels.jsonl")
out_good = Path("data/mcq_event_timeline_strong_good.jsonl")
out_bad = Path("data/mcq_event_timeline_strong_bad.jsonl")

labels = {}
with labels_path.open("r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        i = row.get("id")
        l = str(row.get("label", "")).strip().lower()
        if i and l in {"good", "bad"}:
            labels[i] = l

good_count = bad_count = 0
with dataset_path.open("r", encoding="utf-8") as src, \
     out_good.open("w", encoding="utf-8") as fg, \
     out_bad.open("w", encoding="utf-8") as fb:
    for line in src:
        row = json.loads(line)
        lbl = labels.get(row.get("id"))
        if lbl == "good":
            fg.write(json.dumps(row, ensure_ascii=True) + "\n")
            good_count += 1
        elif lbl == "bad":
            fb.write(json.dumps(row, ensure_ascii=True) + "\n")
            bad_count += 1

print(f"wrote good={good_count} -> {out_good}")
print(f"wrote bad={bad_count} -> {out_bad}")
PY
```

### Build MCQ-RELATION dataset

```bash
make build-mcq-relation-dataset
```

This builds `data/mcq_relation_timeline_strong.jsonl` from strong annotations with:
- the same row/event/audio filtering pipeline as `MCQ-ORDER`
- pairwise Event A vs Event B questions
- fixed relation options:
  - `A starts before B`
  - `A starts after B`
  - `A overlaps B`
  - `A and B start at the same time`
  - `cannot be determined`
- deterministic option ordering controlled by `--seed`
- default quality caps (`--max-pairs-per-audio 4`, `--max-questions 12000`)
- optional relation-balanced global sampling (`--balance-relations`)

Deterministic rebuild example:

```bash
uv run python src/utils/build_relation_mcq_dataset.py \
  --input data/annotations_strong.csv \
  --output data/mcq_relation_timeline_strong.jsonl \
  --seed 7 \
  --same-start-epsilon 0.05 \
  --include-start-same-time
```

Quality rules implemented:
- exact timestamps are never included in question/options text
- near-start ties are filtered unless the `start at the same time` option is enabled
- quality-over-quantity defaults keep only top-quality pairs deterministically

### Build MCQ-SAFETY dataset

```bash
make build-mcq-safety-dataset
```

This builds `data/mcq_safety_presence_100.jsonl` from strong annotations with:
- exactly 100 audios / 100 questions by default
- exactly 50 `True` and 50 `False` answers
- one binary question per audio:
  - `Is this event in the audio: "{event}"?`
- answer options randomized between `True` and `False` labels for bias control
- negatives sampled from other audios and rejected if normalized text matches target audio events

### Unified benchmark launcher (task/model/samples)

Use one target for all benchmark combinations:

```bash
make run-benchmark \
  BENCH_TASK=mcq-order \
  BENCH_MODEL=random \
  BENCH_SAMPLES=200
```

Swap task/model/sample count without changing scripts:

```bash
make run-benchmark BENCH_TASK=mcq-relation BENCH_MODEL=llm-qwen BENCH_SAMPLES=500
make run-benchmark BENCH_TASK=mcq-order BENCH_MODEL=qwen2-audio BENCH_SAMPLES=100
make run-benchmark BENCH_TASK=mcq-order BENCH_MODEL=audioflamingo BENCH_SAMPLES=100 BENCH_USE_AUDIO=0
make run-benchmark BENCH_TASK=mcq-safety BENCH_MODEL=random BENCH_SAMPLES=100
```

Useful flags:
- `BENCH_SAMPLES=0` for full dataset
- `BENCH_PREPARE_DATA=1` to auto run data prep steps
- `BENCH_INSTALL_DEPS=1` to auto run needed `install-*` targets
- `BENCH_ARGS='--dry-run'` to print planned commands only

Direct CLI equivalent:

```bash
uv run python src/utils/run_benchmark.py --task mcq-order --model qwen2-audio --samples 100
```

Generic SLURM entrypoint:

```bash
sbatch --export=ALL,BENCH_TASK=mcq-relation,BENCH_MODEL=qwen2-audio,BENCH_SAMPLES=500 \
  scripts/slurm/run_benchmark_a40.slurm
```

Dedicated safety-check SLURM launcher (one model per job):

```bash
sbatch --export=ALL,BENCH_MODEL=qwen2-audio scripts/slurm/eval_mcq_safety_a40.slurm
sbatch --export=ALL,BENCH_MODEL=qwen2-5-omni scripts/slurm/eval_mcq_safety_a40.slurm
sbatch --export=ALL,BENCH_MODEL=voxtral scripts/slurm/eval_mcq_safety_a40.slurm
sbatch --export=ALL,BENCH_MODEL=audioflamingo scripts/slurm/eval_mcq_safety_a40.slurm
```

### Underperformance debug workflow (human-in-the-loop)

Build a complete debug bundle (CSVs + plots) from current `MCQ-ORDER` runs:

```bash
make debug-mcq-bundle
```

This writes to `results/mcq-order/debug_bundle/`:
- `latest_runs.csv` / `run_registry_enriched.csv`
- `latest_decisions_long.csv`
- `latest_decisions_with_slices.csv` (adds per-example ambiguity slices)
- `audio_ablation_pairs.csv` (when audio vs no-audio pairs exist)
- `model_baselines.csv` / `answer_type_diagnostics.csv`
- `audioflamingo_id_integrity.csv` (if latest AudioFlamingo runs exist)
- `human_review_queue.csv` and `human_review_queue_topk.csv`
- plots:
  - `plot_accuracy_by_model.png`
  - `plot_invalid_missing_rates.png`
  - `plot_accuracy_by_option_count.png`
  - `plot_prediction_bias.png`
  - `plot_accuracy_by_answer_type.png`
  - `plot_model_vs_baselines.png`
  - `plot_none_calibration.png`
  - `plot_audio_ablation_accuracy.png` (if pairs exist)

Interactive notebook for review sessions:

```bash
jupyter notebook notebooks/mcq_underperformance_debug.ipynb
```

The notebook includes:
- full debug plan/checklist
- auto-built diagnostics from latest runs
- human triage queue with per-example question/options/predictions
- helper to inspect specific examples and play local audio (`data/audio/...`)

### One-command setup from scratch

```bash
make setup-from-scratch
``` 

This target:
- installs local dev + llm + tracking extras (`uv sync --extra dev --extra llm --extra tracking`)
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

### Run text-only open-source LLM baselines (local Hugging Face)

Qwen (A40-friendly default, 100-example smoke run):

```bash
make eval-mcq-order-qwen
```

Llama (requires HF access token for gated Meta checkpoints):

```bash
export HF_TOKEN=your_hf_token
make eval-mcq-order-llama
```

Useful overrides:

```bash
make eval-mcq-order-qwen \
  QWEN_MODEL_ID=Qwen/Qwen2.5-7B-Instruct \
  LOCAL_DTYPE=float16 \
  LOCAL_LIMIT=200 \
  LOCAL_MAX_NEW_TOKENS=16
```

### Cloud run tracking (Weights & Biases)

Set your API key once (or add to `.env`):

```bash
export WANDB_API_KEY=your_wandb_key
```

W&B-enabled targets:

```bash
make eval-mcq-order-random
make eval-mcq-order-openai
make eval-mcq-order-qwen2-audio-smoke
make eval-mcq-order-qwen2-audio-full
make eval-mcq-order-qwen2-audio-no-audio-smoke
make eval-mcq-order-qwen2-audio-no-audio-full
make eval-mcq-order-qwen2-5-omni-smoke
make eval-mcq-order-qwen2-5-omni-full
make eval-mcq-order-qwen2-5-omni-no-audio-smoke
make eval-mcq-order-qwen2-5-omni-no-audio-full
make eval-mcq-order-voxtral-smoke
make eval-mcq-order-voxtral-full
make eval-mcq-order-voxtral-no-audio-smoke
make eval-mcq-order-voxtral-no-audio-full
make eval-mcq-order-audioflamingo-smoke
make eval-mcq-order-audioflamingo-full
make eval-mcq-order-audioflamingo-no-audio-smoke
make eval-mcq-order-audioflamingo-no-audio-full
```

All standard evaluation targets now log to W&B by default.

These log:
- live progress (`accuracy_so_far`, progress fraction, per-step correctness/latency)
- final summary metrics
- rich analysis views:
  - answer-label distribution vs prediction distribution
  - accuracy by answer label
  - accuracy by option count
  - answer-type accuracy (`event` vs `none`)
  - confusion matrix (answer label vs predicted label)
- output artifacts (`decisions.jsonl`, `metrics.json`, `results_table.md`, `analysis.json`, and AF3 raw outputs when applicable)

Optional W&B Make variables:

```bash
make eval-mcq-order-audioflamingo-smoke \
  WANDB_PROJECT=tacobelal \
  WANDB_ENTITY=your_team \
  WANDB_LOG_EVERY=25 \
  WANDB_RUN_NAME=af3_smoke_a40
```

If you need to disable W&B for a one-off run:

```bash
uv run python src/utils/evaluate_mcq_order.py --no-wandb ...
uv run python src/utils/evaluate_mcq_order_audioflamingo.py --no-wandb ...
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

No-audio ablation (same wrapper, text-only prompt path):

```bash
make eval-mcq-order-audioflamingo-no-audio-smoke
```

SLURM template for cluster runs:
- `scripts/slurm/eval_mcq_order_audioflamingo_a40.slurm`
- `scripts/slurm/eval_mcq_order_text_llm_a40.slurm`
- `scripts/slurm/run_benchmark_a40.slurm` (generic launcher: choose task/model/samples via env vars)

### Run audio-capable LALM baseline (Qwen2-Audio)

Smoke test (default 100 examples):

```bash
make eval-mcq-order-qwen2-audio-smoke
```

Full run:

```bash
make eval-mcq-order-qwen2-audio-full
```

Default configuration:
- model base: `Qwen/Qwen2-Audio-7B-Instruct`
- batch size: `2`
- max new tokens: `16`
- dtype: `float16`
- device map: `auto`

Useful overrides:

```bash
make eval-mcq-order-qwen2-audio-smoke \
  QWEN2_AUDIO_MODEL_ID=Qwen/Qwen2-Audio-7B-Instruct \
  QWEN2_AUDIO_BATCH_SIZE=1 \
  QWEN2_AUDIO_DTYPE=bfloat16 \
  QWEN2_AUDIO_SMOKE_LIMIT=200
```

No-audio ablation (same wrapper, no audio input):

```bash
make eval-mcq-order-qwen2-audio-no-audio-smoke
```

SLURM template for cluster runs:
- `scripts/slurm/eval_mcq_order_qwen2_audio_a40.slurm`
- `scripts/slurm/run_benchmark_a40.slurm`

### Run audio-capable LALM baseline (Qwen2.5-Omni)

Smoke test (default 100 examples):

```bash
make eval-mcq-order-qwen2-5-omni-smoke
```

Full run:

```bash
make eval-mcq-order-qwen2-5-omni-full
```

Default configuration:
- model base: `Qwen/Qwen2.5-Omni-7B`
- batch size: `1` (conservative default)
- max new tokens: `16`
- dtype: `float16`
- device map: `auto`

Runtime dependency note:
- this target runs with `uv run --with "transformers>=4.57.0"` so it can use Qwen2.5-Omni APIs without changing the project's pinned base dependencies used by other evaluators.

Useful overrides:

```bash
make eval-mcq-order-qwen2-5-omni-smoke \
  QWEN2_5_OMNI_MODEL_ID=Qwen/Qwen2.5-Omni-7B \
  QWEN2_5_OMNI_BATCH_SIZE=1 \
  QWEN2_5_OMNI_DTYPE=bfloat16 \
  QWEN2_5_OMNI_ATTN=flash_attention_2 \
  QWEN2_5_OMNI_SMOKE_LIMIT=200
```

No-audio ablation (same wrapper, no audio input):

```bash
make eval-mcq-order-qwen2-5-omni-no-audio-smoke
```

SLURM template for cluster runs:
- `scripts/slurm/eval_mcq_order_qwen2_5_omni_a40.slurm`
- `scripts/slurm/run_benchmark_a40.slurm`

### Run audio-capable LALM baseline (Voxtral)

Smoke test (default 100 examples):

```bash
make eval-mcq-order-voxtral-smoke
```

Full run:

```bash
make eval-mcq-order-voxtral-full
```

Default configuration:
- model base: `mistralai/Voxtral-Mini-3B-2507`
- batch size: `2`
- max new tokens: `16`
- dtype: `float16`
- device map: `auto`

Runtime dependency note:
- this target runs with `uv run --with "transformers>=4.57.0"` so it can use Voxtral APIs without changing the project's pinned base dependencies used by other evaluators.

Useful overrides:

```bash
make eval-mcq-order-voxtral-smoke \
  VOXTRAL_MODEL_ID=mistralai/Voxtral-Mini-3B-2507 \
  VOXTRAL_BATCH_SIZE=1 \
  VOXTRAL_DTYPE=bfloat16 \
  VOXTRAL_ATTN=flash_attention_2 \
  VOXTRAL_SMOKE_LIMIT=200
```

No-audio ablation (same wrapper, no audio input):

```bash
make eval-mcq-order-voxtral-no-audio-smoke
```

### No-audio naming convention for LALM wrappers

When running any audio-capable wrapper with `--disable-audio` (or the `*-no-audio-*` Make targets), run artifacts use a `-no-audio` suffix in `model_name`, for example:
- `qwen2-audio-7b-instruct-no-audio`
- `qwen2.5-omni-7b-no-audio`
- `voxtral-mini-3b-2507-no-audio`
- `audio-flamingo-3-no-audio`

SLURM template for cluster runs:
- `scripts/slurm/eval_mcq_order_voxtral_a40.slurm`
- `scripts/slurm/run_benchmark_a40.slurm`

## Task B: Temporal grounding

- Give the model the audio plus a query caption, and ask it to output onset and offset times.

Why this is important: MCQ can also just be shallow guessing, so it should be compared to an LLM guessing the answer, without even looking at the audio.





# TODOs

- [ ] Fragestellung vereinfachung (z.b. Ist dieses event in der Audio erhalten? -> Testet pipeline für sanity check) (ESC 50 https://www.kaggle.com/datasets/mmoreaux/environmental-sound-classification-50)
- [ ] Bessere Qualität von Daten (Durch LLM, Manual tagging) -> Schauen ob text gleich ist (Beep, Beep) -> Vlt Similarites (BERT, CLAP -> Embedding + similiarity)
- [ ] Manual tagging -> Dokumentation
- [ ] Recheck parsing
- [ ] OPTIONAL: Reasonging + Output (sowie ReAct??)


If all is good in the pipeline
- Easier questions and see if the events are really different
- Every quetion should have the same amount of answers
- 
