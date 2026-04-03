# Do LALMs reliably solve very easy temporal audio events? 

The answer from our temporal trivial suite is no. Performance is near random across multiple models, even on intentionally simple order/count/gap tasks.

## This repo includes:

- Temporal trivial benchmarks (7 families, 100 samples each)
- Safety benchmark (to show the pipeline runs end-to-end)
- Core runner + model evaluators (random, LLM Qwen, Qwen2-Audio, Audio Flamingo)
- A short SLURM sweep for the full temporal suite

## Temporal Trivial Suite

The suite isolates one temporal property per task, with large separations and minimal confounds:

- Pitch order
- Loudness order
- Duration order
- Count beeps
- Gap length
- Pitch pattern
- Dog vs car order (bundled synthetic assets)

Build all temporal datasets:

```bash
make \
  build-mcq-synth-pitch-order-trivial \
  build-mcq-synth-loudness-order-trivial \
  build-mcq-synth-duration-order-trivial \
  build-mcq-synth-count-beeps-trivial \
  build-mcq-synth-gap-trivial \
  build-mcq-synth-pattern-pitch-trivial \
  build-mcq-synth-dog-car-order-trivial \
  TRIVIAL_SYNTH_SCENES=100
```

Run the sweep on SLURM:

```bash
sbatch scripts/slurm/eval_mcq_synth_temporal_suite_a40.slurm
```

## Simple HTML Viewer

For a static GitHub Pages view of example audio + model answers, open:
`docs/index.html` (it reads datasets and runs from `data/` and `results/`).

## Safety Benchmark

The safety dataset builder remains for pipeline validation:

```bash
make build-mcq-safety-dataset
```

## Results Summary (Temporal Suite)

We see near-random performance across models. The combined table below summarizes the
trivial suite runs.

| Benchmark | Model | Examples | Correct | Accuracy | Elapsed (s) | Avg latency (ms) | Run ID |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| MCQ-SYNTH-COUNT-BEEPS-TRIVIAL | audio-flamingo-3 | 100 | 37 | 0.37 | 58.12 | 581.21 | 20260325_082040 |
| MCQ-SYNTH-COUNT-BEEPS-TRIVIAL | llm-qwen | 100 | 32 | 0.32 | 15.28 | 152.74 | 20260325_081904 |
| MCQ-SYNTH-COUNT-BEEPS-TRIVIAL | qwen2-audio-7b-instruct | 100 | 31 | 0.31 | 12.27 | 122.69 | 20260325_081946 |
| MCQ-SYNTH-COUNT-BEEPS-TRIVIAL | random | 100 | 39 | 0.39 | 0.00 | 0.00 | 20260325_081823 |
| MCQ-SYNTH-DOG-CAR-ORDER-TRIVIAL | audio-flamingo-3 | 100 | 50 | 0.50 | 53.81 | 538.14 | 20260325_083134 |
| MCQ-SYNTH-DOG-CAR-ORDER-TRIVIAL | llm-qwen | 100 | 50 | 0.50 | 15.30 | 152.96 | 20260325_082939 |
| MCQ-SYNTH-DOG-CAR-ORDER-TRIVIAL | qwen2-audio-7b-instruct | 100 | 45 | 0.45 | 11.88 | 118.76 | 20260325_083026 |
| MCQ-SYNTH-DOG-CAR-ORDER-TRIVIAL | random | 100 | 54 | 0.54 | 0.00 | 0.00 | 20260325_082914 |
| MCQ-SYNTH-DURATION-ORDER-TRIVIAL | audio-flamingo-3 | 100 | 50 | 0.50 | 53.70 | 536.97 | 20260325_081705 |
| MCQ-SYNTH-DURATION-ORDER-TRIVIAL | llm-qwen | 100 | 50 | 0.50 | 15.18 | 151.75 | 20260325_081521 |
| MCQ-SYNTH-DURATION-ORDER-TRIVIAL | qwen2-audio-7b-instruct | 100 | 47 | 0.47 | 11.82 | 118.25 | 20260325_081617 |
| MCQ-SYNTH-DURATION-ORDER-TRIVIAL | random | 100 | 64 | 0.64 | 0.00 | 0.00 | 20260325_081454 |
| MCQ-SYNTH-GAP-TRIVIAL | audio-flamingo-3 | 100 | 50 | 0.50 | 77.06 | 770.61 | 20260325_082336 |
| MCQ-SYNTH-GAP-TRIVIAL | llm-qwen | 100 | 54 | 0.54 | 15.07 | 150.63 | 20260325_082218 |
| MCQ-SYNTH-GAP-TRIVIAL | qwen2-audio-7b-instruct | 100 | 71 | 0.71 | 11.81 | 118.08 | 20260325_082248 |
| MCQ-SYNTH-GAP-TRIVIAL | random | 100 | 59 | 0.59 | 0.00 | 0.00 | 20260325_082201 |
| MCQ-SYNTH-LOUDNESS-ORDER-TRIVIAL | audio-flamingo-3 | 100 | 50 | 0.50 | 52.83 | 528.30 | 20260325_081328 |
| MCQ-SYNTH-LOUDNESS-ORDER-TRIVIAL | llm-qwen | 100 | 50 | 0.50 | 15.32 | 153.12 | 20260325_081203 |
| MCQ-SYNTH-LOUDNESS-ORDER-TRIVIAL | qwen2-audio-7b-instruct | 100 | 55 | 0.55 | 11.74 | 117.41 | 20260325_081247 |
| MCQ-SYNTH-LOUDNESS-ORDER-TRIVIAL | random | 100 | 46 | 0.46 | 0.00 | 0.00 | 20260325_081135 |
| MCQ-SYNTH-PATTERN-PITCH-TRIVIAL | audio-flamingo-3 | 100 | 55 | 0.55 | 53.41 | 534.07 | 20260325_082745 |
| MCQ-SYNTH-PATTERN-PITCH-TRIVIAL | llm-qwen | 100 | 55 | 0.55 | 15.16 | 151.59 | 20260325_082549 |
| MCQ-SYNTH-PATTERN-PITCH-TRIVIAL | qwen2-audio-7b-instruct | 100 | 45 | 0.45 | 11.54 | 115.37 | 20260325_082635 |
| MCQ-SYNTH-PATTERN-PITCH-TRIVIAL | random | 100 | 60 | 0.60 | 0.00 | 0.00 | 20260325_082532 |
| MCQ-SYNTH-PITCH-ORDER-TRIVIAL | audio-flamingo-3 | 100 | 50 | 0.50 | 66.69 | 666.93 | 20260325_080950 |
| MCQ-SYNTH-PITCH-ORDER-TRIVIAL | llm-qwen | 100 | 54 | 0.54 | 25.76 | 257.56 | 20260325_080733 |
| MCQ-SYNTH-PITCH-ORDER-TRIVIAL | qwen2-audio-7b-instruct | 100 | 46 | 0.46 | 13.15 | 131.45 | 20260325_080823 |
| MCQ-SYNTH-PITCH-ORDER-TRIVIAL | random | 100 | 47 | 0.47 | 0.00 | 0.00 | 20260325_080720 |

**Conclusion:** On these intentionally easy temporal tasks, LALM performance is near random, suggesting weak temporal reasoning over audio events even when the perceptual differences are large.
