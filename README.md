# TACoBeLAL

**T**emporally-aligned **A**udio **C**apti**O**ns **BE**nchmark for **L**arge **A**udio **L**anguage models 


TACoBeLAL is a benchmark testing Large Audio Language models (LALMs) on the task of temporally-aligned Audio Question Answering (AQA).


## Task A: MCQ

The following will be assessed through Multiple Choice Question (MCQ) tasks:

- Event ordering (Ask which event happened first)
- Duration comparison (Pick two regions with clearly different lengths. Ask which lasted longer)
- to be continued...

## Task B: Temporal grounding

- Give the model the audio plus a query caption, and ask it to output onset and offset times.

Why this is important: MCQ can also just be shallow guessing, so it should be compared to an LLM guessing the answer, without even looking at the audio.