"""Helpers for deterministic synthetic audio rendering."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_FADE_SECONDS = 0.01
TARGET_PEAK = 10 ** (-1.0 / 20.0)


def midi_to_hz(midi_note: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def dbfs_to_amplitude(dbfs: float) -> float:
    """Convert dBFS to linear amplitude."""
    return 10.0 ** (dbfs / 20.0)


def _base_waveform(
    *,
    waveform: str,
    frequency_hz: float,
    num_samples: int,
    sample_rate: int,
) -> np.ndarray:
    t = np.arange(num_samples, dtype=np.float32) / float(sample_rate)
    phase = 2.0 * math.pi * frequency_hz * t
    sine = np.sin(phase).astype(np.float32)

    if waveform == "sine":
        return sine
    if waveform == "square":
        return np.where(sine >= 0.0, 1.0, -1.0).astype(np.float32)
    if waveform == "triangle":
        return (2.0 / math.pi * np.arcsin(sine)).astype(np.float32)

    raise ValueError(f"Unsupported waveform '{waveform}'.")


def _apply_fade(samples: np.ndarray, *, sample_rate: int, fade_seconds: float) -> np.ndarray:
    fade_samples = int(round(fade_seconds * sample_rate))
    fade_samples = max(0, min(fade_samples, samples.shape[0] // 2))
    if fade_samples == 0:
        return samples

    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, fade_samples, dtype=np.float32))
    samples[:fade_samples] *= ramp
    samples[-fade_samples:] *= ramp[::-1]
    return samples


def synthesize_tone(
    *,
    waveform: str,
    frequency_hz: float,
    duration_seconds: float,
    level_dbfs: float,
    sample_rate: int,
    fade_seconds: float = DEFAULT_FADE_SECONDS,
) -> np.ndarray:
    """Render a single tone with the requested waveform, pitch, and level."""
    num_samples = max(1, int(round(duration_seconds * sample_rate)))
    signal = _base_waveform(
        waveform=waveform,
        frequency_hz=frequency_hz,
        num_samples=num_samples,
        sample_rate=sample_rate,
    )
    signal = signal * dbfs_to_amplitude(level_dbfs)
    return _apply_fade(signal.astype(np.float32), sample_rate=sample_rate, fade_seconds=fade_seconds)


def render_timeline(
    *,
    events: list[dict[str, float | str]],
    total_duration_seconds: float,
    sample_rate: int,
    fade_seconds: float = DEFAULT_FADE_SECONDS,
) -> np.ndarray:
    """Render a mono waveform from a list of onset-timed tone events."""
    total_samples = max(1, int(round(total_duration_seconds * sample_rate)))
    waveform = np.zeros(total_samples, dtype=np.float32)

    for event in events:
        onset = float(event["onset"])
        duration = float(event["duration"])
        tone = synthesize_tone(
            waveform=str(event["waveform"]),
            frequency_hz=float(event["pitch_hz"]),
            duration_seconds=duration,
            level_dbfs=float(event["dbfs"]),
            sample_rate=sample_rate,
            fade_seconds=fade_seconds,
        )
        start = max(0, int(round(onset * sample_rate)))
        end = min(total_samples, start + tone.shape[0])
        waveform[start:end] += tone[: end - start]

    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 0.0:
        waveform *= TARGET_PEAK / peak

    return waveform.astype(np.float32)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read mono WAV data and return float32 samples plus sample rate."""
    samples, sample_rate = sf.read(str(path), dtype="float32")
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)
    return samples.astype(np.float32), int(sample_rate)


def resample_audio(samples: np.ndarray, *, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample audio with linear interpolation for small bundled assets."""
    if source_rate == target_rate:
        return samples.astype(np.float32)
    if samples.size == 0:
        return samples.astype(np.float32)

    duration_seconds = samples.shape[0] / float(source_rate)
    target_size = max(1, int(round(duration_seconds * target_rate)))
    source_times = np.linspace(0.0, duration_seconds, samples.shape[0], endpoint=False, dtype=np.float32)
    target_times = np.linspace(0.0, duration_seconds, target_size, endpoint=False, dtype=np.float32)
    return np.interp(target_times, source_times, samples).astype(np.float32)


def normalize_peak(samples: np.ndarray, *, target_peak: float = TARGET_PEAK) -> np.ndarray:
    """Scale samples to the requested peak while preserving relative shape."""
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak <= 0.0:
        return samples.astype(np.float32)
    return (samples * (target_peak / peak)).astype(np.float32)


def render_clip_timeline(
    *,
    clips: list[dict[str, float | np.ndarray]],
    total_duration_seconds: float,
    sample_rate: int,
) -> np.ndarray:
    """Render a mono waveform from onset-timed sample clips."""
    total_samples = max(1, int(round(total_duration_seconds * sample_rate)))
    waveform = np.zeros(total_samples, dtype=np.float32)

    for clip in clips:
        samples = np.asarray(clip["samples"], dtype=np.float32)
        onset = float(clip["onset"])
        start = max(0, int(round(onset * sample_rate)))
        end = min(total_samples, start + samples.shape[0])
        waveform[start:end] += samples[: end - start]

    return normalize_peak(waveform)


def write_wav(path: Path, *, samples: np.ndarray, sample_rate: int) -> None:
    """Write a mono WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples.astype(np.float32), sample_rate, subtype="PCM_16")
