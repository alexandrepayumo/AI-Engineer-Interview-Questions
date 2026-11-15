"""Exercise 03 - Mel filterbank and log-mel spectrogram from a synthetic sine wave.

TASK
----
Implement these functions (numpy only):

    hz_to_mel(f) -> np.ndarray | float      mel = 2595 * log10(1 + f / 700)   (HTK formula)
    mel_to_hz(m) -> np.ndarray | float      inverse of the above

    frame_signal(x, frame_len, hop) -> np.ndarray
        Slice a 1-D signal into overlapping frames, shape (n_frames, frame_len) with
        n_frames = 1 + (len(x) - frame_len) // hop. ValueError if len(x) < frame_len.

    mel_filterbank(n_mels, n_fft, sr, fmin=0.0, fmax=None) -> np.ndarray
        Triangular filters, shape (n_mels, n_fft // 2 + 1). fmax defaults to sr / 2.
        Take n_mels + 2 points equally spaced in mel between hz_to_mel(fmin) and
        hz_to_mel(fmax), convert back to Hz, and for filter m use points (left, centre,
        right) = (hz[m], hz[m + 1], hz[m + 2]). The weight at FFT bin frequency
        f_k = k * sr / n_fft is max(0, min((f_k - left) / (centre - left),
        (right - f_k) / (right - centre))). Peak weight is therefore at most 1.

    power_spectrogram(x, n_fft, hop) -> np.ndarray
        Frame the signal with frame_len = n_fft, multiply each frame by a periodic Hann
        window w[n] = 0.5 - 0.5 * cos(2 * pi * n / n_fft), take np.fft.rfft and return
        |X|^2, shape (n_frames, n_fft // 2 + 1).

    log_mel_spectrogram(x, sr, n_fft=512, hop=160, n_mels=40) -> np.ndarray
        power_spectrogram @ mel_filterbank.T, then log10(max(energy, 1e-10)).
        Shape (n_frames, n_mels).

HINTS
-----
- With sr = 16000 and n_fft = 512 the bin spacing is 31.25 Hz, so a 1000 Hz sine lands
  exactly on bin 32.
- A 25 ms window at 16 kHz is 400 samples and a 10 ms hop is 160 samples. Whisper uses
  these values with n_fft = 400 and 80 mel bins.
- Doubling amplitude multiplies power by 4, which adds log10(4) = 0.602 to log-mel values.

WHAT YOU SHOULD LEARN
---------------------
- How raw audio becomes a 2-D "image" that a ViT-like or conformer encoder can read.
- Why the mel scale gives more resolution at low frequencies, where speech information lives.
- How many frames per second and how many values per frame the encoder has to process.
"""
from __future__ import annotations

import numpy as np


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=float) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=float) / 2595.0) - 1.0)


def frame_signal(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    if len(x) < frame_len:
        raise ValueError("signal shorter than one frame")
    n_frames = 1 + (len(x) - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n_frames)[:, None]
    return x[idx]


def mel_filterbank(n_mels: int, n_fft: int, sr: int, fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    if fmax is None:
        fmax = sr / 2
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mel_pts)
    freqs = np.arange(n_fft // 2 + 1) * sr / n_fft
    fb = np.zeros((n_mels, len(freqs)))
    for m in range(n_mels):
        left, centre, right = hz[m], hz[m + 1], hz[m + 2]
        up = (freqs - left) / (centre - left)       # rising edge
        down = (right - freqs) / (right - centre)   # falling edge
        fb[m] = np.maximum(0.0, np.minimum(up, down))
    return fb


def power_spectrogram(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    frames = frame_signal(x, n_fft, hop)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    spec = np.fft.rfft(frames * window, axis=1)
    return np.abs(spec) ** 2


def log_mel_spectrogram(x: np.ndarray, sr: int, n_fft: int = 512, hop: int = 160, n_mels: int = 40) -> np.ndarray:
    power = power_spectrogram(x, n_fft, hop)
    fb = mel_filterbank(n_mels, n_fft, sr)
    energy = power @ fb.T
    return np.log10(np.maximum(energy, 1e-10))


def _run_tests() -> None:
    # mel scale: 1000 Hz is about 1000 mel, and the mapping round-trips
    assert abs(float(hz_to_mel(1000.0)) - 1000.0) < 1.0
    f = np.array([0.0, 100.0, 440.0, 4000.0, 8000.0])
    assert np.allclose(mel_to_hz(hz_to_mel(f)), f)
    # compression: equal mel steps cover wider and wider Hz ranges
    steps = np.diff(mel_to_hz(np.linspace(0, 2840, 6)))
    assert np.all(np.diff(steps) > 0)

    # framing
    x = np.arange(10, dtype=float)
    fr = frame_signal(x, 4, 2)
    assert fr.shape == (4, 4) and np.array_equal(fr[1], [2, 3, 4, 5]) and np.array_equal(fr[3], [6, 7, 8, 9])
    try:
        frame_signal(x, 11, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")

    # filterbank structure
    sr, n_fft = 16000, 512
    fb = mel_filterbank(40, n_fft, sr)
    assert fb.shape == (40, 257)
    assert fb.min() >= 0.0 and fb.max() <= 1.0 + 1e-12
    assert (fb.sum(axis=1) > 0).all()                     # no empty filter
    peaks = fb.argmax(axis=1)
    assert np.all(np.diff(peaks) >= 0)                    # centres increase with the filter index
    widths = (fb > 0).sum(axis=1)
    assert widths[-1] > 3 * widths[0]                     # high-frequency filters are much wider in Hz

    # a 1 s, 1000 Hz sine
    t = np.arange(sr) / sr
    sine = np.sin(2 * np.pi * 1000.0 * t)
    power = power_spectrogram(sine, n_fft, 160)
    assert power.shape == (97, 257)                       # 1 + (16000 - 512) // 160
    assert np.all(power.argmax(axis=1) == 32)             # 1000 / 31.25 = bin 32

    lm = log_mel_spectrogram(sine, sr)
    assert lm.shape == (97, 40)
    best = int(lm.mean(axis=0).argmax())
    centre_hz = fb[best].argmax() * sr / n_fft
    assert abs(centre_hz - 1000.0) < 150.0                # the loudest mel band is near 1 kHz
    # energy far from the tone is orders of magnitude lower
    assert lm[:, best].mean() - lm[:, 35].mean() > 4.0

    # doubling the amplitude adds log10(4) in the bands that carry the tone
    lm2 = log_mel_spectrogram(2.0 * sine, sr)
    assert abs((lm2[:, best] - lm[:, best]).mean() - np.log10(4.0)) < 1e-6

    # silence hits the floor
    silent = log_mel_spectrogram(np.zeros(sr), sr)
    assert np.all(silent == -10.0)

    # sizes the encoder sees: 30 s at a 10 ms hop is about 3000 frames (2998 without padding)
    frames_30s = 1 + (30 * sr - 400) // 160
    assert frames_30s == 2998


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
