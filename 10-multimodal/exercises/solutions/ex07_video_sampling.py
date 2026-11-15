"""Exercise 07 - Video frame sampling under a token budget.

TASK
----
Implement (numpy and math only):

    uniform_indices(n_frames, k) -> list[int]
        k frame indices that sit at the centres of k equal segments:
        int((i + 0.5) * n_frames / k) for i in 0..k-1. If k >= n_frames return
        list(range(n_frames)). ValueError if n_frames < 1 or k < 1.

    video_tokens(duration_s, fps, tokens_per_frame, audio_tokens_per_s=0.0) -> int
        Total tokens: frames * tokens_per_frame + round(duration_s * audio_tokens_per_s),
        where frames = max(1, int(duration_s * fps)).

    max_fps_for_budget(duration_s, tokens_per_frame, budget, audio_tokens_per_s=0.0,
                       max_fps=1.0) -> tuple[float, int]
        The highest sampling rate (capped at max_fps) whose total cost fits the budget.
        audio = round(duration_s * audio_tokens_per_s); frames = (budget - audio) //
        tokens_per_frame; fps = min(max_fps, frames / duration_s). Return (fps, n_frames)
        with n_frames = min(frames, int(duration_s * max_fps)). ValueError if not even one
        frame fits.

    select_keyframes(feats, threshold, max_frames) -> list[int]
        feats has shape (n, d), one feature vector per frame. Always select frame 0. Then
        scan i = 1..n-1 and select i whenever the L1 distance between feats[i] and the most
        recently selected frame's features exceeds threshold. If more than max_frames were
        selected, keep max_frames of them by taking uniform_indices(len(selected),
        max_frames) positions within the selected list.

    two_pass_tokens(duration_s, coarse_fps, n_windows, window_s, fine_fps,
                    tokens_per_frame) -> int
        Hierarchical cost: a coarse pass over the whole video plus n_windows zoom-ins of
        window_s seconds at fine_fps. Each pass uses frames = max(1, int(seconds * fps)),
        cost frames * tokens_per_frame, no audio.

HINTS
-----
- Centre-of-segment sampling never picks frame 0 or the last frame by accident of
  rounding, and spreads evenly even when k does not divide n.
- An hour at 1 fps and 258 tokens per frame is 3600 * 258 = 928,800 tokens before audio.

WHAT YOU SHOULD LEARN
---------------------
- The budget arithmetic that decides whether a long video fits in context at all.
- Why uniform sampling misses short events and what keyframe selection buys you.
- Why a cheap coarse pass plus targeted high-fps windows beats dense sampling everywhere.
"""
from __future__ import annotations

import numpy as np


def uniform_indices(n_frames: int, k: int) -> list[int]:
    if n_frames < 1 or k < 1:
        raise ValueError("n_frames and k must be >= 1")
    if k >= n_frames:
        return list(range(n_frames))
    return [int((i + 0.5) * n_frames / k) for i in range(k)]


def video_tokens(duration_s: float, fps: float, tokens_per_frame: int, audio_tokens_per_s: float = 0.0) -> int:
    frames = max(1, int(duration_s * fps))
    return frames * tokens_per_frame + round(duration_s * audio_tokens_per_s)


def max_fps_for_budget(duration_s: float, tokens_per_frame: int, budget: int,
                       audio_tokens_per_s: float = 0.0, max_fps: float = 1.0) -> tuple[float, int]:
    audio = round(duration_s * audio_tokens_per_s)
    frames = (budget - audio) // tokens_per_frame
    if frames < 1:
        raise ValueError("budget too small for a single frame")
    fps = min(max_fps, frames / duration_s)
    return fps, min(frames, int(duration_s * max_fps))


def select_keyframes(feats: np.ndarray, threshold: float, max_frames: int) -> list[int]:
    selected = [0]
    for i in range(1, len(feats)):
        if np.abs(feats[i] - feats[selected[-1]]).sum() > threshold:
            selected.append(i)
    if len(selected) > max_frames:
        pos = uniform_indices(len(selected), max_frames)   # thin the keyframes evenly
        selected = [selected[p] for p in pos]
    return selected


def two_pass_tokens(duration_s: float, coarse_fps: float, n_windows: int, window_s: float,
                    fine_fps: float, tokens_per_frame: int) -> int:
    coarse = max(1, int(duration_s * coarse_fps)) * tokens_per_frame
    fine = n_windows * max(1, int(window_s * fine_fps)) * tokens_per_frame
    return coarse + fine


def _run_tests() -> None:
    # uniform sampling
    assert uniform_indices(10, 5) == [1, 3, 5, 7, 9]
    assert uniform_indices(10, 3) == [1, 5, 8]
    assert uniform_indices(4, 10) == [0, 1, 2, 3]
    idx = uniform_indices(1000, 7)
    assert len(idx) == 7 and idx == sorted(idx) and idx[0] > 0 and idx[-1] < 999
    gaps = np.diff(idx)
    assert gaps.max() - gaps.min() <= 1                    # evenly spread
    for bad in [(0, 3), (3, 0)]:
        try:
            uniform_indices(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")

    # budget arithmetic: one hour at 1 fps
    assert video_tokens(3600, 1.0, 258) == 928_800
    assert video_tokens(3600, 1.0, 258, audio_tokens_per_s=32) == 928_800 + 115_200
    assert video_tokens(0.2, 1.0, 258) == 258              # at least one frame
    assert video_tokens(60, 0.5, 70) == 30 * 70

    # the hour does not fit in 200k tokens at 1 fps, so the sampler must slow down
    fps, n = max_fps_for_budget(3600, 258, 200_000)
    assert n == 775 and abs(fps - 775 / 3600) < 1e-12
    assert video_tokens(3600, fps, 258) <= 200_000
    # with room to spare the cap applies
    fps2, n2 = max_fps_for_budget(60, 258, 200_000)
    assert fps2 == 1.0 and n2 == 60
    # audio eats into the frame budget
    _, n_audio = max_fps_for_budget(3600, 258, 200_000, audio_tokens_per_s=32)
    assert n_audio == (200_000 - 115_200) // 258 == 328
    try:
        max_fps_for_budget(3600, 258, 100)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")

    # keyframes: three scenes of 10 frames each, tiny jitter inside a scene
    rng = np.random.default_rng(3)
    scene_means = np.array([[0.0] * 4, [5.0] * 4, [-3.0] * 4])
    feats = np.vstack([m + 0.01 * rng.normal(size=(10, 4)) for m in scene_means])
    assert select_keyframes(feats, threshold=1.0, max_frames=10) == [0, 10, 20]
    # a tiny threshold selects every frame, which is then thinned to max_frames
    all_sel = select_keyframes(feats, threshold=0.0, max_frames=100)
    assert all_sel == list(range(30))
    thin = select_keyframes(feats, threshold=0.0, max_frames=5)
    assert len(thin) == 5 and thin == sorted(thin)
    # a static video yields exactly one keyframe, which uniform sampling would never do
    static = np.ones((50, 4))
    assert select_keyframes(static, threshold=0.1, max_frames=8) == [0]

    # a short event between uniform samples is missed, keyframes catch it
    flash = np.zeros((100, 4))
    flash[40:42] = 9.0
    assert 40 not in uniform_indices(100, 10) and 41 not in uniform_indices(100, 10)
    assert 40 in select_keyframes(flash, threshold=1.0, max_frames=10)

    # hierarchical cost: coarse pass at 0.1 fps plus three 20 s windows at 2 fps
    assert two_pass_tokens(3600, 0.1, 3, 20, 2.0, 258) == (360 + 3 * 40) * 258 == 123_840
    assert two_pass_tokens(3600, 0.1, 3, 20, 2.0, 258) < video_tokens(3600, 1.0, 258) / 7


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
