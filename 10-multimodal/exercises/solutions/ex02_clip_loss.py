"""Exercise 02 - CLIP-style symmetric contrastive loss, SigLIP loss and zero-shot classification.

TASK
----
Implement these functions (numpy only):

    l2_normalize(x) -> np.ndarray
        Normalise each row of a 2-D array to unit L2 norm (add 1e-12 to the norm).

    clip_loss(img, txt, temperature=0.07) -> float
        img and txt have shape (N, d); row i of each is a matched pair. L2-normalise both,
        compute logits = img @ txt.T / temperature, and return the mean of the
        image-to-text cross-entropy (rows) and the text-to-image cross-entropy (columns),
        where the correct class for row/column i is i. Must be numerically stable for
        temperature = 1e-3 (use a log-sum-exp with the max subtracted).

    siglip_loss(img, txt, t=10.0, b=-10.0) -> float
        Pairwise sigmoid loss. Normalise both, z_ij = +1 if i == j else -1,
        logit_ij = t * <img_i, txt_j> + b, loss = sum_ij -log(sigmoid(z_ij * logit_ij)) / N.
        Use np.logaddexp(0, -a) for -log(sigmoid(a)).

    zero_shot_classify(img, class_txt) -> np.ndarray
        For each image row, the index of the class text embedding with the highest cosine
        similarity. Shapes (N, d) and (C, d).

    recall_at_k(img, txt, k) -> float
        Image-to-text retrieval. For each image i, is its matching text i among the k texts
        with the highest cosine similarity? Return the fraction of images for which it is.

HINTS
-----
- The stubbed helper _log_softmax(logits) (row-wise, max-subtracted) is optional but handy.
- The N diagonal logits are the positives. Row-wise log-softmax gives the image-to-text
  loss, column-wise (transpose) gives text-to-image.
- When all embeddings are identical every logit is equal, so the loss is exactly log(N).
- When embeddings are orthonormal and temperature is 1, logits are the identity matrix.

WHAT YOU SHOULD LEARN
---------------------
- Why the loss is symmetric and what the batch contributes (N - 1 negatives per pair).
- What temperature does: smaller values sharpen the softmax and punish hard negatives more.
- How the SigLIP pairwise loss removes the batch-wide softmax normalisation.
- Why zero-shot classification is just nearest-neighbour search against prompt embeddings.
"""
from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=1, keepdims=True)  # subtract the max so exp() cannot overflow
    shifted = logits - m
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def clip_loss(img: np.ndarray, txt: np.ndarray, temperature: float = 0.07) -> float:
    logits = l2_normalize(img) @ l2_normalize(txt).T / temperature
    n = logits.shape[0]
    idx = np.arange(n)
    i2t = -_log_softmax(logits)[idx, idx].mean()      # each image picks its text among N
    t2i = -_log_softmax(logits.T)[idx, idx].mean()    # each text picks its image among N
    return float((i2t + t2i) / 2)


def siglip_loss(img: np.ndarray, txt: np.ndarray, t: float = 10.0, b: float = -10.0) -> float:
    n = img.shape[0]
    sims = l2_normalize(img) @ l2_normalize(txt).T
    z = 2 * np.eye(n) - 1                              # +1 on the diagonal, -1 elsewhere
    a = z * (t * sims + b)
    return float(np.logaddexp(0.0, -a).sum() / n)


def zero_shot_classify(img: np.ndarray, class_txt: np.ndarray) -> np.ndarray:
    sims = l2_normalize(img) @ l2_normalize(class_txt).T
    return sims.argmax(axis=1)


def recall_at_k(img: np.ndarray, txt: np.ndarray, k: int) -> float:
    sims = l2_normalize(img) @ l2_normalize(txt).T
    hits = 0
    for i in range(sims.shape[0]):
        top = np.argsort(-sims[i], kind="stable")[:k]
        hits += int(i in top)
    return hits / sims.shape[0]


def _run_tests() -> None:
    n = 8
    same = np.ones((n, 5))
    # identical embeddings: every logit equal, loss is log N
    assert abs(clip_loss(same, same, 0.07) - np.log(n)) < 1e-9

    # orthonormal embeddings, T = 1: logits are the identity matrix
    eye = np.eye(4)
    expected = np.log(1 + 3 * np.exp(-1.0))
    assert abs(clip_loss(eye, eye, 1.0) - expected) < 1e-9

    # symmetry under swapping the two modalities
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(16, 12)), rng.normal(size=(16, 12))
    assert abs(clip_loss(a, b) - clip_loss(b, a)) < 1e-12

    # lower temperature sharpens: loss of a perfectly aligned batch falls toward 0
    losses = [clip_loss(eye, eye, T) for T in (1.0, 0.3, 0.1, 0.03)]
    assert all(x > y for x, y in zip(losses, losses[1:]))
    assert losses[-1] < 1e-6

    # but a misaligned batch is punished harder at low temperature
    shifted = np.roll(eye, 1, axis=0)
    assert clip_loss(eye, shifted, 0.03) > clip_loss(eye, shifted, 1.0)

    # numerical stability with a tiny temperature
    v = clip_loss(a, b, 1e-3)
    assert np.isfinite(v) and v > 0

    # scale invariance (embeddings are normalised inside)
    assert abs(clip_loss(3.0 * a, 0.2 * b) - clip_loss(a, b)) < 1e-9

    # SigLIP: one aligned pair with t=10, b=-10 gives logit 0 -> loss log 2
    one = np.ones((1, 4))
    assert abs(siglip_loss(one, one) - np.log(2)) < 1e-9
    # orthonormal batch: positives have logit 0, negatives have logit -10
    exp_loss = (np.log(2) * 4 + 12 * np.log1p(np.exp(-10.0))) / 4
    assert abs(siglip_loss(eye, eye) - exp_loss) < 1e-9
    # raising t and b makes the aligned batch cheaper
    assert siglip_loss(eye, eye, t=20.0, b=-10.0) < siglip_loss(eye, eye, t=10.0, b=-10.0)

    # zero-shot classification on synthetic class prototypes
    rng = np.random.default_rng(7)
    protos = rng.normal(size=(5, 32))
    labels = rng.integers(0, 5, size=200)
    imgs = protos[labels] + 0.4 * rng.normal(size=(200, 32))
    pred = zero_shot_classify(imgs, protos)
    assert pred.shape == (200,)
    assert (pred == labels).mean() > 0.95
    # the prompt embedding of the true class is what wins, so swapping two classes swaps predictions
    perm = np.array([1, 0, 2, 3, 4])
    pred_perm = zero_shot_classify(imgs, protos[perm])
    assert np.array_equal(perm[pred_perm], pred)

    # retrieval recall
    img = np.eye(5)
    txt = np.eye(5)
    assert recall_at_k(img, txt, 1) == 1.0
    txt_bad = np.roll(np.eye(5), 1, axis=0)
    assert recall_at_k(img, txt_bad, 1) == 0.0   # every image prefers the wrong text
    assert recall_at_k(img, txt_bad, 5) == 1.0   # but the true text is always somewhere in the top 5
    txt_noisy = np.eye(5)
    txt_noisy[0] = [0.1, 1.0, 0, 0, 0]           # text 0 now looks a lot like image 1
    txt_noisy[1] = [0, 0.5, 0.5, 0, 0]           # and the true text for image 1 is weaker
    assert abs(recall_at_k(img, txt_noisy, 1) - 0.8) < 1e-12
    assert recall_at_k(img, txt_noisy, 2) == 1.0


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
