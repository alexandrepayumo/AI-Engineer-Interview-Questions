"""Exercise 01 - SFT loss with prompt-token masking, multi-turn masks and sequence packing.

TASK
----
Implement (IGNORE = -100 is the label value that the loss skips):

    build_example(prompt_ids, response_ids, eos_id) -> tuple[list[int], list[int]]
        input_ids = prompt + response + [eos]. labels = [IGNORE] * len(prompt) + response + [eos].

    build_multi_turn(turns, eos_id) -> tuple[list[int], list[int]]
        turns: list of (role, token_ids) with role "user" or "assistant".
        Every turn is followed by eos_id in input_ids. Labels are the real token ids
        (including the trailing eos) for assistant turns and IGNORE for user turns.

    shift_for_next_token(input_ids, labels) -> tuple[list[int], list[int]]
        Position t predicts token t+1. Returns (input_ids[:-1], labels[1:]).

    masked_cross_entropy(logits, targets) -> tuple[float, int]
        logits: (T, V) float array, targets: (T,) int array where IGNORE entries are skipped.
        Returns (mean negative log-likelihood over non-ignored positions, number of such positions).
        If nothing is left, return (0.0, 0).

    pack_sequences(examples, max_len, pad_id) -> dict[str, np.ndarray]
        examples: list of (input_ids, labels) pairs. Pack them greedily, in order, into rows of
        length max_len (start a new row when the next example does not fit; pad the tail).
        Returns arrays of shape (n_rows, max_len):
          input_ids   (pad_id on padding)
          labels      (IGNORE on padding)
          position_ids (restart at 0 for every packed example, 0 on padding)
          segment_ids  (1, 2, 3, ... per example inside a row, 0 on padding)
        Raise ValueError if one example is longer than max_len.

    packed_targets(labels, segment_ids) -> np.ndarray
        1-D inputs for one packed row. Next-token targets: targets[t] = labels[t+1] when position
        t+1 belongs to the same non-zero segment as t, otherwise IGNORE. The last position gets IGNORE.

    block_diagonal_causal_mask(segment_ids) -> np.ndarray
        (L, L) bool. mask[i, j] is True when j <= i, segment_ids[i] == segment_ids[j] and the
        segment is not 0 (padding). Packed examples must not attend to each other.

HINTS
-----
- log-softmax: subtract the row max, then logits - max - log(sum(exp(...))).
- Pick target log-probs with np.arange(T) and the target index, only on the kept rows.
- The first token of an example has no predictor inside its own segment. That is exactly the
  case packed_targets must mask: the last token of example A must not "predict" the first
  token of example B.

WHAT YOU SHOULD LEARN
---------------------
- Loss masking is a label convention, not a model change. The prompt still conditions the answer.
- Why the mean must be taken over unmasked tokens, and why per-example mean and per-token mean
  give different weightings.
- Packing removes padding waste but needs correct boundaries (targets, positions, attention),
  otherwise examples leak into each other.
"""
from __future__ import annotations

import numpy as np

IGNORE = -100


def build_example(prompt_ids: list[int], response_ids: list[int], eos_id: int) -> tuple[list[int], list[int]]:
    # Prompt tokens are inputs only; the model is trained on the response and the EOS.
    input_ids = list(prompt_ids) + list(response_ids) + [eos_id]
    labels = [IGNORE] * len(prompt_ids) + list(response_ids) + [eos_id]
    return input_ids, labels


def build_multi_turn(turns: list[tuple[str, list[int]]], eos_id: int) -> tuple[list[int], list[int]]:
    input_ids: list[int] = []
    labels: list[int] = []
    for role, ids in turns:
        seg = list(ids) + [eos_id]
        input_ids += seg
        # Only assistant turns (and their EOS) carry a training signal.
        labels += seg if role == "assistant" else [IGNORE] * len(seg)
    return input_ids, labels


def shift_for_next_token(input_ids: list[int], labels: list[int]) -> tuple[list[int], list[int]]:
    return list(input_ids[:-1]), list(labels[1:])


def masked_cross_entropy(logits: np.ndarray, targets: np.ndarray) -> tuple[float, int]:
    keep = targets != IGNORE
    n = int(keep.sum())
    if n == 0:
        return 0.0, 0
    z = logits[keep]
    t = targets[keep]
    z = z - z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    nll = -logp[np.arange(n), t]
    return float(nll.mean()), n


def pack_sequences(examples: list[tuple[list[int], list[int]]], max_len: int, pad_id: int) -> dict[str, np.ndarray]:
    rows: list[list[tuple[list[int], list[int]]]] = []
    used = 0
    for ids, labs in examples:
        if len(ids) > max_len:
            raise ValueError("example longer than max_len")
        if not rows or used + len(ids) > max_len:
            rows.append([])
            used = 0
        rows[-1].append((ids, labs))
        used += len(ids)
    n = len(rows)
    out = {
        "input_ids": np.full((n, max_len), pad_id, dtype=np.int64),
        "labels": np.full((n, max_len), IGNORE, dtype=np.int64),
        "position_ids": np.zeros((n, max_len), dtype=np.int64),
        "segment_ids": np.zeros((n, max_len), dtype=np.int64),
    }
    for r, row in enumerate(rows):
        pos = 0
        for seg, (ids, labs) in enumerate(row, start=1):
            L = len(ids)
            out["input_ids"][r, pos:pos + L] = ids
            out["labels"][r, pos:pos + L] = labs
            out["position_ids"][r, pos:pos + L] = np.arange(L)  # positions restart per example
            out["segment_ids"][r, pos:pos + L] = seg
            pos += L
    return out


def packed_targets(labels: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
    L = len(labels)
    targets = np.full(L, IGNORE, dtype=np.int64)
    for t in range(L - 1):
        # Only predict inside a segment; never across a boundary or into padding.
        if segment_ids[t] != 0 and segment_ids[t] == segment_ids[t + 1]:
            targets[t] = labels[t + 1]
    return targets


def block_diagonal_causal_mask(segment_ids: np.ndarray) -> np.ndarray:
    L = len(segment_ids)
    causal = np.tril(np.ones((L, L), dtype=bool))
    same = segment_ids[:, None] == segment_ids[None, :]
    return causal & same & (segment_ids[:, None] != 0)


def _run_tests() -> None:
    # build_example
    ids, labs = build_example([1, 2, 3], [7, 8], eos_id=9)
    assert ids == [1, 2, 3, 7, 8, 9]
    assert labs == [IGNORE, IGNORE, IGNORE, 7, 8, 9]

    # multi-turn: only assistant tokens are trained
    ids, labs = build_multi_turn([("user", [1, 2]), ("assistant", [5]), ("user", [3]), ("assistant", [6, 7])], 9)
    assert ids == [1, 2, 9, 5, 9, 3, 9, 6, 7, 9]
    assert labs == [IGNORE, IGNORE, IGNORE, 5, 9, IGNORE, IGNORE, 6, 7, 9]

    # shift
    x, y = shift_for_next_token([1, 2, 3, 4], [IGNORE, IGNORE, 3, 4])
    assert x == [1, 2, 3] and y == [IGNORE, 3, 4]

    # masked cross entropy: uniform logits -> log V, ignored rows do not count
    V = 5
    logits = np.zeros((4, V))
    targets = np.array([IGNORE, 2, IGNORE, 4])
    loss, n = masked_cross_entropy(logits, targets)
    assert n == 2 and abs(loss - np.log(V)) < 1e-12
    assert masked_cross_entropy(logits, np.full(4, IGNORE)) == (0.0, 0)
    # a confident correct logit gives a tiny loss; huge logits do not overflow
    big = np.zeros((1, 3)); big[0, 1] = 1000.0
    loss, n = masked_cross_entropy(big, np.array([1]))
    assert loss < 1e-9 and n == 1
    # masked prompt rows can hold any garbage without changing the loss
    garb = logits.copy(); garb[0] = 123.0; garb[2] = -50.0
    assert abs(masked_cross_entropy(garb, targets)[0] - np.log(V)) < 1e-12

    # packing layout
    a = build_example([1, 2], [3], 9)        # length 4
    b = build_example([4], [5, 6], 9)        # length 4
    c = build_example([7, 8], [3, 3, 3], 9)  # length 6
    packed = pack_sequences([a, b, c], max_len=8, pad_id=0)
    assert packed["input_ids"].shape == (2, 8)
    assert packed["input_ids"][0].tolist() == [1, 2, 3, 9, 4, 5, 6, 9]
    assert packed["segment_ids"][0].tolist() == [1, 1, 1, 1, 2, 2, 2, 2]
    assert packed["position_ids"][0].tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    assert packed["segment_ids"][1].tolist() == [1, 1, 1, 1, 1, 1, 0, 0]
    assert packed["labels"][1, 6:].tolist() == [IGNORE, IGNORE]
    assert packed["input_ids"][1, 6:].tolist() == [0, 0]
    try:
        pack_sequences([a], max_len=3, pad_id=0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    # targets never cross a boundary
    t0 = packed_targets(packed["labels"][0], packed["segment_ids"][0])
    # position 3 is the EOS of example 1 -> must not predict example 2's first token
    assert t0[3] == IGNORE and t0[7] == IGNORE
    assert t0[2] == 9 and t0[4] == 5 and t0[0] == IGNORE  # label of position 1 is IGNORE (prompt)
    t1 = packed_targets(packed["labels"][1], packed["segment_ids"][1])
    assert (t1[5:] == IGNORE).all()

    # attention mask is block diagonal and causal
    m = block_diagonal_causal_mask(packed["segment_ids"][0])
    assert m[3, 0] and m[3, 3] and not m[3, 4]
    assert not m[4, 3] and m[4, 4] and m[5, 4] and not m[4, 5]
    pm = block_diagonal_causal_mask(packed["segment_ids"][1])
    assert not pm[7].any() and not pm[:, 7].any()

    # Packing does not change the total loss. Use a per-token model: logits depend only on the
    # current token, so no attention is involved and the equality is exact.
    rng = np.random.default_rng(0)
    table = rng.normal(size=(10, 10))
    exs = [build_example([1, 2], [3, 4], 9), build_example([5], [6, 7, 8], 9),
           build_example([2, 2, 2], [1], 9), build_example([4], [4], 9)]
    unpacked_sum, unpacked_n, per_example_means = 0.0, 0, []
    for ids, labs in exs:
        x, y = shift_for_next_token(ids, labs)
        lo, cnt = masked_cross_entropy(table[np.array(x)], np.array(y))
        unpacked_sum += lo * cnt
        unpacked_n += cnt
        per_example_means.append(lo)
    pk = pack_sequences(exs, max_len=10, pad_id=0)
    packed_sum, packed_n = 0.0, 0
    for r in range(pk["input_ids"].shape[0]):
        tg = packed_targets(pk["labels"][r], pk["segment_ids"][r])
        lo, cnt = masked_cross_entropy(table[pk["input_ids"][r]], tg)
        packed_sum += lo * cnt
        packed_n += cnt
    assert packed_n == unpacked_n
    assert abs(packed_sum - unpacked_sum) < 1e-9
    # token-level mean and mean-of-example-means weight examples differently
    assert abs(unpacked_sum / unpacked_n - float(np.mean(per_example_means))) > 1e-6


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
