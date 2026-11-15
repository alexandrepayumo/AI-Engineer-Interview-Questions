"""Exercise 05 - Trie longest-match tokenizer with a streaming encoder.

TASK
----
Build a greedy longest-match tokenizer (the "MaxMatch" family used by WordPiece-style
tokenizers, and handy for stop strings and special tokens) on a trie, then make it
work on text that arrives in arbitrary chunks.

    Trie
        insert(token: str, token_id: int) -> None
        scan(text: str, start: int = 0) -> tuple[tuple[int, int] | None, bool]
            Walk the trie from text[start:]. Return (best, open_ended) where
            best is (token_id, length) for the LONGEST vocabulary token that is a
            prefix of text[start:], or None, and open_ended is True only when the walk
            consumed ALL of text[start:] and the node it stopped on still has children
            (so more text could still extend the match).
        longest_match(text: str, start: int = 0) -> tuple[int, int] | None
            Just the best match from scan.
    encode(trie, text, unk_id) -> list[int]
        Repeatedly take the longest match. If nothing matches at a position, emit
        unk_id and advance one character.
    StreamingEncoder(trie, unk_id)
        feed(chunk) -> list[int]   tokens that are now final
        flush() -> list[int]       end of stream: resolve whatever is pending
        pending -> str             text held back, waiting for more input
    The concatenation of everything returned by feed() calls plus flush() must equal
    encode(trie, whole_text, unk_id) for EVERY way of cutting the text into chunks.

HINTS
-----
- Greedy longest match needs backtracking inside the walk: remember the last node that
  was a token, keep walking, and fall back to it when the path dies. Example with
  vocabulary {"ab", "abcd", "c"}: "abce" walks a, b, c, then dies at e. The best match
  is "ab", not "abc" (not a token). Resume at "c".
- Streaming is the same decision made late. You may only emit when the answer cannot
  change with more input: the walk ended on a missing child, or on a leaf. If the walk
  ran out of text while the node still had children, hold the text back.
- So the held-back text is always shorter than the longest vocabulary token.

WHAT YOU SHOULD LEARN
---------------------
- A trie answers "longest vocabulary entry starting here" in O(token length).
- Streaming correctness means deciding only when the decision is final, not guessing.
- The same pattern powers stop-sequence detection in token streams: hold back any text
  that could still become a stop string.
- Greedy longest match is NOT BPE: BPE applies merges by rank, which can give a different
  split (see the deep dive, Part 5).
"""
from __future__ import annotations

import random


class Trie:
    def __init__(self) -> None:
        self.children: list[dict[str, int]] = [{}]    # node -> {char: child node}
        self.value: list[int | None] = [None]         # node -> token id if a token ends here

    def insert(self, token: str, token_id: int) -> None:
        raise NotImplementedError("TODO")

    def scan(self, text: str, start: int = 0) -> tuple[tuple[int, int] | None, bool]:
        raise NotImplementedError("TODO")

    def longest_match(self, text: str, start: int = 0) -> tuple[int, int] | None:
        raise NotImplementedError("TODO")


def encode(trie: Trie, text: str, unk_id: int) -> list[int]:
    raise NotImplementedError("TODO")


class StreamingEncoder:
    def __init__(self, trie: Trie, unk_id: int) -> None:
        self.trie = trie
        self.unk_id = unk_id
        self._buf = ""

    @property
    def pending(self) -> str:
        return self._buf

    def _drain(self, final: bool) -> list[int]:
        raise NotImplementedError("TODO")

    def feed(self, chunk: str) -> list[int]:
        raise NotImplementedError("TODO")

    def flush(self) -> list[int]:
        raise NotImplementedError("TODO")


def _build(vocab: dict[str, int]) -> Trie:
    t = Trie()
    for tok, i in vocab.items():
        t.insert(tok, i)
    return t


def _run_tests() -> None:
    UNK = 99
    vocab = {"a": 0, "b": 1, "c": 2, "e": 3, "ab": 4, "abcd": 5}
    trie = _build(vocab)

    # longest match and backtracking
    assert trie.longest_match("abcd!") == (5, 4)
    assert trie.longest_match("abce") == (4, 2)         # "abc" is not a token: fall back to "ab"
    assert trie.longest_match("xab") is None
    assert trie.longest_match("xab", 1) == (4, 2)
    assert trie.longest_match("") is None
    assert encode(trie, "abce", UNK) == [4, 2, 3]
    assert encode(trie, "abcdab", UNK) == [5, 4]
    assert encode(trie, "azb", UNK) == [0, UNK, 1]
    assert encode(trie, "", UNK) == []

    # scan reports whether the answer is still open
    assert trie.scan("ab") == ((4, 2), True)            # "abcd" could still follow
    assert trie.scan("abc") == ((4, 2), True)
    assert trie.scan("abcd") == ((5, 4), False)         # leaf: nothing can extend it
    assert trie.scan("abx") == ((4, 2), False)
    assert trie.scan("e") == ((3, 1), False)

    # streaming, hand-checked
    s = StreamingEncoder(trie, UNK)
    assert s.feed("a") == [] and s.pending == "a"
    assert s.feed("b") == [] and s.pending == "ab"
    assert s.feed("c") == [] and s.pending == "abc"
    assert s.feed("e") == [4, 2, 3] and s.pending == ""
    assert s.feed("abcd") == [5]
    assert s.feed("ab") == [] and s.flush() == [4] and s.pending == ""
    assert s.flush() == []
    s2 = StreamingEncoder(trie, UNK)
    assert s2.feed("z") == [UNK]                        # unknown char is final immediately

    # streaming equals batch for every chunking, on random vocabularies and texts
    rnd = random.Random(0)
    alphabet = "abcd"
    for trial in range(60):
        words = {"".join(rnd.choice(alphabet) for _ in range(rnd.randint(1, 5))) for _ in range(12)}
        v = {w: i for i, w in enumerate(sorted(words))}
        tr = _build(v)
        max_len = max(len(w) for w in v)
        text = "".join(rnd.choice(alphabet + "z") for _ in range(rnd.randint(0, 80)))
        want = encode(tr, text, UNK)
        enc = StreamingEncoder(tr, UNK)
        got, i = [], 0
        while i < len(text):
            n = rnd.randint(1, 4)
            got += enc.feed(text[i:i + n])
            assert len(enc.pending) < max_len, "pending text must be shorter than the longest token"
            i += n
        got += enc.flush()
        assert got == want, (trial, text)
        # one character at a time is the hardest chunking
        enc1 = StreamingEncoder(tr, UNK)
        got1 = [t for ch in text for t in enc1.feed(ch)] + enc1.flush()
        assert got1 == want

    # round trip when every character is a token: no information lost
    chars = {c: i for i, c in enumerate("abcd ")}
    chars.update({"ab": 10, "cd": 11, "abcd": 12})
    inv = {i: t for t, i in chars.items()}
    tr = _build(chars)
    sample = "abcd ab cd abcdabc d"
    ids = encode(tr, sample, UNK)
    assert "".join(inv[i] for i in ids) == sample
    assert ids == [12, 4, 10, 4, 11, 4, 12, 10, 2, 4, 3]

    # non-ASCII text
    uni = _build({"\u00e9": 0, "\u00e9a": 1, "\u65e5": 2, "\u65e5\u672c": 3, "a": 4})
    text = "\u65e5\u672c\u00e9a\u65e5a"
    assert encode(uni, text, UNK) == [3, 1, 2, 4]
    e = StreamingEncoder(uni, UNK)
    assert e.feed("\u00e9") == [] and e.feed("a") == [1]


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
