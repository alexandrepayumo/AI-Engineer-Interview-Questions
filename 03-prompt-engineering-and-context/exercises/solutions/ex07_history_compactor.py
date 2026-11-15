"""Exercise 07 - Conversation history compaction with a rolling summary.

TASK
----
Keep a chat history under a token budget by replacing the oldest turns with a
rolling summary. Tokens are whitespace-separated words.

    @dataclass(frozen=True)
    class Message:
        role: str
        text: str

    def count_tokens(text: str) -> int
    def history_tokens(summary: str, messages: list[Message]) -> int
        '''tokens(summary) + sum of tokens of message texts.'''

    def compact(summary: str, messages: list[Message], budget: int, summarize,
                summary_cap: int, keep_last: int = 2) -> tuple[str, list[Message]]
        '''Return (new_summary, kept_messages).
        1. If history_tokens(summary, messages) <= budget, return inputs unchanged
           (same summary, a copy of the list).
        2. Otherwise compaction triggers. Reserve summary_cap tokens for the summary:
           pop messages from the FRONT until the remaining messages have at most
           budget - summary_cap tokens, but never leave fewer than keep_last messages.
        3. new_summary = summarize(summary, popped, summary_cap). If it returns more
           than summary_cap tokens, keep only its LAST summary_cap words (newest info).
        4. If even keep_last messages plus summary_cap exceed the budget, raise ValueError.
        `summarize(prev_summary, popped_messages, max_tokens) -> str` is supplied by the
        caller (in production, an LLM call).'''

HINTS
-----
- Work on a copy of the list. Compute the message-token target once.
- Raise ValueError before calling summarize if the budget is impossible.
- Rolling means the previous summary is an INPUT to the next summarisation.

WHAT YOU SHOULD LEARN
---------------------
Compaction trades fidelity for space, so the policy matters: how much is kept
verbatim (recent turns), how large the summary may grow, and when to trigger.
Each compaction rewrites the prompt prefix and invalidates the prompt cache, which
is why you compact in batches at a threshold instead of trimming on every turn.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    role: str
    text: str


def count_tokens(text: str) -> int:
    return len(text.split())


def history_tokens(summary: str, messages: list[Message]) -> int:
    return count_tokens(summary) + sum(count_tokens(m.text) for m in messages)


def compact(summary: str, messages: list[Message], budget: int, summarize,
            summary_cap: int, keep_last: int = 2) -> tuple[str, list[Message]]:
    kept = list(messages)
    if history_tokens(summary, kept) <= budget:
        return summary, kept
    target = budget - summary_cap  # token allowance for verbatim messages
    # impossible if the protected tail alone does not fit next to a full summary
    tail = sum(count_tokens(m.text) for m in kept[-keep_last:]) if keep_last > 0 else 0
    if target < 0 or tail > target:
        raise ValueError("budget too small for keep_last messages plus summary_cap")
    popped: list[Message] = []
    msg_tokens = sum(count_tokens(m.text) for m in kept)
    while msg_tokens > target and len(kept) > keep_last:
        m = kept.pop(0)
        popped.append(m)
        msg_tokens -= count_tokens(m.text)
    new_summary = summarize(summary, popped, summary_cap)
    words = new_summary.split()
    if len(words) > summary_cap:
        new_summary = " ".join(words[-summary_cap:])  # keep the newest information
    return new_summary, kept


def fake_summarize(prev: str, popped: list[Message], max_tokens: int) -> str:
    """Deterministic stand-in for an LLM: previous summary + first 3 words of each popped turn."""
    parts = [prev] if prev else []
    parts += [f"{m.role[0]}:" + "_".join(m.text.split()[:3]) for m in popped]
    return " ".join(parts)


def _run_tests() -> None:
    def msg(i: int, n: int = 10) -> Message:
        return Message("user" if i % 2 == 0 else "assistant", " ".join(f"w{i}x{j}" for j in range(n)))

    assert history_tokens("a b", [Message("user", "c d e")]) == 5

    # under budget: unchanged
    msgs = [msg(i) for i in range(4)]            # 40 tokens
    s, kept = compact("", msgs, 100, fake_summarize, 20)
    assert s == "" and kept == msgs and kept is not msgs

    # over budget: 8 messages (80 tokens), budget 60, cap 20 -> messages must be <= 40
    msgs = [msg(i) for i in range(8)]
    s, kept = compact("", msgs, 60, fake_summarize, 20)
    assert kept == msgs[4:]
    assert s.split()[0].startswith("u:w0x0_w0x1_w0x2") and len(s.split()) == 4
    assert history_tokens(s, kept) <= 60

    # keep_last protects the tail even if popping everything else is not enough to hit target
    msgs = [msg(i, 30) for i in range(3)]        # 90 tokens, target = 50-20 = 30
    s, kept = compact("", msgs, 50, fake_summarize, 20, keep_last=1)
    assert kept == msgs[2:] and history_tokens(s, kept) <= 50

    # impossible budget raises before summarising
    calls: list[int] = []

    def spy(prev, popped, cap):
        calls.append(1)
        return "x"

    try:
        compact("", [msg(0, 30), msg(1, 30)], 40, spy, 20, keep_last=2)
        raise AssertionError("expected ValueError")
    except ValueError:
        assert calls == []

    # an over-long summary is cut to its last summary_cap words
    def verbose(prev, popped, cap):
        return " ".join(f"s{i}" for i in range(100))

    s, kept = compact("", [msg(i) for i in range(8)], 60, verbose, 20)
    assert len(s.split()) == 20 and s.split()[-1] == "s99" and s.split()[0] == "s80"

    # rolling: stream 40 turns, stay under budget at every step, summary stays under cap
    summary: str = ""
    history: list[Message] = []
    compactions = 0
    for i in range(40):
        history.append(msg(i))
        new_summary, new_history = compact(summary, history, 80, fake_summarize, 24)
        if len(new_history) != len(history):
            compactions += 1
        summary, history = new_summary, new_history
        assert history_tokens(summary, history) <= 80
        assert count_tokens(summary) <= 24
    assert 0 < compactions < 40          # compaction runs in batches, not on every turn
    assert history[-1] == msg(39)        # newest turn always kept verbatim


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
