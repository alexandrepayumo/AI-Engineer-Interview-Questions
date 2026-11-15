"""Exercise 02 - Token-budgeted context assembler.

TASK
----
Build the function that decides what goes into the prompt when everything does
not fit. Tokens are approximated as whitespace-separated words.

    @dataclass(frozen=True)
    class Block:
        name: str
        text: str
        priority: int            # higher = more important
        slot: int                # position in the final prompt (ascending)
        truncatable: bool = False
        required: bool = False

    @dataclass
    class Assembly:
        text: str                # included blocks joined by "\\n\\n" in slot order
        tokens: int              # count_tokens(text)
        included: list[str]      # names, slot order (truncated blocks included)
        dropped: list[str]       # names, in the order they were rejected
        truncated: list[str]     # names of blocks that were cut

    def count_tokens(text: str) -> int
    def truncate_to(text: str, n: int) -> str
    def assemble(blocks: list[Block], budget: int) -> Assembly

Rules for assemble:
1. Required blocks are included whole and reserved first. If they alone exceed
   the budget, raise ValueError.
2. The remaining blocks are considered in order of priority (highest first, ties
   broken by lower slot). If a block fits in the remaining budget, include it whole.
3. If it does not fit and it is truncatable and at least 2 tokens remain,
   include truncate_to(text, remaining). That uses the whole remaining budget.
4. Otherwise drop it (append to `dropped`) and continue with the next block;
   a smaller, lower-priority block may still fit.
5. The final prompt is ordered by slot, not by priority.

truncate_to(text, n): if text has <= n tokens return it unchanged. Otherwise
return the first n-1 words followed by the marker "[...]" (the marker counts as
one token), so the result has exactly n tokens. n < 1 raises ValueError.

HINTS
-----
- Priority decides who survives; slot decides where it sits. Keep them separate.
- Sort the optional blocks with key=(-priority, slot).

WHAT YOU SHOULD LEARN
---------------------
Context engineering is budget allocation. Separating "importance" from "position"
lets you keep instructions at the edges (good for lost-in-the-middle) while still
sacrificing low-value history or retrieved chunks first. Production assemblers
do exactly this, plus real tokenisers and per-section caps.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Block:
    name: str
    text: str
    priority: int
    slot: int
    truncatable: bool = False
    required: bool = False


@dataclass
class Assembly:
    text: str
    tokens: int
    included: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    truncated: list[str] = field(default_factory=list)


def count_tokens(text: str) -> int:
    return len(text.split())


def truncate_to(text: str, n: int) -> str:
    if n < 1:
        raise ValueError("n must be >= 1")
    words = text.split()
    if len(words) <= n:
        return text
    return " ".join(words[: n - 1] + ["[...]"])


def assemble(blocks: list[Block], budget: int) -> Assembly:
    chosen: dict[str, str] = {}  # name -> final text
    dropped: list[str] = []
    truncated: list[str] = []

    # 1. reserve required blocks
    required = [b for b in blocks if b.required]
    remaining = budget - sum(count_tokens(b.text) for b in required)
    if remaining < 0:
        raise ValueError("required blocks exceed the budget")
    for b in required:
        chosen[b.name] = b.text

    # 2-4. optional blocks by priority
    optional = sorted((b for b in blocks if not b.required), key=lambda b: (-b.priority, b.slot))
    for b in optional:
        n = count_tokens(b.text)
        if n <= remaining:
            chosen[b.name] = b.text
            remaining -= n
        elif b.truncatable and remaining >= 2:
            chosen[b.name] = truncate_to(b.text, remaining)
            truncated.append(b.name)
            remaining = 0
        else:
            dropped.append(b.name)

    # 5. emit in slot order
    ordered = sorted((b for b in blocks if b.name in chosen), key=lambda b: b.slot)
    text = "\n\n".join(chosen[b.name] for b in ordered)
    return Assembly(
        text=text,
        tokens=count_tokens(text),
        included=[b.name for b in ordered],
        dropped=dropped,
        truncated=truncated,
    )


def _run_tests() -> None:
    assert count_tokens("a b  c\nd") == 4
    assert truncate_to("a b c", 5) == "a b c"
    assert truncate_to("a b c d e", 3) == "a b [...]"
    assert count_tokens(truncate_to("a b c d e", 3)) == 3
    assert truncate_to("a b c d e", 1) == "[...]"
    try:
        truncate_to("a", 0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    sys_b = Block("system", "you are a helpful bot " * 1, priority=10, slot=0, required=True)  # 5 tokens
    task = Block("task", "summarise the documents", priority=9, slot=9, required=True)         # 3 tokens
    docs = Block("docs", " ".join(f"d{i}" for i in range(20)), priority=5, slot=2, truncatable=True)
    hist = Block("history", " ".join(f"h{i}" for i in range(10)), priority=3, slot=1)
    tiny = Block("tiny", "ok thanks", priority=1, slot=3)

    # everything fits: 5 + 3 + 20 + 10 + 2 = 40
    a = assemble([sys_b, task, docs, hist, tiny], 40)
    assert a.tokens == 40 and a.dropped == [] and a.truncated == []
    assert a.included == ["system", "history", "docs", "tiny", "task"]  # slot order
    assert a.text.startswith("you are") and a.text.endswith("documents")

    # budget 30: required 8, remaining 22. docs (20) fits -> 2 left. history (10) dropped, tiny (2) fits.
    a = assemble([sys_b, task, docs, hist, tiny], 30)
    assert a.dropped == ["history"] and a.included == ["system", "docs", "tiny", "task"]
    assert a.tokens == 30

    # budget 20: required 8, remaining 12. docs truncatable -> cut to 12 tokens, nothing else fits.
    a = assemble([sys_b, task, docs, hist, tiny], 20)
    assert a.truncated == ["docs"] and a.tokens == 20
    assert a.dropped == ["history", "tiny"]
    assert "d10" in a.text and "d11" not in a.text and "[...]" in a.text

    # a block that does not fit and is not truncatable is skipped, a smaller later one still fits
    big = Block("big", " ".join("w" for _ in range(50)), priority=8, slot=1)
    a = assemble([sys_b, task, big, tiny], 15)
    assert a.dropped == ["big"] and a.included == ["system", "tiny", "task"]

    # required overflow
    try:
        assemble([sys_b, task], 7)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    # priority beats slot when choosing survivors
    hi = Block("hi", "a b c d", priority=7, slot=5)
    lo = Block("lo", "e f g h", priority=2, slot=0)
    a = assemble([hi, lo], 4)
    assert a.included == ["hi"] and a.dropped == ["lo"]

    # tie on priority broken by lower slot
    x = Block("x", "a b", priority=1, slot=3)
    y = Block("y", "c d", priority=1, slot=1)
    a = assemble([x, y], 2)
    assert a.included == ["y"] and a.dropped == ["x"]

    # property: never exceeds the budget
    for budget in range(8, 45):
        a = assemble([sys_b, task, docs, hist, tiny], budget)
        assert a.tokens <= budget


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
