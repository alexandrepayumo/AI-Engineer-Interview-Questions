"""Exercise 01 - PII detector with checksum validation, span resolution and reversible redaction.

TASK
----
Implement a small, deterministic PII layer of the kind that sits in front of an LLM API.

    luhn_valid(digits) -> bool
        True if the string of digits passes the Luhn checksum. Non-digit input or
        fewer than 2 digits returns False.

    find_pii(text) -> list[Span]
        Return resolved, non-overlapping spans sorted by start. Detect four labels:
          EMAIL  r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}"
          CARD   13-19 digits, optionally separated by single spaces or hyphens, and
                 the digits must pass luhn_valid. Candidate regex:
                 r"(?<![\\d-])(?:\\d[ -]?){12,18}\\d(?![\\d-])"
          SSN    r"(?<!\\d)(\\d{3})-(\\d{2})-(\\d{4})(?!\\d)" with the US rules: area not
                 000, 666 or 900-999, group not 00, serial not 0000.
          PHONE  r"(?<![\\d-])(?:\\+1[ -]?)?\\(?\\d{3}\\)?[ .-]\\d{3}[ .-]\\d{4}(?!\\d)"
        Overlap rules (see resolve_spans): same-label overlaps are merged into one span,
        different-label overlaps are resolved by PRIORITY (CARD > SSN > EMAIL > PHONE),
        then by longer span.

    resolve_spans(spans) -> list[Span]
        Apply the overlap rules above. Touching spans (end == next start) are NOT
        overlapping.

    redact(text) -> tuple[str, dict[str, str]]
        Replace each span with a placeholder "[LABEL_n]" where n counts distinct values
        of that label from 1 in order of first appearance. The same value always gets
        the same placeholder. Return (redacted_text, mapping) where mapping is
        placeholder -> original string.

    rehydrate(text, mapping) -> str
        Put the originals back. rehydrate(*redact(t)) == t for every input.

HINTS
-----
- Validate cards with Luhn AFTER stripping spaces and hyphens, and check the length.
- Build spans from every regex first, then resolve. Do not try to be clever inside
  the regexes.
- Replace from the end of the string backwards, or build the output left to right, so
  offsets stay valid.

WHAT YOU SHOULD LEARN
---------------------
- Why checksums cut false positives: a random 16-digit order number passes Luhn only
  about 1 time in 10.
- Why overlap resolution matters (a card number contains a substring that looks like a
  phone number).
- Why reversible placeholders let the model reason about "[PERSON_1]" without seeing the
  value, and why the mapping must stay on your side of the trust boundary.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str


PRIORITY = ["CARD", "SSN", "EMAIL", "PHONE"]

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_CARD = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")
_SSN = re.compile(r"(?<!\d)(\d{3})-(\d{2})-(\d{4})(?!\d)")
_PHONE = re.compile(r"(?<![\d-])(?:\+1[ -]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)")


def luhn_valid(digits: str) -> bool:
    if len(digits) < 2 or not digits.isdigit():
        return False
    total = 0
    # Walk from the right. Every second digit (starting with the second from the right)
    # is doubled, and a doubled value above 9 has 9 subtracted (same as summing digits).
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _candidates(text: str) -> list[Span]:
    out: list[Span] = []
    for m in _EMAIL.finditer(text):
        out.append(Span(m.start(), m.end(), "EMAIL"))
    for m in _CARD.finditer(text):
        digits = re.sub(r"[ -]", "", m.group())
        if 13 <= len(digits) <= 19 and luhn_valid(digits):
            out.append(Span(m.start(), m.end(), "CARD"))
    for m in _SSN.finditer(text):
        area, group, serial = m.group(1), m.group(2), m.group(3)
        if area in ("000", "666") or area.startswith("9") or group == "00" or serial == "0000":
            continue
        out.append(Span(m.start(), m.end(), "SSN"))
    for m in _PHONE.finditer(text):
        out.append(Span(m.start(), m.end(), "PHONE"))
    return out


def resolve_spans(spans: list[Span]) -> list[Span]:
    # Step 1: merge overlapping spans that share a label.
    merged: list[Span] = []
    for label in PRIORITY:
        same = sorted((s for s in spans if s.label == label), key=lambda s: s.start)
        cur: Span | None = None
        for s in same:
            if cur is not None and s.start < cur.end:
                cur = Span(cur.start, max(cur.end, s.end), label)
            else:
                if cur is not None:
                    merged.append(cur)
                cur = s
        if cur is not None:
            merged.append(cur)
    # Step 2: greedy selection by priority, then by length (longer first).
    ranked = sorted(merged, key=lambda s: (PRIORITY.index(s.label), -(s.end - s.start), s.start))
    chosen: list[Span] = []
    for s in ranked:
        if all(s.end <= c.start or s.start >= c.end for c in chosen):
            chosen.append(s)
    return sorted(chosen, key=lambda s: s.start)


def find_pii(text: str) -> list[Span]:
    return resolve_spans(_candidates(text))


def redact(text: str) -> tuple[str, dict[str, str]]:
    spans = find_pii(text)
    mapping: dict[str, str] = {}
    value_to_ph: dict[tuple[str, str], str] = {}
    counters: dict[str, int] = {}
    pieces: list[str] = []
    pos = 0
    for s in spans:
        original = text[s.start:s.end]
        key = (s.label, original)
        if key not in value_to_ph:
            counters[s.label] = counters.get(s.label, 0) + 1
            ph = f"[{s.label}_{counters[s.label]}]"
            value_to_ph[key] = ph
            mapping[ph] = original
        pieces.append(text[pos:s.start])
        pieces.append(value_to_ph[key])
        pos = s.end
    pieces.append(text[pos:])
    return "".join(pieces), mapping


def rehydrate(text: str, mapping: dict[str, str]) -> str:
    # Placeholders are bracketed, so "[EMAIL_1]" can never be a prefix of "[EMAIL_10]".
    for ph, original in mapping.items():
        text = text.replace(ph, original)
    return text


def _run_tests() -> None:
    # Luhn: known good and bad numbers
    assert luhn_valid("4111111111111111")
    assert luhn_valid("79927398713")
    assert not luhn_valid("4111111111111112")
    assert not luhn_valid("") and not luhn_valid("7") and not luhn_valid("12a4")
    assert luhn_valid("378282246310005")  # 15-digit test number

    text = ("Email alice@example.com or bob@example.com, card 4111 1111 1111 1111, "
            "ssn 123-45-6789, call (415) 555-0132. Again alice@example.com.")
    spans = find_pii(text)
    labels = [s.label for s in spans]
    assert labels == ["EMAIL", "EMAIL", "CARD", "SSN", "PHONE", "EMAIL"], labels
    assert all(a.end <= b.start for a, b in zip(spans, spans[1:]))
    assert text[spans[2].start:spans[2].end] == "4111 1111 1111 1111"

    red, mapping = redact(text)
    assert red == ("Email [EMAIL_1] or [EMAIL_2], card [CARD_1], ssn [SSN_1], "
                   "call [PHONE_1]. Again [EMAIL_1]."), red
    assert mapping["[EMAIL_1]"] == "alice@example.com"
    assert mapping["[CARD_1]"] == "4111 1111 1111 1111"
    assert len(mapping) == 5
    assert rehydrate(red, mapping) == text

    # Luhn kills false positives: a 16-digit order id that fails the checksum is ignored
    assert find_pii("order 4111 1111 1111 1112 shipped") == []
    # ... and an invalid SSN pattern is ignored
    for bad in ("000-12-3456", "666-12-3456", "912-12-3456", "123-00-4567", "123-45-0000"):
        assert find_pii(f"id {bad} end") == [], bad

    # A card number must win over a phone-like substring and must not be split
    only_card = "pay 5500 0000 0000 0004 now"
    assert [s.label for s in find_pii(only_card)] == ["CARD"]
    assert luhn_valid("5500000000000004")

    # Overlap resolution on hand-built spans
    out = resolve_spans([Span(0, 10, "PHONE"), Span(4, 12, "CARD")])
    assert out == [Span(4, 12, "CARD")]
    out = resolve_spans([Span(0, 6, "EMAIL"), Span(4, 12, "EMAIL")])
    assert out == [Span(0, 12, "EMAIL")]
    out = resolve_spans([Span(0, 5, "EMAIL"), Span(5, 9, "PHONE")])  # touching, both kept
    assert out == [Span(0, 5, "EMAIL"), Span(5, 9, "PHONE")]
    out = resolve_spans([Span(0, 4, "PHONE"), Span(2, 20, "PHONE"), Span(10, 12, "SSN")])
    assert out == [Span(10, 12, "SSN")]  # same-label merge to (0, 20), then SSN outranks it
    assert resolve_spans([]) == []

    # Placeholder numbering counts distinct values, and round trip holds for no-PII text
    red2, m2 = redact("nothing to see here")
    assert red2 == "nothing to see here" and m2 == {}
    many = " ".join(f"u{i}@x.org" for i in range(11)) + " u0@x.org"
    red3, m3 = redact(many)
    assert "[EMAIL_11]" in red3 and red3.endswith("[EMAIL_1]")
    assert rehydrate(red3, m3) == many


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
