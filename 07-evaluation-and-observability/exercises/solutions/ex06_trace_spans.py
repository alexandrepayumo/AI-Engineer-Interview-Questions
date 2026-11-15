"""Exercise 06 - Trace span trees: self time, subtree cost and the critical path.

TASK
----
A trace is a list of span dicts with keys:
    id (str), parent (str or None), name (str), start (float, seconds), end (float),
    model (str or None), tokens_in (int), tokens_out (int).
Exactly one span has parent None (the root). Children may overlap (parallel tool calls).

Implement:

    children_map(spans) -> dict[str, list[str]]
        Map every span id to the ids of its children, ordered by start time (ties by id).
        Leaves map to an empty list. Raise ValueError if there is not exactly one root.
    self_time(spans) -> dict[str, float]
        Duration of each span minus the length of the UNION of its children's intervals,
        each clipped to the parent's [start, end]. Parallel children are not double counted.
    subtree_cost(spans, prices) -> dict[str, float]
        prices maps model -> (usd_per_million_input_tokens, usd_per_million_output_tokens).
        A span's own cost is tokens_in * p_in / 1e6 + tokens_out * p_out / 1e6 (0 if model
        is None). Return, for every span, its own cost plus the cost of all descendants.
    critical_path(spans) -> list[str]
        Span ids on the critical path, sorted by start time. Walk back from the end of a
        span: among its children not yet chosen, pick the one with the latest end that is
        <= the current cursor (initially the parent's end), recurse into it, and move the
        cursor to that child's start. Repeat until no child fits. The span itself is always
        on the path.
    percentile(values, q) -> float
        q in [0, 100], linear interpolation between order statistics (same as
        numpy.percentile default). Raise ValueError on empty input.

HINTS
-----
- Union length: sort intervals by start, sweep, and merge overlapping ones.
- Critical path is the chain of spans the root actually waited on. A parallel branch that
  finishes earlier than its sibling is not on it.

WHAT YOU SHOULD LEARN
---------------------
- Why wall-clock time does not equal the sum of span durations when work is parallel.
- Self time points at the span to optimise; the critical path says which optimisation
  would actually shorten the request.
- Cost rolls up the tree, so a per-span price table is enough to attribute spend.
"""
from __future__ import annotations

import numpy as np


def children_map(spans: list[dict]) -> dict[str, list[str]]:
    roots = [s for s in spans if s["parent"] is None]
    if len(roots) != 1:
        raise ValueError("trace must have exactly one root")
    by_id = {s["id"]: s for s in spans}
    kids: dict[str, list[str]] = {s["id"]: [] for s in spans}
    for s in spans:
        if s["parent"] is not None:
            kids[s["parent"]].append(s["id"])
    for pid in kids:
        kids[pid].sort(key=lambda c: (by_id[c]["start"], c))
    return kids


def _union_length(intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    cur_s = cur_e = None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:  # gap: close the previous merged block
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def self_time(spans: list[dict]) -> dict[str, float]:
    by_id = {s["id"]: s for s in spans}
    kids = children_map(spans)
    out = {}
    for s in spans:
        clipped = [
            (max(by_id[c]["start"], s["start"]), min(by_id[c]["end"], s["end"]))
            for c in kids[s["id"]]
        ]
        clipped = [(a, b) for a, b in clipped if b > a]
        out[s["id"]] = (s["end"] - s["start"]) - _union_length(clipped)
    return out


def subtree_cost(spans: list[dict], prices: dict[str, tuple[float, float]]) -> dict[str, float]:
    by_id = {s["id"]: s for s in spans}
    kids = children_map(spans)
    total: dict[str, float] = {}

    def visit(sid: str) -> float:
        s = by_id[sid]
        own = 0.0
        if s["model"] is not None:
            p_in, p_out = prices[s["model"]]
            own = s["tokens_in"] * p_in / 1e6 + s["tokens_out"] * p_out / 1e6
        total[sid] = own + sum(visit(c) for c in kids[sid])  # own cost plus descendants
        return total[sid]

    root = next(s["id"] for s in spans if s["parent"] is None)
    visit(root)
    return total


def critical_path(spans: list[dict]) -> list[str]:
    by_id = {s["id"]: s for s in spans}
    kids = children_map(spans)
    path: list[str] = []

    def walk(sid: str) -> None:
        path.append(sid)
        cursor = by_id[sid]["end"]
        remaining = list(kids[sid])
        while True:
            fits = [c for c in remaining if by_id[c]["end"] <= cursor + 1e-12]
            if not fits:
                return
            nxt = max(fits, key=lambda c: (by_id[c]["end"], by_id[c]["start"]))
            remaining.remove(nxt)
            walk(nxt)  # blocking child: recurse into it
            cursor = by_id[nxt]["start"]

    root = next(s["id"] for s in spans if s["parent"] is None)
    walk(root)
    return sorted(path, key=lambda i: (by_id[i]["start"], -by_id[i]["end"], i))


def percentile(values: list[float], q: float) -> float:
    if len(values) == 0:
        raise ValueError("empty input")
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100.0  # fractional index of the order statistic
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    return float(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo))


def _span(i, parent, name, start, end, model=None, tin=0, tout=0):
    return {"id": i, "parent": parent, "name": name, "start": start, "end": end,
            "model": model, "tokens_in": tin, "tokens_out": tout}


def _run_tests() -> None:
    spans = [
        _span("root", None, "agent", 0.0, 10.0),
        _span("ret", "root", "retrieve", 0.0, 2.0),
        _span("llm1", "root", "plan", 2.0, 6.0, "m1", 1000, 200),
        _span("t1", "root", "tool_a", 6.0, 9.0),
        _span("t2", "root", "tool_b", 6.0, 8.0),       # parallel with t1, finishes earlier
        _span("http", "t1", "http_call", 6.5, 8.5),
        _span("llm2", "root", "answer", 9.0, 10.0, "m1", 2000, 500),
    ]
    kids = children_map(spans)
    assert kids["root"] == ["ret", "llm1", "t1", "t2", "llm2"]
    assert kids["t1"] == ["http"] and kids["llm2"] == []

    st = self_time(spans)
    assert abs(st["root"] - 0.0) < 1e-12     # children tile the whole interval
    assert abs(st["t1"] - 1.0) < 1e-12       # 3.0 - 2.0 covered by http
    assert abs(st["llm1"] - 4.0) < 1e-12     # leaf: whole duration
    # parallel children are merged, not summed: t1 and t2 overlap on [6, 8]
    assert sum(s["end"] - s["start"] for s in spans if s["parent"] == "root") > 10.0

    cost = subtree_cost(spans, {"m1": (3.0, 15.0)})
    assert abs(cost["llm1"] - (0.003 + 0.003)) < 1e-12
    assert abs(cost["llm2"] - (0.006 + 0.0075)) < 1e-12
    assert abs(cost["root"] - 0.0195) < 1e-12
    assert cost["t1"] == 0.0

    # t2 is parallel and finishes before t1, so it is not on the critical path
    cp = critical_path(spans)
    assert cp == ["root", "ret", "llm1", "t1", "http", "llm2"], cp
    assert "t2" not in cp

    # a trace with a gap: the span after the gap is still found
    gap = [_span("r", None, "r", 0, 10), _span("a", "r", "a", 1, 3), _span("b", "r", "b", 6, 9)]
    assert critical_path(gap) == ["r", "a", "b"]

    # percentiles match numpy
    vals = [0.3, 1.2, 0.8, 2.5, 0.9, 7.0, 1.1, 0.7, 1.6, 0.4]
    for q in (0, 25, 50, 90, 95, 99, 100):
        assert abs(percentile(vals, q) - float(np.percentile(vals, q))) < 1e-12
    assert percentile([5.0], 99) == 5.0

    # exactly one root
    for bad in ([_span("a", None, "a", 0, 1), _span("b", None, "b", 0, 1)], []):
        try:
            children_map(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
    try:
        percentile([], 50)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
