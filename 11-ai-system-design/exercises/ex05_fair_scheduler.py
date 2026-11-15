"""Exercise 05 - Tenant-fair scheduling: token bucket and weighted fair queue.

TASK
----
Implement two building blocks of a multi-tenant LLM gateway.

    class TokenBucket:
        __init__(self, rate: float, burst: float)      starts full
        allow(self, now: float, cost: float = 1.0) -> bool
            Refill (now - last) * rate tokens, capped at burst. If at least `cost`
            tokens are available, spend them and return True, otherwise return False and
            spend nothing. `now` is in seconds and supplied by the caller (no real clock).
            If now is earlier than the last call, treat the elapsed time as 0.
            A cost larger than burst can never succeed.

    class WeightedFairQueue:        (self-clocked fair queueing)
        __init__(self, weights: dict[str, float])
        enqueue(self, tenant: str, cost: float, item) -> None
            Raise KeyError for an unknown tenant, ValueError if cost <= 0.
            start  = max(virtual_time, last_finish[tenant])
            finish = start + cost / weights[tenant]
            Remember last_finish[tenant] = finish.
        dequeue(self) -> tuple[str, object] | None
            Pop the item with the smallest finish tag (ties: earliest enqueued). Set
            virtual_time to that finish tag. Return (tenant, item), or None if empty.
        __len__(self) -> int

    jain_index(values: list[float]) -> float
        (sum x)^2 / (n * sum x^2). 1.0 is perfectly equal, 1/n is one tenant taking all.
        An all-zero or empty list returns 1.0.

HINTS
-----
- heapq with tuples (finish, seq, tenant, item) gives the tie-break for free. Use a
  counter for seq so items are never compared.
- virtual_time only moves forward when something is served. That is what stops a tenant
  that was idle from banking credit and then bursting.
- Cost can be tokens, so a tenant sending long prompts is charged more per request.

WHAT YOU SHOULD LEARN
---------------------
- Rate limits protect the provider quota, fair queues divide scarce capacity. You need both.
- With weights 1 and 3, a backlogged tenant gets served in a 1:3 ratio of cost, whatever
  its request sizes.
- Why a noisy neighbour cannot starve others under weighted fair queueing, and why FIFO
  across tenants fails.
"""
from __future__ import annotations

import heapq
import itertools


class TokenBucket:
    def __init__(self, rate: float, burst: float):
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last = 0.0

    def allow(self, now: float, cost: float = 1.0) -> bool:
        raise NotImplementedError("TODO")


class WeightedFairQueue:
    def __init__(self, weights: dict[str, float]):
        self.weights = dict(weights)
        self.virtual_time = 0.0
        self._last_finish = {t: 0.0 for t in weights}
        self._heap: list[tuple[float, int, str, object]] = []
        self._seq = itertools.count()

    def enqueue(self, tenant: str, cost: float, item) -> None:
        raise NotImplementedError("TODO")

    def dequeue(self):
        raise NotImplementedError("TODO")

    def __len__(self) -> int:
        raise NotImplementedError("TODO")


def jain_index(values: list[float]) -> float:
    raise NotImplementedError("TODO")


def _run_tests() -> None:
    # token bucket: burst 5, 2 tokens per second
    tb = TokenBucket(rate=2.0, burst=5.0)
    assert [tb.allow(0.0) for _ in range(6)] == [True] * 5 + [False]
    assert tb.allow(0.5) is True            # one token refilled
    assert tb.allow(0.5) is False
    assert tb.allow(100.0, cost=5.0) is True   # long idle refills only up to burst
    assert tb.allow(100.0) is False
    assert tb.allow(50.0) is False            # clock going backwards adds nothing
    big = TokenBucket(rate=10.0, burst=3.0)
    assert big.allow(1000.0, cost=4.0) is False   # larger than burst can never pass
    assert big.allow(1000.0, cost=3.0) is True
    # a rejected request spends nothing
    tb2 = TokenBucket(rate=1.0, burst=2.0)
    assert tb2.allow(0.0, cost=3.0) is False and tb2.allow(0.0, cost=2.0) is True

    # weights 1:3, both tenants backlogged with unit cost
    wfq = WeightedFairQueue({"A": 1.0, "B": 3.0})
    for i in range(10):
        wfq.enqueue("A", 1.0, f"a{i}")
    for i in range(30):
        wfq.enqueue("B", 1.0, f"b{i}")
    assert len(wfq) == 40
    first = [wfq.dequeue()[0] for _ in range(20)]
    assert first.count("A") == 5 and first.count("B") == 15
    # per-tenant FIFO is preserved
    rest = []
    while (x := wfq.dequeue()) is not None:
        rest.append(x)
    assert len(wfq) == 0 and wfq.dequeue() is None
    assert [i for t, i in rest if t == "A"] == [f"a{i}" for i in range(5, 10)]

    # a flood from one tenant does not starve the other
    wfq = WeightedFairQueue({"noisy": 1.0, "quiet": 1.0})
    for i in range(1000):
        wfq.enqueue("noisy", 1.0, i)
    for i in range(5):
        wfq.enqueue("quiet", 1.0, i)
    served = [wfq.dequeue()[0] for _ in range(10)]
    assert served.count("quiet") == 5   # all quiet requests finish in the first 10 slots

    # large requests are charged more: equal weights, A sends cost 4, B sends cost 1
    wfq = WeightedFairQueue({"A": 1.0, "B": 1.0})
    for i in range(10):
        wfq.enqueue("A", 4.0, i)
    for i in range(40):
        wfq.enqueue("B", 1.0, i)
    cost_served = {"A": 0.0, "B": 0.0}
    for _ in range(25):
        t, _ = wfq.dequeue()
        cost_served[t] += 4.0 if t == "A" else 1.0
    assert abs(cost_served["A"] - cost_served["B"]) <= 4.0

    # an idle tenant gets no banked credit when it returns
    wfq = WeightedFairQueue({"A": 1.0, "B": 1.0})
    for i in range(6):
        wfq.enqueue("B", 1.0, i)
    for _ in range(3):
        wfq.dequeue()
    for i in range(3):
        wfq.enqueue("A", 1.0, i)
    order = [wfq.dequeue()[0] for _ in range(6)]
    assert order == ["B", "A", "B", "A", "B", "A"]

    # error handling
    wfq = WeightedFairQueue({"A": 1.0})
    for call in (lambda: wfq.enqueue("Z", 1.0, 0), lambda: wfq.enqueue("A", 0.0, 0)):
        try:
            call()
        except (KeyError, ValueError):
            pass
        else:
            raise AssertionError("expected an error")

    # Jain's fairness index
    assert jain_index([1, 1, 1, 1]) == 1.0
    assert abs(jain_index([1, 0, 0, 0]) - 0.25) < 1e-12
    assert abs(jain_index([3, 1]) - 16 / 20) < 1e-12
    assert jain_index([]) == 1.0 and jain_index([0, 0]) == 1.0


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
