"""Exercise 06 - Async fan-out with safe retries, idempotency keys and a budget.

TASK
----
Fan out many LLM-style API calls at once, survive transient failures, never repeat a
side effect, and stop cleanly when a budget runs out. A deterministic fake API, an
error type and a budget counter are provided. Implement:

    make_key(run_id, index, payload) -> str
        Idempotency key: the first 16 hex chars of sha256 of f"{run_id}:{index}:{payload}".
        The key names the LOGICAL operation, so every retry of it reuses the same key.
    is_retryable(exc) -> bool
        True for ApiError with status 429 or any 5xx, and for TimeoutError. Everything
        else (4xx other than 429, ValueError, BudgetExhausted, ...) is not retryable.
    backoff_delay(attempt, base, cap, rng) -> float
        "Full jitter": rng.uniform(0, min(cap, base * 2 ** attempt)). attempt starts at 0.
        Must not overflow for huge attempt numbers (for example 5000).
    async call_with_retry(api, key, payload, *, timeout, max_attempts, base, cap,
                          rng, budget, sleep=asyncio.sleep) -> tuple[str, int]
        Up to max_attempts tries. Before EVERY attempt call budget.take() (it raises
        BudgetExhausted when spent, and that must propagate). Each attempt runs
        api.call(key, payload) under a per-call timeout. On a retryable error, sleep
        backoff_delay(...) and try again, but do not sleep after the final attempt. On a
        non-retryable error, raise CallFailed(attempts, cause) immediately. After the
        last attempt fails, raise CallFailed(max_attempts, last_exception).
        Returns (value, attempts_used).
    async fan_out(api, payloads, *, run_id, concurrency, timeout, max_attempts, base,
                  cap, seed, budget) -> list[Result]
        Run one task per payload, at most `concurrency` calls at a time
        (asyncio.Semaphore). Each task uses its own random.Random(seed * 10007 + index)
        so results do not depend on scheduling order. Return one Result per payload, in
        input order, with status "ok", "failed" (CallFailed) or "cancelled" (never
        finished because the budget ran out or fan_out itself was cancelled).
        When BudgetExhausted escapes any task, cancel every other task and wait for them
        all. When fan_out itself is cancelled, do the same. No task may be left running.

HINTS
-----
- asyncio.wait_for(coro, timeout) cancels the inner coroutine on timeout and raises
  TimeoutError. The fake API releases its in-flight counter in a finally block, so a
  correct implementation ends with api.in_flight == 0.
- CancelledError is not an Exception subclass in Python 3.8+. A bare `except Exception`
  will not swallow it, which is what you want. Never catch it and carry on.
- asyncio.wait(tasks, return_when=FIRST_EXCEPTION) wakes up as soon as the budget
  error appears. Put the cleanup in `finally`: cancel unfinished tasks, then
  `await asyncio.gather(*tasks, return_exceptions=True)` so nothing is orphaned and no
  exception goes unretrieved.
- A retry after a lost response is only safe because the server remembers the key.
  The fake API counts executed side effects per key so the tests can prove it.

WHAT YOU SHOULD LEARN
---------------------
- The four pieces of a safe fan-out: concurrency cap, per-call timeout, jittered backoff
  on retryable errors only, and idempotency keys that outlive a single attempt.
- Why a timeout does not mean "it did not happen", and how a key makes the retry safe.
- Structured cleanup: cancelling siblings and awaiting them, even when the parent is
  itself cancelled.
- How to test async code deterministically: fake clock for sleeps, scripted failures,
  millisecond latencies, assertions on counters instead of wall time.
"""
from __future__ import annotations

import asyncio
import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass


class ApiError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"status {status}")
        self.status = status


class BudgetExhausted(Exception):
    pass


class CallFailed(Exception):
    def __init__(self, attempts: int, cause: BaseException) -> None:
        super().__init__(f"failed after {attempts} attempts: {cause!r}")
        self.attempts = attempts
        self.cause = cause


@dataclass(frozen=True)
class Result:
    index: int
    status: str            # "ok", "failed" or "cancelled"
    value: str | None
    attempts: int


class Budget:
    """Shared cap on the total number of attempts across the whole fan-out."""

    def __init__(self, max_attempts: int) -> None:
        self.max_attempts = max_attempts
        self.used = 0

    def take(self) -> None:
        if self.used >= self.max_attempts:
            raise BudgetExhausted(f"budget of {self.max_attempts} attempts spent")
        self.used += 1


class FakeAPI:
    """Deterministic fake. plan maps payload -> scripted outcomes, one per REAL call.

    Outcomes: "ok", "429", "503", "400", "hang" (never answers),
    "slow" (answers far too late) and "lost" (performs the side effect, then the
    response never arrives). A repeated idempotency key replays the stored result and
    does not execute the side effect again. Unlisted calls succeed.
    """

    def __init__(self, plan: dict[str, list[str]] | None = None, latency: float = 0.002) -> None:
        self.plan = {k: list(v) for k, v in (plan or {}).items()}
        self.latency = latency
        self.calls = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self.effects: dict[str, int] = defaultdict(int)       # key -> times executed
        self.keys_seen: dict[str, set[str]] = defaultdict(set)  # payload -> keys used
        self._done: dict[str, str] = {}

    async def call(self, key: str, payload: str) -> str:
        self.calls += 1
        self.keys_seen[payload].add(key)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.latency)
            if key in self._done:
                return self._done[key]
            script = self.plan.get(payload, [])
            outcome = script.pop(0) if script else "ok"
            if outcome in ("429", "503", "400"):
                raise ApiError(int(outcome))
            if outcome in ("hang", "slow"):
                await asyncio.sleep(30)
            self.effects[key] += 1
            self._done[key] = f"result:{payload}"
            if outcome == "lost":
                await asyncio.sleep(30)
            return self._done[key]
        finally:
            self.in_flight -= 1


def make_key(run_id: str, index: int, payload: str) -> str:
    raise NotImplementedError("TODO")


def is_retryable(exc: BaseException) -> bool:
    raise NotImplementedError("TODO")


def backoff_delay(attempt: int, base: float, cap: float, rng: random.Random) -> float:
    raise NotImplementedError("TODO")


async def call_with_retry(api: FakeAPI, key: str, payload: str, *, timeout: float,
                          max_attempts: int, base: float, cap: float, rng: random.Random,
                          budget: Budget, sleep=asyncio.sleep) -> tuple[str, int]:
    raise NotImplementedError("TODO")


async def fan_out(api: FakeAPI, payloads: list[str], *, run_id: str, concurrency: int,
                  timeout: float, max_attempts: int, base: float, cap: float, seed: int,
                  budget: Budget) -> list[Result]:
    raise NotImplementedError("TODO")


def _run_tests() -> None:
    BASE, CAP = 0.001, 0.004

    def no_orphans() -> bool:
        me = asyncio.current_task()
        return not [t for t in asyncio.all_tasks() if t is not me and not t.done()]

    async def main() -> None:
        # ---- keys
        k1 = make_key("run", 0, "hello")
        assert k1 == make_key("run", 0, "hello") and len(k1) == 16
        int(k1, 16)  # hex
        assert len({k1, make_key("run", 1, "hello"), make_key("run", 0, "hellO"),
                    make_key("other", 0, "hello")}) == 4

        # ---- retryable classification
        assert is_retryable(ApiError(429)) and is_retryable(ApiError(500))
        assert is_retryable(ApiError(503)) and is_retryable(ApiError(599))
        assert is_retryable(TimeoutError())
        for exc in (ApiError(400), ApiError(401), ApiError(404), ApiError(422),
                    ValueError("x"), BudgetExhausted()):
            assert not is_retryable(exc), exc

        # ---- backoff: bounded, capped, jittered, overflow safe
        rng = random.Random(0)
        for attempt in range(12):
            bound = min(CAP, BASE * 2 ** attempt)
            draws = [backoff_delay(attempt, BASE, CAP, rng) for _ in range(200)]
            assert all(0.0 <= d <= bound + 1e-12 for d in draws), attempt
        assert 0.0 <= backoff_delay(5000, BASE, CAP, rng) <= CAP
        many = [backoff_delay(10, BASE, CAP, random.Random(i)) for i in range(2000)]
        assert abs(sum(many) / len(many) - CAP / 2) < 0.0003 and len(set(many)) > 1500
        assert backoff_delay(3, BASE, CAP, random.Random(5)) == backoff_delay(3, BASE, CAP, random.Random(5))

        # ---- call_with_retry with a recording fake sleep (no real waiting)
        delays: list[float] = []

        async def fake_sleep(d: float) -> None:
            delays.append(d)

        def kw(**over):
            d = dict(timeout=0.1, max_attempts=4, base=BASE, cap=CAP,
                     rng=random.Random(1), budget=Budget(100), sleep=fake_sleep)
            d.update(over)
            return d

        api = FakeAPI({"p": ["429", "503"]})
        value, attempts = await call_with_retry(api, "k", "p", **kw())
        assert (value, attempts) == ("result:p", 3) and api.calls == 3
        assert len(delays) == 2 and delays[0] <= BASE and delays[1] <= 2 * BASE

        delays.clear()
        api = FakeAPI({"p": ["400"]})
        try:
            await call_with_retry(api, "k", "p", **kw())
        except CallFailed as exc:
            assert exc.attempts == 1 and isinstance(exc.cause, ApiError)
        else:
            raise AssertionError("400 must fail immediately")
        assert api.calls == 1 and delays == []

        api = FakeAPI({"p": ["503"] * 10})
        try:
            await call_with_retry(api, "k", "p", **kw(max_attempts=3))
        except CallFailed as exc:
            assert exc.attempts == 3
        else:
            raise AssertionError("must give up")
        assert api.calls == 3 and len(delays) == 2, "no sleep after the final attempt"

        budget = Budget(2)
        api = FakeAPI({"p": ["503"] * 10})
        try:
            await call_with_retry(api, "k", "p", **kw(budget=budget))
        except BudgetExhausted:
            pass
        else:
            raise AssertionError("BudgetExhausted must propagate")
        assert api.calls == 2 and budget.used == 2

        # timeout: a hung call is cancelled, then retried
        api = FakeAPI({"p": ["hang"]})
        value, attempts = await call_with_retry(api, "k", "p", **kw(timeout=0.1))
        assert attempts == 2 and api.in_flight == 0

        # lost response: the side effect ran once, and the retry replays it
        api = FakeAPI({"p": ["lost"]})
        value, attempts = await call_with_retry(api, "key-1", "p", **kw(timeout=0.1))
        assert (value, attempts) == ("result:p", 2)
        assert api.effects["key-1"] == 1, "a retry must not repeat the side effect"
        assert api.calls == 2 and api.in_flight == 0
        assert no_orphans()

        # ---- fan_out: success, retries, failures, cap
        payloads = [f"q{i}" for i in range(9)]
        api = FakeAPI({"q1": ["429"], "q4": ["503", "429"], "q7": ["hang"], "q8": ["lost"]})
        res = await fan_out(api, payloads, run_id="r1", concurrency=3, timeout=0.1,
                            max_attempts=4, base=BASE, cap=CAP, seed=7, budget=Budget(1000))
        assert [r.index for r in res] == list(range(9))
        assert all(r.status == "ok" for r in res)
        assert [r.value for r in res] == [f"result:q{i}" for i in range(9)]
        assert res[0].attempts == 1 and res[1].attempts == 2 and res[4].attempts == 3
        assert res[7].attempts == 2 and res[8].attempts == 2
        assert api.max_in_flight == 3, api.max_in_flight
        assert all(n == 1 for n in api.effects.values()) and len(api.effects) == 9
        assert all(len(keys) == 1 for keys in api.keys_seen.values()), "key must be reused on retry"
        assert api.in_flight == 0 and no_orphans()

        # one bad payload does not stop the others
        api = FakeAPI({"bad": ["400"], "flaky": ["503"] * 20})
        res = await fan_out(api, ["a", "bad", "flaky", "b"], run_id="r2", concurrency=2,
                            timeout=0.1, max_attempts=3, base=BASE, cap=CAP, seed=1,
                            budget=Budget(1000))
        assert [r.status for r in res] == ["ok", "failed", "failed", "ok"]
        assert res[1].attempts == 1 and res[2].attempts == 3 and res[1].value is None
        assert no_orphans()

        # deterministic: same seed, same outcome
        def run_once():
            return fan_out(FakeAPI({"q1": ["429", "503"], "q3": ["hang"]}), payloads[:5],
                           run_id="r3", concurrency=2, timeout=0.1, max_attempts=4,
                           base=BASE, cap=CAP, seed=3, budget=Budget(100))
        assert await run_once() == await run_once()

        # ---- budget exhaustion, serial: five attempts, then everything left is cancelled
        api = FakeAPI()
        budget = Budget(5)
        res = await fan_out(api, [f"s{i}" for i in range(10)], run_id="r4", concurrency=1,
                            timeout=0.1, max_attempts=3, base=BASE, cap=CAP, seed=1, budget=budget)
        assert [r.status for r in res] == ["ok"] * 5 + ["cancelled"] * 5
        assert budget.used == 5 and api.calls == 5
        assert api.in_flight == 0 and no_orphans()

        # ---- budget exhaustion while calls are in flight: they must be cancelled
        api = FakeAPI({"p0": ["slow"], "p1": ["503"] * 100, "p2": ["503"] * 100})
        budget = Budget(6)
        res = await fan_out(api, ["p0", "p1", "p2", "p3"], run_id="r5", concurrency=3,
                            timeout=5.0, max_attempts=1000, base=BASE, cap=CAP, seed=2, budget=budget)
        assert all(r.status == "cancelled" for r in res), res
        assert budget.used == 6 and api.calls == 6
        assert api.in_flight == 0, "in-flight calls must be cancelled, not abandoned"
        assert no_orphans()

        # ---- the caller cancels fan_out itself
        api = FakeAPI({f"w{i}": ["slow"] for i in range(6)})
        task = asyncio.create_task(fan_out(
            api, [f"w{i}" for i in range(6)], run_id="r6", concurrency=3, timeout=5.0,
            max_attempts=3, base=BASE, cap=CAP, seed=1, budget=Budget(100)))
        await asyncio.sleep(0.02)
        assert api.in_flight == 3
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("fan_out should propagate its own cancellation")
        assert api.in_flight == 0 and no_orphans()

    asyncio.run(main())


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
