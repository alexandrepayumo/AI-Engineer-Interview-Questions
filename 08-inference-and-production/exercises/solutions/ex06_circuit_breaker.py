"""Exercise 06 - A circuit breaker, jittered backoff and a fallback router.

TASK
----
Implement a deterministic (injected clock) reliability toolkit for calling LLM providers.

    backoff_delay(attempt, base, cap, rng) -> float
        Full jitter: rng.uniform(0, min(cap, base * 2 ** attempt)). `attempt` starts at 0.
        `rng` is a random.Random.

    CircuitBreaker(failure_threshold, reset_timeout, clock, half_open_max_calls=1)
        `clock` is a zero-argument callable returning seconds.
        state -> "closed" | "open" | "half_open"   (property)
        allow() -> bool
        record_success() -> None
        record_failure() -> None
      Behaviour:
        closed:    allow() is True. A failure increments a consecutive-failure counter, a success
                   resets it. Reaching failure_threshold opens the breaker and stamps opened_at.
        open:      allow() is False. After reset_timeout seconds the state reads "half_open".
        half_open: allow() admits at most half_open_max_calls probe calls (count them as they are
                   admitted). A probe success closes the breaker and clears counters. A probe
                   failure re-opens it with a fresh timer.
        Results recorded while the breaker is open are ignored (late responses).

    AllProvidersFailed(Exception)

    FallbackRouter(providers, breakers)
        providers: list of (name, callable(request) -> result), in priority order.
        breakers: dict name -> CircuitBreaker.
        call(request) -> (name, result)
          For each provider in order: skip it if its breaker does not allow() the call. Otherwise
          call it. On success record_success and return. On any Exception record_failure and
          move to the next provider. If none succeeded raise AllProvidersFailed.

HINTS
-----
- Make `state` do the open -> half_open transition lazily by comparing clock() with opened_at.
- Reset the half-open probe counter whenever you enter the open state.
- A skipped provider must not be called at all. That is the point of the breaker.

WHAT YOU SHOULD LEARN
---------------------
- A breaker converts a slow failure (timeouts that pile up) into a fast one and gives the
  provider room to recover. Half-open limits how much traffic a recovering provider sees.
- Full jitter spreads retries out so clients do not synchronise into a retry storm.
- Failover order is policy: the router never calls a provider whose breaker is open.
"""
from __future__ import annotations

import random
from typing import Any, Callable


def backoff_delay(attempt: int, base: float, cap: float, rng: random.Random) -> float:
    """Full-jitter exponential backoff delay in seconds."""
    return rng.uniform(0.0, min(cap, base * (2 ** attempt)))


class CircuitBreaker:
    def __init__(self, failure_threshold: int, reset_timeout: float,
                 clock: Callable[[], float], half_open_max_calls: int = 1) -> None:
        """Create a closed breaker."""
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.clock = clock
        self.half_open_max_calls = half_open_max_calls
        self._failures = 0
        self._opened_at = 0.0
        self._open = False                                     # True from the trip until a probe succeeds
        self._probes = 0

    @property
    def state(self) -> str:
        """Current state, applying the timed open -> half_open transition."""
        if not self._open:
            return "closed"
        if self.clock() - self._opened_at >= self.reset_timeout:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        """May a call proceed right now?"""
        s = self.state
        if s == "closed":
            return True
        if s == "open":
            return False
        if self._probes < self.half_open_max_calls:            # half-open: limited probes
            self._probes += 1
            return True
        return False

    def _trip(self) -> None:
        self._open = True
        self._opened_at = self.clock()
        self._probes = 0

    def record_success(self) -> None:
        """Report a successful call."""
        s = self.state
        if s == "open":
            return                                             # late result, ignore
        self._failures = 0
        if s == "half_open":
            self._open = False                                 # probe succeeded: close
            self._probes = 0

    def record_failure(self) -> None:
        """Report a failed call."""
        s = self.state
        if s == "open":
            return
        if s == "half_open":
            self._trip()                                       # probe failed: fresh timer
            return
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._trip()


class AllProvidersFailed(Exception):
    """Raised when every provider failed or was skipped."""


class FallbackRouter:
    def __init__(self, providers: list[tuple[str, Callable[[Any], Any]]],
                 breakers: dict[str, CircuitBreaker]) -> None:
        """Ordered providers with one breaker each."""
        self.providers = providers
        self.breakers = breakers

    def call(self, request: Any) -> tuple[str, Any]:
        """Try providers in priority order, honouring breakers."""
        for name, fn in self.providers:
            breaker = self.breakers[name]
            if not breaker.allow():
                continue                                       # fail fast, never touch the provider
            try:
                result = fn(request)
            except Exception:
                breaker.record_failure()
                continue
            breaker.record_success()
            return name, result
        raise AllProvidersFailed("no provider available")


def _run_tests() -> None:
    # backoff: bounded by min(cap, base * 2^attempt), and the ceiling grows then saturates
    rng = random.Random(0)
    for attempt, ceiling in [(0, 1.0), (1, 2.0), (2, 4.0), (5, 30.0), (10, 30.0)]:
        draws = [backoff_delay(attempt, 1.0, 30.0, rng) for _ in range(500)]
        assert 0.0 <= min(draws) and max(draws) <= ceiling
        assert max(draws) > 0.9 * ceiling                       # the range is really used
    assert abs(sum(backoff_delay(3, 1.0, 30.0, random.Random(i)) for i in range(2000)) / 2000 - 4.0) < 0.3

    # breaker state machine with a fake clock
    now = [0.0]
    br = CircuitBreaker(failure_threshold=3, reset_timeout=10.0, clock=lambda: now[0])
    assert br.state == "closed" and br.allow()
    br.record_failure()
    br.record_failure()
    br.record_success()                                         # resets the consecutive count
    br.record_failure()
    br.record_failure()
    assert br.state == "closed"
    br.record_failure()                                         # third consecutive failure trips it
    assert br.state == "open" and not br.allow()
    now[0] = 9.9
    assert br.state == "open" and not br.allow()
    br.record_success()                                         # late success while open is ignored
    assert br.state == "open"
    now[0] = 10.0
    assert br.state == "half_open"
    assert br.allow() is True                                   # the single probe
    assert br.allow() is False                                  # no second concurrent probe
    br.record_failure()                                         # probe failed: open again, new timer
    assert br.state == "open"
    now[0] = 19.9
    assert br.state == "open"
    now[0] = 20.0
    assert br.state == "half_open" and br.allow()
    br.record_success()                                         # probe succeeded: closed again
    assert br.state == "closed" and br.allow()
    br.record_failure()
    assert br.state == "closed"                                 # counter was cleared

    # several probes allowed
    now[0] = 0.0
    br2 = CircuitBreaker(1, 5.0, lambda: now[0], half_open_max_calls=2)
    br2.record_failure()
    now[0] = 5.0
    assert [br2.allow(), br2.allow(), br2.allow()] == [True, True, False]

    # router: primary fails, secondary serves, then the breaker keeps traffic off the primary
    calls = {"primary": 0, "secondary": 0, "local": 0}

    def primary(req):
        calls["primary"] += 1
        raise TimeoutError("provider down")

    def secondary(req):
        calls["secondary"] += 1
        return f"secondary:{req}"

    def local(req):
        calls["local"] += 1
        return f"local:{req}"

    now[0] = 0.0
    breakers = {n: CircuitBreaker(2, 30.0, lambda: now[0]) for n in ("primary", "secondary", "local")}
    router = FallbackRouter([("primary", primary), ("secondary", secondary), ("local", local)], breakers)
    assert router.call("a") == ("secondary", "secondary:a")
    assert router.call("b") == ("secondary", "secondary:b")
    assert calls["primary"] == 2 and breakers["primary"].state == "open"
    assert router.call("c") == ("secondary", "secondary:c")
    assert calls["primary"] == 2                                # breaker open: primary not called at all
    assert calls["local"] == 0

    # after the timeout one probe reaches the primary, fails, and the breaker re-opens
    now[0] = 30.0
    router.call("d")
    assert calls["primary"] == 3 and breakers["primary"].state == "open"

    # everything down -> AllProvidersFailed, and breakers of failing providers trip
    def boom(req):
        raise RuntimeError("down")

    dead = FallbackRouter([("x", boom), ("y", boom)],
                          {"x": CircuitBreaker(1, 5.0, lambda: now[0]), "y": CircuitBreaker(1, 5.0, lambda: now[0])})
    try:
        dead.call("z")
        raise AssertionError("expected AllProvidersFailed")
    except AllProvidersFailed:
        pass
    try:
        dead.call("z")                                          # both breakers open: nothing is even called
        raise AssertionError("expected AllProvidersFailed")
    except AllProvidersFailed:
        pass
    assert dead.breakers["x"].state == "open" and dead.breakers["y"].state == "open"


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
