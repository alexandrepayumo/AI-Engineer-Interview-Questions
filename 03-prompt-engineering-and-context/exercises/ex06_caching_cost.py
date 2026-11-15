"""Exercise 06 - Prompt caching cost calculator.

TASK
----
Prices are per million tokens. A Pricing has: input (USD per 1M input tokens),
output (USD per 1M output tokens), write_mult (cache write price as a multiple of
input, e.g. 1.25) and read_mult (cache read price as a multiple, e.g. 0.1).

    @dataclass(frozen=True)
    class Pricing:
        input: float
        output: float
        write_mult: float
        read_mult: float

    def request_cost(prefix: int, suffix: int, out: int, p: Pricing, mode: str) -> float
        '''USD cost of one request. `prefix` tokens are the cacheable stable prefix,
        `suffix` tokens are the uncached tail (volatile part), `out` output tokens.
        mode "none": no caching, all input at the base rate.
        mode "write": prefix billed at write_mult x input (cache miss that stores).
        mode "hit": prefix billed at read_mult x input.
        Suffix is always base rate. Unknown mode -> ValueError.'''

    def expected_cost(prefix: int, suffix: int, out: int, p: Pricing, hit_rate: float) -> float
        '''Average cost per request when a fraction hit_rate of requests are hits and
        the rest are misses that write the cache.'''

    def break_even_hit_rate(write_mult: float, read_mult: float) -> float
        '''Hit rate h at which caching costs the same as no caching on the prefix:
        h * read_mult + (1 - h) * write_mult = 1. Return max(0.0, that h) (0 when
        writes are free or cheaper than base). Raise ValueError if read_mult >= write_mult.'''

    def agent_loop_input_cost(initial: int, step: int, steps: int, price_per_m: float,
                              write_mult: float, read_mult: float, cached: bool) -> float
        '''Input-side USD cost of an agent loop. At step i (1-based) the request
        carries initial + i*step tokens (history grows by `step` each turn).
        Uncached: every token at the base price every step.
        Cached: step 1 writes all its tokens; at step i > 1 the previous request's
        tokens (initial + (i-1)*step) are read at read_mult and the `step` new tokens
        are written at write_mult.'''

    def hit_probability(requests_per_minute: float, ttl_minutes: float) -> float
        '''For Poisson arrivals where every hit refreshes the TTL, the chance that the
        next request arrives before the entry expires: 1 - exp(-rate * ttl).'''

HINTS
-----
- Work in "token-equivalents" (tokens x multiplier) and convert to dollars at the end.
- Break-even: solve h*r + (1-h)*w = 1 for h, giving h = (w - 1) / (w - r).

WHAT YOU SHOULD LEARN
---------------------
Caching is not free: a write costs more than a plain read of the same tokens, so a
prefix that is never re-used loses money. The break-even hit rate is low (about 22%
for 1.25x writes and 0.1x reads), and agent loops, which re-send a growing prefix,
sit far above it. Traffic spacing versus TTL decides whether you actually get hits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Pricing:
    input: float
    output: float
    write_mult: float
    read_mult: float


def request_cost(prefix: int, suffix: int, out: int, p: Pricing, mode: str) -> float:
    raise NotImplementedError("TODO")


def expected_cost(prefix: int, suffix: int, out: int, p: Pricing, hit_rate: float) -> float:
    raise NotImplementedError("TODO")


def break_even_hit_rate(write_mult: float, read_mult: float) -> float:
    raise NotImplementedError("TODO")


def agent_loop_input_cost(initial: int, step: int, steps: int, price_per_m: float,
                          write_mult: float, read_mult: float, cached: bool) -> float:
    raise NotImplementedError("TODO")


def hit_probability(requests_per_minute: float, ttl_minutes: float) -> float:
    raise NotImplementedError("TODO")


def _run_tests() -> None:
    p = Pricing(input=3.0, output=15.0, write_mult=1.25, read_mult=0.1)

    # 10,000-token prefix, 500 suffix, 300 output
    none = request_cost(10_000, 500, 300, p, "none")
    assert abs(none - (10_500 * 3 + 300 * 15) / 1e6) < 1e-12 and abs(none - 0.036) < 1e-9
    write = request_cost(10_000, 500, 300, p, "write")
    assert abs(write - (12_500 + 500) * 3 / 1e6 - 300 * 15 / 1e6) < 1e-12   # 0.0435
    hit = request_cost(10_000, 500, 300, p, "hit")
    assert abs(hit - 0.0090) < 1e-9
    assert hit < none < write
    try:
        request_cost(1, 1, 1, p, "bogus")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    # expected cost interpolates between write and hit
    assert abs(expected_cost(10_000, 500, 300, p, 0.0) - write) < 1e-12
    assert abs(expected_cost(10_000, 500, 300, p, 1.0) - hit) < 1e-12
    assert abs(expected_cost(10_000, 500, 300, p, 0.5) - (write + hit) / 2) < 1e-12

    # break-even
    h = break_even_hit_rate(1.25, 0.1)
    assert abs(h - 0.25 / 1.15) < 1e-12 and abs(h - 0.2174) < 1e-4
    assert abs(break_even_hit_rate(2.0, 0.1) - 1.0 / 1.9) < 1e-12
    assert break_even_hit_rate(1.0, 0.1) == 0.0       # free writes: always profitable
    assert break_even_hit_rate(0.9, 0.1) == 0.0
    try:
        break_even_hit_rate(1.0, 1.0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    # at the break-even rate the prefix costs the same as uncached
    eq = h * 0.1 + (1 - h) * 1.25
    assert abs(eq - 1.0) < 1e-12

    # agent loop worked example: S=10,000, d=1,000, 10 steps, $3 / 1M
    unc = agent_loop_input_cost(10_000, 1_000, 10, 3.0, 1.25, 0.1, cached=False)
    cac = agent_loop_input_cost(10_000, 1_000, 10, 3.0, 1.25, 0.1, cached=True)
    assert abs(unc - 0.465) < 1e-9                      # 155,000 tokens
    assert abs(cac - 0.1155) < 1e-9                     # 38,500 token-equivalents
    assert 4.0 < unc / cac < 4.1
    # one step: caching only costs extra (pure write)
    one_u = agent_loop_input_cost(10_000, 1_000, 1, 3.0, 1.25, 0.1, cached=False)
    one_c = agent_loop_input_cost(10_000, 1_000, 1, 3.0, 1.25, 0.1, cached=True)
    assert abs(one_c / one_u - 1.25) < 1e-12
    # longer loops widen the gap
    r20 = (agent_loop_input_cost(10_000, 1_000, 20, 3.0, 1.25, 0.1, False)
           / agent_loop_input_cost(10_000, 1_000, 20, 3.0, 1.25, 0.1, True))
    assert r20 > unc / cac

    # traffic vs TTL
    assert abs(hit_probability(1.0, 5.0) - (1 - math.exp(-5))) < 1e-12
    assert hit_probability(1.0, 5.0) > 0.99
    assert hit_probability(0.01, 5.0) < 0.05             # one request per 100 minutes: cache mostly expires
    assert hit_probability(0.0, 5.0) == 0.0


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
