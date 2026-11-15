"""Exercise 06 - Output moderation pipeline with severity tiers and policy actions.

TASK
----
Classifier scores per harm category arrive as a dict {category: score in [0, 1]}. Turn
them into one policy decision. (No model is called here. You are building the decision
layer that sits after the classifiers.)

    TIERS = ["none", "low", "medium", "high", "critical"]
    ACTIONS = {"none": "allow", "low": "allow_log", "medium": "warn",
               "high": "block", "critical": "block_and_escalate"}

    tier_for(score, thresholds, strict=1.0) -> str
        thresholds is a 4-tuple (low, medium, high, critical) in ascending order. The
        effective thresholds are each multiplied by strict (strict < 1 is stricter). The
        tier is TIERS[k], where k is the number of effective thresholds the score meets
        or exceeds (score >= threshold).

    moderate(scores, policy, strict=1.0, required=(), fail_closed=True) -> dict
        policy maps category -> thresholds tuple. Categories in scores but not in policy
        are ignored. Return
            {"tier": overall_tier, "action": ACTIONS[overall_tier],
             "reasons": [(category, tier), ...]}
        where overall_tier is the highest tier over categories, and reasons lists every
        category whose tier is not "none", sorted by tier severity descending, then by
        category name.
        A category listed in required but missing from scores means a classifier failed
        or timed out. If fail_closed, treat it as tier "high" with reason (category,
        "missing"). If not fail_closed, ignore it. "missing" sorts as "high".

    tally(decisions) -> dict[str, int]
        Count decisions by action (only actions that occurred).

    stream_gate(chunks, score_fn, policy, strict=1.0) -> tuple[list[str], dict | None]
        Streaming output check. Process chunks in order, with the text so far being the
        concatenation of the chunks released plus the new chunk. score_fn(text) returns a
        scores dict. If moderate(...)'s action for the cumulative text is "block" or
        "block_and_escalate", stop and return (released_chunks, decision) without
        releasing the offending chunk. Otherwise release it. If every chunk passes return
        (all_chunks, None).

HINTS
-----
- Keep severity as an index into TIERS so max() and sorting are trivial.
- Decide the "missing" semantics before coding: it is a safety control, so the default
  is to fail closed.

WHAT YOU SHOULD LEARN
---------------------
- Moderation is a policy table, not a boolean. Tiers let you log cheap events, warn on
  borderline ones and escalate rare severe ones to humans.
- Per-category thresholds and a strictness multiplier (for example a child-safe surface)
  are configuration, not code.
- A failed classifier is a decision you must make explicitly (fail open or closed).
- In streaming you can only withhold what you have not yet sent, so the check must run on
  the cumulative text and before release.
"""
from __future__ import annotations

TIERS = ["none", "low", "medium", "high", "critical"]
ACTIONS = {"none": "allow", "low": "allow_log", "medium": "warn",
           "high": "block", "critical": "block_and_escalate"}


def tier_for(score: float, thresholds: tuple[float, float, float, float],
             strict: float = 1.0) -> str:
    raise NotImplementedError("TODO")


def moderate(scores: dict[str, float], policy: dict[str, tuple[float, float, float, float]],
             strict: float = 1.0, required: tuple[str, ...] = (),
             fail_closed: bool = True) -> dict:
    raise NotImplementedError("TODO")


def tally(decisions: list[dict]) -> dict[str, int]:
    raise NotImplementedError("TODO")


def stream_gate(chunks: list[str], score_fn, policy: dict[str, tuple[float, float, float, float]],
                strict: float = 1.0) -> tuple[list[str], dict | None]:
    raise NotImplementedError("TODO")


def _run_tests() -> None:
    thr = (0.2, 0.5, 0.8, 0.95)
    assert tier_for(0.0, thr) == "none"
    assert tier_for(0.19, thr) == "none"
    assert tier_for(0.2, thr) == "low"          # boundary is inclusive
    assert tier_for(0.5, thr) == "medium"
    assert tier_for(0.79, thr) == "medium"
    assert tier_for(0.8, thr) == "high"
    assert tier_for(0.95, thr) == "critical"
    assert tier_for(1.0, thr) == "critical"
    # strict < 1 shifts every threshold down: at 0.5 the cuts are 0.1, 0.25, 0.4, 0.475
    assert tier_for(0.45, thr, strict=0.5) == "high"
    assert tier_for(0.45, thr, strict=1.0) == "low"
    assert tier_for(0.45, thr, strict=3.0) == "none"

    policy = {
        "violence": (0.2, 0.5, 0.8, 0.95),
        "self_harm": (0.1, 0.3, 0.6, 0.9),
        "harassment": (0.3, 0.6, 0.85, 2.0),   # 2.0 disables "critical"
    }
    d = moderate({"violence": 0.1, "self_harm": 0.05, "harassment": 0.1}, policy)
    assert d == {"tier": "none", "action": "allow", "reasons": []}

    d = moderate({"violence": 0.55, "self_harm": 0.35, "harassment": 0.99}, policy)
    assert d["tier"] == "high" and d["action"] == "block"  # harassment 0.99 caps at high
    assert d["reasons"] == [("harassment", "high"), ("self_harm", "medium"),
                            ("violence", "medium")]

    d = moderate({"violence": 0.97, "unknown_cat": 1.0}, policy)
    assert d["tier"] == "critical" and d["action"] == "block_and_escalate"
    assert d["reasons"] == [("violence", "critical")]  # unknown categories ignored

    d = moderate({"violence": 0.25}, policy)
    assert d["tier"] == "low" and d["action"] == "allow_log"

    # strictness changes the outcome for the same scores
    sc = {"violence": 0.3}
    assert moderate(sc, policy)["action"] == "allow_log"
    assert moderate(sc, policy, strict=0.5)["action"] == "warn"

    # a failed classifier: fail closed by default, and only for required categories
    d = moderate({"violence": 0.0}, policy, required=("violence", "self_harm"))
    assert d["tier"] == "high" and d["reasons"] == [("self_harm", "missing")]
    d = moderate({"violence": 0.0}, policy, required=("self_harm",), fail_closed=False)
    assert d["tier"] == "none"
    d = moderate({"violence": 0.0}, policy)
    assert d["tier"] == "none"  # not required, so missing is fine
    # missing and a real high tie on severity, so they sort by name
    d = moderate({"violence": 0.85}, policy, required=("self_harm",))
    assert d["reasons"] == [("self_harm", "missing"), ("violence", "high")]

    decisions = [moderate({"violence": v}, policy) for v in (0.0, 0.1, 0.25, 0.3, 0.9, 0.99)]
    assert tally(decisions) == {"allow": 2, "allow_log": 2, "block": 1, "block_and_escalate": 1}
    assert tally([]) == {}

    # streaming: the threat only becomes visible once the second chunk arrives
    def score_fn(text: str) -> dict[str, float]:
        return {"violence": 0.9 if "attack" in text and "plan" in text else 0.0}

    chunks = ["Here is the ", "plan to ", "attack the ", "server."]
    released, dec = stream_gate(chunks, score_fn, policy)
    assert released == ["Here is the ", "plan to "] and dec is not None
    assert dec["action"] == "block"
    released, dec = stream_gate(["all ", "fine"], score_fn, policy)
    assert released == ["all ", "fine"] and dec is None
    released, dec = stream_gate([], score_fn, policy)
    assert released == [] and dec is None


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
