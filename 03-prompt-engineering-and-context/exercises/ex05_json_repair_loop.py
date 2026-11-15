"""Exercise 05 - JSON validation and a repair loop against a mock LLM.

TASK
----
Schemas use a small JSON-Schema-like subset (dicts):
    {"type": "object", "required": [...], "properties": {name: schema},
     "additionalProperties": False}
    {"type": "string", "enum": [...]}   {"type": "integer", "minimum": 0, "maximum": 5}
    {"type": "number"}  {"type": "boolean"}  {"type": "array", "items": schema}

    def extract_json(text: str) -> object
        '''Find the first JSON object in model output that may contain prose and
        ```json fences. Scan for the first "{", then find its matching "}" while
        respecting string literals and escapes, and json.loads that slice.
        Raise ValueError if there is no balanced object or it does not parse.'''

    def validate(value: object, schema: dict, path: str = "$") -> list[str]
        '''Return a list of human-readable error strings (empty list = valid).
        Messages start with the path, e.g. "$.age: expected integer, got str".
        booleans are NOT integers or numbers. Missing required keys:
        "$: missing required key 'name'". Extra keys when additionalProperties is
        False: "$: unexpected key 'x'". Enum: "$.label: 'x' not in ['a', 'b']".
        Report all errors, not just the first.'''

    def repair_loop(llm, prompt: str, schema: dict, max_attempts: int = 3)
            -> tuple[object | None, int]
        '''Call llm(prompt_i) up to max_attempts times. On a parse or validation
        failure, the next prompt is the ORIGINAL prompt plus a feedback section:
            "\\n\\nYour previous reply was invalid:\\n" + previous_reply +
            "\\nErrors:\\n- " + "\\n- ".join(errors) + "\\nReturn only corrected JSON."
        Return (parsed_value, attempts_used) on success, (None, max_attempts) on failure.'''

HINTS
-----
- Recurse in validate; build child paths like f"{path}.{key}" and f"{path}[{i}]".
- A parse failure is an error too: feed the exception message back.
- Never loop forever: the cap is part of the contract.

WHAT YOU SHOULD LEARN
---------------------
Prompting for JSON gives you "usually valid" output, not a guarantee, so a
production pipeline validates and retries with specific error feedback. Constrained
decoding removes syntax errors but not semantic ones (a valid enum value can still
be the wrong label), so validation stays even when decoding is constrained.
"""
from __future__ import annotations

import json

_TYPES = {
    "string": str,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def extract_json(text: str) -> object:
    raise NotImplementedError("TODO")


def validate(value: object, schema: dict, path: str = "$") -> list[str]:
    raise NotImplementedError("TODO")


def repair_loop(llm, prompt: str, schema: dict, max_attempts: int = 3) -> tuple[object | None, int]:
    raise NotImplementedError("TODO")


class MockLLM:
    """Returns scripted replies in order and records every prompt it receives."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.replies[min(len(self.prompts) - 1, len(self.replies) - 1)]


SCHEMA = {
    "type": "object",
    "required": ["label", "score"],
    "additionalProperties": False,
    "properties": {
        "label": {"type": "string", "enum": ["spam", "ham"]},
        "score": {"type": "integer", "minimum": 0, "maximum": 5},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
}


def _run_tests() -> None:
    assert extract_json('Sure! ```json\n{"a": 1}\n``` done') == {"a": 1}
    assert extract_json('x {"a": "}{", "b": {"c": [1, 2]}} y') == {"a": "}{", "b": {"c": [1, 2]}}
    assert extract_json('{"q": "say \\"hi\\" {"}') == {"q": 'say "hi" {'}
    for bad in ("no braces", '{"a": 1', "{a: 1}"):
        try:
            extract_json(bad)
            raise AssertionError("expected ValueError for " + bad)
        except ValueError:
            pass

    assert validate({"label": "spam", "score": 3, "tags": ["a"]}, SCHEMA) == []
    assert validate({"label": "spam", "score": 3}, SCHEMA) == []
    errs = validate({"label": "maybe", "score": "3", "extra": 1}, SCHEMA)
    assert "$.label: 'maybe' not in ['spam', 'ham']" in errs
    assert "$.score: expected integer, got str" in errs
    assert "$: unexpected key 'extra'" in errs
    assert len(errs) == 3
    assert validate({"label": "ham"}, SCHEMA) == ["$: missing required key 'score'"]
    assert validate({"label": "ham", "score": True}, SCHEMA) == ["$.score: expected integer, got bool"]
    assert validate({"label": "ham", "score": 9}, SCHEMA) == ["$.score: 9 is above maximum 5"]
    assert validate({"label": "ham", "score": 1, "tags": ["a", 2]}, SCHEMA) == ["$.tags[1]: expected string, got int"]
    assert validate([1], SCHEMA) == ["$: expected object, got list"]
    assert validate(1.5, {"type": "number"}) == []
    assert validate(True, {"type": "number"}) == ["$: expected number, got bool"]

    # succeeds first time
    llm = MockLLM(['{"label": "ham", "score": 1}'])
    assert repair_loop(llm, "classify", SCHEMA) == ({"label": "ham", "score": 1}, 1)
    assert llm.prompts == ["classify"]

    # fenced + wrong type first, valid second; feedback is in the 2nd prompt
    llm = MockLLM(['```json\n{"label": "ham", "score": "high"}\n```', '{"label": "ham", "score": 4}'])
    value, attempts = repair_loop(llm, "classify", SCHEMA)
    assert value == {"label": "ham", "score": 4} and attempts == 2
    assert llm.prompts[1].startswith("classify\n\nYour previous reply was invalid:")
    assert "$.score: expected integer, got str" in llm.prompts[1]
    assert llm.prompts[1].endswith("Return only corrected JSON.")

    # prompts do not snowball: third prompt is built from the original, not the second
    llm = MockLLM(["not json", "still {not json", '{"label": "spam", "score": 0}'])
    value, attempts = repair_loop(llm, "classify", SCHEMA, max_attempts=3)
    assert attempts == 3 and value == {"label": "spam", "score": 0}
    assert llm.prompts[2].count("Your previous reply was invalid") == 1
    assert "still {not json" in llm.prompts[2] and "Your previous reply was invalid:\nnot json" not in llm.prompts[2]

    # gives up
    llm = MockLLM(["nope"])
    assert repair_loop(llm, "classify", SCHEMA, max_attempts=2) == (None, 2)
    assert len(llm.prompts) == 2


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
