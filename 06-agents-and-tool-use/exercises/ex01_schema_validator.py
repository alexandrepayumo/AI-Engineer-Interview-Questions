"""Exercise 01 - A JSON-Schema subset validator that produces model-readable errors.

TASK
----
Implement:

    json_type(value) -> str
        One of "null", "boolean", "integer", "number", "string", "array", "object".
        Python bool is NOT an integer here. A float is "number" even when it is whole
        (3.0 is "number", 3 is "integer").

    validate(schema, value, path="$") -> list[str]
        Return a list of error strings (empty list means valid). Supported keywords:
        type ("integer" also accepts nothing but ints; "number" accepts ints and floats),
        enum, minimum, maximum, minLength, maxLength, minItems, maxItems, items,
        properties, required, additionalProperties (only the value False is supported).
        If the type check fails return just that one error and check nothing else.
        Otherwise check in this order: enum, numeric bounds, string lengths, array
        checks (minItems, maxItems, then each item in order), object checks (required
        keys in the order listed in the schema, then properties in schema order for keys
        that are present, then unexpected properties in the order they appear in value).
        Error formats (exact):
            type:       "{path}: expected {type}, got {actual}"
            required:   "{path}.{key}: required property missing"
            additional: "{path}.{key}: unexpected property"
            enum:       "{path}: {value!r} not in {enum}"
            minimum:    "{path}: {value} < minimum {m}"
            maximum:    "{path}: {value} > maximum {m}"
            minLength:  "{path}: length {n} < minLength {m}"
            maxLength:  "{path}: length {n} > maxLength {m}"
            minItems:   "{path}: {n} items < minItems {m}"
            maxItems:   "{path}: {n} items > maxItems {m}"
        Child paths: "{path}.{key}" for object properties, "{path}[{i}]" for array items.

    validate_call(tools, name, args) -> tuple[bool, str]
        tools maps tool name -> input schema. Returns (ok, message_for_the_model).
        Unknown tool: (False, "Unknown tool 'x'. Available tools: a, b") with names sorted.
        If difflib.get_close_matches(name, tools, n=1) finds a match, insert
        " Did you mean 'a'?" before " Available tools: ...".
        Invalid args: (False, "Invalid arguments for 'name':\\n- err1\\n- err2").
        Valid: (True, "").

HINTS
-----
- bool is a subclass of int in Python. Check bool before int in json_type.
- Return early after a type mismatch: "expected object, got string" makes every other
  message about that value meaningless.
- The message is an observation the model will read. Name the path and say what would be
  valid (enum values, bounds) instead of a bare "invalid".

WHAT YOU SHOULD LEARN
---------------------
- A well-formed JSON call is not a valid call: schema validation is a separate layer.
- Why validation errors belong in the tool result (the model can repair them) rather
  than in an exception that kills the run.
- The bool-is-int trap and why strict types matter for tool arguments.
"""
from __future__ import annotations

import difflib


def json_type(value) -> str:
    # bool first: isinstance(True, int) is True in Python
    raise NotImplementedError("TODO")


def _type_ok(expected: str, actual: str) -> bool:
    if expected == "number":
        return actual in ("integer", "number")  # every integer is also a number
    return expected == actual


def validate(schema: dict, value, path: str = "$") -> list[str]:
    raise NotImplementedError("TODO")


def validate_call(tools: dict[str, dict], name: str, args) -> tuple[bool, str]:
    raise NotImplementedError("TODO")


def _run_tests() -> None:
    # json_type, including the bool trap
    assert json_type(None) == "null"
    assert json_type(True) == "boolean"
    assert json_type(3) == "integer"
    assert json_type(3.0) == "number"
    assert json_type("x") == "string"
    assert json_type([1]) == "array"
    assert json_type({"a": 1}) == "object"

    # type mismatches stop further checks
    assert validate({"type": "string", "minLength": 5}, 7) == ["$: expected string, got integer"]
    assert validate({"type": "integer"}, True) == ["$: expected integer, got boolean"]
    assert validate({"type": "integer"}, 3.0) == ["$: expected integer, got number"]
    assert validate({"type": "number"}, 3) == []
    assert validate({"type": "number"}, 2.5) == []

    # enum and bounds
    status = {"type": "string", "enum": ["open", "closed"]}
    assert validate(status, "open") == []
    assert validate(status, "done") == ["$: 'done' not in ['open', 'closed']"]
    num = {"type": "integer", "minimum": 1, "maximum": 100}
    assert validate(num, 50) == []
    assert validate(num, 0) == ["$: 0 < minimum 1"]
    assert validate(num, 101) == ["$: 101 > maximum 100"]
    s = {"type": "string", "minLength": 2, "maxLength": 4}
    assert validate(s, "a") == ["$: length 1 < minLength 2"]
    assert validate(s, "abcde") == ["$: length 5 > maxLength 4"]
    assert validate(s, "abc") == []

    # arrays
    arr = {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 3}
    assert validate(arr, [1, 2]) == []
    assert validate(arr, []) == ["$: 0 items < minItems 1"]
    assert validate(arr, [1, 2, 3, 4]) == ["$: 4 items > maxItems 3"]
    assert validate(arr, [1, "x", 2.5]) == [
        "$[1]: expected integer, got string",
        "$[2]: expected integer, got number",
    ]

    # objects: required, nested paths, additionalProperties
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            "filters": {
                "type": "object",
                "properties": {"status": status},
                "required": ["status"],
            },
        },
        "required": ["query", "limit"],
        "additionalProperties": False,
    }
    assert validate(schema, {"query": "cats", "limit": 10}) == []
    assert validate(schema, {}) == [
        "$.query: required property missing",
        "$.limit: required property missing",
    ]
    assert validate(schema, {"query": "cats", "limit": 10, "filters": {"status": "x"}}) == [
        "$.filters.status: 'x' not in ['open', 'closed']"
    ]
    assert validate(schema, {"query": "cats", "limit": 10, "filters": {}}) == [
        "$.filters.status: required property missing"
    ]
    assert validate(schema, {"query": "cats", "limit": 10, "extra": 1, "zzz": 2}) == [
        "$.extra: unexpected property",
        "$.zzz: unexpected property",
    ]
    assert validate(schema, "not an object") == ["$: expected object, got string"]
    # without additionalProperties False, extra keys are accepted
    assert validate({"type": "object"}, {"anything": 1}) == []
    # a schema with no keywords accepts everything
    assert validate({}, [1, "a", None]) == []

    # validate_call: messages for the model
    tools = {"search": schema, "fetch": {"type": "object", "required": ["url"]}}
    assert validate_call(tools, "search", {"query": "a", "limit": 5}) == (True, "")
    ok, msg = validate_call(tools, "search", {"query": "a", "limit": 500})
    assert not ok
    assert msg == "Invalid arguments for 'search':\n- $.limit: 500 > maximum 50"
    ok, msg = validate_call(tools, "serch", {})
    assert not ok
    assert msg == "Unknown tool 'serch'. Did you mean 'search'? Available tools: fetch, search"
    ok, msg = validate_call(tools, "zzzzzz", {})
    assert msg == "Unknown tool 'zzzzzz'. Available tools: fetch, search"
    ok, msg = validate_call(tools, "fetch", {})
    assert msg == "Invalid arguments for 'fetch':\n- $.url: required property missing"


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
