"""Exercise 01 - Strict prompt template renderer.

TASK
----
Implement a tiny prompt template engine where variables are written
``{{ name }}`` (optional spaces inside the braces).

    class TemplateError(Exception): ...

    def find_variables(template: str) -> list[str]:
        '''Variable names in order of first appearance, without duplicates.
        Raises TemplateError for a malformed placeholder: an inner text that is
        not a valid identifier ([A-Za-z_][A-Za-z0-9_]*) or a "{{" with no "}}".'''

    def render(template: str, variables: dict[str, object]) -> str:
        '''Single-pass substitution with strict checking:
        - a variable used in the template but missing from `variables` -> TemplateError
        - a key in `variables` never used in the template -> TemplateError
        - a value that is not str, int or float (None, list, dict, bool) -> TemplateError
        Substituted values are NOT re-scanned: a value containing "{{x}}" must
        appear literally in the output.'''

HINTS
-----
- re.finditer(r"\\{\\{(.*?)\\}\\}", template) finds placeholders; strip() the inner text.
- After removing all matched placeholders, any remaining "{{" is unclosed.
- Use re.sub with a replacement function so output is built in one pass.
- bool is a subclass of int in Python, so check it explicitly.

WHAT YOU SHOULD LEARN
---------------------
Prompts are code. Silent failures (a literal "{{user_name}}" sent to the model,
or a forgotten variable that never reaches the prompt) are some of the most common
production prompt bugs. Strict rendering turns them into errors at build time.
Single-pass substitution also stops a user-supplied value from smuggling in new
template syntax, which is a small instance of the injection problem.
"""
from __future__ import annotations

import re

_PLACEHOLDER = re.compile(r"\{\{(.*?)\}\}", re.S)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class TemplateError(Exception):
    """Raised for any template or variable problem."""


def find_variables(template: str) -> list[str]:
    names: list[str] = []
    for m in _PLACEHOLDER.finditer(template):
        name = m.group(1).strip()
        if not _IDENT.fullmatch(name):
            raise TemplateError(f"invalid placeholder: {m.group(0)!r}")
        if name not in names:
            names.append(name)
    # whatever is left after removing complete placeholders must not contain "{{"
    if "{{" in _PLACEHOLDER.sub("", template):
        raise TemplateError("unclosed '{{'")
    return names


def render(template: str, variables: dict[str, object]) -> str:
    used = find_variables(template)
    missing = [n for n in used if n not in variables]
    if missing:
        raise TemplateError(f"missing variables: {missing}")
    extra = [k for k in variables if k not in used]
    if extra:
        raise TemplateError(f"unused variables: {extra}")
    for k, v in variables.items():
        if isinstance(v, bool) or not isinstance(v, (str, int, float)):
            raise TemplateError(f"bad value type for {k!r}: {type(v).__name__}")
    # single pass: the replacement text is never scanned again
    return _PLACEHOLDER.sub(lambda m: str(variables[m.group(1).strip()]), template)


def _run_tests() -> None:
    assert find_variables("Hi {{ name }}, {{name}} and {{ topic }}") == ["name", "topic"]
    assert find_variables("no variables here") == []
    assert render("Hello {{name}}!", {"name": "Ada"}) == "Hello Ada!"
    assert render("{{ a }}-{{a}}-{{ b }}", {"a": 1, "b": 2.5}) == "1-1-2.5"

    def raises(fn, *args):
        try:
            fn(*args)
        except TemplateError:
            return True
        return False

    assert raises(render, "Hello {{name}}", {})                       # missing
    assert raises(render, "Hello {{name}}", {"name": "x", "y": "z"})  # unused
    assert raises(render, "Hello {{name}}", {"name": None})           # bad type
    assert raises(render, "Hello {{name}}", {"name": True})           # bool rejected
    assert raises(render, "Hello {{name}}", {"name": ["a"]})
    assert raises(find_variables, "Hello {{ 1bad }}")
    assert raises(find_variables, "Hello {{ two words }}")
    assert raises(find_variables, "Hello {{name")

    # values are not re-scanned (no template injection through data)
    out = render("Q: {{q}} / {{other}}", {"q": "{{other}}", "other": "SAFE"})
    assert out == "Q: {{other}} / SAFE"

    # multi-line templates work
    tpl = "<doc>\n{{ doc }}\n</doc>\nTask: {{task}}"
    assert render(tpl, {"doc": "line1\nline2", "task": "summarise"}) == (
        "<doc>\nline1\nline2\n</doc>\nTask: summarise"
    )


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
