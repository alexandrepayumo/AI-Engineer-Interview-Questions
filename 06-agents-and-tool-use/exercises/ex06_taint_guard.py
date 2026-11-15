"""Exercise 06 - A capability guard with taint tracking (the lethal trifecta).

TASK
----
Implement CapabilityGuard, a policy layer that sits between the model's tool calls and
their execution. Each tool is described by a ToolSpec (provided) with flags:
    reads_private    brings private data into context
    reads_untrusted  brings attacker-controllable text into context (web, email, issues)
    sends_external   can move data out of the system (HTTP request, email, upload)
    irreversible     cannot be undone (send, delete, pay)

    host_of(dest) -> str
        Strip a "scheme://" prefix, cut at the first "/", then if "@" is present keep what
        follows the last "@". So "https://a.com/x" -> "a.com", "boss@corp.com" -> "corp.com".

    CapabilityGuard(tools, egress_allowlist)
        tools: dict[name, ToolSpec]; egress_allowlist: set of allowed destination hosts.
        State: untrusted_seen and private_seen, both False at start.

    check(name, args) -> Decision   (must NOT change state)
        Rules, applied in this order, first match wins:
          1. unknown tool name           -> deny, reason "unknown tool"
          2. lethal trifecta: private data AND untrusted content AND an exfiltration
             channel would all be present once this tool runs. Private is present if
             private_seen or the tool reads_private; untrusted likewise; the channel is
             the tool's own sends_external.
                                         -> deny, reason "lethal trifecta"
          3. sends_external and host_of(args.get("dest", "")) is not in egress_allowlist
                                         -> deny, reason "egress not allowed: {host}"
          4. irreversible                -> ask; reason "irreversible action" and, if
                                            untrusted_seen, "irreversible action after untrusted content"
          5. otherwise                   -> allow, reason "ok"

    record(name) -> None
        Update state after a tool actually ran: untrusted_seen |= reads_untrusted,
        private_seen |= reads_private.

    step(name, args, approver=None) -> Decision
        check(); if the action is "ask", call approver(reason) (a bool). With no approver
        or a False answer, return Decision("deny", "not approved: " + reason). If approved
        return Decision("allow", "approved: " + reason). When the final action is
        "allow", call record(name).

    reset() -> None   clear the state (new session).

Decision(action, reason) and ToolSpec are provided.

HINTS
-----
- Evaluate the trifecta on the PROSPECTIVE state (current state plus this tool's own flags).
  A single browser tool that reads the web and can make requests may complete the
  trifecta all by itself once private data is in context.
- Keep check() pure so you can call it for dry runs and logging.

WHAT YOU SHOULD LEARN
---------------------
- Prompt injection cannot be solved in the prompt. You bound the damage structurally by
  never letting one context hold private data, untrusted input and an exfiltration path.
- Taint is sticky: once untrusted text has been read, everything later in the session is
  suspect until the context is discarded.
- Egress allowlists and approval tiers are the other two layers of defence in depth.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolSpec:
    reads_private: bool = False
    reads_untrusted: bool = False
    sends_external: bool = False
    irreversible: bool = False


@dataclass(frozen=True)
class Decision:
    action: str  # "allow" | "deny" | "ask"
    reason: str


def host_of(dest: str) -> str:
    raise NotImplementedError("TODO")


class CapabilityGuard:
    def __init__(self, tools: dict[str, ToolSpec], egress_allowlist: set[str]):
        self.tools = tools
        self.egress_allowlist = set(egress_allowlist)
        self.untrusted_seen = False
        self.private_seen = False

    def check(self, name: str, args: dict) -> Decision:
        raise NotImplementedError("TODO")

    def record(self, name: str) -> None:
        raise NotImplementedError("TODO")

    def step(self, name: str, args: dict, approver=None) -> Decision:
        raise NotImplementedError("TODO")

    def reset(self) -> None:
        raise NotImplementedError("TODO")


def _run_tests() -> None:
    tools = {
        "calc": ToolSpec(),
        "read_file": ToolSpec(reads_private=True),
        "read_inbox": ToolSpec(reads_private=True, reads_untrusted=True),
        "fetch_url": ToolSpec(reads_untrusted=True, sends_external=True),
        "send_email": ToolSpec(sends_external=True, irreversible=True),
        "delete_file": ToolSpec(irreversible=True),
    }
    allow = {"docs.python.org", "corp.com"}

    assert host_of("https://docs.python.org/3/library") == "docs.python.org"
    assert host_of("boss@corp.com") == "corp.com"
    assert host_of("") == ""

    g = CapabilityGuard(tools, allow)
    # harmless tools and unknown tools
    assert g.step("calc", {}).action == "allow"
    d = g.step("rm_rf", {})
    assert (d.action, d.reason) == ("deny", "unknown tool")

    # clean state: an allowlisted fetch is fine, a fetch to anywhere else is not
    assert g.check("fetch_url", {"dest": "https://docs.python.org/x"}) == Decision("allow", "ok")
    d = g.check("fetch_url", {"dest": "https://evil.example/steal?d=secret"})
    assert (d.action, d.reason) == ("deny", "egress not allowed: evil.example")
    assert g.check("fetch_url", {}).reason == "egress not allowed: "

    # check() is pure
    assert not g.untrusted_seen and not g.private_seen

    # reading private data is fine; afterwards the browser tool would complete the trifecta
    assert g.step("read_file", {}).action == "allow"
    assert g.private_seen and not g.untrusted_seen
    d = g.check("fetch_url", {"dest": "https://docs.python.org/x"})
    assert (d.action, d.reason) == ("deny", "lethal trifecta")  # even to an allowlisted host

    # untrusted first, private second: the exfiltration channel is what gets blocked
    g.reset()
    assert not g.private_seen
    assert g.step("fetch_url", {"dest": "docs.python.org"}).action == "allow"
    assert g.untrusted_seen
    d = g.check("read_file", {})  # reading private data after untrusted content is allowed...
    assert d.action == "allow"
    g.step("read_file", {})
    d = g.check("send_email", {"dest": "boss@corp.com"})  # ...but now no exfil channel may run
    assert (d.action, d.reason) == ("deny", "lethal trifecta")

    # irreversible actions need approval, with a distinct reason when tainted
    g.reset()
    d = g.check("delete_file", {})
    assert (d.action, d.reason) == ("ask", "irreversible action")
    g.step("fetch_url", {"dest": "docs.python.org"})
    d = g.check("delete_file", {})
    assert (d.action, d.reason) == ("ask", "irreversible action after untrusted content")

    # approver decisions
    asked: list[str] = []

    def yes(reason):
        asked.append(reason)
        return True

    def no(reason):
        asked.append(reason)
        return False

    d = g.step("delete_file", {}, approver=yes)
    assert d == Decision("allow", "approved: irreversible action after untrusted content")
    d = g.step("delete_file", {}, approver=no)
    assert d.action == "deny" and d.reason.startswith("not approved")
    d = g.step("delete_file", {})  # no approver means no
    assert d.action == "deny"
    assert len(asked) == 2

    # send_email: clean state asks, allowlist still applies
    g.reset()
    assert g.check("send_email", {"dest": "boss@corp.com"}).action == "ask"
    assert g.check("send_email", {"dest": "x@evil.example"}).reason == "egress not allowed: evil.example"

    # an inbox read (private + untrusted) blocks every later exfiltration channel
    g.reset()
    assert g.step("read_inbox", {}).action == "allow"
    assert g.untrusted_seen and g.private_seen
    assert g.step("send_email", {"dest": "boss@corp.com"}, approver=yes).reason == "lethal trifecta"
    assert g.step("fetch_url", {"dest": "docs.python.org"}, approver=yes).reason == "lethal trifecta"
    assert g.step("calc", {}).action == "allow"  # tools without a channel keep working

    # a denied call does not taint the session
    g2 = CapabilityGuard(tools, allow)
    g2.step("fetch_url", {"dest": "https://evil.example/"})  # denied by egress
    assert not g2.untrusted_seen


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
