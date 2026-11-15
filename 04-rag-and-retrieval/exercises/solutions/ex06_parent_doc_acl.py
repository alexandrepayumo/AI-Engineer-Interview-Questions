"""Exercise 06 - Parent-document (small-to-big) retrieval with ACL pre-filtering.

TASK
----
Small chunks give precise matches, large parents give the model enough context. Index small
child chunks, retrieve children, then return their parent documents. Every chunk carries the
ACL groups of its document, and permissions are enforced inside the retrieval step.

A hashed bag-of-words `embed` is provided so the exercise needs no model.

Implement:

    split_children(text, child_words, stride) -> list[str]
        Sliding windows over the whitespace-split words. Window i starts at i*stride and holds
        up to child_words words. Stop after the first window that reaches the final word
        (no trailing window that is a strict subset of the previous one). Empty text -> [].

    class ParentDocIndex
        .add_document(doc_id, text, allowed_groups, child_words=20, stride=10)
        .delete_document(doc_id)            remove the parent and all its children at once
        .search(query, user_groups, k_parents) -> list[tuple[str, float]]
            PRE-filter: only chunks whose allowed_groups intersect user_groups are scored.
            Parent score = max child score. Return up to k_parents (doc_id, score), best first.
        .search_postfilter(query, user_groups, k_children) -> list[str]
            The naive way: take the global top-k_children chunks first, THEN drop the ones the
            user cannot see. Return the unique parent ids in rank order.
        .get_context(parent_ids, user_groups, max_words) -> str
            Join parent texts in order, skipping any parent the user cannot see (re-check the
            ACL at read time), stopping once max_words words are used. The last parent may
            be truncated to fit the budget.

HINTS
-----
- Store child embeddings in one 2-D array rebuilt (or appended) on add/delete so scoring is a
  single matrix-vector product, then mask with a boolean visibility vector before top-k.
- Visibility = bool(chunk_groups & user_groups). A document with an empty group set is visible
  to nobody.
- Deleting must remove vectors too, otherwise a revoked document still matches.

WHAT YOU SHOULD LEARN
---------------------
- Why post-filtering can return fewer than k results (or none) while pre-filtering cannot.
- Why the ACL check belongs in retrieval (and again at context assembly), never in the prompt.
- Max-pooling child scores to rank parents, and spending a word budget on parents.
"""
from __future__ import annotations

import re
import zlib

import numpy as np

DIM = 4096


def embed(text: str) -> np.ndarray:
    """Hashed bag-of-words embedding, L2-normalised (provided)."""
    v = np.zeros(DIM)
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        v[zlib.crc32(w.encode()) % DIM] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


def split_children(text: str, child_words: int, stride: int) -> list[str]:
    words = text.split()
    out: list[str] = []
    start = 0
    while start < len(words):
        out.append(" ".join(words[start:start + child_words]))
        if start + child_words >= len(words):
            break  # this window already reaches the last word
        start += stride
    return out


class ParentDocIndex:
    def __init__(self) -> None:
        self.parents: dict[str, str] = {}
        self.parent_groups: dict[str, set[str]] = {}
        self.child_parent: list[str] = []
        self.child_groups: list[set[str]] = []
        self.child_vecs: list[np.ndarray] = []

    def add_document(
        self, doc_id: str, text: str, allowed_groups: set[str], child_words: int = 20, stride: int = 10
    ) -> None:
        self.parents[doc_id] = text
        self.parent_groups[doc_id] = set(allowed_groups)
        for child in split_children(text, child_words, stride):
            self.child_parent.append(doc_id)
            self.child_groups.append(set(allowed_groups))
            self.child_vecs.append(embed(child))

    def delete_document(self, doc_id: str) -> None:
        self.parents.pop(doc_id, None)
        self.parent_groups.pop(doc_id, None)
        keep = [i for i, p in enumerate(self.child_parent) if p != doc_id]
        self.child_parent = [self.child_parent[i] for i in keep]
        self.child_groups = [self.child_groups[i] for i in keep]
        self.child_vecs = [self.child_vecs[i] for i in keep]

    def search(self, query: str, user_groups: set[str], k_parents: int) -> list[tuple[str, float]]:
        if not self.child_vecs:
            return []
        scores = np.stack(self.child_vecs) @ embed(query)
        visible = np.array([bool(g & user_groups) for g in self.child_groups])
        best: dict[str, float] = {}
        for i in np.where(visible)[0]:  # only visible chunks ever compete
            p = self.child_parent[i]
            best[p] = max(best.get(p, -1.0), float(scores[i]))
        ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k_parents]

    def search_postfilter(self, query: str, user_groups: set[str], k_children: int) -> list[str]:
        if not self.child_vecs:
            return []
        scores = np.stack(self.child_vecs) @ embed(query)
        top = np.argsort(-scores, kind="stable")[:k_children]  # global top-k first
        out: list[str] = []
        for i in top:
            if self.child_groups[i] & user_groups and self.child_parent[i] not in out:
                out.append(self.child_parent[i])
        return out

    def get_context(self, parent_ids: list[str], user_groups: set[str], max_words: int) -> str:
        used: list[str] = []
        budget = max_words
        for pid in parent_ids:
            if pid not in self.parents or not (self.parent_groups[pid] & user_groups):
                continue  # defence in depth: re-check ACL when assembling context
            words = self.parents[pid].split()
            if len(words) > budget:
                words = words[:budget]
            used.extend(words)
            budget -= len(words)
            if budget <= 0:
                break
        return " ".join(used)


def _run_tests() -> None:
    # split_children
    assert split_children("", 5, 3) == []
    assert split_children("a b c", 5, 3) == ["a b c"]
    words = " ".join(f"w{i}" for i in range(10))
    ch = split_children(words, 4, 2)
    assert ch[0] == "w0 w1 w2 w3" and ch[1] == "w2 w3 w4 w5"
    assert ch[-1].endswith("w9") and len(ch) == 4  # starts 0,2,4,6; window at 6 reaches w9
    assert split_children(words, 10, 5) == [words]  # one window covers everything

    idx = ParentDocIndex()
    idx.add_document(
        "refunds", "Customers can request a refund within thirty days. A refund needs the "
        "order number. Refund requests are handled by the billing team.", {"support", "all"},
        child_words=8, stride=4,
    )
    idx.add_document(
        "vpn", "Install the vpn client then sign in with your badge. The vpn gateway rejects "
        "expired certificates. Contact it security for a new vpn certificate.", {"all"},
        child_words=8, stride=4,
    )
    idx.add_document(
        "salaries", "Payroll salary bands for engineering. Payroll bands are reviewed yearly.",
        {"hr"}, child_words=50,
    )
    idx.add_document(
        "bonus", "Payroll bonus targets for engineering. Payroll bonus payouts follow salary review.",
        {"hr"}, child_words=50,
    )
    idx.add_document(
        "payday", "Payroll runs on the last working day of each month.", {"all"}, child_words=50,
    )

    # parents are returned, scored by best child, best first
    res = idx.search("how do I get a refund", {"all"}, 2)
    assert res[0][0] == "refunds" and len(res) == 2
    assert res[0][1] >= res[1][1]
    assert all(isinstance(s, float) for _, s in res)

    # ACL pre-filter: restricted docs never appear, even when they are the best textual match
    res = idx.search("payroll salary bands engineering", {"all"}, 5)
    ids = [d for d, _ in res]
    assert "salaries" not in ids and "bonus" not in ids and ids[0] == "payday"
    res_hr = idx.search("payroll salary bands engineering", {"hr"}, 5)
    assert res_hr[0][0] == "salaries"
    assert idx.search("anything", set(), 5) == []  # no groups, no results

    # post-filter failure mode: top-2 chunks are both HR-only, so the user sees nothing
    assert idx.search_postfilter("payroll salary bonus engineering", {"all"}, 2) == []
    # while the pre-filter still fills the list for the same user
    assert [d for d, _ in idx.search("payroll salary bonus engineering", {"all"}, 1)] == ["payday"]
    # with a deep enough global k, post-filtering recovers (but k had to be guessed)
    assert "payday" in idx.search_postfilter("payroll salary bonus engineering", {"all"}, 20)

    # context assembly: order kept, ACL re-checked, budget respected
    ctx = idx.get_context(["salaries", "payday", "vpn"], {"all"}, 1000)
    assert "Payroll runs" in ctx and "salary bands" not in ctx and "vpn client" in ctx
    short = idx.get_context(["payday", "vpn"], {"all"}, 12)
    assert len(short.split()) == 12 and short.startswith("Payroll runs on")
    assert idx.get_context(["payday"], {"all"}, 0) == ""

    # deleting a document removes it from search at once
    idx.delete_document("payday")
    assert "payday" not in [d for d, _ in idx.search("payroll runs month", {"all"}, 5)]
    assert idx.get_context(["payday"], {"all"}, 100) == ""
    idx.delete_document("does-not-exist")  # no error


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
