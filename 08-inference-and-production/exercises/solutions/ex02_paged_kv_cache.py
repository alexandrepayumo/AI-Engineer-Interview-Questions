"""Exercise 02 - A paged KV-cache block allocator with prefix sharing.

TASK
----
Implement `PagedKVCache`, a toy version of the block manager inside a PagedAttention server.
It tracks which physical block holds which tokens. It does not store tensors.

    PagedKVCache(num_blocks, block_size)
    num_free_blocks() -> int
        Blocks with reference count 0 (this includes blocks that still hold a reusable
        cached prefix and can be evicted).
    allocate(seq_id, tokens) -> int
        Create a sequence from its prompt tokens. Return the number of prompt tokens served
        from the prefix cache. Raise MemoryError, leaving the cache unchanged, if there are
        not enough blocks.
    append(seq_id, token) -> bool
        Add one generated token. Return False (and change nothing) if a block is needed and
        none is free.
    fork(parent_id, child_id) -> None
        Child shares every block of the parent (as in beam search), ref counts go up.
    free(seq_id) -> None
        Release the sequence. Blocks whose ref count reaches 0 become free but keep their
        cached content hash until they are reused.
    block_table(seq_id) -> list[int]
    ref_count(block_id) -> int

Rules
-----
- A FULL block (block_size tokens) is identified by a chained hash of (previous block's hash,
  its tokens). Only full blocks are shared through the prefix cache. The partial last block of a
  sequence is always private.
- allocate() reuses a block when its hash is in the cache, but only for a contiguous run of
  hits from the start of the prompt (stop at the first miss).
- Free blocks are reused in least-recently-freed order. Reusing a block removes its old hash
  from the cache.
- append() when the last block is shared (ref_count > 1) must copy on write: take a new block,
  drop one reference on the old one.
- When an append fills a block, register its hash if that hash is not already cached.

HINTS
-----
- Keep: ref counts per block, the tokens held by each block, a dict hash -> block id, and an
  OrderedDict of free block ids (oldest first) for LRU reuse.
- Precompute how many hits are on free (ref 0) blocks: claiming them removes them from the
  free list, so they cannot also serve as new blocks.

WHAT YOU SHOULD LEARN
---------------------
- Why block tables remove external fragmentation: any free block can extend any sequence.
- Why prefix caching works at block granularity and what a "chained" hash buys you.
- Why copy-on-write is needed for beam search and parallel sampling.
- Why freed blocks keep their content: a later request can hit them until memory pressure
  evicts them.
"""
from __future__ import annotations

from collections import OrderedDict


class PagedKVCache:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        """Set up empty bookkeeping."""
        self.block_size = block_size
        self.num_blocks = num_blocks
        self._refs = [0] * num_blocks
        self._tokens: list[list[int]] = [[] for _ in range(num_blocks)]
        self._hash_of: list[int | None] = [None] * num_blocks   # block -> hash while cached
        self._by_hash: dict[int, int] = {}                      # hash -> block
        self._free: OrderedDict[int, None] = OrderedDict((i, None) for i in range(num_blocks))
        self._tables: dict[str, list[int]] = {}
        self._hashes: dict[str, list[int]] = {}                 # per-seq hash of each FULL block
        self._len: dict[str, int] = {}

    # ---- internal helpers -------------------------------------------------
    def _pop_free(self) -> int:
        block, _ = self._free.popitem(last=False)               # least recently freed
        h = self._hash_of[block]
        if h is not None and self._by_hash.get(h) == block:
            del self._by_hash[h]                                 # evict its cached content
        self._hash_of[block] = None
        self._tokens[block] = []
        self._refs[block] = 1
        return block

    def _claim(self, block: int) -> None:
        if self._refs[block] == 0:
            del self._free[block]                                # no longer evictable
        self._refs[block] += 1

    def _release(self, block: int) -> None:
        self._refs[block] -= 1
        if self._refs[block] == 0:
            self._free[block] = None                             # goes to the MRU end

    @staticmethod
    def _chain(prev: int, tokens: tuple[int, ...]) -> int:
        return hash((prev, tokens))

    # ---- public API -------------------------------------------------------
    def num_free_blocks(self) -> int:
        """Number of blocks with ref count 0."""
        return len(self._free)

    def allocate(self, seq_id: str, tokens: list[int]) -> int:
        """Create a sequence, reusing cached full-block prefixes. Returns cached token count."""
        bs = self.block_size
        n_full = len(tokens) // bs
        n_total = -(-len(tokens) // bs)
        # 1. find the contiguous run of cache hits
        hashes: list[int] = []
        prev = 0
        hit_blocks: list[int] = []
        for i in range(n_full):
            h = self._chain(prev, tuple(tokens[i * bs:(i + 1) * bs]))
            hashes.append(h)
            prev = h
        for h in hashes:
            b = self._by_hash.get(h)
            if b is None:
                break
            hit_blocks.append(b)
        # 2. check capacity before touching anything
        hits_on_free = sum(1 for b in hit_blocks if self._refs[b] == 0)
        need_new = n_total - len(hit_blocks)
        if need_new > len(self._free) - hits_on_free:
            raise MemoryError("not enough free KV blocks")
        # 3. commit
        table: list[int] = []
        for b in hit_blocks:
            self._claim(b)
            table.append(b)
        for i in range(len(hit_blocks), n_total):
            b = self._pop_free()
            chunk = tokens[i * bs:(i + 1) * bs]
            self._tokens[b] = list(chunk)
            if i < n_full:                                       # full block: register hash
                self._hash_of[b] = hashes[i]
                self._by_hash.setdefault(hashes[i], b)
            table.append(b)
        self._tables[seq_id] = table
        self._hashes[seq_id] = hashes
        self._len[seq_id] = len(tokens)
        return len(hit_blocks) * bs

    def append(self, seq_id: str, token: int) -> bool:
        """Append one generated token. False if out of memory."""
        table = self._tables[seq_id]
        bs = self.block_size
        length = self._len[seq_id]
        if length % bs == 0:                                     # last block full (or empty seq)
            if not self._free:
                return False
            table.append(self._pop_free())
        else:
            last = table[-1]
            if self._refs[last] > 1:                             # shared partial block: copy on write
                if not self._free:
                    return False
                new = self._pop_free()
                self._tokens[new] = list(self._tokens[last])
                self._release(last)
                table[-1] = new
        last = table[-1]
        self._tokens[last].append(token)
        self._len[seq_id] = length + 1
        if len(self._tokens[last]) == bs:                        # block just filled
            hashes = self._hashes[seq_id]
            prev = hashes[-1] if hashes else 0
            h = self._chain(prev, tuple(self._tokens[last]))
            hashes.append(h)
            if h not in self._by_hash:
                self._by_hash[h] = last
                self._hash_of[last] = h
        return True

    def fork(self, parent_id: str, child_id: str) -> None:
        """Share all of the parent's blocks with a new child sequence."""
        for b in self._tables[parent_id]:
            self._refs[b] += 1
        self._tables[child_id] = list(self._tables[parent_id])
        self._hashes[child_id] = list(self._hashes[parent_id])
        self._len[child_id] = self._len[parent_id]

    def free(self, seq_id: str) -> None:
        """Release a sequence's blocks."""
        for b in self._tables.pop(seq_id):
            self._release(b)
        del self._hashes[seq_id]
        del self._len[seq_id]

    def block_table(self, seq_id: str) -> list[int]:
        """Physical block ids of the sequence, in order."""
        return list(self._tables[seq_id])

    def ref_count(self, block_id: int) -> int:
        """Number of sequences using the block."""
        return self._refs[block_id]


def _run_tests() -> None:
    # block_size 4, 8 blocks. Sequence A: 10 tokens -> 2 full blocks + 1 partial.
    kv = PagedKVCache(num_blocks=8, block_size=4)
    a_tokens = list(range(10))
    assert kv.allocate("A", a_tokens) == 0                       # cold cache, nothing reused
    assert len(kv.block_table("A")) == 3
    assert kv.num_free_blocks() == 5

    # B shares the first 8 tokens (2 full blocks) and diverges in the partial block
    b_tokens = list(range(8)) + [100, 101, 102]
    assert kv.allocate("B", b_tokens) == 8
    ta, tb = kv.block_table("A"), kv.block_table("B")
    assert ta[:2] == tb[:2] and ta[2] != tb[2]
    assert kv.ref_count(ta[0]) == 2 and kv.ref_count(ta[1]) == 2
    assert kv.num_free_blocks() == 4                             # only one new block for B

    # a prompt that differs in block 0 cannot reuse block 1 even if its tokens match
    c_tokens = [999, 1, 2, 3] + list(range(4, 8))
    assert kv.allocate("C", c_tokens) == 0                       # chained hash: no hit
    assert kv.num_free_blocks() == 2
    kv.free("C")
    assert kv.num_free_blocks() == 4

    # freeing A keeps shared blocks alive (ref 1) and releases only its private partial block
    kv.free("A")
    assert kv.ref_count(tb[0]) == 1 and kv.ref_count(ta[2]) == 0
    assert kv.num_free_blocks() == 5
    kv.free("B")
    assert kv.num_free_blocks() == 8
    assert all(kv.ref_count(i) == 0 for i in range(8))

    # cached content survives free: D hits the 2-block prefix again
    assert kv.allocate("D", a_tokens) == 8
    kv.free("D")

    # capacity failure leaves the cache unchanged
    before = kv.num_free_blocks()
    kv.allocate("E", list(range(1000, 1004)))                    # 1 block, evicts oldest free
    try:
        kv.allocate("F", list(range(2000, 2000 + 32)))           # needs 8 blocks, only 7 free
        raise AssertionError("expected MemoryError")
    except MemoryError:
        pass
    assert kv.num_free_blocks() == before - 1
    kv.free("E")

    # --- fork and copy on write (beam search) ---
    kv = PagedKVCache(num_blocks=6, block_size=4)
    kv.allocate("P", [1, 2, 3, 4, 5, 6])                         # block 0 full, block 1 holds 2 tokens
    kv.fork("P", "Q")
    tp, tq = kv.block_table("P"), kv.block_table("Q")
    assert tp == tq and kv.ref_count(tp[0]) == 2 and kv.ref_count(tp[1]) == 2
    free_before = kv.num_free_blocks()
    assert kv.append("P", 7) is True                             # P copies the shared partial block
    assert kv.num_free_blocks() == free_before - 1
    assert kv.block_table("P")[0] == tq[0] and kv.block_table("P")[1] != tq[1]
    assert kv.ref_count(tq[1]) == 1                              # Q is now the sole owner
    assert kv.append("Q", 8) is True                             # Q writes in place, no new block
    assert kv.num_free_blocks() == free_before - 1
    assert kv.ref_count(tq[0]) == 2                              # full prefix block still shared

    # appending past a block boundary allocates a new block; filled blocks become cacheable
    kv.append("P", 9)                                            # P now has 8 tokens: block full
    assert len(kv.block_table("P")) == 2
    kv.append("P", 10)                                           # 9th token opens a third block
    assert len(kv.block_table("P")) == 3
    kv.free("P")
    kv.free("Q")
    assert kv.num_free_blocks() == 6
    # the filled block [5,6,7,9] is reusable by a new prompt with the same first 8 tokens
    assert kv.allocate("R", [1, 2, 3, 4, 5, 6, 7, 9]) == 8

    # out-of-memory on append returns False and changes nothing
    kv = PagedKVCache(num_blocks=2, block_size=2)
    kv.allocate("X", [1, 2])
    kv.allocate("Y", [3, 4])
    assert kv.num_free_blocks() == 0
    assert kv.append("X", 5) is False
    assert len(kv.block_table("X")) == 1 and kv.num_free_blocks() == 0

    # internal waste is bounded by one block per sequence: 100 tokens in 16-token blocks
    kv = PagedKVCache(num_blocks=64, block_size=16)
    kv.allocate("big", list(range(100)))
    assert len(kv.block_table("big")) == 7                       # ceil(100 / 16)
    assert 7 * 16 - 100 < 16                                     # wasted slots < block_size


if __name__ == "__main__":
    _run_tests()
    print("All tests passed.")
