"""Small integer reference for owner/head layouts; NOT a vLLM/NPU test.

No KDA recurrence, quantization, async stream or graph runtime is simulated.
The check isolates the permutation and reduction contracts in the design.
"""

import itertools
import json
from pathlib import Path
import unittest

DP, C, H, HEADS, D, I, VOCAB = 32, 3, 4, 8, 2, 16, 32
P = HEADS * D
FIELDS = ("mlp", "embedding", "lmhead", "qkv_iproj", "gproj", "oproj")


def weight(rows, cols, salt=0):
    return [[(row * 7 + col * 3 + salt) % 11 - 5 for col in range(cols)] for row in range(rows)]


def matvec(w, x):
    return [sum(a * b for a, b in zip(row, x)) for row in w]


INPUTS = [
    [[(rank * 5 + row * 3 + col) % 7 - 3 for col in range(H)] for row in range(rank % (C + 1))]
    for rank in range(DP)
]


def pad(rows, width):
    return rows + [[0] * width for _ in range(C - len(rows))]


def column_return(w, inputs, size, segments=None):
    """AG owner rows, column-project head slices, A2A and restore per segment."""
    segments = segments or [len(w)]
    assert sum(segments) == len(w) and all(width % size == 0 for width in segments)
    out = [[[0] * len(w) for _ in range(C)] for _ in range(DP)]
    for base in range(0, DP, size):
        gathered = [row for owner in range(base, base + size) for row in pad(inputs[owner], len(w[0]))]
        for source in range(size):
            indices, start = [], 0
            for width in segments:
                shard = width // size
                indices.extend(range(start + source * shard, start + (source + 1) * shard))
                start += width
            projected = [matvec([w[index] for index in indices], row) for row in gathered]
            # A2A receiver gets [source, owner_capacity, local_features].
            for owner in range(size):
                for row in range(C):
                    received = projected[owner * C + row]
                    for index, value in zip(indices, received):
                        out[base + owner][row][index] = value
    return [rows[:len(inputs[rank])] for rank, rows in enumerate(out)]


def row_return(w, inputs, size):
    """Head A2A -> each O column shard -> sum/RS to original token owner."""
    width = len(w[0])
    assert width % size == 0
    shard = width // size
    out = []
    for base in range(0, DP, size):
        owner_rows = [pad(inputs[owner], width) for owner in range(base, base + size)]
        partials = []
        for rank in range(size):
            start, end = rank * shard, (rank + 1) * shard
            shard_w = [row[start:end] for row in w]
            partials.append([[matvec(shard_w, row[start:end]) for row in rows] for rows in owner_rows])
        for owner in range(size):
            rows = [[sum(partials[s][owner][r][h] for s in range(size)) for h in range(len(w))] for r in range(C)]
            out.append(rows[:len(inputs[base + owner])])
    return out


def dense(wup, wdown, inputs, size, chunks=1):
    """Owner-row chunking, AG -> local gate/up+activation -> down -> RS."""
    assert I % size == 0
    out = [[] for _ in range(DP)]
    # Fixed chunk boundaries shared by all owners, including idle ones.
    bounds = [(C * k // chunks, C * (k + 1) // chunks) for k in range(chunks)]
    for a, b in bounds:
        for base in range(0, DP, size):
            for owner in range(base, base + size):
                for x in pad(inputs[owner], H)[a:b]:
                    result = [0] * H
                    for s in range(size):
                        indices = list(range(s * I // size, (s + 1) * I // size))
                        up = matvec([wup[j] for j in indices], x)
                        gate = matvec([wup[I + j] for j in indices], x)
                        # Integer stand-in activation; does not test SiTU math.
                        activated = [u * g for u, g in zip(up, gate)]
                        partial = matvec([[row[j] for j in indices] for row in wdown], activated)
                        result = [v + p for v, p in zip(result, partial)]
                    out[owner].append(result)
    return [rows[:len(inputs[rank])] for rank, rows in enumerate(out)]


def embedding(w, ids, size):
    shard = VOCAB // size
    return [
        [[sum(w[token][h] if s * shard <= token < (s + 1) * shard else 0 for s in range(size)) for h in range(H)] for token in rows]
        for rows in ids
    ]


def lmhead(w, inputs, size, chunks):
    outputs = [[] for _ in range(DP)]
    for k in range(chunks):
        a, b = C * k // chunks, C * (k + 1) // chunks
        chunk_inputs = [pad(rows, H)[a:b] for rows in inputs]
        # column_return uses C for scratch capacity; valid rows remain b-a.
        result = column_return(w, chunk_inputs, size)
        for rank in range(DP):
            outputs[rank].extend(result[rank])
    return [rows[:len(inputs[rank])] for rank, rows in enumerate(outputs)]


def plan(sizes):
    assert len(sizes) == 6
    active = {name: (size if size > 1 else 1) for name, size in zip(FIELDS, sizes)}
    if any(size < 0 for size in sizes) or any(DP % size for size in active.values()):
        raise ValueError("invalid size")
    return {
        "sizes": active,
        "kda_projection": "localtp" if active["qkv_iproj"] > 1 or active["gproj"] > 1 else "legacy",
        "pg_sizes": sorted({s for s in active.values() if s > 1}),
        "state_heads": HEADS,
    }


class ReferenceChecks(unittest.TestCase):
    def test_all_64_switch_combinations(self):
        qkvb = weight(3 * P + HEADS, H, 1)
        g, o = weight(P, H, 2), weight(H, P, 3)
        up, down, vocab = weight(2 * I, H, 4), weight(H, I, 5), weight(VOCAB, H, 6)
        ids = [[(rank + row) % VOCAB for row in range(len(rows))] for rank, rows in enumerate(INPUTS)]
        q_ref = [[matvec(qkvb, x) for x in rows] for rows in INPUTS]
        g_ref = [[matvec(g, x) for x in rows] for rows in INPUTS]
        go_ref = [[matvec(o, x) for x in rows] for rows in g_ref]
        dense_ref = dense(up, down, INPUTS, 1)
        head_ref = [[matvec(vocab, x) for x in rows] for rows in INPUTS]
        embed_ref = [[vocab[token] for token in rows] for rows in ids]
        for sizes in itertools.product((1, 8), repeat=6):
            with self.subTest(sizes=sizes):
                m, e, l, q, gate, output = sizes
                p = plan(sizes)
                self.assertEqual(p["state_heads"], HEADS)
                self.assertEqual(column_return(qkvb, INPUTS, q, [P, P, P, HEADS]), q_ref)
                gated = column_return(g, INPUTS, gate)
                self.assertEqual(gated, g_ref)
                # G -> O composes the independently sized layout domains only.
                self.assertEqual(row_return(o, gated, output), go_ref)
                self.assertEqual(dense(up, down, INPUTS, m, 2 if m > 1 else 1), dense_ref)
                self.assertEqual(embedding(vocab, ids, e), embed_ref)
                self.assertEqual(lmhead(vocab, INPUTS, l, 2 if l > 1 else 1), head_ref)

    def test_different_qkv_g_o_sizes(self):
        qkv = weight(3 * P + HEADS, H)
        g, o = weight(P, H, 1), weight(H, P, 2)
        expected_q = [[matvec(qkv, x) for x in rows] for rows in INPUTS]
        expected_go = [[matvec(o, matvec(g, x)) for x in rows] for rows in INPUTS]
        for q, gate, output in itertools.product((2, 4, 8), repeat=3):
            with self.subTest(q=q, gate=gate, output=output):
                self.assertEqual(column_return(qkv, INPUTS, q, [P, P, P, HEADS]), expected_q)
                self.assertEqual(row_return(o, column_return(g, INPUTS, gate), output), expected_go)

    def test_disabled_alias_and_rejections(self):
        self.assertEqual(plan([0] * 6), plan([1] * 6))
        self.assertEqual(plan([1] * 6)["pg_sizes"], [])
        self.assertEqual(plan([1] * 6)["kda_projection"], "legacy")
        for size in (-1, 3):
            with self.assertRaises(ValueError):
                plan([size, 1, 1, 1, 1, 1])

    def test_qkv_source_major_flatten_is_wrong(self):
        # A regression witness for the exact permutation the plan prohibits.
        shards = [[10, 20, 30], [11, 21, 31]]  # each shard's Q,K,V
        naive = [x for shard in shards for x in shard]
        correct = [shards[s][qkv] for qkv in range(3) for s in range(2)]
        self.assertNotEqual(naive, correct)
        self.assertEqual(correct, [10, 11, 20, 21, 30, 31])


if __name__ == "__main__":
    unittest.main(verbosity=2)
