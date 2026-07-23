"""Cloud engine C1 — the device-materialized moment cloud as a CERTIFIED
accelerated view of vanish.VsSpace (issue #16, chunk 1).

The Rust space is the convention authority; this engine builds the same
object at GPU scale and refuses to serve data until it reproduces the
space's certificate:

  - ranking vectors: engine unrank(rank) == certificate subsets,
  - moment rows: engine rows at pinned ranks == certificate rows exactly,
  - coordinate cuts: zero-counts of columns e_1..e_4 == certificate values
    (full-population check; at s = 32 the certificate cuts are streamed by
    the Rust kernels — minutes — or checked against the census JSONs).

Row convention (from src/vs.rs): row(rank) = raw elementary symmetric
vector (e_0 = 1, e_1, ..., e_{s-r}) of the COMPLEMENT of the rank-th
lex subset. dtype uint32 (p < 2^31), shape [C(s,r), s-r+1].

Persistence: shards cloud_<i>.npy + manifest.json carrying (p, s, k),
the full certificate, and shard checksums; reload re-verifies the
certificate pins before use.

Usage:
  python cloud_engine.py --selfcheck                # s=16 gate, CPU-ok
  python cloud_engine.py --build --p 65537 --s 32 --k 15 --out DIR
  python cloud_engine.py --verify --dir DIR         # re-gate a stored cloud
Requires: vanish wheel (the authority); cupy optional (numpy fallback).
"""
import argparse
import hashlib
import json
import math
import os
import time

import numpy as np
import vanish

try:
    import cupy as cp
    GPU = True
except ImportError:
    cp = np
    GPU = False


# ---------------------------------------------------------------------------

def binom_table(s, r):
    T = np.zeros((s + 1, r + 1), dtype=np.int64)
    T[:, 0] = 1
    for n in range(1, s + 1):
        for k in range(1, min(n, r) + 1):
            T[n][k] = T[n - 1][k - 1] + T[n - 1][k]
    return T


def unrank_block(ranks, s, r, T, xp):
    """Lex unrank (the vs.rs convention): ranks (B,) -> subsets (B, r)."""
    B = ranks.shape[0]
    R = ranks.copy()
    need = xp.full(B, r, dtype=xp.int64)
    out = xp.zeros((B, r), dtype=xp.int64)
    Tx = xp.asarray(T)
    for pos in range(s):
        cnt = Tx[s - pos - 1, xp.maximum(need - 1, 0)]
        take = (need > 0) & (R < cnt)
        rows = xp.where(take)[0]
        out[rows, (r - need)[rows]] = pos
        R = xp.where((need > 0) & ~take, R - cnt, R)
        need = need - take.astype(xp.int64)
    return out


def complement_rows(idx, dom_x, p, s, r, xp):
    """(B, r) subset indices -> (B, s-r+1) raw e-vectors of complements."""
    B = idx.shape[0]
    in_set = xp.zeros((B, s), dtype=bool)
    xp.put_along_axis(in_set, idx, True, axis=1)
    comp_mask = ~in_set
    m = s - r
    # gather complement elements row-wise (each row has exactly m of them)
    comp_idx = xp.argsort(~comp_mask, axis=1, kind="stable")[:, :m]
    pts = dom_x[comp_idx]
    e = xp.zeros((B, m + 1), dtype=xp.int64)
    e[:, 0] = 1
    for c in range(m):
        v = pts[:, c:c + 1] % p
        e[:, 1:c + 2] = (e[:, 1:c + 2] + v * e[:, 0:c + 1]) % p
    return e.astype(xp.uint32)


class CloudEngine:
    def __init__(self, p, s, k, chunk=1 << 20):
        self.space = vanish.VsSpace(p, s, k)
        self.p, self.s, self.k = p, s, k
        self.r = self.space.r
        self.cols = s - self.r + 1
        self.total = math.comb(s, self.r)
        self.chunk = chunk
        self.T = binom_table(s, self.r)
        self.dom = np.array(self.space.domain(), dtype=np.int64)
        self.cert = self.space.certificate()

    def rows_for_ranks(self, ranks_np):
        xp = cp if GPU else np
        ranks = xp.asarray(ranks_np, dtype=xp.int64)
        idx = unrank_block(ranks, self.s, self.r, self.T, xp)
        rows = complement_rows(idx, xp.asarray(self.dom), self.p,
                               self.s, self.r, xp)
        return (cp.asnumpy(rows) if GPU else rows), \
               (cp.asnumpy(idx) if GPU else idx)

    def verify_certificate(self):
        """The gate: reproduce every certificate pin exactly."""
        cert = self.cert
        ranks = np.array([rk for rk, _ in cert["ranking"]], dtype=np.int64)
        rows, idx = self.rows_for_ranks(ranks)
        for (rk, sub), got in zip(cert["ranking"], idx):
            assert list(got) == list(sub), \
                f"ranking pin failed at rank {rk}: {list(got)} != {list(sub)}"
        for (rk, row), got in zip(cert["moment_rows"], rows):
            assert list(int(x) for x in got) == list(row), \
                f"moment-row pin failed at rank {rk}"
        assert list(self.dom[:8]) == list(cert["domain_head"]), "domain order"
        return True

    def verify_coordinate_cuts(self, log=print):
        """Full-population check: column zero-counts == certificate cuts."""
        zeros = np.zeros(self.cols, dtype=np.int64)
        for lo in range(0, self.total, self.chunk):
            ranks = np.arange(lo, min(lo + self.chunk, self.total),
                              dtype=np.int64)
            rows, _ = self.rows_for_ranks(ranks)
            zeros += (rows == 0).sum(axis=0)
        want = self.cert["coordinate_cuts"]
        got = [int(zeros[j]) for j in range(1, len(want) + 1)]
        assert got == [int(w) for w in want], \
            f"coordinate cuts {got} != certificate {want}"
        log(f"  coordinate cuts verified: {got}")
        return True

    def cut_counts(self, bs, log=None):
        """|Z(b)| for a list of syndromes, streamed over the cloud with
        overflow-safe arithmetic: per-term reduction BEFORE the row sum
        (the 2026-07-22 ad-hoc consumer bug — 17 unreduced products of
        ~2^62 overflow int64 — is structurally excluded here; see the
        s=32 strata incident). Gated in selfcheck() against the
        authority's cut_counts."""
        xp = cp if GPU else np
        B = [xp.asarray(b, dtype=xp.int64) for b in bs]
        zeros = np.zeros(len(bs), dtype=np.int64)
        for lo in range(0, self.total, self.chunk):
            ranks = np.arange(lo, min(lo + self.chunk, self.total),
                              dtype=np.int64)
            rows, _ = self._device_rows(ranks)
            rows64 = rows.astype(xp.int64)
            for i, b in enumerate(B):
                acc = ((rows64 * b[None, :]) % self.p).sum(axis=1) % self.p
                zeros[i] += int((acc == 0).sum())
        return zeros.tolist()

    def strata_counts(self, b):
        """Antipodal strata Z^(F) of one cut (F = pairs inside the rank
        subset), same safe arithmetic. Gated against the authority at
        s = 16 and the pinned s = 32 value."""
        xp = cp if GPU else np
        bd = xp.asarray(b, dtype=xp.int64)
        half = self.s // 2
        nf = self.r // 2 + 1
        strata = np.zeros(nf, dtype=np.int64)
        for lo in range(0, self.total, self.chunk):
            ranks = np.arange(lo, min(lo + self.chunk, self.total),
                              dtype=np.int64)
            rows, idx = self._device_rows(ranks)
            rows64 = rows.astype(xp.int64)
            acc = ((rows64 * bd[None, :]) % self.p).sum(axis=1) % self.p
            hit = acc == 0
            if bool(hit.any()):
                in_set = xp.zeros((idx.shape[0], self.s), dtype=bool)
                xp.put_along_axis(in_set, xp.asarray(idx), True, axis=1)
                pairs = (in_set[:, :half] & in_set[:, half:]).sum(axis=1)
                h = xp.bincount(pairs[hit], minlength=nf)
                strata += (cp.asnumpy(h) if GPU else h)[:nf]
        return strata.tolist()

    def _device_rows(self, ranks_np):
        """rows_for_ranks, but keeping arrays on-device when GPU."""
        xp = cp if GPU else np
        ranks = xp.asarray(ranks_np, dtype=xp.int64)
        idx = unrank_block(ranks, self.s, self.r, self.T, xp)
        rows = complement_rows(idx, xp.asarray(self.dom), self.p,
                               self.s, self.r, xp)
        return rows, (cp.asnumpy(idx) if GPU else idx)

    def build(self, out_dir, log=print):
        os.makedirs(out_dir, exist_ok=True)
        self.verify_certificate()
        log(f"certificate pins verified; building {self.total:,} rows")
        shards = []
        t0 = time.time()
        for i, lo in enumerate(range(0, self.total, self.chunk)):
            ranks = np.arange(lo, min(lo + self.chunk, self.total),
                              dtype=np.int64)
            rows, _ = self.rows_for_ranks(ranks)
            fn = os.path.join(out_dir, f"cloud_{i:05d}.npy")
            np.save(fn, rows)
            shards.append({
                "file": os.path.basename(fn), "lo": int(lo),
                "n": int(len(ranks)),
                "sha256": hashlib.sha256(rows.tobytes()).hexdigest()[:16],
            })
            if i % 50 == 0:
                log(f"  shard {i}: {100 * lo / self.total:5.1f}% "
                    f"({time.time() - t0:.0f}s)")
        manifest = {
            "p": self.p, "s": self.s, "k": self.k, "r": self.r,
            "cols": self.cols, "total": self.total, "chunk": self.chunk,
            "dtype": "uint32", "certificate": _cert_jsonable(self.cert),
            "shards": shards,
        }
        json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"),
                  indent=1)
        log(f"built {len(shards)} shards in {time.time() - t0:.0f}s")
        return manifest


def _cert_jsonable(cert):
    return {
        "version": cert["version"], "p": cert["p"], "s": cert["s"],
        "k": cert["k"],
        "ranking": [[int(rk), list(sub)] for rk, sub in cert["ranking"]],
        "moment_rows": [[int(rk), [int(x) for x in row]]
                        for rk, row in cert["moment_rows"]],
        "domain_head": [int(x) for x in cert["domain_head"]],
        "coordinate_cuts": [int(x) for x in cert["coordinate_cuts"]],
    }


def verify_stored(dir_, log=print):
    man = json.load(open(os.path.join(dir_, "manifest.json")))
    eng = CloudEngine(man["p"], man["s"], man["k"], chunk=man["chunk"])
    assert _cert_jsonable(eng.cert) == man["certificate"], \
        "stored certificate != authority certificate (convention drift?)"
    eng.verify_certificate()
    for sh in man["shards"][:3] + man["shards"][-1:]:
        rows = np.load(os.path.join(dir_, sh["file"]))
        assert hashlib.sha256(rows.tobytes()).hexdigest()[:16] == sh["sha256"]
    log(f"stored cloud verified: {man['total']:,} rows, "
        f"{len(man['shards'])} shards")
    return True


def selfcheck():
    """The s = 16 gate: every pin + full coordinate cuts, CPU-feasible."""
    eng = CloudEngine(65537, 16, 7)
    eng.verify_certificate()
    print("  certificate pins: PASS")
    eng.verify_coordinate_cuts()
    # spot-check 500 random rows against the authority
    rng = np.random.default_rng(3)
    ranks = rng.integers(0, eng.total, size=500)
    rows, idx = eng.rows_for_ranks(ranks.astype(np.int64))
    for rk, row, sub in zip(ranks, rows, idx):
        want = eng.space.moment_row([int(x) for x in sub])
        assert [int(x) for x in row] == want, f"row mismatch at rank {rk}"
        assert eng.space.subset_rank([int(x) for x in sub]) == rk
    print("  500 random rows vs authority: PASS")
    # C2 gate: engine cut/strata vs the Rust authority at s = 16
    top = eng.space.top_word(4)
    b_top = eng.space.syndrome(top)
    w18 = [14274, 45571, 60798, 30803, 16774, 53622, 23957, 63873, 57198,
           44950, 44028, 28126, 25267, 3166, 17634, 55356]
    b18 = eng.space.syndrome(w18)
    got = eng.cut_counts([b_top, b18])
    want = eng.space.cut_counts([b_top, b18])
    assert got == want == [810, 404], f"cut gate: {got} vs {want}"
    print(f"  engine cut vs authority: {got} PASS")
    st = eng.strata_counts(b18)
    assert st == [0, 48, 164, 180, 12], f"strata gate: {st}"
    print(f"  engine strata (18-word): {st} PASS")
    print("SELFCHECK PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--p", type=int, default=65537)
    ap.add_argument("--s", type=int, default=32)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--out", type=str, default="cloud_out")
    ap.add_argument("--dir", type=str, default="cloud_out")
    a = ap.parse_args()
    if a.selfcheck:
        selfcheck()
    elif a.build:
        CloudEngine(a.p, a.s, a.k).build(a.out)
    elif a.verify:
        verify_stored(a.dir)
