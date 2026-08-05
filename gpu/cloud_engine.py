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


KERNEL_POOL_ZERO = r"""
__device__ __forceinline__ unsigned long long bmod64(
        unsigned long long x, unsigned int p, unsigned long long minv) {
    unsigned long long q = __umul64hi(x, minv);
    unsigned long long r = x - q * (unsigned long long)p;
    while (r >= p) r -= p;
    return r;
}

extern "C" __global__ void pool_zero_count(
        const long long n_rows, const int n_pool,
        const unsigned int p, const unsigned long long minv,
        const unsigned int* __restrict__ rows,
        const unsigned int* __restrict__ pool,
        unsigned long long* counts) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows) return;
    unsigned int r[COLS];
    #pragma unroll
    for (int l = 0; l < COLS; ++l) r[l] = rows[i * COLS + l];
    for (int j = 0; j < n_pool; ++j) {
        unsigned long long acc = 0;
        #pragma unroll
        for (int l = 0; l < COLS; ++l)
            acc += bmod64((unsigned long long)r[l] * pool[j * COLS + l],
                          p, minv);
        if (bmod64(acc, p, minv) == 0)
            atomicAdd(&counts[j], 1ULL);
    }
}
"""

_POOLK_CACHE = {}


def _pool_kernel(cols):
    """Compile (or fetch) the cols-templated fused zero-count kernel
    (#16 optimization P2): one thread per row, per-term Barrett
    reduction in registers, atomics only on the rare zero hits — no
    materialized accumulator matrices, so the pool screen's cost is
    compute alone."""
    if cols not in _POOLK_CACHE:
        src = KERNEL_POOL_ZERO.replace("COLS", str(cols))
        _POOLK_CACHE[cols] = cp.RawKernel(src, "pool_zero_count")
    return _POOLK_CACHE[cols]


def mod_matmul(A, B, p, xp):
    """(A @ B) % p for int64 device arrays with entries in [0, p),
    p < 2^31: the matmul accumulates 16-bit-split halves of B so every
    partial sum stays below 2^52 for inner dimensions up to ~2^5 rows
    of slack (2^31 * 2^16 * dim) — the per-term-reduction doctrine in
    matmul form."""
    lo = (A @ (B & 0xFFFF)) % p
    hi = (A @ (B >> 16)) % p
    return (lo + (hi << 16)) % p


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
    def __init__(self, p, s, k, chunk=1 << 20, light=False):
        """light=True skips the precomputed certificate — at s = 32 the
        Rust certificate streams full coordinate cuts (10+ min CPU per
        prime). Light engines must gate through verify_pins() instead:
        direct authority spot-calls (subset_unrank + moment_row), the
        same pins verify_certificate checks, without the cut census."""
        self.space = vanish.VsSpace(p, s, k)
        self.p, self.s, self.k = p, s, k
        self.r = self.space.r
        self.cols = s - self.r + 1
        self.total = math.comb(s, self.r)
        self.chunk = chunk
        self.T = binom_table(s, self.r)
        self.dom = np.array(self.space.domain(), dtype=np.int64)
        self.cert = None if light else self.space.certificate()
        self._rows_res = None
        self._pairs_res = None

    def verify_pins(self, n=8, seed=0):
        """Authority spot-gate for light engines: n random ranks must
        unrank and row-build identically to the Rust space."""
        rng = np.random.RandomState(seed)
        ranks = np.unique(rng.randint(0, self.total, size=n))
        rows, idx = self.rows_for_ranks(ranks)
        for rk, sub, row in zip(ranks, idx, rows):
            want_sub = list(self.space.subset_unrank(int(rk)))
            assert list(int(x) for x in sub) == want_sub, \
                f"unrank pin failed at rank {rk}"
            want_row = list(self.space.moment_row(want_sub))
            assert [int(x) for x in row] == want_row, \
                f"moment-row pin failed at rank {rk}"
        assert list(self.dom[:8]) == list(self.space.domain()[:8])
        return True

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
        for i, lo, hi in self._chunk_bounds():
            rows64 = self._chunk_rows(i, lo, hi, xp).astype(xp.int64)
            for j, b in enumerate(B):
                acc = ((rows64 * b[None, :]) % self.p).sum(axis=1) % self.p
                zeros[j] += int((acc == 0).sum())
        return zeros.tolist()

    def pool_cut_counts(self, B, tile_pool=512, out="host", path=None):
        """Batched syndrome fitness (#16 pool extension): |Z(b)| for a
        pool of syndromes against the cloud — the rank-3 search's
        inner loop.

        B: [N, cols] syndrome pool, host or device; out="device" keeps
        the counts on-device for a resident optimizer (the pool API —
        no per-candidate host round-trip).

        Two arithmetic paths, gated equal in selfcheck. The fused
        kernel (GPU default, path="fused"): one thread per row,
        per-term Barrett reduction in registers, atomics only on the
        rare zero hits — no materialized accumulators, so cost is
        compute alone and the marginal price per candidate is ~ms.
        The matmul path (CPU always, path="matmul"): 16-bit-split
        accumulation via mod_matmul, every partial sum below 2^52 —
        the per-term-reduction doctrine in matmul form."""
        xp = cp if GPU else np
        Bd = xp.asarray(B, dtype=xp.int64) % self.p
        n = Bd.shape[0]
        assert Bd.shape[1] == self.cols, f"pool cols {Bd.shape[1]}"
        use_fused = GPU if path is None else path == "fused"
        if use_fused and not GPU:
            raise ValueError("fused path requires cupy")
        if use_fused:
            pool_dev = cp.ascontiguousarray(Bd.astype(cp.uint32))
            counts = cp.zeros(n, dtype=cp.uint64)
            minv = np.uint64((1 << 64) // self.p)
            kern = _pool_kernel(self.cols)
            threads = 256
            for i, lo, hi in self._chunk_bounds():
                rows = self._chunk_rows(i, lo, hi, cp)
                rows_u = cp.ascontiguousarray(rows.astype(cp.uint32))
                m = rows_u.shape[0]
                blocks = (m + threads - 1) // threads
                kern((blocks,), (threads,),
                     (np.int64(m), np.int32(n), np.uint32(self.p),
                      minv, rows_u, pool_dev, counts))
            cp.cuda.Stream.null.synchronize()
            zeros = counts.astype(cp.int64)
        else:
            zeros = xp.zeros(n, dtype=xp.int64)
            Bt = Bd.T.copy()
            for i, lo, hi in self._chunk_bounds():
                rows64 = self._chunk_rows(i, lo, hi, xp).astype(xp.int64)
                for j in range(0, n, tile_pool):
                    acc = mod_matmul(rows64, Bt[:, j:j + tile_pool],
                                     self.p, xp)
                    zeros[j:j + tile_pool] += (acc == 0).sum(axis=0)
        if out == "device":
            return zeros
        return (cp.asnumpy(zeros) if GPU else zeros)

    def strata_counts(self, b):
        """Antipodal strata Z^(F) of one cut (F = pairs inside the rank
        subset), same safe arithmetic. Gated against the authority at
        s = 16 and the pinned s = 32 value."""
        xp = cp if GPU else np
        bd = xp.asarray(b, dtype=xp.int64)
        nf = self.r // 2 + 1
        strata = np.zeros(nf, dtype=np.int64)
        for i, lo, hi in self._chunk_bounds():
            rows, pairs = self._chunk_rows_pairs(i, lo, hi, xp)
            rows64 = rows.astype(xp.int64)
            acc = ((rows64 * bd[None, :]) % self.p).sum(axis=1) % self.p
            hit = acc == 0
            if bool(hit.any()):
                h = xp.bincount(pairs[hit].astype(xp.int64),
                                minlength=nf)
                strata += (cp.asnumpy(h) if GPU else np.asarray(h))[:nf]
        return strata.tolist()

    def value_histograms(self, cols=None, top=5):
        """C6 — exact value spectra per cloud column: the multiplicity
        histogram of {e_j(complement)} over all C(s, r) subsets. This is
        the L^2 instrument: A_j(p) = max_mult of column j. Returns
        {j: {"distinct", "max_mult", "top": [(value, mult), ...]}}.
        bincount path for p <= 2^27, sort/run-length path above. Gated
        in selfcheck() against the authority's cloud."""
        xp = cp if GPU else np
        cols = list(range(1, self.cols)) if cols is None else list(cols)
        out = {}
        if self.p <= (1 << 24):
            # incremental bincount: one cloud pass, O(p) memory per
            # column, no column materialization (601M-row safe)
            cnts = {j: xp.zeros(self.p, dtype=xp.int64) for j in cols}
            for i, lo, hi in self._chunk_bounds():
                rows = self._chunk_rows(i, lo, hi, xp)
                for j in cols:
                    cnts[j] += xp.bincount(rows[:, j].astype(xp.int64),
                                           minlength=self.p)
            for j in cols:
                cnt = cnts[j]
                distinct = int((cnt > 0).sum())
                order = xp.argsort(cnt)[::-1][:top]
                ov = cp.asnumpy(order) if GPU else order
                tops = [(int(v), int(cnt[int(v)])) for v in ov
                        if int(cnt[int(v)]) > 0]
                out[j] = {"distinct": distinct, "max_mult": tops[0][1],
                          "top": tops}
            return out
        parts = {j: [] for j in cols}
        for i, lo, hi in self._chunk_bounds():
            rows = self._chunk_rows(i, lo, hi, xp)
            for j in cols:
                parts[j].append(rows[:, j].astype(xp.int64))
        for j in cols:
            col = xp.concatenate(parts[j])
            srt = xp.sort(col)
            edge = xp.flatnonzero(srt[1:] != srt[:-1]) + 1
            z = xp.zeros(1, dtype=edge.dtype)
            starts = xp.concatenate([z, edge])
            ends = xp.concatenate([edge, xp.asarray([srt.size])])
            lens = ends - starts
            distinct = int(lens.size)
            order = xp.argsort(lens)[::-1][:top]
            ov = cp.asnumpy(order) if GPU else order
            tops = [(int(srt[int(starts[int(i)])]), int(lens[int(i)]))
                    for i in ov]
            out[j] = {"distinct": distinct, "max_mult": tops[0][1],
                      "top": tops}
        return out

    def _build_chunk(self, ranks_np, xp):
        """One chunk of the cloud, everything on-device: (rows uint32
        [n, cols], subset indices int64 [n, r])."""
        ranks = xp.asarray(ranks_np, dtype=xp.int64)
        idx = unrank_block(ranks, self.s, self.r, self.T, xp)
        rows = complement_rows(idx, xp.asarray(self.dom), self.p,
                               self.s, self.r, xp)
        return rows, idx

    def _device_rows(self, ranks_np):
        """rows_for_ranks, but keeping row arrays on-device when GPU."""
        xp = cp if GPU else np
        rows, idx = self._build_chunk(ranks_np, xp)
        return rows, (cp.asnumpy(idx) if GPU else idx)

    def materialize(self, with_pairs=False, log=print):
        """The resident cloud (#16 optimization P1): build every row
        chunk once (uint32, per-chunk device arrays — cols * 4 bytes
        per row, ~43 GB at (32, 15), within an 80 GB A100) and serve
        all counters from the resident copy. The per-call unrank +
        e-vector rebuild (~38 s per full pass at (32, 15)) disappears
        from every subsequent operation. with_pairs adds one uint8 per
        row (the antipodal pair count) so strata_counts is resident
        too."""
        xp = cp if GPU else np
        half = self.s // 2
        rows_res = []
        pairs_res = [] if with_pairs else None
        t0 = time.time()
        for i, lo, hi in self._chunk_bounds():
            rows, idx = self._build_chunk(
                np.arange(lo, hi, dtype=np.int64), xp)
            rows_res.append(rows.astype(xp.uint32))
            if with_pairs:
                in_set = xp.zeros((idx.shape[0], self.s), dtype=bool)
                xp.put_along_axis(in_set, idx, True, axis=1)
                pairs_res.append(
                    (in_set[:, :half] & in_set[:, half:])
                    .sum(axis=1).astype(xp.uint8))
            if i % 100 == 0:
                log(f"  materialize {100 * lo / self.total:5.1f}% "
                    f"({time.time() - t0:.0f}s)")
        self._rows_res, self._pairs_res = rows_res, pairs_res
        log(f"  resident: {self.total:,} rows, {len(rows_res)} chunks "
            f"({time.time() - t0:.0f}s)")

    def _chunk_bounds(self):
        for i, lo in enumerate(range(0, self.total, self.chunk)):
            yield i, lo, min(lo + self.chunk, self.total)

    def _chunk_rows(self, i, lo, hi, xp):
        """Row chunk (uint32, on-device when GPU), resident if built."""
        if self._rows_res is not None:
            return self._rows_res[i]
        rows, _ = self._build_chunk(np.arange(lo, hi, dtype=np.int64),
                                    xp)
        return rows

    def _chunk_rows_pairs(self, i, lo, hi, xp):
        """(rows, antipodal pair counts) for one chunk, resident when
        materialize(with_pairs=True) was run; with rows-only residency
        the pair counts recompute from the unrank alone (no e-vector
        rebuild)."""
        if self._rows_res is not None and self._pairs_res is not None:
            return self._rows_res[i], self._pairs_res[i]
        ranks = np.arange(lo, hi, dtype=np.int64)
        if self._rows_res is not None:
            rows = self._rows_res[i]
            idx = unrank_block(xp.asarray(ranks, dtype=xp.int64),
                               self.s, self.r, self.T, xp)
        else:
            rows, idx = self._build_chunk(ranks, xp)
        half = self.s // 2
        in_set = xp.zeros((idx.shape[0], self.s), dtype=bool)
        xp.put_along_axis(in_set, idx, True, axis=1)
        pairs = (in_set[:, :half] & in_set[:, half:]).sum(axis=1)
        return rows, pairs

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


class DescentTier:
    """#16 descent tier: channel splits, channel syndromes, and psi_Y
    fiber statistics for batches of words, on device — the s = 64
    stratum sweeps' acceleration path.

    Certification before use (the engine's contract): the tier refuses
    to serve until it reproduces the authority (vanish.Descent) on the
    canonical word w[i] = dom[i] and on random words — channels,
    interpolant coefficients, channel syndromes, and psi_Y statistics
    at the head core and a random core. When the engine carries a full
    certificate, the descent_pins block is checked as well.

    Conventions mirrored from src/rs/descent.rs: half_points[j] =
    dom[2j] with lifts dom[j], dom[j + s/2]; w_even[j] =
    (w[j] + w[j+s/2])/2, w_odd[j] = (w[j] - w[j+s/2])/(2 dom[j]);
    B_t[j] = c[k + 2j + t]; psi_Y from the core-interpolated channel
    pair, normalized by the vanishing product."""

    def __init__(self, engine, checks=8, seed=0):
        self.eng = engine
        p, s, k = engine.p, engine.s, engine.k
        self.p, self.s, self.k = p, s, k
        self.half = s // 2
        assert s % 2 == 0, "descent needs an even level"
        self.auth = vanish.Descent(p, s, k)
        self.k_odd = self.auth.k_odd()
        xp = cp if GPU else np
        dom = engine.dom
        self.dom_d = xp.asarray(dom)
        self.inv2 = pow(2, p - 2, p)
        self.inv_dom_d = xp.asarray(np.array(
            [pow(int(x), p - 2, p) for x in dom[: self.half]],
            dtype=np.int64))
        # inverse-DFT interpolation: c_j = s^{-1} sum_i w_i dom_i^{-j}
        sinv = pow(s, p - 2, p)
        Mi = np.zeros((s, s), dtype=np.int64)
        for i in range(s):
            xinv = pow(int(dom[i]), p - 2, p)
            v = sinv
            for j in range(s):
                Mi[j, i] = v
                v = (v * xinv) % p
        self.interp_t = xp.asarray(Mi.T.copy())
        self._certify(checks, seed)

    # -- batch operations (words: [N, s] int64 in [0, p)) --------------

    def channels(self, W):
        xp = cp if GPU else np
        Wd = xp.asarray(W, dtype=xp.int64)
        lo, hi = Wd[:, : self.half], Wd[:, self.half:]
        wev = ((lo + hi) % self.p) * self.inv2 % self.p
        wod = (((lo - hi) % self.p) * self.inv2 % self.p
               * self.inv_dom_d[None, :]) % self.p
        return wev, wod

    def coeffs(self, W):
        xp = cp if GPU else np
        Wd = xp.asarray(W, dtype=xp.int64)
        return mod_matmul(Wd, self.interp_t, self.p, xp)

    def channel_syndromes(self, W):
        """[N, 3, m] with B_t[j] = c[k + 2j + t] (zero past s - 1)."""
        xp = cp if GPU else np
        c = self.coeffs(W)
        m = self.half - self.k // 2
        out = xp.zeros((c.shape[0], 3, m), dtype=xp.int64)
        for t in range(3):
            js = np.arange(m)
            keep = self.k + 2 * js + t < self.s
            out[:, t, keep] = c[:, self.k + 2 * js[keep] + t]
        return out

    def _core_tables(self, core):
        """Host precompute for a fixed core: the Lagrange evaluation
        matrix over the available half points and the vanishing-product
        inverses."""
        p = self.p
        dom = self.eng.dom
        core = list(core)
        assert len(core) == self.k_odd and all(
            0 <= y < self.half for y in core), "bad core"
        nodes = [int(dom[2 * y % self.s]) for y in core]
        avail = [j for j in range(self.half) if j not in core]
        us = [int(dom[2 * j % self.s]) for j in avail]
        Lg = np.zeros((len(us), len(nodes)), dtype=np.int64)
        for a, u in enumerate(us):
            for c, n in enumerate(nodes):
                num = den = 1
                for c2, n2 in enumerate(nodes):
                    if c2 != c:
                        num = num * ((u - n2) % p) % p
                        den = den * ((n - n2) % p) % p
                Lg[a, c] = num * pow(den, p - 2, p) % p
        v_inv = np.array(
            [pow(int(np.prod([(u - n) % p for n in nodes]) % p),
                 p - 2, p) for u in us], dtype=np.int64)
        return avail, Lg, v_inv

    def psi_y(self, W, core):
        """psi_Y values for a batch of words at one shared core:
        [N, 2 * navail], columns (2a, 2a+1) = the two lifts (j_a,
        j_a + s/2) of the a-th available half point, ascending."""
        xp = cp if GPU else np
        p = self.p
        Wd = xp.asarray(W, dtype=xp.int64)
        wev, wod = self.channels(Wd)
        avail, Lg, v_inv = self._core_tables(core)
        core_ix = xp.asarray(np.array(core, dtype=np.int64))
        Lg_t = xp.asarray(Lg.T.copy())
        gs = mod_matmul(wev[:, core_ix], Lg_t, p, xp)
        hs = mod_matmul(wod[:, core_ix], Lg_t, p, xp)
        vd = xp.asarray(v_inv)
        out = xp.zeros((Wd.shape[0], 2 * len(avail)), dtype=xp.int64)
        for a, j in enumerate(avail):
            for lift, i in enumerate((j, j + self.half)):
                x = int(self.eng.dom[i])
                num = (Wd[:, i] - gs[:, a] - x * hs[:, a] % p) % p
                out[:, 2 * a + lift] = num * vd[a] % p
        return out

    def psi_stats(self, W, core, chunk=1 << 14):
        """Per-word psi_Y fiber statistics [N, 4]: (total, distinct,
        max_fiber, collisions), collisions counting unordered
        non-antipodal equal-value pairs — the authority's psi_y_stats,
        batched. The two lifts of one half point are antipodal, so the
        correction is the per-column lift agreement count."""
        xp = cp if GPU else np
        V = self.psi_y(W, core)
        n, m = V.shape
        out = np.zeros((n, 4), dtype=np.int64)
        out[:, 0] = m
        for lo in range(0, n, chunk):
            v = V[lo:lo + chunk]
            eq = v[:, :, None] == v[:, None, :]
            pairs = (eq.sum(axis=(1, 2)) - m) // 2
            antip = (v[:, 0::2] == v[:, 1::2]).sum(axis=1)
            fibers = eq.sum(axis=2)
            vs = xp.sort(v, axis=1)
            distinct = 1 + (vs[:, 1:] != vs[:, :-1]).sum(axis=1)
            blk = np.stack([
                (cp.asnumpy(x) if GPU else np.asarray(x)) for x in (
                    distinct, fibers.max(axis=1), pairs - antip)], axis=1)
            out[lo:lo + chunk, 1:] = blk
        return out

    # -- certification -------------------------------------------------

    def _certify(self, checks, seed):
        p, s = self.p, self.s
        rng = np.random.default_rng(seed)
        canonical = np.array(self.eng.dom, dtype=np.int64)
        words = np.vstack([canonical[None, :],
                           rng.integers(0, p, (checks, s))])
        head_core = list(range(self.k_odd))
        wev, wod = self.channels(words)
        cs = self.coeffs(words)
        syn = self.channel_syndromes(words)
        stats = self.psi_stats(words, head_core)
        rand_core = sorted(
            rng.choice(self.half, self.k_odd, replace=False).tolist())
        stats_r = self.psi_stats(words, rand_core)
        for i, w in enumerate(words):
            wl = [int(x) for x in w]
            aev, aod = self.auth.channels(wl)
            assert [int(x) for x in wev[i]] == aev, f"tier channels {i}"
            assert [int(x) for x in wod[i]] == aod, f"tier channels {i}"
            assert [int(x) for x in cs[i]] == \
                self.auth.monomial_coeffs(wl), f"tier coeffs {i}"
            view = self.auth.word(wl)
            for t, bt in enumerate(view.channel_syndromes()):
                assert [int(x) for x in syn[i, t]] == bt, \
                    f"tier syndrome slice {t} word {i}"
            for core, got in ((head_core, stats[i]), (rand_core,
                                                      stats_r[i])):
                want = view.psi_y_stats(core)
                assert tuple(int(x) for x in got) == want, \
                    f"tier psi stats word {i} core {core}: " \
                    f"{tuple(got)} vs {want}"
        cert = self.eng.cert
        if cert is not None and cert.get("descent"):
            pins = cert["descent"]
            assert [int(x) for x in wev[0][:4]] == \
                list(pins["wev_head"]), "descent_pins wev"
            assert [int(x) for x in wod[0][:4]] == \
                list(pins["wod_head"]), "descent_pins wod"
            for t in range(3):
                assert [int(x) for x in syn[0, t, :4]] == \
                    list(pins["slice_heads"][t]), f"descent_pins B{t}"
            v = self.psi_y(words[:1], head_core)
            avail, _, _ = self._core_tables(head_core)
            assert (avail[0], int(v[0, 0])) == \
                tuple(pins["psi_sample"]), "descent_pins psi"


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
    # pool extension gate: the matmul path == the streamed path == the
    # authority, on the pins plus a random pool (odd size to exercise
    # the tile remainder)
    pool = [b_top, b18] + [[int(x) for x in rng.integers(0, eng.p, eng.cols)]
                           for rng in [np.random.default_rng(7)]
                           for _ in range(21)]
    got_pool = eng.pool_cut_counts(np.array(pool), tile_pool=8,
                                   path="matmul").tolist()
    want_pool = [int(x) for x in eng.space.cut_counts(pool)]
    assert got_pool == want_pool, f"pool gate: {got_pool} vs {want_pool}"
    assert got_pool[:2] == [810, 404]
    print(f"  pool matmul cut vs authority (23-pool): PASS")
    # P1/P2 gates: the resident cloud reproduces every counter, and on
    # GPU the fused kernel path equals the matmul path equals the
    # authority
    eng.materialize(with_pairs=True, log=lambda *a, **k: None)
    assert eng.cut_counts([b_top, b18]) == [810, 404], "resident cut"
    assert eng.strata_counts(b18) == [0, 48, 164, 180, 12], \
        "resident strata"
    assert eng.pool_cut_counts(np.array(pool), tile_pool=8,
                               path="matmul").tolist() == want_pool, \
        "resident pool matmul"
    if GPU:
        fused = eng.pool_cut_counts(np.array(pool), path="fused").tolist()
        assert fused == want_pool, f"fused path: {fused} vs {want_pool}"
        print("  resident cloud + fused kernel vs authority: PASS")
    else:
        print("  resident cloud (matmul path; fused needs GPU): PASS")
    # C6 gate: value histograms vs the authority cloud
    rows_auth = np.array(
        [eng.space.moment_row(eng.space.subset_unrank(i))
         for i in range(eng.total)], dtype=np.int64)
    vh = eng.value_histograms(cols=[1, 2])
    for j in (1, 2):
        vals, cnts = np.unique(rows_auth[:, j], return_counts=True)
        assert vh[j]["distinct"] == int(vals.size), f"distinct col {j}"
        assert vh[j]["max_mult"] == int(cnts.max()), f"max_mult col {j}"
        want = sorted(cnts.tolist(), reverse=True)[: len(vh[j]["top"])]
        got = sorted((c for _, c in vh[j]["top"]), reverse=True)
        assert got == want, f"top counts col {j}: {got} vs {want}"
    print("  value histograms (cols 1,2) vs authority: PASS")
    lite = CloudEngine(eng.p, eng.s, eng.k, light=True)
    lite.verify_pins(n=8)
    print("  light-engine pins vs authority: PASS")
    # descent tier: construction IS the gate (authority + descent_pins
    # checks on the canonical and random words); light engines certify
    # through the authority path alone
    DescentTier(eng, checks=8)
    DescentTier(lite, checks=4)
    print("  descent tier certified (full + light engines): PASS")
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
