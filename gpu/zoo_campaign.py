"""GPU zoo-scaling campaign driver — extend every coordinate-family series
to s = 32 and beyond, and sample the dense bulk.

Data-first program (2026-07-22): the purpose is series, not certificates.
Per (s, p) this driver produces the rows that extend the small-s CPU census
(experiments/landscape/census_zoo_scaling.py) to GPU scale:

  cloud   — exact zero-fiber of every coordinate e_j over ALL C(s, r)
            r-subsets (r = s/2), plus the top-class histogram (e_r is
            mu_s-valued up to sign). Chunked unrank -> poly-build on GPU;
            601M subsets at s = 32 in minutes on an A100.
  zoo     — decoded L of every coordinate word x^{k+j} (k = r-1) at
            agreement r (optionally r+1, r+2: the depth sweep) via the
            validated GPU decoder (decode_gpu.gpu_list_size).
  bulk    — L for N uniform-random words: the list-size distribution of
            the dense bulk (where does the top word sit vs the tail?).

Correctness gate: `python zoo_campaign.py --validate` runs the identical
cloud/decode code paths on numpy + the CPU vanish decoder at s = 16
(pins: top class 810, e1 zero-fiber 70, e4 zero-fiber 266, L(x^8) = 70,
L(x^11) = 266) and s = 12 (e1 zero-fiber 24 = the Lam-Leung value).
Run it on the pod BEFORE believing any s = 32 output.

Usage (pod, A100, after decode_gpu.py --validate passes):
  python zoo_campaign.py --validate
  python zoo_campaign.py --cloud --s 32 --p 2130706433 --out cloud_s32_kb.json
  python zoo_campaign.py --zoo   --s 32 --p 2130706433 --depth 2 --out zoo_s32_kb.json
  python zoo_campaign.py --bulk  --s 32 --p 65537 --n 20000 --out bulk_s32.json
Requires: cupy-cuda12x on the pod (numpy fallback for --validate), vanish
wheel for CPU cross-checks.
"""
import argparse
import json
import time
from math import comb

import numpy as np

try:
    import cupy as cp
    GPU = True
except ImportError:  # validation path on a CPU box
    cp = np
    GPU = False


# ---------------------------------------------------------------------------
# cloud census: chunked unrank -> product polynomial -> coordinate fibers
# ---------------------------------------------------------------------------

def _binom_table(s, r):
    T = np.zeros((s + 1, r + 1), dtype=np.int64)
    T[:, 0] = 1
    for n in range(1, s + 1):
        for k in range(1, min(n, r) + 1):
            T[n][k] = T[n - 1][k - 1] + T[n - 1][k]
    return T

def unrank_block(ranks, s, r, T):
    """Vectorized lex unrank: ranks (B,) -> combinations (B, r) of [0..s)."""
    xp = cp.get_array_module(ranks) if GPU else np
    B = ranks.shape[0]
    R = ranks.copy()
    need = xp.full(B, r, dtype=xp.int64)
    out = xp.zeros((B, r), dtype=xp.int64)
    Tx = xp.asarray(T)
    for pos in range(s):
        cnt = Tx[s - pos - 1, xp.maximum(need - 1, 0)]
        take = (need > 0) & (R < cnt)
        slot = r - need
        rows = xp.where(take)[0]
        out[rows, slot[rows]] = pos
        R = xp.where((need > 0) & ~take, R - cnt, R)
        need = need - take.astype(xp.int64)
    return out

def cloud_census(s, p, dom, chunk=1 << 22, progress=True):
    """Exact zero-fiber per coordinate j=1..r and the top-coefficient
    histogram, over all C(s,r) r-subsets. Convention: coef[:, m] holds the
    z^m coefficient of prod_{x in T}(z - x), so e_j(T) = +/- coef[:, r-j]
    and the constant term coef[:, 0] = +/- e_r(T); zero-fibers and class
    sizes are sign-invariant."""
    r = s // 2
    T = _binom_table(s, r)
    total = comb(s, r)
    domx = cp.asarray(np.array(dom, dtype=np.int64))
    zero = np.zeros(r + 1, dtype=np.int64)
    top = {}
    spectra = p < (1 << 24)
    hists = [cp.zeros(p, dtype=cp.int64) for _ in range(r)] if spectra else None
    t0 = time.time()
    for lo in range(0, total, chunk):
        hi = min(lo + chunk, total)
        ranks = cp.arange(lo, hi, dtype=cp.int64)
        idx = unrank_block(ranks, s, r, T)
        pts = domx[idx]                                   # (B, r)
        B = pts.shape[0]
        coef = cp.zeros((B, r + 1), dtype=cp.int64)
        coef[:, 0] = 1
        for m in range(r):
            xm = pts[:, m:m + 1]
            shifted = cp.zeros_like(coef)
            shifted[:, 1:] = coef[:, :-1]
            coef = (shifted + (p - xm) * coef) % p
        for j in range(1, r + 1):
            zero[j] += int((coef[:, r - j] == 0).sum())
            if spectra:
                hists[j - 1] += cp.bincount(coef[:, r - j], minlength=p)
        vals, cnts = (cp.unique(coef[:, 0], return_counts=True)
                      if GPU else np.unique(coef[:, 0], return_counts=True))
        for v, c in zip(vals.tolist(), cnts.tolist()):
            top[v] = top.get(v, 0) + c
        if progress and GPU:
            done = hi / total
            print(f"  cloud {100*done:5.1f}%  ({time.time()-t0:.0f}s)",
                  flush=True)
    out = {"s": s, "p": p, "total": total,
           "zero_fiber": {j: int(zero[j]) for j in range(1, r + 1)},
           "top_class_max": max(top.values()),
           "top_hist": sorted(top.values(), reverse=True)}
    if spectra:
        spec = {}
        for j in range(1, r + 1):
            h = hists[j - 1]
            sizes, mult = (cp.unique(h[h > 0], return_counts=True) if GPU
                           else np.unique(h[h > 0], return_counts=True))
            spec[j] = {int(sz): int(m) for sz, m in
                       zip(sizes.tolist(), mult.tolist())}
        out["fiber_size_spectrum"] = spec
    return out


# ---------------------------------------------------------------------------
# decode paths
# ---------------------------------------------------------------------------

def decode_L(p, dom, word, k, t):
    if GPU:
        from decode_gpu import gpu_list_size
        return int(gpu_list_size(p, list(dom), list(word), k, t))
    import vanish
    return len(vanish.list_decode(p, list(dom), k, list(word), t))

def zoo_decodes(s, p, dom, depth=0):
    r, k = s // 2, s // 2 - 1
    rows = []
    for j in range(1, r):
        w = [pow(int(x), k + j, p) for x in dom]
        for t in range(r, r + depth + 1):
            t0 = time.time()
            L = decode_L(p, dom, w, k, t)
            rows.append({"s": s, "p": p, "j": j, "t": t, "L": L,
                         "secs": round(time.time() - t0, 1)})
            print(f"  x^{k+j} t={t}: L = {L} ({rows[-1]['secs']}s)",
                  flush=True)
    return rows

def bulk_sample(s, p, dom, n, seed=1):
    r, k = s // 2, s // 2 - 1
    rng = np.random.default_rng(seed)
    hist = {}
    for i in range(n):
        w = rng.integers(0, p, size=s).tolist()
        L = decode_L(p, dom, w, k, r)
        hist[L] = hist.get(L, 0) + 1
        if (i + 1) % 500 == 0:
            print(f"  bulk {i+1}/{n}", flush=True)
    return {"s": s, "p": p, "n": n, "L_hist": dict(sorted(hist.items()))}


# ---------------------------------------------------------------------------

def subgroup(p, s):
    try:
        import vanish
        return list(vanish.subgroup(p, s))
    except ImportError:
        for g in range(2, p):
            e = pow(g, (p - 1) // s, p)
            os_ = {pow(e, i, p) for i in range(s)}
            if len(os_) == s:
                return [pow(e, i, p) for i in range(s)]
        raise ValueError("no order-s subgroup")

def validate():
    ok = True
    dom = subgroup(65537, 16)
    c = cloud_census(16, 65537, dom, progress=False)
    for lbl, got, want in (("top class", c["top_class_max"], 810),
                           ("e1 zero", c["zero_fiber"][1], 70),
                           ("e4 zero", c["zero_fiber"][4], 266)):
        good = got == want
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] s=16 {lbl}: {got} (want {want})")
    for j, want in ((1, 70), (4, 266)):
        w = [pow(int(x), 7 + j, 65537) for x in dom]
        L = decode_L(65537, dom, w, 7, 8)
        good = L == want
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] s=16 decode x^{7+j}: L={L} "
              f"(want {want})")
    dom12 = subgroup(30013, 12)
    c12 = cloud_census(12, 30013, dom12, progress=False)
    good = c12["zero_fiber"][1] == 24
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] s=12 e1 zero (Lam-Leung): "
          f"{c12['zero_fiber'][1]} (want 24)")
    print("ALL PASS" if ok else "FAILURES PRESENT")
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--cloud", action="store_true")
    ap.add_argument("--zoo", action="store_true")
    ap.add_argument("--bulk", action="store_true")
    ap.add_argument("--s", type=int, default=32)
    ap.add_argument("--p", type=int, default=2130706433)
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    if a.validate:
        raise SystemExit(0 if validate() else 1)
    dom = subgroup(a.p, a.s)
    out = {}
    if a.cloud:
        out["cloud"] = cloud_census(a.s, a.p, dom)
    if a.zoo:
        out["zoo"] = zoo_decodes(a.s, a.p, dom, depth=a.depth)
    if a.bulk:
        out["bulk"] = bulk_sample(a.s, a.p, dom, a.n)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")
    else:
        print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
