"""s=32 concentration experiment — the pod driver that turns the GPU decoder
into a max-vs-mean / bucket-vs-entropy measurement.

WHAT IT TESTS
  The char-p entropy defense = "the list-size distribution CONCENTRATES": the
  MAX list tracks the entropy (average) count, so it crosses the threshold at
  the same radius (0.468) the average does. We cannot see the 2^lambdaF
  threshold at small s, so we measure the *exponent* instead:

      list_exponent(s, delta) := log_q(maxList) / n          (n = s, q = p)

  and compare it, per delta, to
      entropy_exponent(delta)  := rho + H_q(delta) - 1        (the average / count)
      bucket_exponent(s,delta) := log_q(m_struct) / n         (the structured max)

  SIGNATURES (what would confirm the entropy picture):
    * list_exponent is ~s-INDEPENDENT (overlays across s=16 and s=32)  -> entropy scaling
    * list_exponent > bucket_exponent, and the gap GROWS with s        -> non-bucket wins
    * list_exponent approaches entropy_exponent as n grows             -> concentration
    * the max word stays STRUCTURELESS at s=32 (structure probe)       -> hypothesis (b)

HOW
  The GPU kernel (decode_gpu) decodes ONE word over C(n,k) info-sets in seconds
  on an A100. The optimizer is a Python greedy hill-climb to convergence that
  calls the GPU decoder for its pool each step (mirrors vanish rs::cluster
  optimize). Per cell: N random-pencil seeds -> max over converged local maxima.

  Run AFTER `python decode_gpu.py --validate` passes.

  python concentration.py --p 2130706433 --s 32 --cells "16:17,16:18,16:19"
       (cells = k:t,...  ; e.g. rate 1/2 at several radii)

Requires: cupy-cuda12x, vanish wheel, CUDA GPU.
"""
import argparse, math, time
import numpy as np
import cupy as cp
import vanish
from decode_gpu import _MOD  # reuse the validated kernel


# ----- host helpers -------------------------------------------------------
def modinv(a, p):
    return pow(int(a) % p, p - 2, p)


def interp_full(xs, ys, dom, p):
    """Reconstruct a degree-<k codeword's full eval vector from k (point,value)
    pairs (used to expand emitted codeword IDs into full codewords)."""
    k = len(xs)
    w = []
    for j in range(k):
        d = 1
        for m in range(k):
            if m != j:
                d = d * ((xs[j] - xs[m]) % p) % p
        w.append(modinv(d, p))
    out = []
    for x in dom:
        if x in xs:
            out.append(ys[xs.index(x)]); continue
        num = den = 0
        for j in range(k):
            tt = w[j] * modinv((x - xs[j]) % p, p) % p
            num = (num + tt * ys[j]) % p; den = (den + tt) % p
        out.append(num * modinv(den, p) % p)
    return out


# ----- GPU decode returning the codewords (not just size) -----------------
def gpu_decode(p, dom, word, k, t, cap=1 << 26, tile=1 << 26):
    """Distinct codewords (as full eval vectors) of RS[F_p, dom, k] at agreement
    >= t. Reuses the kernel; expands emitted IDs (values at first k pts) to full
    codewords on host."""
    n = len(dom)
    total = math.comb(n, k)
    dom_d = cp.asarray(dom, dtype=cp.uint32)
    word_d = cp.asarray(word, dtype=cp.uint32)
    inv = np.zeros((n, n), dtype=np.uint32)
    for i in range(n):
        for j in range(n):
            if i != j:
                inv[i, j] = modinv((dom[i] - dom[j]) % p, p)
    inv_d = cp.asarray(inv.ravel(), dtype=cp.uint32)
    binom = np.zeros((n + 1, k + 1), dtype=np.uint64)
    for a in range(n + 1):
        for b in range(min(a, k) + 1):
            binom[a, b] = math.comb(a, b)
    binom_d = cp.asarray(binom.ravel(), dtype=cp.uint64)
    out_ids = cp.zeros(cap * k, dtype=cp.uint32)
    out_count = cp.zeros(1, dtype=cp.int32)
    threads = 256
    for base in range(0, total, tile):
        span = min(tile, total - base)
        blocks = (span + threads - 1) // threads
        _MOD((blocks,), (threads,),
             (np.int32(n), np.int32(k), np.int32(t), np.uint32(p),
              dom_d, word_d, inv_d, binom_d, np.int64(total), np.int64(base),
              out_ids, out_count, np.int32(cap)))
    cp.cuda.Stream.null.synchronize()
    hits = int(out_count[0])
    assert hits <= cap, f"OVERFLOW: {hits} > cap {cap}; raise cap"
    ids = cp.asnumpy(out_ids[: hits * k].reshape(hits, k)) if hits else np.zeros((0, k))
    ids = np.unique(ids, axis=0)
    xs = dom[:k]
    return [interp_full(xs, list(map(int, row)), dom, p) for row in ids]


# ----- Python greedy optimizer to convergence (GPU pool per step) ---------
def pencil_seed(p, dom, k, petals, rng):
    n = len(dom)
    idx = list(rng.permutation(n))
    core, pt = idx[:k - 1], idx[k - 1]
    cv = [int(rng.integers(0, p)) for _ in range(k - 1)]
    xs = [dom[i] for i in core] + [dom[pt]]
    pv = list({int(rng.integers(0, p)) for _ in range(petals * 3)})[:petals]
    cws = [interp_full(xs, cv + [y], dom, p) for y in pv]
    seed = [0] * n
    for i, c in enumerate(core):
        seed[c] = cv[i]
    fill = 0
    for z in range(n):
        if z in core:
            continue
        seed[z] = cws[fill % len(cws)][z]; fill += 1
    return seed


def optimize_gpu(p, dom, k, t, seed, rng, max_flips=200):
    """Greedy list-size climb to convergence, GPU-decoding the (t-1) pool each
    step; mirrors vanish rs::cluster::optimize."""
    n = len(dom); w = list(seed); relaxed = max(t - 1, k)
    traj = []
    for _ in range(max_flips):
        pool = gpu_decode(p, dom, w, k, relaxed)
        ag = [sum(1 for a, b in zip(c, w) if a == b) for c in pool]
        cur = sum(1 for a in ag if a >= t); traj.append(cur)
        cands = {(x, c[x]) for c in pool for x in range(n) if c[x] != w[x]}
        best = None
        for (x, v) in cands:
            nl = sum(1 for i, c in enumerate(pool)
                     if ag[i] + (c[x] == v) - (c[x] == w[x]) >= t)
            if best is None or nl > best[2]:
                best = (x, v, nl)
        if best and best[2] > cur:
            w[best[0]] = best[1]
        else:
            break
    members = gpu_decode(p, dom, w, k, t)
    return w, members, traj


# ----- entropy / bucket exponents -----------------------------------------
def h_q(delta, q):  # q-ary entropy (bits normalized: returns log_q units)
    if delta <= 0 or delta >= 1:
        return 0.0
    return (delta * math.log(q - 1) - delta * math.log(delta)
            - (1 - delta) * math.log(1 - delta)) / math.log(q)


def run_cell(p, s, k, t, nseed):
    dom = list(vanish.subgroup(p, s))
    rng = np.random.default_rng(0)
    petals = max(1, (s - k + 1) // (t - k + 1))
    best = None
    for sd in range(nseed):
        seed = pencil_seed(p, dom, k, petals, np.random.default_rng(sd))
        w, m, _ = optimize_gpu(p, dom, k, t, seed, rng)
        if best is None or len(m) > best[1]:
            best = (w, len(m), m)
    maxL = best[1]
    bucket = int(vanish.m_struct(s, t, t - k))
    rho, delta, q = k / s, 1 - t / s, p
    lx = math.log(max(maxL, 1), q) / s
    bx = math.log(max(bucket, 1), q) / s
    ex = rho + h_q(delta, q) - 1
    # structure of the max (e1 entropy + affine-rank proxy)
    center, members = best[0], best[2]
    e1 = [sum(dom[i] for i in range(s) if members[j][i] == center[i]) % p for j in range(maxL)]
    e1H = (-sum((c / maxL) * math.log2(c / maxL)
               for c in np.bincount([e % maxL for e in e1]) if c) if maxL else 0.0)
    return dict(s=s, rho=round(rho, 3), delta=round(delta, 3), q=t - k, maxL=maxL,
                bucket=bucket, list_exp=round(lx, 4), bucket_exp=round(bx, 4),
                entropy_exp=round(ex, 4), e1H_max=round(e1H, 2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, required=True)
    ap.add_argument("--s", type=int, required=True)
    ap.add_argument("--cells", required=True, help="k:t,k:t,...")
    ap.add_argument("--nseed", type=int, default=16)
    a = ap.parse_args()
    print(f"{'s':>3}{'rho':>6}{'delta':>7}{'q':>3}{'maxL':>8}{'bucket':>8}"
          f"{'list_exp':>10}{'bucket_exp':>11}{'entropy_exp':>12}{'e1H_max':>9}")
    for cell in a.cells.split(","):
        k, t = map(int, cell.split(":"))
        r = run_cell(a.p, a.s, k, t, a.nseed)
        print(f"{r['s']:>3}{r['rho']:>6}{r['delta']:>7}{r['q']:>3}{r['maxL']:>8}"
              f"{r['bucket']:>8}{r['list_exp']:>10}{r['bucket_exp']:>11}"
              f"{r['entropy_exp']:>12}{r['e1H_max']:>9}")
