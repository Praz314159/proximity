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
import argparse, faulthandler, math, signal, time
import numpy as np
import cupy as cp
import vanish
from decode_gpu import _mod_tmpl, barrett_minv  # validated templated kernel


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


# ----- batch interpolation (host) -----------------------------------------
_CTX = {}


def _ctx(p, dom, k, cap):
    """Per-(p,dom,k) constants, built once: device tables, the emit buffer, and
    the batch-interpolation matrices. The interpolation NODES are always
    dom[:k] (the kernel emits values there), so the barycentric weights and the
    tt matrix are the same for every codeword of every pool — hoist them."""
    key = (p, id(dom), k)
    if key in _CTX:
        return _CTX[key]
    n = len(dom)
    inv = np.zeros((n, n), dtype=np.uint32)
    for i in range(n):
        for j in range(n):
            if i != j:
                inv[i, j] = modinv((dom[i] - dom[j]) % p, p)
    binom = np.zeros((n + 1, k + 1), dtype=np.uint64)
    for a in range(n + 1):
        for b in range(min(a, k) + 1):
            binom[a, b] = math.comb(a, b)
    xs, rest = dom[:k], dom[k:]
    wj = []
    for j in range(k):
        d = 1
        for m in range(k):
            if m != j:
                d = d * ((xs[j] - xs[m]) % p) % p
        wj.append(modinv(d, p))
    TT = np.zeros((len(rest), k), dtype=np.int64)
    inv_den = np.zeros(len(rest), dtype=np.int64)
    for i, x in enumerate(rest):
        den = 0
        for j in range(k):
            t_ = wj[j] * modinv((x - xs[j]) % p, p) % p
            TT[i, j] = t_
            den = (den + t_) % p
        inv_den[i] = modinv(den, p)
    c = dict(dom_d=cp.asarray(dom, dtype=cp.uint32),
             inv_d=cp.asarray(inv.ravel(), dtype=cp.uint32),
             binom_d=cp.asarray(binom.ravel(), dtype=cp.uint64),
             out_ids=cp.empty(cap * k, dtype=cp.uint32),
             out_count=cp.empty(1, dtype=cp.int32),
             TT=TT, inv_den=inv_den, n=n, total=math.comb(n, k))
    _CTX[key] = c
    return c


def interp_batch(ids, dom, p, k, ctx):
    """Expand emitted codeword IDs (values at dom[:k]) to full eval vectors —
    one matmul mod p for the whole pool instead of a Lagrange loop per
    codeword (validated exactly against interp_full; ~100x). The 16-bit split
    keeps the int64 accumulators from overflowing at p < 2^31."""
    Y = np.asarray(ids, dtype=np.int64)
    if len(Y) == 0:
        return np.zeros((0, ctx["n"]), dtype=np.int64)
    TT, inv_den = ctx["TT"], ctx["inv_den"]
    hi, lo = Y >> 16, Y & 0xFFFF
    num = ((hi @ TT.T % p) * 65536 + (lo @ TT.T % p)) % p
    out = np.empty((len(Y), ctx["n"]), dtype=np.int64)
    out[:, :k] = Y
    out[:, k:] = num * inv_den[None, :] % p
    return out


# ----- GPU decode returning the codewords (not just size) -----------------
def gpu_decode(p, dom, word, k, t, cap=1 << 27, tile=1 << 26):
    """Distinct codewords (as full eval vectors, (L, n) int64) of
    RS[F_p, dom, k] at agreement >= t."""
    c = _ctx(p, dom, k, cap)
    word_d = cp.asarray(word, dtype=cp.uint32)
    c["out_count"].fill(0)
    threads = 256
    kern = _mod_tmpl(c["n"], k)
    minv = barrett_minv(p)
    for base in range(0, c["total"], tile):
        span = min(tile, c["total"] - base)
        blocks = (span + threads - 1) // threads
        kern((blocks,), (threads,),
             (np.int32(t), np.uint32(p), minv,
              c["dom_d"], word_d, c["inv_d"], c["binom_d"],
              np.int64(c["total"]), np.int64(base),
              c["out_ids"], c["out_count"], np.int32(cap)))
    cp.cuda.Stream.null.synchronize()
    hits = int(c["out_count"][0])
    if hits > cap:
        raise OverflowError(f"decode overflow: {hits} hits > cap {cap}")
    if not hits:
        return np.zeros((0, c["n"]), dtype=np.int64)
    # host dedup: cp.unique(axis=0) stalls at structured-word hit counts
    # (see decode_gpu module docs)
    ids = np.unique(
        cp.asnumpy(c["out_ids"][: hits * k]).reshape(hits, k), axis=0)
    return interp_batch(ids, dom, p, k, c)


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
    step; mirrors vanish rs::cluster::optimize.

    Flip scoring is exact O(n*L) via the boundary-count identity (validated in
    experiments/landscape/descent_verify.py against direct decodes):
      nl(x,v) = |ag>=t+1| + |ag==t| - |ag==t & P[:,x]==w[x]|
                + |ag==t-1 & P[:,x]==v|
    with the best v per coordinate = mode of P[ag==t-1, x].

    traj records per-step L2 slice statistics (survivors/recruits of the
    accepted flip; exact_t = members at agreement exactly t) so the s=32 run
    measures the growth law for free: growth = L_next/L, compare to n/(n-t)
    [measured law at s=16] vs 1 + t/n [naive shell]."""
    n = len(dom); w = list(seed); relaxed = max(t - 1, k)
    traj = []
    for step in range(max_flips):
        t_step = time.time()
        P = gpu_decode(p, dom, w, k, relaxed)   # (L, n) int64
        wv = np.array(w, dtype=np.int64)
        ag = (P == wv[None, :]).sum(axis=1)
        cur = int((ag >= t).sum())
        exact_t = int((ag == t).sum())
        best = None
        hi = int((ag >= t + 1).sum()) + exact_t
        rows_t, rows_b = P[ag == t], P[ag == t - 1]
        for x in range(n):
            base = hi - int((rows_t[:, x] == wv[x]).sum())
            if len(rows_b) == 0:
                continue
            col = rows_b[:, x]
            col = col[col != wv[x]]
            if len(col) == 0:
                continue
            vals, counts = np.unique(col, return_counts=True)
            j = int(counts.argmax())
            nl = base + int(counts[j])
            if best is None or nl > best[2]:
                best = (x, int(vals[j]), nl, base, int(counts[j]))
        if best is not None and best[2] > cur:
            traj.append(dict(L=cur, exact_t=exact_t,
                             survivors=best[3], recruits=best[4]))
            w[best[0]] = best[1]
            print(f"      [step {step}] L={cur} -> {best[2]} "
                  f"pool={len(P)} ({time.time()-t_step:.1f}s)", flush=True)
        else:
            traj.append(dict(L=cur, exact_t=exact_t))
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
    growths, exact_fr = [], []
    for sd in range(nseed):
        t0 = time.time()
        seed = pencil_seed(p, dom, k, petals, np.random.default_rng(sd))
        try:
            w, m, traj = optimize_gpu(p, dom, k, t, seed, rng)
        except OverflowError as e:
            print(f"  [k{k}t{t} seed {sd:>2}] ABORTED near-code attractor "
                  f"({e})", flush=True)
            continue
        for a, b in zip(traj, traj[1:]):        # L2 growth law, late phase
            if a["L"] >= 20:
                growths.append(b["L"] / a["L"])
                exact_fr.append(a["exact_t"] / a["L"])
        if best is None or len(m) > best[1]:
            best = (w, len(m), m)
        print(f"  [k{k}t{t} seed {sd:>2}] L={len(m):<8} steps={len(traj):<3} "
              f"best={best[1]:<8} {time.time() - t0:6.1f}s", flush=True)
    maxL = best[1]
    # persist the best word: replays should never be needed again
    import json as _json
    _json.dump(
        {"p": p, "s": s, "k": k, "t": t, "L": maxL,
         "w": [int(x) for x in best[0]]},
        open(f"best_k{k}t{t}_L{maxL}.json", "w"))
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
    g = np.array(growths) if growths else np.array([0.0])
    return dict(s=s, rho=round(rho, 3), delta=round(delta, 3), q=t - k, maxL=maxL,
                bucket=bucket, list_exp=round(lx, 4), bucket_exp=round(bx, 4),
                entropy_exp=round(ex, 4), e1H_max=round(e1H, 2),
                growth_med=round(float(np.median(g)), 3),
                growth_law=round(s / (s - t), 3),
                exact_t_fr=round(float(np.mean(exact_fr)), 4) if exact_fr else None)


def _install_faulthandler():
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1)


if __name__ == "__main__":
    _install_faulthandler()
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, required=True)
    ap.add_argument("--s", type=int, required=True)
    ap.add_argument("--cells", required=True, help="k:t,k:t,...")
    ap.add_argument("--nseed", type=int, default=16)
    a = ap.parse_args()
    print(f"{'s':>3}{'rho':>6}{'delta':>7}{'q':>3}{'maxL':>8}{'bucket':>8}"
          f"{'list_exp':>10}{'bucket_exp':>11}{'entropy_exp':>12}{'e1H_max':>9}"
          f"{'growth':>8}{'n/(n-t)':>9}{'exact_t':>9}")
    for cell in a.cells.split(","):
        k, t = map(int, cell.split(":"))
        r = run_cell(a.p, a.s, k, t, a.nseed)
        fmt = lambda v, w: f"{v:>{w}}" if v is not None else " " * (w - 2) + "--"
        print(f"{r['s']:>3}{r['rho']:>6}{r['delta']:>7}{r['q']:>3}{r['maxL']:>8}"
              f"{r['bucket']:>8}{r['list_exp']:>10}{r['bucket_exp']:>11}"
              f"{r['entropy_exp']:>12}{r['e1H_max']:>9}"
              + fmt(r['growth_med'], 8) + fmt(r['growth_law'], 9)
              + fmt(r['exact_t_fr'], 9))
