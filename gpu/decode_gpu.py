"""GPU exact list decoder (CuPy raw kernel) — the s=32 discovery engine.

Same math as vanish::rs::decode (validated in gpu/decode_ref.py against the CPU
decoder), GPU-shaped: one thread per information set. Each thread unranks its
C(n,k)-combination, interpolates the degree-<k polynomial through those k
(point, word-value) pairs, evaluates it on all n domain points, counts agreement
with the word, and — if agreement >= t — atomically emits the codeword's ID (its
values at the first k domain points). Host deduplicates the IDs (cp.unique) to
the list.

The interpolation uses a precomputed inverse-difference table and the identity
den(X)*prod_m(X - x_m) = 1, so there is NO modular inverse in the per-thread
eval loop: pure u32 mulmod with u64 accumulators (primes < 2^31; KoalaBear /
BabyBear / 65537 all qualify).

Correctness gate (run FIRST on the pod): `python decode_gpu.py --validate`
reproduces the exhaustively-known s=16 lists (bucket x^8 -> 70) and cross-checks
against vanish.list_decode. Do not trust s=32 output from a pod that has not
passed this gate.

Usage:
  python decode_gpu.py --validate
  python decode_gpu.py --p 2130706433 --s 32 --k 16 --t 17 --word-file w.npy
Requires: cupy-cuda12x, vanish wheel (subgroup/list_decode for validation), CUDA GPU.
"""
import argparse
import math
import numpy as np
import cupy as cp
import vanish

# Barrett modular arithmetic — the kernels' hot path.
#
# The reference kernels below use `% p` on 64-bit values. NVIDIA hardware has no
# integer-division instruction: each `%` expands to a ~50-cycle sequence, and a
# thread runs ~1800 of them (k*(k-1) for the weights + n*3k in the eval loop),
# so at s=32/k=16 (601M threads) the modulos ARE the runtime.
#
# Barrett: for a, b < p < 2^31 let t = a*b < 2^62 and minv = floor(2^64/p).
# With q = umul64hi(t, minv) = floor(t*minv/2^64), writing minv = (2^64 - e)/p
# with 0 <= e < p gives t*minv/2^64 = t/p - t*e/(p*2^64) and
# t*e/(p*2^64) < 2^62*p/(p*2^64) = 1/4, hence q in {floor(t/p)-1, floor(t/p)}
# and r = t - q*p lies in [0, 2p): ONE conditional subtract is provably enough.
# (Bound verified exhaustively in Python over all our primes; the observed worst
# quotient deficit is exactly 1.) Cost: 2 multiplies + a select, ~10 cycles.
#
# addmod/submod are safe in u32 because p < 2^31 => a+b < 2^32.
_MODMATH = r"""
__device__ __forceinline__ unsigned int mulmod(
    unsigned int a, unsigned int b, unsigned int p, unsigned long long minv)
{
    unsigned long long t = (unsigned long long)a * b;
    unsigned long long q = __umul64hi(t, minv);
    unsigned long long r = t - q * (unsigned long long)p;
    return (unsigned int)(r >= p ? r - p : r);
}
__device__ __forceinline__ unsigned int addmod(
    unsigned int a, unsigned int b, unsigned int p)
{
    unsigned int s = a + b;
    return s >= p ? s - p : s;
}
__device__ __forceinline__ unsigned int submod(
    unsigned int a, unsigned int b, unsigned int p)
{
    return a >= b ? a - b : a + p - b;
}
"""

KERNEL = r"""
extern "C" __global__ void list_decode(
    const int n, const int k, const int t,
    const unsigned int p,
    const unsigned int* dom,          // [n] domain points
    const unsigned int* word,         // [n] received word
    const unsigned int* inv_diff,     // [n*n] inv(dom[i]-dom[j]) mod p (diag unused)
    const unsigned long long* binom,  // [(n+1)*(k+1)] binomial table for unranking
    const long long total,            // C(n,k)
    const long long base,             // thread 0 handles info-set index `base`
    unsigned int* out_ids,            // [cap*k] emitted codeword IDs
    int* out_count,                   // global atomic counter
    const int cap)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x + base;
    if (tid >= total) return;

    int I[32];
    {   // unrank tid -> the k ascending indices (combinatorial number system)
        long long rem = tid; int x = 0;
        for (int pos = 0; pos < k; pos++) {
            while (1) {
                unsigned long long cnt = binom[(long long)(n - x - 1) * (k + 1) + (k - pos - 1)];
                if ((unsigned long long)rem < cnt) { I[pos] = x; x++; break; }
                rem -= cnt; x++;
            }
        }
    }

    unsigned int wt[32];
    for (int m = 0; m < k; m++) {
        unsigned long long acc = 1;
        for (int l = 0; l < k; l++)
            if (l != m) acc = acc * inv_diff[(long long)I[m] * n + I[l]] % p;
        wt[m] = (unsigned int)acc;
    }

    unsigned int idvals[32];
    int agree = 0;
    for (int j = 0; j < n; j++) {
        int inI = 0;
        for (int m = 0; m < k; m++) if (I[m] == j) { inI = 1; break; }
        unsigned int cj;
        if (inI) {
            cj = word[j];
        } else {
            unsigned long long num = 0, invden = 1;
            for (int m = 0; m < k; m++) {
                num = (num + (unsigned long long)wt[m] * word[I[m]] % p
                             * inv_diff[(long long)j * n + I[m]]) % p;
                unsigned int diff = (dom[j] + p - dom[I[m]]) % p;
                invden = invden * diff % p;
            }
            cj = (unsigned int)(num * invden % p);
        }
        if (cj == word[j]) agree++;
        if (j < k) idvals[j] = cj;
    }

    if (agree >= t) {
        int slot = atomicAdd(out_count, 1);
        if (slot < cap)
            for (int j = 0; j < k; j++) out_ids[(long long)slot * k + j] = idvals[j];
    }
}
"""

_MOD = cp.RawKernel(KERNEL, "list_decode")

# Barrett-reduced twins of the two kernels. Same algorithm, same emissions —
# only the modular arithmetic changes. `validate()` A/B-checks them against the
# reference kernels AND the CPU oracle before any result is believed.
KERNEL_FAST = _MODMATH + r"""
extern "C" __global__ void list_decode_fast(
    const int n, const int k, const int t,
    const unsigned int p, const unsigned long long minv,
    const unsigned int* dom, const unsigned int* word,
    const unsigned int* inv_diff, const unsigned long long* binom,
    const long long total, const long long base,
    unsigned int* out_ids, int* out_count, const int cap)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x + base;
    if (tid >= total) return;

    int I[32];
    {
        long long rem = tid; int x = 0;
        for (int pos = 0; pos < k; pos++) {
            while (1) {
                unsigned long long cnt = binom[(long long)(n - x - 1) * (k + 1) + (k - pos - 1)];
                if ((unsigned long long)rem < cnt) { I[pos] = x; x++; break; }
                rem -= cnt; x++;
            }
        }
    }

    unsigned int wt[32];
    for (int m = 0; m < k; m++) {
        unsigned int acc = 1;
        for (int l = 0; l < k; l++)
            if (l != m) acc = mulmod(acc, inv_diff[(long long)I[m] * n + I[l]], p, minv);
        wt[m] = acc;
    }

    unsigned int idvals[32];
    int agree = 0;
    for (int j = 0; j < n; j++) {
        int inI = 0;
        for (int m = 0; m < k; m++) if (I[m] == j) { inI = 1; break; }
        unsigned int cj;
        if (inI) {
            cj = word[j];
        } else {
            unsigned int num = 0, invden = 1;
            for (int m = 0; m < k; m++) {
                unsigned int tmp = mulmod(wt[m], word[I[m]], p, minv);
                tmp = mulmod(tmp, inv_diff[(long long)j * n + I[m]], p, minv);
                num = addmod(num, tmp, p);
                invden = mulmod(invden, submod(dom[j], dom[I[m]], p), p, minv);
            }
            cj = mulmod(num, invden, p, minv);
        }
        if (cj == word[j]) agree++;
        if (j < k) idvals[j] = cj;
    }

    if (agree >= t) {
        int slot = atomicAdd(out_count, 1);
        if (slot < cap)
            for (int j = 0; j < k; j++) out_ids[(long long)slot * k + j] = idvals[j];
    }
}
"""

_MOD_FAST = cp.RawKernel(KERNEL_FAST, "list_decode_fast")


def barrett_minv(p):
    """floor(2^64 / p) — the Barrett constant (host side)."""
    return np.uint64((1 << 64) // int(p))

# Count-mode kernel: dedup ON DEVICE via the lex-first rule — a thread counts
# its codeword only if its information set I is the lexicographically first
# k-subset of the codeword's agreement set. Each distinct codeword is counted
# EXACTLY once (no emission buffer, no cap on the count), so lists far above
# the emit cap (e.g. the predicted 17.7M multiplicative class at s=32, hit by
# C(17,16)=17 threads each) are counted exactly. A hash-sampled subset of
# codeword IDs is emitted for host-side structure checks (frozen product).
KERNEL_COUNT = _MODMATH + r"""
extern "C" __global__ void list_count(
    const int n, const int k, const int t,
    const unsigned int p, const unsigned long long minv,
    const unsigned int* dom,
    const unsigned int* word,
    const unsigned int* inv_diff,
    const unsigned long long* binom,
    const long long total,
    const long long base,
    const unsigned int sample_mask,   // emit if (fnv(idvals) & mask) == 0
    unsigned int* out_ids,            // [scap*k] sampled codeword IDs
    int* out_counts,                  // [0]=distinct count, [1]=emitted
    const int scap)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x + base;
    if (tid >= total) return;

    int I[32];
    {
        long long rem = tid; int x = 0;
        for (int pos = 0; pos < k; pos++) {
            while (1) {
                unsigned long long cnt = binom[(long long)(n - x - 1) * (k + 1) + (k - pos - 1)];
                if ((unsigned long long)rem < cnt) { I[pos] = x; x++; break; }
                rem -= cnt; x++;
            }
        }
    }

    unsigned int wt[32];
    for (int m = 0; m < k; m++) {
        unsigned int acc = 1;
        for (int l = 0; l < k; l++)
            if (l != m) acc = mulmod(acc, inv_diff[(long long)I[m] * n + I[l]], p, minv);
        wt[m] = acc;
    }

    unsigned int idvals[32];
    int agree = 0, lexok = 1;
    for (int j = 0; j < n; j++) {
        int inI = 0;
        for (int m = 0; m < k; m++) if (I[m] == j) { inI = 1; break; }
        unsigned int cj;
        if (inI) {
            cj = word[j];
        } else {
            unsigned int num = 0, invden = 1;
            for (int m = 0; m < k; m++) {
                unsigned int tmp = mulmod(wt[m], word[I[m]], p, minv);
                tmp = mulmod(tmp, inv_diff[(long long)j * n + I[m]], p, minv);
                num = addmod(num, tmp, p);
                invden = mulmod(invden, submod(dom[j], dom[I[m]], p), p, minv);
            }
            cj = mulmod(num, invden, p, minv);
        }
        if (cj == word[j]) {
            if (agree < k && I[agree] != j) lexok = 0;
            agree++;
        }
        if (j < k) idvals[j] = cj;
    }

    if (agree >= t && lexok) {
        atomicAdd(&out_counts[0], 1);
        unsigned int h = 2166136261u;
        for (int m = 0; m < k; m++) h = (h ^ idvals[m]) * 16777619u;
        if ((h & sample_mask) == 0u) {
            int slot = atomicAdd(&out_counts[1], 1);
            if (slot < scap)
                for (int m = 0; m < k; m++) out_ids[(long long)slot * k + m] = idvals[m];
        }
    }
}
"""

_MOD_COUNT = cp.RawKernel(KERNEL_COUNT, "list_count")


def _tables(p, dom, k):
    n = len(dom)
    inv = np.zeros((n, n), dtype=np.uint32)
    for i in range(n):
        for j in range(n):
            if i != j:
                inv[i, j] = pow(int((dom[i] - dom[j]) % p), p - 2, p)
    binom = np.zeros((n + 1, k + 1), dtype=np.uint64)
    for a in range(n + 1):
        for b in range(min(a, k) + 1):
            binom[a, b] = math.comb(a, b)
    return (cp.asarray(inv.ravel(), dtype=cp.uint32),
            cp.asarray(binom.ravel(), dtype=cp.uint64))


def gpu_list_count(p, dom, word, k, t, sample_mask=0xFF, scap=1 << 20,
                   tile=1 << 26):
    """Exact DISTINCT-codeword count at agreement >= t via lex-first on-device
    dedup (no cap on the count), plus a hash-sampled batch of codeword IDs
    (values at the first k domain points) for structure checks.
    Returns (count, sampled_ids ndarray[m, k], sample_overflowed)."""
    n = len(dom)
    total = math.comb(n, k)
    dom_d = cp.asarray(dom, dtype=cp.uint32)
    word_d = cp.asarray(word, dtype=cp.uint32)
    inv_d, binom_d = _tables(p, dom, k)
    minv = barrett_minv(p)
    out_ids = cp.zeros(scap * k, dtype=cp.uint32)
    out_counts = cp.zeros(2, dtype=cp.int32)
    threads = 256
    for base in range(0, total, tile):
        span = min(tile, total - base)
        blocks = (span + threads - 1) // threads
        _MOD_COUNT((blocks,), (threads,),
                   (np.int32(n), np.int32(k), np.int32(t), np.uint32(p), minv,
                    dom_d, word_d, inv_d, binom_d,
                    np.int64(total), np.int64(base), np.uint32(sample_mask),
                    out_ids, out_counts, np.int32(scap)))
    cp.cuda.Stream.null.synchronize()
    count, emitted = int(out_counts[0]), int(out_counts[1])
    m = min(emitted, scap)
    ids = cp.asnumpy(out_ids[: m * k].reshape(m, k)) if m else np.zeros((0, k), dtype=np.uint32)
    return count, ids, emitted > scap


def _modinv(a, p):
    return pow(int(a) % p, p - 2, p)


def _interp_full(xs, ys, dom, p):
    """Expand a codeword ID (values at k points) to its full eval vector."""
    k = len(xs)
    w = []
    for j in range(k):
        d = 1
        for m in range(k):
            if m != j:
                d = d * ((xs[j] - xs[m]) % p) % p
        w.append(_modinv(d, p))
    out = []
    for x in dom:
        if x in xs:
            out.append(ys[xs.index(x)])
            continue
        num = den = 0
        for j in range(k):
            tt = w[j] * _modinv((x - xs[j]) % p, p) % p
            num = (num + tt * ys[j]) % p
            den = (den + tt) % p
        out.append(num * _modinv(den, p) % p)
    return out


def gpu_list_size(p, dom, word, k, t, cap=1 << 26, tile=1 << 26, fast=True):
    """Exact list size of RS[F_p, dom, k] at agreement >= t. Returns
    (list_size, overflowed). `fast=False` selects the reference (`% p`) kernel —
    used by validate() to A/B-check the Barrett kernel."""
    n = len(dom)
    total = math.comb(n, k)
    dom_d = cp.asarray(dom, dtype=cp.uint32)
    word_d = cp.asarray(word, dtype=cp.uint32)
    inv_d, binom_d = _tables(p, dom, k)
    minv = barrett_minv(p)

    out_ids = cp.zeros(cap * k, dtype=cp.uint32)
    out_count = cp.zeros(1, dtype=cp.int32)
    threads = 256
    for base in range(0, total, tile):
        span = min(tile, total - base)
        blocks = (span + threads - 1) // threads
        if fast:
            _MOD_FAST((blocks,), (threads,),
                      (np.int32(n), np.int32(k), np.int32(t), np.uint32(p),
                       minv, dom_d, word_d, inv_d, binom_d,
                       np.int64(total), np.int64(base),
                       out_ids, out_count, np.int32(cap)))
        else:
            _MOD((blocks,), (threads,),
                 (np.int32(n), np.int32(k), np.int32(t), np.uint32(p),
                  dom_d, word_d, inv_d, binom_d, np.int64(total), np.int64(base),
                  out_ids, out_count, np.int32(cap)))
    cp.cuda.Stream.null.synchronize()
    hits = int(out_count[0])
    overflow = hits > cap
    m = min(hits, cap)
    ids = out_ids[: m * k].reshape(m, k)
    distinct = cp.unique(ids, axis=0).shape[0] if m else 0
    return int(distinct), overflow


def validate():
    p, s, k, t = 65537, 16, 7, 8
    dom = list(vanish.subgroup(p, s))
    ok = True
    for name, w in [("bucket x^8", [pow(x, 8, p) for x in dom]),
                    ("shifted", [(pow(x, 8, p) + 3 * x) % p for x in dom])]:
        gpu, ov = gpu_list_size(p, dom, w, k, t)              # Barrett kernel
        ref, ovr = gpu_list_size(p, dom, w, k, t, fast=False)  # reference kernel
        cpu = len(vanish.list_decode(p, dom, k, w, t))
        good = (gpu == ref == cpu) and not ov and not ovr
        ok &= good
        print(f"  {name:<12} fast={gpu} ref={ref} cpu={cpu} overflow={ov}  "
              f"{'PASS' if good else 'FAIL'}")

    # multiplicative-word gate: pre-registered answer for the rate-1/2 q=1
    # cell at s=16 (the exact scale-half of the s=32 target). Requires
    # `python construct_word.py --s 16 --p 65537` to have been run (same dir).
    import json as _json
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    wf, mf = os.path.join(here, "w16_mult.npy"), os.path.join(here, "w16_mult.json")
    if os.path.exists(wf) and os.path.exists(mf):
        meta = _json.load(open(mf))
        km, tm, pred, cstar = meta["k"], meta["t"], meta["predicted_count"], meta["cstar"]
        w = [int(v) for v in np.load(wf)]
        cpu = len(vanish.list_decode(meta["p"], dom, km, w, tm))
        gpu, ov = gpu_list_size(meta["p"], dom, w, km, tm)
        ref, _ = gpu_list_size(meta["p"], dom, w, km, tm, fast=False)
        cnt, ids, sov = gpu_list_count(meta["p"], dom, w, km, tm, sample_mask=0)
        assert gpu == ref, f"Barrett/reference kernel disagree: {gpu} vs {ref}"
        frozen = True
        for row in ids:
            c = _interp_full(dom[:km], [int(v) for v in row], dom, meta["p"])
            A = [i for i in range(s) if c[i] == w[i]]
            frozen &= (sum(A) % s == cstar) and (len(A) == tm)
        good = (cpu == gpu == cnt == pred == len(ids)) and frozen and not ov and not sov
        ok &= good
        print(f"  mult-word    cpu={cpu} gpu={gpu} count={cnt} predicted={pred} "
              f"sampled={len(ids)} frozen={frozen}  {'PASS' if good else 'FAIL'}")
    else:
        ok = False
        print("  mult-word    MISSING w16_mult.npy/.json — run construct_word.py first  FAIL")

    print("VALIDATE:", "PASS" if ok else "FAIL")
    return ok


def bench(p=2130706433, s=32, k=16, t=17):
    """Time one full decode at the real workload size, Barrett vs reference.
    A/B-checks the counts too — a speedup that changes the answer is a bug."""
    import time
    dom = list(vanish.subgroup(p, s))
    rng = np.random.default_rng(0)
    w = [int(v) for v in rng.integers(0, p, s)]
    print(f"bench: n={s} k={k} t={t} p={p}  C(n,k)={math.comb(s, k):,} threads")
    out = {}
    for tag, fastflag in [("reference (% p)", False), ("Barrett", True)]:
        gpu_list_size(p, dom, w, k, t, fast=fastflag)      # warm up / JIT
        t0 = time.time()
        L, ov = gpu_list_size(p, dom, w, k, t, fast=fastflag)
        el = time.time() - t0
        out[fastflag] = (L, el)
        print(f"  {tag:<16} {el:7.2f}s   L={L}")
    assert out[True][0] == out[False][0], "KERNELS DISAGREE — do not use fast"
    print(f"  speedup: {out[False][1] / out[True][1]:.1f}x   (counts agree: "
          f"{out[True][0]})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--bench", action="store_true",
                    help="time Barrett vs reference kernel at s=32 k=16")
    ap.add_argument("--p", type=int)
    ap.add_argument("--s", type=int)
    ap.add_argument("--k", type=int)
    ap.add_argument("--t", type=int)
    ap.add_argument("--word-file", help="npy file: uint32 word of length s")
    ap.add_argument("--count", action="store_true",
                    help="count mode: exact distinct count via lex-first "
                         "dedup (no emit cap) + hash-sampled member IDs")
    ap.add_argument("--sample-mask", type=lambda x: int(x, 0), default=0xFF,
                    help="emit codeword IDs where (fnv & mask)==0 (count mode)")
    ap.add_argument("--out-ids", help="npy path for sampled IDs (count mode)")
    a = ap.parse_args()
    if a.validate:
        raise SystemExit(0 if validate() else 1)
    if a.bench:
        bench()
        raise SystemExit(0)
    dom = list(vanish.subgroup(a.p, a.s))
    word = list(np.load(a.word_file).astype(np.uint32))
    if a.count:
        cnt, ids, sov = gpu_list_count(a.p, dom, word, a.k, a.t,
                                       sample_mask=a.sample_mask)
        print(f"distinct list size = {cnt}  sampled ids = {len(ids)}"
              + ("  [SAMPLE OVERFLOW: raise scap or mask]" if sov else ""))
        if a.out_ids:
            np.save(a.out_ids, ids)
            print(f"wrote {a.out_ids}")
    else:
        size, ov = gpu_list_size(a.p, dom, word, a.k, a.t)
        print(f"list size = {size}" + ("  [OVERFLOW: raise cap]" if ov else ""))
