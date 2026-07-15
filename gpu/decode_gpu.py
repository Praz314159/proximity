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


def gpu_list_size(p, dom, word, k, t, cap=1 << 26, tile=1 << 26):
    """Exact list size of RS[F_p, dom, k] at agreement >= t. Returns
    (list_size, overflowed)."""
    n = len(dom)
    total = math.comb(n, k)
    dom_d = cp.asarray(dom, dtype=cp.uint32)
    word_d = cp.asarray(word, dtype=cp.uint32)
    # inverse-difference table (host: Fermat via numpy pow is slow; use vanish? small n -> fine)
    inv = np.zeros((n, n), dtype=np.uint32)
    for i in range(n):
        for j in range(n):
            if i != j:
                inv[i, j] = pow(int((dom[i] - dom[j]) % p), p - 2, p)
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
        gpu, ov = gpu_list_size(p, dom, w, k, t)
        cpu = len(vanish.list_decode(p, dom, k, w, t))
        good = (gpu == cpu) and not ov
        ok &= good
        print(f"  {name:<12} gpu={gpu} cpu={cpu} overflow={ov}  {'PASS' if good else 'FAIL'}")
    print("VALIDATE:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--p", type=int)
    ap.add_argument("--s", type=int)
    ap.add_argument("--k", type=int)
    ap.add_argument("--t", type=int)
    ap.add_argument("--word-file", help="npy file: uint32 word of length s")
    a = ap.parse_args()
    if a.validate:
        raise SystemExit(0 if validate() else 1)
    dom = list(vanish.subgroup(a.p, a.s))
    word = list(np.load(a.word_file).astype(np.uint32))
    size, ov = gpu_list_size(a.p, dom, word, a.k, a.t)
    print(f"list size = {size}" + ("  [OVERFLOW: raise cap]" if ov else ""))
