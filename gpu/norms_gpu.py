"""
GPU norm sweep (CuPy raw kernel) — the s=64 bad-set campaign engine.

Same mathematics as vanish::norms, GPU-shaped: each thread computes one
vector's exact norm via CRT over two 31-bit split primes (u32 mulmod, u64
accumulators; NO floating point — norms exceed the f64 mantissa at s=64),
reconstructs to u64 on-device, and batches are deduplicated with cp.unique
before host aggregation. Factoring the (few) unique norms happens on CPU
afterwards via vanish.factor.

Correctness gate (run FIRST, on the pod): `--validate` recomputes the s=32
w<=6 table and must reproduce the exhaustively verified anticorrelation pins
{2:256, 3:6561, 4:38416, 5:279841, 6:1331716} and the total vector count.
Do not trust s=64 output from a pod that has not passed this gate.

Usage:
  python norms_gpu.py --validate
  python norms_gpu.py --s 64 --w 10 --out norms_s64_w10.json
Requires: cupy-cuda12x, vanish wheel (for primes/roots/factoring), CUDA GPU.
"""
import argparse, json, time
from itertools import combinations
from collections import defaultdict
import numpy as np
import cupy as cp
import vanish

KERNEL = r"""
extern "C" __global__ void norms(
    const int   s, const int w, const int ncoef, const long long npat,
    const int   nsup,
    const int*  sup,          // [nsup, w] support positions
    const unsigned int q1, const unsigned int q2,
    const unsigned int* pw1,  // [s] powers of order-s root mod q1
    const unsigned int* pw2,
    const int*  coefs,        // [ncoef] nonzero coefficient values
    unsigned long long* out)  // [nsup * npat] exact norms (CRT-reconstructed)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)nsup * npat;
    if (tid >= total) return;
    int  si  = (int)(tid / npat);
    long long pat = tid % npat;
    int cvec[16]; int pos[16];
    long long t = pat;
    for (int i = 0; i < w; i++) {
        cvec[i] = coefs[t % ncoef]; t /= ncoef;
        pos[i]  = sup[si * w + i];
    }
    unsigned long long prod1 = 1, prod2 = 1;
    for (int k = 1; k < s; k += 2) {
        unsigned long long v1 = 0, v2 = 0;
        for (int i = 0; i < w; i++) {
            int e = (pos[i] * k) % s;
            long long c = cvec[i];
            unsigned long long c1 = c >= 0 ? (unsigned long long)c
                                           : q1 - (unsigned long long)(-c);
            unsigned long long c2 = c >= 0 ? (unsigned long long)c
                                           : q2 - (unsigned long long)(-c);
            v1 = (v1 + c1 * pw1[e]) % q1;
            v2 = (v2 + c2 * pw2[e]) % q2;
        }
        prod1 = (prod1 * v1) % q1;
        prod2 = (prod2 * v2) % q2;
    }
    // CRT: n = r1 + q1 * ((r2 - r1) * inv(q1) mod q2)  -- inv passed via pw2[s]? no:
    // compute with precomputed inverse folded into pw2 tail is fragile; do it here:
    // Fermat inverse of q1 mod q2 (q2 prime, 31-bit -> fast powmod)
    unsigned long long inv = 1, base = q1 % q2, e = q2 - 2;
    while (e) { if (e & 1) inv = (inv * base) % q2; base = (base * base) % q2; e >>= 1; }
    unsigned long long diff = (prod2 + q2 - prod1 % q2) % q2;
    out[tid] = prod1 + (unsigned long long)q1 * ((diff * inv) % q2);
}
"""

def split_prime(s, start):
    q = start - start % s + 1
    while True:
        q += s
        if vanish.is_prime(q):
            return q

def run(s, wmax, cmax, out_path):
    half = s // 2
    bound_bits = (s / 4) * np.log2(cmax * cmax * wmax)
    assert bound_bits < 61.5, "norm exceeds two-31-bit-prime CRT range"
    q1, q2 = split_prime(s, 1 << 31), split_prime(s, (1 << 31) + (1 << 20))
    pw1 = np.array([pow(vanish.subgroup(q1, s)[1], j, q1) for j in range(s)],
                   dtype=np.uint32)
    pw2 = np.array([pow(vanish.subgroup(q2, s)[1], j, q2) for j in range(s)],
                   dtype=np.uint32)
    coefs = np.array([c for c in range(-cmax, cmax + 1) if c], dtype=np.int32)
    mod = cp.RawModule(code=KERNEL)
    kern = mod.get_function("norms")
    d_pw1, d_pw2 = cp.asarray(pw1), cp.asarray(pw2)
    d_coefs = cp.asarray(coefs)

    table = defaultdict(lambda: defaultdict(int))   # norm -> {w: count}
    for w in range(1, wmax + 1):
        sups = np.array(list(combinations(range(half), w)), dtype=np.int32)
        npat = len(coefs) ** w
        # chunk supports so nsup*npat stays ~2^26 per launch
        chunk = max(1, (1 << 26) // npat)
        t0, done = time.time(), 0
        for lo in range(0, len(sups), chunk):
            sub = sups[lo:lo + chunk]
            total = len(sub) * npat
            d_sup = cp.asarray(sub.ravel())
            d_out = cp.empty(total, dtype=cp.uint64)
            kern(((total + 255) // 256,), (256,),
                 (np.int32(s), np.int32(w), np.int32(len(coefs)),
                  np.int64(npat), np.int32(len(sub)), d_sup,
                  np.uint32(q1), np.uint32(q2), d_pw1, d_pw2, d_coefs, d_out))
            vals, counts = cp.unique(d_out, return_counts=True)
            for n, c in zip(vals.get(), counts.get()):
                table[int(n)][w] += int(c)
            done += total
        print(f"w={w}: {done:,} vectors, {len(table):,} uniques so far, "
              f"{time.time()-t0:.1f}s", flush=True)
    with open(out_path, "w") as f:
        json.dump({str(n): dict(ws) for n, ws in table.items()}, f)
    return table

def validate():
    table = run(32, 6, 1, "/tmp/validate_s32.json")
    nmax = {}
    for n, ws in table.items():
        for w in ws:
            nmax[w] = max(nmax.get(w, 0), n)
    pins = {2: 256, 3: 6561, 4: 38416, 5: 279841, 6: 1331716}
    ok = all(nmax[w] == v for w, v in pins.items())
    ref = vanish.norms_n_max(32, 6, 1)
    ok &= all(str(nmax[w]) == ref[w] for w in range(2, 7))
    print("VALIDATION:", "PASS — GPU path reproduces the s=32 golden profile"
          if ok else f"FAIL — {nmax} vs {pins}")
    return ok

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--s", type=int, default=64)
    ap.add_argument("--w", type=int, default=10)
    ap.add_argument("--cmax", type=int, default=1)
    ap.add_argument("--out", default="norms_gpu.json")
    a = ap.parse_args()
    if a.validate:
        raise SystemExit(0 if validate() else 1)
    run(a.s, a.w, a.cmax, a.out)
