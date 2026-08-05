"""
GPU norm sweep (CuPy raw kernel) — the s=64 bad-set campaign engine.

Same mathematics as vanish::norms, GPU-shaped: each thread computes one
vector's exact norm via CRT over two 31-bit split primes (u32 mulmod, u64
accumulators; NO floating point — norms exceed the f64 mantissa at s=64),
reconstructs to u64 on-device, and batches are deduplicated with cp.unique
before host aggregation. Factoring the (few) unique norms happens on CPU
afterwards via vanish.factor.

Exemplar retention (cmax = 1): each thread also emits its vector packed
2 bits/slot into a u64, and the dedup carries the FIRST vector per norm
through cp.unique(return_index=True) — so the vector <-> norm
association survives to the dumps as `.exemplars.bin` (s/2 i8 per row,
row-aligned with the norms), the format vanish's events-retaining
ingest validates by norm recompute. One exemplar per (norm, weight,
shard): sufficient for event rows, not an orbit census.

Correctness gate (run FIRST, on the pod): `--validate` recomputes the s=32
w<=6 table and must reproduce the exhaustively verified anticorrelation pins
{2:256, 3:6561, 4:38416, 5:279841, 6:1331716} and the total vector count,
and every retained exemplar must recompute (exact Cyclo norm, CPU) to the
norm it is filed under. Do not trust s=64 output from a pod that has not
passed this gate.

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
    const unsigned long long inv,   // inverse of q1 mod q2 (hoisted)
    unsigned long long* out,  // [nsup * npat] exact norms (CRT-reconstructed)
    unsigned long long* enc)  // [nsup * npat] packed vectors (2 bits/slot:
                              //   1 = +1, 2 = -1; cmax = 1 only)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)nsup * npat;
    if (tid >= total) return;
    int  si  = (int)(tid / npat);
    long long pat = tid % npat;
    int cvec[16]; int pos[16];
    long long t = pat;
    unsigned long long packed = 0;
    for (int i = 0; i < w; i++) {
        cvec[i] = coefs[t % ncoef]; t /= ncoef;
        pos[i]  = sup[si * w + i];
        packed |= (unsigned long long)(cvec[i] > 0 ? 1 : 2) << (2 * pos[i]);
    }
    enc[tid] = packed;
    unsigned long long prod1 = 1, prod2 = 1;
    // Lazy accumulation: |c| <= 4, pw < 2^31, w <= 16 -> sums < 2^37, so the
    // per-term %q disappears (one reduction per embedding, ~8x fewer mods).
    // Exponent mask is valid because s is a power of two.
    for (int k = 1; k < s; k += 2) {
        unsigned long long a1 = 0, a2 = 0;
        for (int i = 0; i < w; i++) {
            int e = (pos[i] * k) & (s - 1);
            long long c = cvec[i];
            unsigned long long m = c >= 0 ? (unsigned long long)c
                                          : (unsigned long long)(-c);
            a1 += m * (c >= 0 ? pw1[e] : q1 - pw1[e]);
            a2 += m * (c >= 0 ? pw2[e] : q2 - pw2[e]);
        }
        prod1 = (prod1 * (a1 % q1)) % q1;
        prod2 = (prod2 * (a2 % q2)) % q2;
    }
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

def decode_exemplars(enc, half):
    """Unpack 2-bit/slot u64 encodings to [n, half] i8 coefficient rows."""
    slots = (enc[:, None] >> (2 * np.arange(half, dtype=np.uint64))) & 3
    return np.select([slots == 1, slots == 2], [1, -1], 0).astype(np.int8)


def run(s, wmax, cmax, out_path, shard=0, nshard=1, wmin=1):
    half = s // 2
    assert cmax == 1, "exemplar packing is 2 bits/slot: cmax = 1 only"
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
    exemplars = {}                                  # (w, norm) -> i8 row
    for w in range(wmin, wmax + 1):
        sups = np.array(list(combinations(range(half), w))[shard::nshard],
                        dtype=np.int32)
        if len(sups) == 0:
            continue
        npat = len(coefs) ** w
        # chunk supports so nsup*npat stays ~2^26 per launch
        chunk = max(1, (1 << 26) // npat)
        t0, done = time.time(), 0
        # running dedup lives ON the GPU: the accumulated unique set is small
        # (tens of millions of u64), while host-side merging at w >= 10 was
        # observed to starve the GPUs (0% util, 100% CPU, tens of GB RSS).
        # The exemplar array rides the same dedup: unique's first-occurrence
        # indices point at the earliest vector per norm, and because the
        # accumulator is concatenated FIRST, an exemplar never changes once
        # chosen (stable across chunks).
        acc_v = cp.empty(0, dtype=cp.uint64)
        acc_c = cp.empty(0, dtype=cp.float64)  # exact: totals < 2^53
        acc_e = cp.empty(0, dtype=cp.uint64)
        for lo in range(0, len(sups), chunk):
            sub = sups[lo:lo + chunk]
            total = len(sub) * npat
            d_sup = cp.asarray(sub.ravel())
            d_out = cp.empty(total, dtype=cp.uint64)
            d_enc = cp.empty(total, dtype=cp.uint64)
            kern(((total + 255) // 256,), (256,),
                 (np.int32(s), np.int32(w), np.int32(len(coefs)),
                  np.int64(npat), np.int32(len(sub)), d_sup,
                  np.uint32(q1), np.uint32(q2), d_pw1, d_pw2, d_coefs,
                  np.uint64(pow(q1, q2 - 2, q2)), d_out, d_enc))
            vals, first, counts = cp.unique(d_out, return_index=True,
                                            return_counts=True)
            allv = cp.concatenate([acc_v, vals])
            alle = cp.concatenate([acc_e, d_enc[first]])
            allc = cp.concatenate([acc_c, counts.astype(cp.float64)])
            acc_v, keep, inv = cp.unique(allv, return_index=True,
                                         return_inverse=True)
            acc_e = alle[keep]
            acc_c = cp.bincount(inv, weights=allc)
            done += total
        if out_path.endswith(".json"):
            # small runs / --validate: in-memory table, JSON at the end
            # (exemplars stay in memory; JSON carries no vectors)
            ex = decode_exemplars(acc_e.get(), half)
            for i, (n, c) in enumerate(zip(acc_v.get(), acc_c.get())):
                table[int(n)][w] += int(c)
                exemplars[(w, int(n))] = ex[i]
        else:
            # campaign runs: binary per-weight dumps -- no Python dict, no
            # JSON (a w=12 shard is ~516M uniques; the dict path cost ~300GB
            # RAM and OOM'd 3 of 4 shards)
            acc_v.get().tofile(f"{out_path}.w{w}.norms.bin")
            acc_c.get().astype(np.uint64).tofile(f"{out_path}.w{w}.counts.bin")
            decode_exemplars(acc_e.get(), half).tofile(
                f"{out_path}.w{w}.exemplars.bin")
        print(f"w={w}: {done:,} vectors, {len(table):,} uniques so far, "
              f"{time.time()-t0:.1f}s", flush=True)
    if out_path.endswith(".json"):
        with open(out_path, "w") as f:
            json.dump({str(n): dict(ws) for n, ws in table.items()}, f)
    return table, exemplars

def validate():
    table, exemplars = run(32, 6, 1, "/tmp/validate_s32.json")
    nmax = {}
    for n, ws in table.items():
        for w in ws:
            nmax[w] = max(nmax.get(w, 0), n)
    pins = {2: 256, 3: 6561, 4: 38416, 5: 279841, 6: 1331716}
    ok = all(nmax[w] == v for w, v in pins.items())
    ref = vanish.norms_n_max(32, 6, 1)
    ok &= all(str(nmax[w]) == ref[w] for w in range(2, 7))
    # exemplar association gate: every retained vector must recompute
    # (exact Cyclo norm, CPU) to the norm it is filed under, at its weight
    bad = 0
    for (w, n), row in exemplars.items():
        v = [int(c) for c in row]
        if sum(1 for c in v if c) != w or int(vanish.Cyclo(v).norm_crt()) != n:
            bad += 1
    ok &= bad == 0
    print("VALIDATION:",
          f"PASS — GPU path reproduces the s=32 golden profile and all "
          f"{len(exemplars):,} exemplars recompute to their norms"
          if ok else f"FAIL — {nmax} vs {pins}; {bad} exemplar mismatches")
    return ok

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--s", type=int, default=64)
    ap.add_argument("--w", type=int, default=10)
    ap.add_argument("--cmax", type=int, default=1)
    ap.add_argument("--out", default="norms_gpu.json")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--wmin", type=int, default=1,
                    help="resume: skip weights below this (their .bin dumps already exist)")
    a = ap.parse_args()
    if a.validate:
        raise SystemExit(0 if validate() else 1)
    run(a.s, a.w, a.cmax, a.out, a.shard, a.nshard, a.wmin)
