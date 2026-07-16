"""Construct the multiplicative-C.5 word — the Experiment-1 artifact.

For the q=1 cell (t = k+1) on mu_s in F_p: the extremal list is (hypothesis,
verified at s=16) the full Graham-Sloane class
    C* = { T subset Z_s, |T| = t : sum_{i in T} i == c* (mod s) }
(exponent coordinates, dom[i] = zeta^i), realized by a word w whose order-t
divided difference vanishes on every T in C*:
    D_T(w) = sum_{x in T} w(x) / prod_{y in T, y != x} (x - y)  ==  0.
D_T annihilates RS_k, so the solution space contains the code; the RANK
IDENTITY (verified at s=16: rank = k+1) says the kernel is exactly RS_k plus
ONE extra line — that line is the word.

This script:
  1. computes the exact class counts N_c(s,t) (DP) and picks the max class c*
  2. solves the system on sampled class rows, checks the kernel stabilizes at
     dim k+2? no — dim k+1, and reduces mod RS_k to the unique word
  3. verifies on FRESH samples (in-class rows vanish, out-of-class don't),
     and for s=16 fully verifies against the CPU decoder (list == 810 ==
     class count, members frozen)
  4. writes  w{s}_mult.npy  +  w{s}_mult.json  (predicted count, c*, params)

Usage:
  python construct_word.py --s 16 --p 65537        # self-test cell (810)
  python construct_word.py --s 32 --p 2130706433   # Experiment-1 word (~17.7M)
"""
import argparse
import json
import random
from itertools import combinations

import numpy as np


def modinv(a, p):
    return pow(int(a) % p, p - 2, p)


def class_counts(n, t):
    """N_c = #{t-subsets of Z_n : sum == c mod n} exactly (DP)."""
    dp = np.zeros((t + 1, n), dtype=np.int64)
    dp[0][0] = 1
    for a in range(n):
        for j in range(min(a + 1, t), 0, -1):
            dp[j] = dp[j] + np.roll(dp[j - 1], a)
    return dp[t]


def dd_row(T, dom, p, n):
    """Divided-difference functional of the index set T as a length-n row."""
    row = [0] * n
    for x in T:
        d = 1
        for y in T:
            if y != x:
                d = d * ((dom[x] - dom[y]) % p) % p
        row[x] = modinv(d, p)
    return row


def rref_kernel(rows, n, p):
    """(rank, kernel basis) of the span of `rows` acting on F_p^n."""
    A = [r[:] for r in rows]
    pivots, r = [], 0
    for c in range(n):
        piv = next((i for i in range(r, len(A)) if A[i][c] % p), None)
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        inv = modinv(A[r][c], p)
        A[r] = [v * inv % p for v in A[r]]
        for i in range(len(A)):
            if i != r and A[i][c]:
                f = A[i][c]
                A[i] = [(A[i][j] - f * A[r][j]) % p for j in range(n)]
        pivots.append(c)
        r += 1
    free = [c for c in range(n) if c not in pivots]
    basis = []
    for fc in free:
        v = [0] * n
        v[fc] = 1
        for ri, pc in enumerate(pivots):
            v[pc] = (-A[ri][fc]) % p
        basis.append(v)
    return r, basis


def reduce_mod_span(vecs, span_rows, n, p):
    """Residues of `vecs` after eliminating the pivots of span_rows (RREF)."""
    A = [r[:] for r in span_rows]
    pivots, r = [], 0
    for c in range(n):
        piv = next((i for i in range(r, len(A)) if A[i][c] % p), None)
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        inv = modinv(A[r][c], p)
        A[r] = [v * inv % p for v in A[r]]
        for i in range(len(A)):
            if i != r and A[i][c]:
                f = A[i][c]
                A[i] = [(A[i][j] - f * A[r][j]) % p for j in range(n)]
        pivots.append(c)
        r += 1
    out = []
    for v in vecs:
        w = v[:]
        for ri, pc in enumerate(pivots):
            if w[pc]:
                f = w[pc]
                w = [(w[j] - f * A[ri][j]) % p for j in range(n)]
        out.append(w)
    return out


def sample_class(n, t, cstar, rng):
    while True:
        T = tuple(sorted(rng.sample(range(n), t)))
        if sum(T) % n == cstar:
            return T


def build_mult_word(p, s, rows_init=300, seed=1):
    """Returns (word, cstar, predicted_count, diagnostics)."""
    n, t, k = s, s // 2 + 1, s // 2   # q=1 rate-1/2 cell: k = s/2, t = k+1
    # domain in generator-power order
    import vanish
    dom = list(vanish.subgroup(p, s))
    N = class_counts(n, t)
    cstar = int(N.argmax())
    pred = int(N.max())
    rng = random.Random(seed)

    # small s: use the whole class; large s: sample rows until rank stabilizes
    if pred <= 2000:
        Ts = [T for T in combinations(range(n), t) if sum(T) % n == cstar]
    else:
        Ts = [sample_class(n, t, cstar, rng) for _ in range(rows_init)]
    rows = [dd_row(T, dom, p, n) for T in Ts]
    rank, kern = rref_kernel(rows, n, p)
    # stabilization check for sampled systems
    if pred > 2000:
        rows2 = rows + [dd_row(sample_class(n, t, cstar, rng), dom, p, n)
                        for _ in range(150)]
        rank2, kern2 = rref_kernel(rows2, n, p)
        assert rank2 == rank, f"rank not stabilized: {rank} -> {rank2}"
        kern = kern2
    assert len(kern) == k + 1, (
        f"kernel dim {len(kern)} != k+1 = {k + 1} — rank identity FAILS here")

    # reduce kernel mod RS_k; exactly one independent residue = the word
    rs_basis = [[pow(x, j, p) for x in dom] for j in range(k)]
    residues = [v for v in reduce_mod_span(kern, rs_basis, n, p)
                if any(c % p for c in v)]
    assert residues, "kernel contained in RS_k?!"
    w = residues[0]
    # normalize: first nonzero coordinate = 1
    lead = next(c for c in w if c % p)
    w = [c * modinv(lead, p) % p for c in w]

    # fresh-sample verification
    ok_in = all(
        sum(r * wi for r, wi in zip(
            dd_row(sample_class(n, t, cstar, rng), dom, p, n), w)) % p == 0
        for _ in range(500))
    out_hits = sum(
        1 for _ in range(200)
        if sum(r * wi for r, wi in zip(
            dd_row(tuple(sorted(rng.sample(range(n), t))), dom, p, n), w))
        % p == 0)
    diag = dict(rank=rank, kernel_dim=len(kern), fresh_inclass_allzero=ok_in,
                random_T_zero_fraction=out_hits / 200,
                class_counts_max=pred, class_counts_c=cstar,
                class_second=int(np.sort(N)[-2]))
    assert ok_in, "fresh in-class functional did not vanish"
    return w, cstar, pred, diag


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, required=True)
    ap.add_argument("--p", type=int, required=True)
    a = ap.parse_args()
    import vanish

    w, cstar, pred, diag = build_mult_word(a.p, a.s)
    n, t, k = a.s, a.s // 2 + 1, a.s // 2
    print(f"(n={n}, k={k}, t={t}) at p={a.p}: c*={cstar}, "
          f"predicted list = {pred}")
    print("diagnostics:", diag)

    if a.s <= 16:   # full CPU verification against the exact decoder
        dom = list(vanish.subgroup(a.p, a.s))
        members = vanish.list_decode(a.p, dom, k, w, t)
        P = np.array(members, dtype=np.int64)
        agree = P == np.array(w, dtype=np.int64)[None, :]
        sums = {int(np.where(r)[0].sum() % n) for r in agree}
        sizes = {int(r.sum()) for r in agree}
        print(f"CPU decode: L = {len(members)} (predicted {pred}); "
              f"agreement sizes {sorted(sizes)}; frozen sums {sorted(sums)}")
        assert len(members) == pred and sums == {cstar}, "s=16 GATE FAILED"
        print("s=16 FULL GATE PASSED")

    np.save(f"w{a.s}_mult.npy", np.array(w, dtype=np.uint32))
    with open(f"w{a.s}_mult.json", "w") as f:
        json.dump(dict(p=a.p, s=a.s, k=k, t=t, cstar=cstar,
                       predicted_count=pred, diag=diag), f, indent=1)
    print(f"wrote w{a.s}_mult.npy / .json")
