"""The q-axis by CONSTRUCTION — the additive family's decay, decoded exactly.

Search is the wrong instrument for the structured families: they are isolated
attractors with shallow basins (greedy crawls +1/step at s=32 because the
(t-1)-shell is nearly empty for a generic word). But we do not need search —
Theorem B gives EXACTNESS at k = r - q, so for a fixed k the cells t = k + q
are all exactly decodable from a constructed word:

    lam  = rung_lambda_e(p, s, r, q)          # the rung's optimal lambda vector
    w    = c5_word(p, dom, r, lam)            # sum_i (-1)^i lam_i x^{r-i}
    pred = bucket_e(p, s, r, lam)             # ANALYTIC list size (accidents included)
    GPU decode w at (k, t=r)  ==  pred        # pre-registered, per cell

Each cell is ONE decode (~2.4s at s=32 with the Barrett kernel) instead of a
16-seed optimizer campaign. The printed table is the additive q-decay curve at
rate k/s, next to the multiplicative q=1 value (C(s,k+1)/s, verified 17,678,835
at s=32) and the classical information-set bound.

  python q_axis.py --p 2130706433 --s 32 --k 16 --qmax 4
  python q_axis.py --p 65537 --s 16 --k 7 --qmax 4     # cross-check vs CPU
"""
import argparse
import math

import numpy as np
import vanish
from decode_gpu import gpu_list_count


def c5_word(p, dom, r, lam):
    """w(x) = sum_{i=0}^{q} (-1)^i lam_i x^{r-i}, lam_0 = 1.
    Mirrors rs::code::c5_word (verified against it at s=16)."""
    q = len(lam)
    out = []
    for x in dom:
        acc = 0
        for i in range(q + 1):
            l = 1 if i == 0 else lam[i - 1] % p
            term = l * pow(x, r - i, p) % p
            acc = (acc + term) % p if i % 2 == 0 else (acc - term) % p
        out.append(acc)
    return out


def gs_class_size(s, t):
    """max_c #{t-subsets of Z_s : sum == c mod s} — the multiplicative
    (Graham-Sloane) class size, computed exactly by DP."""
    dp = np.zeros((t + 1, s), dtype=object)
    dp[0][0] = 1
    for a in range(s):
        for j in range(min(a + 1, t), 0, -1):
            dp[j] = dp[j] + np.roll(dp[j - 1], a)
    return int(max(dp[t]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, required=True)
    ap.add_argument("--s", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--qmax", type=int, default=4)
    a = ap.parse_args()
    dom = list(vanish.subgroup(a.p, a.s))

    print(f"q-axis by construction: p={a.p} s={a.s} k={a.k} (rate {a.k/a.s:.3f})")
    print(f"{'q':>2}{'t':>4}{'delta':>8}{'predicted':>12}{'decoded':>12}"
          f"{'m_struct':>10}{'mult(GS)':>13}{'classical':>13}  status")
    allok = True
    for q in range(1, a.qmax + 1):
        r = a.k + q
        if r >= a.s:
            break
        lam = vanish.rung_lambda_e(a.p, a.s, r, q)
        w = c5_word(a.p, dom, r, lam)
        pred = int(vanish.bucket_e(a.p, a.s, r, lam))
        cnt, ids, sov = gpu_list_count(a.p, dom, w, a.k, r, sample_mask=0xFF)
        ms = int(vanish.m_struct(a.s, r, q))
        gs = gs_class_size(a.s, r) if q == 1 else None
        cl = math.comb(a.s, a.k) // math.comb(r, a.k)
        ok = cnt == pred
        allok &= ok
        gss = f"{gs:,}" if gs else "-"
        print(f"{q:>2}{r:>4}{1 - r/a.s:>8.4f}{pred:>12,}{cnt:>12,}{ms:>10,}"
              f"{gss:>13}{cl:>13,}  {'EXACT' if ok else 'MISMATCH'}")
    print()
    print("EXACTNESS (Theorem B at k=r-q):", "CONFIRMED" if allok else "VIOLATED")
    print("mult(GS) at q=1 is the multiplicative class = max_c N_c(s, k+1); the "
          "additive 'predicted' column is the rung bucket at the same cell.")
