"""CPU reference of the GPU decode algorithm — validates the exact math the CUDA
kernel will use (combination unranking + barycentric with NO modular inverse in
the eval loop) against vanish.list_decode before we port to CUDA."""
import vanish, math
from itertools import combinations

def modinv(a, p): return pow(a % p, p - 2, p)

def unrank(idx, n, k):
    """The idx-th k-combination of range(n) in lexicographic order (the CUDA
    thread->info-set map). Colex/comb-number-system."""
    c = []; x = 0; rem = idx
    for pos in range(k):
        # choose next element >= x
        while True:
            cnt = math.comb(n - x - 1, k - pos - 1)
            if rem < cnt: c.append(x); x += 1; break
            rem -= cnt; x += 1
    return c

def gpu_decode(dom, w, k, t, p):
    n = len(dom)
    # precompute inverse-difference table inv_diff[i][j] = 1/(x_i - x_j)
    inv_diff = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j: inv_diff[i][j] = modinv((dom[i]-dom[j]) % p, p)
    seen = set()
    total = math.comb(n, k)
    for idx in range(total):                      # one "thread" per info-set
        I = unrank(idx, n, k)
        # barycentric weights
        wt = []
        for m in range(k):
            acc = 1
            for l in range(k):
                if l != m: acc = acc * inv_diff[I[m]][I[l]] % p
            wt.append(acc)
        ys = [w[i] for i in I]
        c = [0]*n; ok = True
        for j in range(n):
            if j in I: c[j] = w[j]; continue
            num = 0; invden = 1
            for m in range(k):
                num = (num + wt[m]*ys[m] % p * inv_diff[j][I[m]]) % p
                invden = invden * ((dom[j]-dom[I[m]]) % p) % p   # = 1/den, no modinv!
            c[j] = num * invden % p
        agree = sum(1 for a, b in zip(c, w) if a == b)
        if agree >= t:
            seen.add(tuple(c[:k]))                # codeword ID = values at first k pts
    return len(seen)

if __name__ == "__main__":
    P, S = 65537, 16
    DOM = list(vanish.subgroup(P, S))
    # unrank self-check: does it enumerate every combination exactly once?
    got = {tuple(unrank(i, S, 7)) for i in range(math.comb(S, 7))}
    print("unrank enumerates all C(16,7):", got == set(map(tuple, combinations(range(S), 7))))
    # validate decode on a bucket word (x^8) and a shifted word
    for name, w in [("bucket x^8", [pow(x, 8, P) for x in DOM]),
                    ("shifted",   [(pow(x, 8, P) + 3*x) % P for x in DOM])]:
        ref = gpu_decode(DOM, w, 7, 8, P)
        cpu = len(vanish.list_decode(P, DOM, 7, w, 8))
        print(f"{name:<12} gpu-ref={ref}  vanish.list_decode={cpu}  MATCH={ref==cpu}")
