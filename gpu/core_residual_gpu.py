"""GPU core-residual list decoder — the s = 64 attack instrument.

Same math as vanish::rs::core_residual (the Rust authority this module
is gated against): a codeword agreeing with the word on t > n of the
s = 2n paired points fully agrees on at least l = t - n fibers, so
enumerating the C(n, l) fiber cores and Guruswami-Sudan-decoding each
residual is complete. GPU-shaped: one core per warp. The warp unranks
its core (colexicographic, matching the Rust index space, so GPU and
CPU shards partition the same way), interpolates the word on the 2l
core points, builds the residual targets on the free points,
interpolates by Koetter's update, finds y-roots by Roth-Ruckenstein,
reassembles f = q_Y + V_Y g, and emits the members agreeing on >= t
points.

Why a warp and not a thread. The dense Hasse system of
vanish::rs::gs is 126 x 132 at the (64, 31, 43) residual -- far past a
thread, which is why Koetter replaces it. But Koetter's own state is
also too big: measured peak is 330 u32 for a single one of the dy + 1
candidates at (42, 9, 21), and 2436 u32 at (44, 11, 22), against a
255-register hardware ceiling and a ~64 budget for decent occupancy.
Thread-per-core would spill to local memory -- the 50-100x cliff
decode_gpu.py documents. So the candidates live in shared memory, one
core per warp, and the lanes cooperate on the two inner loops that
dominate (the dy + 1 discrepancies, and the elementwise row updates),
both of which are long vector operations over exactly this state.

Re-encoding, evaluated and declined. Koetter-Vardy re-encoding
translates the word so k targets vanish and starts from the basis
V_R^max(0, m-b) y^b, which discharges k m(m+1)/2 constraints at once.
Measured at both production cells it is an exact wash: the degree
front-loaded into that basis is k * sum_{b<m} (m - b) = k m(m+1)/2 --
the same number as the constraints it saves, since each constraint
adds exactly one to the pivot's degree. Peak state is byte-identical
with and without it (330 / 829 u32 at (42, 9, 21)), and constraints
fall 126 -> 99 only because the degree moved into the starting basis.
A real saving needs the transformed coordinates, storing q-hat with
the V_R factors implicit -- worth about 20-25%, at the cost of a
second interpolation algorithm to gate. It cannot change the
architecture (330 u32 does not become 64), so it is deferred; the
path stays here, gated, as the evidence.

This module is the CPU mirror and the host driver. The mirror is a
transliteration of the intended kernel -- same constraint order, same
pivot tie-break, same descent -- so an algorithm bug surfaces here,
on any machine, against the Rust oracle, before pod time is spent.

Gates (no output is believed before its gate):
  selfcheck() -- the mirror equals vanish.list_decode_paired at
    (16, 7, 10) and (16, 5, 9) on random, codeword, near-codeword and
    planted words, at two primes. Runs anywhere; no GPU needed.
  validate()  -- the CUDA kernel equals the mirror and the Rust
    decoder on those cells plus the (32, 15, 21) battery cell. Run on
    the pod before any s = 64 number is quoted.

Cell shapes come from vanish.gs_params (the Rust derivation):
  (64, 31, 43) -> residual (42, 9, 21):  m = 2, d = 41,  dy = 5
  (64, 31, 42) -> residual (44, 11, 22): m = 6, d = 131, dy = 13

Status: the kernel is written but UNVALIDATED -- it has never been
compiled or run, because this workstation has no CUDA device. Nothing
it produces may be quoted until validate() passes on the pod. The
mirror is the gate that stands today.

Usage:
  python core_residual_gpu.py --tabulate    # cell shape table
  python core_residual_gpu.py --layout      # shared-memory plan
  python core_residual_gpu.py --selfcheck   # the mirror gate, anywhere
  python core_residual_gpu.py --validate    # the GPU gate, on the pod
Requires: vanish wheel; cupy-cuda12x for the GPU paths only.
"""

import argparse
import math

import vanish

# ---------------------------------------------------------------------------
# Cell shapes, from the Rust authority


def residual_cell(s, k, t):
    """The residual decode shape at the paired cell (s, k, t):
    (l, n_res, k_res, t_res, m, d, dy). Raises when the cell is not
    core-residual decodable -- vanish.gs_params owns that judgement."""
    n = s // 2
    if not n < t <= s:
        raise ValueError(f"core enumeration needs n < t <= 2n (t = {t}, n = {n})")
    l = t - n
    nr, kr, tr = s - 2 * l, k - 2 * l, t - 2 * l
    if kr < 2:
        raise ValueError(f"residual dimension k - 2l = {kr} below 2")
    m, d = vanish.gs_params(nr, kr, tr)
    return l, nr, kr, tr, m, d, d // (kr - 1)


def tabulate(cells=((64, 31, 43), (64, 31, 42), (32, 15, 21), (16, 7, 10), (16, 5, 9))):
    """The shape table the kernel generator consumes."""
    head = ("cell", "l", "residual", "m", "d", "dy", "constraints", "cores")
    print(f"{head[0]:>14} {head[1]:>3} {head[2]:>14} {head[3]:>3} {head[4]:>4} "
          f"{head[5]:>3} {head[6]:>11} {head[7]:>15}")
    for s, k, t in cells:
        l, nr, kr, tr, m, d, dy = residual_cell(s, k, t)
        print(f"{(s, k, t)!s:>14} {l:>3} {(nr, kr, tr)!s:>14} {m:>3} {d:>4} "
              f"{dy:>3} {nr * m * (m + 1) // 2:>11} {math.comb(s // 2, l):>15,}")


def paired_domain(p, s):
    """The order-s subgroup in fiber-major antipodal layout:
    points[i + n] = -points[i], the layout vanish's paired decoder
    and the kernel both assume."""
    pts = [int(x) for x in vanish.subgroup(p, s)]
    n = s // 2
    return pts[:n] + [(p - x) % p for x in pts[:n]]


# ---------------------------------------------------------------------------
# Univariate F_p[x], dense low-to-high, mirroring vanish::math::poly.
# Trimmed means the last coefficient is nonzero; [] is the zero
# polynomial.


def _trim(f):
    while f and f[-1] == 0:
        f.pop()
    return f


def _horner(f, x, p):
    acc = 0
    for c in reversed(f):
        acc = (acc * x + c) % p
    return acc


def _mul(a, b, p):
    if not a or not b:
        return []
    out = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                out[i + j] = (out[i + j] + ai * bj) % p
    return _trim(out)


def _rem(f, mod, p):
    f = list(f)
    dm = len(mod) - 1
    lead_inv = pow(mod[dm], p - 2, p)
    _trim(f)
    while len(f) > dm:
        c = f[-1] * lead_inv % p
        if c:
            shift = len(f) - 1 - dm
            for i, mi in enumerate(mod):
                f[shift + i] = (f[shift + i] - c * mi) % p
        f.pop()
        _trim(f)
    return f


def _gcd(a, b, p):
    a, b = _trim(list(a)), _trim(list(b))
    while b:
        a, b = b, _rem(a, b, p)
    if a:
        li = pow(a[-1], p - 2, p)
        a = [c * li % p for c in a]
    return a


def _pow_rem(base, e, mod, p):
    acc, base = [1], _rem(base, mod, p)
    while e:
        if e & 1:
            acc = _rem(_mul(acc, base, p), mod, p)
        base = _rem(_mul(base, base, p), mod, p)
        e >>= 1
    return acc


def _div_exact(h, dv, p):
    rest = _trim(list(h))
    dd = len(dv) - 1
    lead_inv = pow(dv[dd], p - 2, p)
    quot = [0] * max(len(rest) - dd, 0)
    while len(rest) > dd:
        c = rest[-1] * lead_inv % p
        shift = len(rest) - 1 - dd
        quot[shift] = c
        for i, di in enumerate(dv):
            rest[shift + i] = (rest[shift + i] - c * di) % p
        rest.pop()
        _trim(rest)
    return _trim(quot)


def _roots(f, p):
    """All roots of a nonzero f in F_p, sorted, each once: gcd with
    x^p - x, then Cantor-Zassenhaus on the deterministic shifts
    1, 2, 3, ... -- vanish::math::poly::roots, transliterated."""
    f = _trim(list(f))
    assert f, "roots of the zero polynomial"
    out = []
    if len(f) > 1 and f[0] == 0:
        out.append(0)
        low = next(i for i, c in enumerate(f) if c)
        f = f[low:]
    if len(f) > 1:
        xp = _pow_rem([0, 1], p, f, p)
        while len(xp) < 2:
            xp.append(0)
        xp[1] = (xp[1] - 1) % p
        _split_linear(_gcd(f, xp, p), p, out)
    return sorted(out)


def _split_linear(h, p, out):
    stack, shift = [h], 1
    while stack:
        h = stack.pop()
        if len(h) <= 1:
            continue
        if len(h) == 2:
            out.append((-h[0]) % p)
            continue
        while True:
            half = _pow_rem([shift % p, 1], (p - 1) // 2, h, p)
            shift += 1
            if not half:
                half = [0]
            half[0] = (half[0] - 1) % p
            _trim(half)
            if not half:
                continue
            dv = _gcd(h, half, p)
            if 1 < len(dv) < len(h):
                stack.append(_div_exact(h, dv, p))
                stack.append(dv)
                break


def _interpolate(xs, ys, p):
    """The interpolant through (xs, ys) as coefficients (Newton)."""
    n = len(xs)
    dd = list(ys)
    coeffs = [dd[0]]
    for level in range(1, n):
        for i in range(n - 1, level - 1, -1):
            den = (xs[i] - xs[i - level]) % p
            assert den, "repeated interpolation node"
            dd[i] = (dd[i] - dd[i - 1]) * pow(den, p - 2, p) % p
        coeffs.append(dd[level])
    f, basis = [0] * n, [1]
    for level, c in enumerate(coeffs):
        for i, b in enumerate(basis):
            f[i] = (f[i] + c * b) % p
        if level + 1 < n:
            neg = (-xs[level]) % p
            basis = [0] + basis
            for i in range(len(basis) - 1):
                basis[i] = (basis[i] + neg * basis[i + 1]) % p
    return f


# ---------------------------------------------------------------------------
# Bivariate Q(x, y) as rows q[b] = the x-polynomial multiplying y^b.


def _binom_table(top, p):
    c = [[0] * (top + 1) for _ in range(top + 1)]
    for a in range(top + 1):
        c[a][0] = 1
        for b in range(1, a + 1):
            c[a][b] = (c[a - 1][b - 1] + c[a - 1][b]) % p
    return c


def _add_poly(a, b, p):
    """Sum of two coefficient vectors."""
    out = list(a) + [0] * max(0, len(b) - len(a))
    for i, c in enumerate(b):
        out[i] = (out[i] + c) % p
    return _trim(out)


def _wdeg(q, w):
    """The (1, w)-weighted degree; -1 for the zero polynomial."""
    best = -1
    for b, row in enumerate(q):
        for a, c in enumerate(row):
            if c:
                best = max(best, a + w * b)
    return best


def _hasse(q, x, y, r, s, binom, p):
    """The (r, s) Hasse derivative of Q evaluated at (x, y)."""
    acc = 0
    for b in range(s, len(q)):
        cb = binom[b][s]
        if not cb:
            continue
        yp = pow(y, b - s, p)
        row = q[b]
        for a in range(r, len(row)):
            if row[a]:
                term = binom[a][r] * cb % p * row[a] % p
                acc = (acc + term * pow(x, a - r, p) * yp) % p
    return acc % p


def _koetter(xs, ys, k, m, d, p, init=None, stats=None):
    """Koetter's interpolation: the minimal (1, k-1)-weighted-degree
    Q vanishing to order m at every (xs[i], ys[i]).

    `init` supplies a starting basis other than y^0..y^dy (the
    re-encoded path passes one); `stats` is an optional dict that
    collects the peak state the kernel must budget for.

    Candidates g_0..g_dy start as y^b and stay a basis of the solution
    module of the constraints processed so far. Each constraint kills
    one candidate: the nonzero-discrepancy candidate of least weighted
    degree is multiplied by (x - x_i) (raising its degree by one), and
    every other nonzero-discrepancy candidate has that pivot subtracted
    off. Ties in weighted degree break to the lower index -- the kernel
    does the same, so mirror and device agree coefficient for
    coefficient, not merely as root sets.

    This replaces the nullspace of vanish::rs::gs: same solution
    module, but dy + 1 running polynomials instead of a dense
    126 x 132 system, which is what makes a thread-per-core kernel
    possible."""
    w = k - 1
    dy = d // w
    if init is None:
        gs = []
        for j in range(dy + 1):
            rows = [[] for _ in range(dy + 1)]
            rows[j] = [1]
            gs.append(rows)
    else:
        gs = [[list(row) for row in g] for g in init]
    # each constraint multiplies one candidate by (x - x_i), so an
    # x-degree can reach the constraint count past the starting basis
    start = max((len(r) for g in gs for r in g), default=1)
    binom = _binom_table(start + len(xs) * m * (m + 1) // 2 + dy + 2, p)
    for xi, yi in zip(xs, ys):
        for r in range(m):
            for s in range(m - r):
                deltas = [_hasse(g, xi, yi, r, s, binom, p) for g in gs]
                live = [j for j, dl in enumerate(deltas) if dl]
                if not live:
                    continue
                star = min(live, key=lambda j: (_wdeg(gs[j], w), j))
                inv_star = pow(deltas[star], p - 2, p)
                pivot = gs[star]
                for j in live:
                    if j == star:
                        continue
                    c = deltas[j] * inv_star % p
                    g = gs[j]
                    for b, prow in enumerate(pivot):
                        row = g[b]
                        if len(row) < len(prow):
                            row.extend([0] * (len(prow) - len(row)))
                        for a, pv in enumerate(prow):
                            row[a] = (row[a] - c * pv) % p
                        _trim(row)
                # pivot *= (x - x_i)
                gs[star] = [_mul(row, [(-xi) % p, 1], p) if row else []
                            for row in pivot]
                if stats is not None:
                    stats["constraints"] = stats.get("constraints", 0) + 1
                    live_state = max(sum(len(r) for r in g) for g in gs)
                    stats["peak_candidate"] = max(stats.get("peak_candidate", 0),
                                                  live_state)
                    stats["peak_total"] = max(
                        stats.get("peak_total", 0),
                        sum(len(r) for g in gs for r in g))
    return min(gs, key=lambda g: _wdeg(g, w))


def _reencode_basis(xs, ys, k, m, d, p):
    """The re-encoding setup at (xs, ys): the translate psi, the
    translated targets, and the starting basis that satisfies the
    first k points' constraints for free.

    Translating by psi -- the degree-<k interpolant of the word on k
    chosen points -- zeroes the targets there, and is a bijection on
    codewords preserving every agreement (f <-> f - psi), so the list
    is unchanged. With those targets zero, vanishing to order m at
    (x_i, 0) is exactly divisibility of q_b by V_R^(m-b), so the
    module they cut out is generated by V_R^max(0, m-b) y^b. Starting
    Koetter there discharges k m(m+1)/2 constraints at once; only the
    remaining points are iterated."""
    psi = _interpolate(xs[:k], ys[:k], p)
    hat = [(y - _horner(psi, x, p)) % p for x, y in zip(xs, ys)]
    v_r = [1]
    for x in xs[:k]:
        v_r = _mul(v_r, [(-x) % p, 1], p)
    dy = d // (k - 1)
    init = []
    for b in range(dy + 1):
        rows = [[] for _ in range(dy + 1)]
        power = [1]
        for _ in range(max(0, m - b)):
            power = _mul(power, v_r, p)
        rows[b] = power
        init.append(rows)
    return psi, hat, init


def _strip_x(q):
    """Divide out the largest common power of x."""
    low = min((next((a for a, c in enumerate(row) if c), len(row))
               for row in q if any(row)), default=0)
    return [row[low:] if len(row) > low else [] for row in q] if low else q


def _shift_y(q, alpha, p):
    """Q(x, x y + alpha)."""
    dy = len(q) - 1
    binom = _binom_table(dy, p)
    out = [[] for _ in range(dy + 1)]
    for j in range(dy + 1):
        acc = []
        for b in range(j, dy + 1):
            c = binom[b][j] * pow(alpha, b - j, p) % p
            if not c or not q[b]:
                continue
            if len(acc) < len(q[b]):
                acc.extend([0] * (len(q[b]) - len(acc)))
            for a, qa in enumerate(q[b]):
                acc[a] = (acc[a] + c * qa) % p
        out[j] = ([0] * j + _trim(acc)) if _trim(acc) else []
    return out


def _roth_ruckenstein(q, k, p):
    """Every f with deg f < k such that (y - f(x)) divides Q, as
    coefficient vectors. Soundness needs no care -- the caller verifies
    every candidate -- so a degenerate all-zero column takes the single
    branch alpha = 0, as the Rust decoder does."""
    out = []

    def descend(q, prefix):
        q = _strip_x(q)
        column = _trim([row[0] if row else 0 for row in q])
        branches = [0] if not column else _roots(column, p)
        for alpha in branches:
            prefix.append(alpha)
            if len(prefix) == k:
                out.append(list(prefix))
            else:
                descend(_shift_y(q, alpha, p), prefix)
            prefix.pop()

    descend(q, [])
    return out


# ---------------------------------------------------------------------------
# The per-core pipeline: exactly what one GPU thread will do.


def members_through_core(p, points, k, word, t, core, reencode=False, stats=None):
    """The members found through one core, as evaluation vectors over
    the full domain, each verified to agree with `word` on >= t
    points."""
    n = len(points) // 2
    l = len(core)
    in_core = [False] * n
    for i in core:
        in_core[i] = True
    cxs, cys = [], []
    for i in core:
        for j in (i, i + n):
            cxs.append(points[j])
            cys.append(word[j])
    q = _interpolate(cxs, cys, p)
    squares = [points[i] * points[i] % p for i in core]

    def v_at(x):
        x2 = x * x % p
        acc = 1
        for y in squares:
            acc = acc * ((x2 - y) % p) % p
        return acc

    free = [j for i in range(n) if not in_core[i] for j in (i, i + n)]
    fxs = [points[j] for j in free]
    targets = [(word[j] - _horner(q, x, p)) * pow(v_at(x), p - 2, p) % p
               for j, x in zip(free, fxs)]
    kr, tr = k - 2 * l, t - 2 * l
    try:
        m, d = vanish.gs_params(len(fxs), kr, tr)
    except Exception:
        return []
    if reencode:
        psi, hat, init = _reencode_basis(fxs, targets, kr, m, d, p)
        qq = _koetter(fxs[kr:], hat[kr:], kr, m, d, p, init=init, stats=stats)
        cands = [_add_poly(g, psi, p) for g in _roth_ruckenstein(qq, kr, p)]
    else:
        qq = _koetter(fxs, targets, kr, m, d, p, stats=stats)
        cands = _roth_ruckenstein(qq, kr, p)
    out = []
    for g in cands:
        f = [(_horner(q, x, p) + v_at(x) * _horner(g, x, p)) % p for x in points]
        if sum(a == b for a, b in zip(f, word)) >= t:
            out.append(tuple(f))
    return out


def list_paired_mirror(p, points, k, word, t, cores=None, reencode=False, stats=None):
    """The mirror's exact list: the members through every core (or the
    given core index range), deduplicated. Equals
    vanish.list_decode_paired -- selfcheck() is that assertion."""
    n = len(points) // 2
    l, *_ = residual_cell(len(points), k, t)
    total = math.comb(n, l)
    rng = range(total) if cores is None else cores
    seen = {}
    for idx in rng:
        for f in members_through_core(p, points, k, word, t,
                                      _unrank_colex(idx, n, l),
                                      reencode=reencode, stats=stats):
            seen.setdefault(f, None)
    return list(seen)


def _unrank_colex(idx, n, l):
    """The idx-th l-subset of 0..n colexicographically -- the same
    index space as vanish's unrank_combination, so a GPU shard and a
    CPU shard of the same range cover the same cores."""
    out, remaining, top = [], l, n
    while remaining:
        top -= 1
        c = math.comb(top, remaining)
        if idx >= c:
            out.append(top)
            idx -= c
            remaining -= 1
    out.reverse()
    return out


# ---------------------------------------------------------------------------
# The CUDA kernel: one core per warp.
#
# Transliterated from the mirror above -- same constraint order, same
# pivot tie-break, same descent -- so validate() compares two
# implementations of one algorithm, not two algorithms.
#
# Shared memory, per core, at the (64, 31, 43) residual (42, 9, 21):
# the dy + 1 = 6 Koetter candidates dominate. Rows are allocated
# triangularly, row b holding WDB - w b + 1 coefficients, because the
# measured row lengths are exactly that profile (75, 67, 59, 51, 43,
# 35 at w = 8) -- a candidate's weighted degree bounds every row at
# once. WDB = 84 against a measured maximum of 74 over 120 cores;
# degree conservation permits 126 in the worst case, so a candidate
# that exceeds the budget is not decoded here: the core is emitted as
# a survivor and finished on the host by the gated Rust decoder. That
# keeps the sweep exact whatever the budget.

_KERNEL = r"""
#define LANES 32
#define WARPS %(WARPS)d

#define SC   %(SC)d          /* paired domain size s        */
#define NF   %(NF)d          /* fibers n = s / 2            */
#define KC   %(KC)d          /* code dimension k            */
#define TC   %(TC)d          /* agreement threshold t       */
#define LC   %(LC)d          /* core size l = t - n         */
#define NR   %(NR)d          /* residual points  s - 2l     */
#define KR   %(KR)d          /* residual dimension k - 2l   */
#define TR   %(TR)d          /* residual agreement t - 2l   */
#define MC   %(MC)d          /* GS multiplicity             */
#define DYC  %(DYC)d         /* interpolant y-degree        */
#define WC   %(WC)d          /* weight k_res - 1            */
#define WDB  %(WDB)d         /* weighted-degree budget      */
#define CANDSZ %(CANDSZ)d    /* u32 per candidate           */
#define ROW0 %(ROW0)d        /* longest row = WDB + 1       */
#define PERCORE %(PERCORE)d  /* u32 of shared per core      */
#define BT   %(BT)d          /* binomial table dimension    */

__constant__ int ROWOFF[DYC + 2] = {%(ROWOFF)s};

/* row b of candidate j */
#define CROW(cand, j, b) ((cand) + (j) * CANDSZ + ROWOFF[b])
#define ROWLEN(b) (ROWOFF[(b) + 1] - ROWOFF[(b)])

__device__ __forceinline__ unsigned int powmod_d(
    unsigned int b, unsigned int e, unsigned int p, unsigned long long minv)
{
    unsigned int r = 1;
    while (e) {
        if (e & 1) r = mulmod(r, b, p, minv);
        b = mulmod(b, b, p, minv);
        e >>= 1;
    }
    return r;
}

__device__ __forceinline__ unsigned int invmod_d(
    unsigned int a, unsigned int p, unsigned long long minv)
{
    return powmod_d(a, p - 2, p, minv);
}

/* ---------------- univariate helpers, lane-serial ----------------
   Only the descent's root finding uses these, and only on a degree
   <= DYC column: measured, one such node per core against ~seven
   degree-<= 1 nodes, so this is ~1% of the work and stays serial. */

#define UMAX (DYC + 2)

__device__ int u_deg(const unsigned int* f, int n)
{
    for (int i = n - 1; i >= 0; i--) if (f[i]) return i;
    return -1;
}

/* a <- a mod b; returns deg a */
__device__ int u_rem(unsigned int* a, int da, const unsigned int* b, int db,
                     unsigned int p, unsigned long long minv)
{
    unsigned int li = invmod_d(b[db], p, minv);
    while (da >= db) {
        unsigned int c = mulmod(a[da], li, p, minv);
        if (c)
            for (int i = 0; i <= db; i++)
                a[da - db + i] = submod(a[da - db + i], mulmod(c, b[i], p, minv), p);
        a[da] = 0;
        da--;
        while (da >= 0 && !a[da]) da--;
    }
    return da;
}

/* g <- monic gcd(a, b); returns deg g */
__device__ int u_gcd(unsigned int* g, const unsigned int* a, int da,
                     const unsigned int* b, int db,
                     unsigned int p, unsigned long long minv)
{
    unsigned int u[UMAX], v[UMAX];
    for (int i = 0; i < UMAX; i++) { u[i] = (i <= da) ? a[i] : 0u;
                                     v[i] = (i <= db) ? b[i] : 0u; }
    int du = da, dv = db;
    while (dv >= 0) {
        du = u_rem(u, du, v, dv, p, minv);
        for (int i = 0; i < UMAX; i++) { unsigned int tmp = u[i]; u[i] = v[i]; v[i] = tmp; }
        int t = du; du = dv; dv = t;
    }
    if (du >= 0) {
        unsigned int li = invmod_d(u[du], p, minv);
        for (int i = 0; i <= du; i++) u[i] = mulmod(u[i], li, p, minv);
    }
    for (int i = 0; i < UMAX; i++) g[i] = u[i];
    return du;
}

/* acc <- base^e mod f */
__device__ int u_powrem(unsigned int* acc, const unsigned int* base0, int db0,
                        unsigned int e, const unsigned int* f, int df,
                        unsigned int p, unsigned long long minv)
{
    unsigned int b[UMAX], t[2 * UMAX];
    for (int i = 0; i < UMAX; i++) { acc[i] = 0; b[i] = (i <= db0) ? base0[i] : 0u; }
    acc[0] = 1;
    int dacc = 0, dbb = u_deg(b, UMAX);
    dbb = (dbb < 0) ? -1 : u_rem(b, dbb, f, df, p, minv);
    while (e) {
        if (e & 1) {
            for (int i = 0; i < 2 * UMAX; i++) t[i] = 0;
            for (int i = 0; i <= dacc; i++)
                if (acc[i])
                    for (int j = 0; j <= dbb; j++)
                        t[i + j] = addmod(t[i + j], mulmod(acc[i], b[j], p, minv), p);
            int dt = u_deg(t, 2 * UMAX);
            dt = (dt < 0) ? -1 : u_rem(t, dt, f, df, p, minv);
            for (int i = 0; i < UMAX; i++) acc[i] = (i <= dt) ? t[i] : 0u;
            dacc = (dt < 0) ? -1 : dt;
        }
        for (int i = 0; i < 2 * UMAX; i++) t[i] = 0;
        for (int i = 0; i <= dbb; i++)
            if (b[i])
                for (int j = 0; j <= dbb; j++)
                    t[i + j] = addmod(t[i + j], mulmod(b[i], b[j], p, minv), p);
        int dt2 = u_deg(t, 2 * UMAX);
        dt2 = (dt2 < 0) ? -1 : u_rem(t, dt2, f, df, p, minv);
        for (int i = 0; i < UMAX; i++) b[i] = (i <= dt2) ? t[i] : 0u;
        dbb = (dt2 < 0) ? -1 : dt2;
        e >>= 1;
    }
    return dacc;
}

/* All roots of f in F_p (f nonzero, deg <= DYC). Returns the count;
   roots land in `out`. Mirrors vanish::math::poly::roots: split off
   the distinct linear factors with gcd(f, x^p - x), then separate by
   Cantor-Zassenhaus on the deterministic shifts 1, 2, 3, ... */
__device__ int u_roots(unsigned int* out, const unsigned int* f0, int df0,
                       unsigned int p, unsigned long long minv)
{
    unsigned int f[UMAX];
    for (int i = 0; i < UMAX; i++) f[i] = (i <= df0) ? f0[i] : 0u;
    int df = df0, n = 0;
    if (df > 0 && f[0] == 0) {              /* factor out x */
        out[n++] = 0;
        int low = 0;
        while (low < df && !f[low]) low++;
        for (int i = 0; i + low < UMAX; i++) f[i] = f[i + low];
        for (int i = UMAX - low; i < UMAX; i++) f[i] = 0;
        df -= low;
    }
    if (df <= 0) return n;
    if (df == 1) {                           /* fast path: the common case */
        out[n++] = mulmod(p - f[0] % p, invmod_d(f[1], p, minv), p, minv);
        return n;
    }
    unsigned int xp[UMAX], g[UMAX], xx[2];
    xx[0] = 0; xx[1] = 1;
    int dxp = u_powrem(xp, xx, 1, p, f, df, p, minv);
    if (dxp < 1) { xp[1] = 0; dxp = 1; }
    xp[1] = submod(xp[1], 1u, p);
    int dg = u_gcd(g, f, df, xp, u_deg(xp, UMAX) < 0 ? 0 : u_deg(xp, UMAX), p, minv);
    if (dg < 1) return n;

    /* explicit stack of squarefree products of distinct linear factors */
    unsigned int st[DYC + 1][UMAX];
    int sd[DYC + 1], sp = 0;
    for (int i = 0; i < UMAX; i++) st[0][i] = g[i];
    sd[0] = dg; sp = 1;
    unsigned int shift = 1;
    int guard = 0;
    while (sp > 0 && guard++ < 4096) {
        sp--;
        unsigned int h[UMAX];
        for (int i = 0; i < UMAX; i++) h[i] = st[sp][i];
        int dh = sd[sp];
        if (dh <= 0) continue;
        if (dh == 1) {
            out[n++] = mulmod(p - h[0] % p, invmod_d(h[1], p, minv), p, minv);
            continue;
        }
        int split = 0;
        while (!split && guard++ < 4096) {
            unsigned int half[UMAX], sx[2];
            sx[0] = shift % p; sx[1] = 1;
            shift++;
            int dhalf = u_powrem(half, sx, 1, (p - 1) >> 1, h, dh, p, minv);
            if (dhalf < 0) { half[0] = 0; dhalf = 0; }
            half[0] = submod(half[0], 1u, p);
            dhalf = u_deg(half, UMAX);
            if (dhalf < 0) continue;         /* this shift squares everywhere */
            unsigned int dvp[UMAX];
            int ddv = u_gcd(dvp, h, dh, half, dhalf, p, minv);
            if (ddv >= 1 && ddv < dh) {
                /* h / dvp, exactly */
                unsigned int rest[UMAX], quo[UMAX];
                for (int i = 0; i < UMAX; i++) { rest[i] = h[i]; quo[i] = 0; }
                int dr = dh;
                unsigned int li = invmod_d(dvp[ddv], p, minv);
                while (dr >= ddv) {
                    unsigned int c = mulmod(rest[dr], li, p, minv);
                    quo[dr - ddv] = c;
                    if (c)
                        for (int i = 0; i <= ddv; i++)
                            rest[dr - ddv + i] = submod(rest[dr - ddv + i],
                                                        mulmod(c, dvp[i], p, minv), p);
                    rest[dr] = 0; dr--;
                    while (dr >= 0 && !rest[dr]) dr--;
                }
                for (int i = 0; i < UMAX; i++) st[sp][i] = dvp[i];
                sd[sp] = ddv; sp++;
                for (int i = 0; i < UMAX; i++) st[sp][i] = quo[i];
                sd[sp] = u_deg(quo, UMAX); sp++;
                split = 1;
            }
        }
    }
    return n;
}

/* ------------------------- the kernel ------------------------- */

extern "C" __global__ void core_residual(
    const unsigned int p, const unsigned long long minv,
    const unsigned int* __restrict__ dom,      /* [SC] fiber-major     */
    const unsigned int* __restrict__ word,     /* [SC]                 */
    const unsigned long long* __restrict__ binom,  /* unranking table  */
    const unsigned int* __restrict__ binmod,   /* [(BT+1)^2] mod p     */
    const long long base, const long long total,
    unsigned int* out_vals, int* out_count, const int cap,
    long long* surv, int* surv_count, const int scap)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const long long tid = base + (long long)blockIdx.x * WARPS + warp;
    if (tid >= total) return;

    extern __shared__ unsigned int smem[];
    unsigned int* S    = smem + (size_t)warp * PERCORE;
    unsigned int* cand = S;                            /* (DYC+1)*CANDSZ */
    unsigned int* qsel = cand + (DYC + 1) * CANDSZ;    /* CANDSZ          */
    unsigned int* work = qsel + CANDSZ;                /* CANDSZ          */
    unsigned int* xpow = work + CANDSZ;                /* ROW0            */
    unsigned int* cxs  = xpow + ROW0;                  /* 2*LC            */
    unsigned int* cys  = cxs + 2 * LC;
    unsigned int* cwt  = cys + 2 * LC;
    unsigned int* fxs  = cwt + 2 * LC;                 /* NR              */
    unsigned int* ftg  = fxs + NR;                     /* NR              */
    unsigned int* vfr  = ftg + NR;                     /* NR              */
    int*          ci   = (int*)(vfr + NR);             /* LC              */
    int*          wdeg = ci + LC;                      /* DYC+1           */
    unsigned int* delt = (unsigned int*)(wdeg + DYC + 1);  /* DYC+1       */
    unsigned int* alph = delt + DYC + 1;               /* KR              */
    unsigned int* memb = alph + KR;                    /* SC              */
    unsigned int* rts  = memb + SC;                    /* UMAX            */
    int*          flag = (int*)(rts + UMAX);           /* 4               */

    /* ---- 1. unrank the core (colex, the Rust index space) ---- */
    if (lane == 0) {
        long long idx = tid;
        int rem = LC, top = NF;
        int pos = LC;
        while (rem > 0) {
            top--;
            unsigned long long c = binom[(long long)top * (LC + 1) + rem];
            if ((unsigned long long)idx >= c) { ci[--pos] = top; idx -= c; rem--; }
        }
        flag[0] = 0;   /* survivor flag */
    }
    __syncwarp();

    /* ---- 2. core points, and the barycentric weights of the word
            interpolant q on them (q is never expanded: every use is
            an evaluation) ---- */
    for (int a = lane; a < 2 * LC; a += LANES) {
        int f = ci[a >> 1];
        int j = (a & 1) ? f + NF : f;
        cxs[a] = dom[j];
        cys[a] = word[j];
    }
    __syncwarp();
    for (int a = lane; a < 2 * LC; a += LANES) {
        unsigned int den = 1;
        for (int b = 0; b < 2 * LC; b++)
            if (b != a) den = mulmod(den, submod(cxs[a], cxs[b], p), p, minv);
        cwt[a] = invmod_d(den, p, minv);
    }
    __syncwarp();

    /* ---- 3. free points, V_Y there, and the residual targets ---- */
    for (int a = lane; a < NR; a += LANES) {
        int fib = a >> 1, seen = 0, f = -1;
        /* the (a/2)-th fiber not in the core */
        for (int i = 0; i < NF; i++) {
            int inc = 0;
            for (int c = 0; c < LC; c++) if (ci[c] == i) { inc = 1; break; }
            if (!inc) { if (seen == fib) { f = i; break; } seen++; }
        }
        int j = (a & 1) ? f + NF : f;
        unsigned int x = dom[j];
        fxs[a] = x;
        unsigned int x2 = mulmod(x, x, p, minv), v = 1;
        for (int c = 0; c < LC; c++) {
            unsigned int y = mulmod(dom[ci[c]], dom[ci[c]], p, minv);
            v = mulmod(v, submod(x2, y, p), p, minv);
        }
        vfr[a] = v;
        /* q(x) barycentrically, then the target (w - q)/V_Y */
        unsigned int num = 0, den = 0;
        for (int b = 0; b < 2 * LC; b++) {
            unsigned int u = mulmod(cwt[b], invmod_d(submod(x, cxs[b], p), p, minv), p, minv);
            num = addmod(num, mulmod(u, cys[b], p, minv), p);
            den = addmod(den, u, p);
        }
        unsigned int qx = mulmod(num, invmod_d(den, p, minv), p, minv);
        ftg[a] = mulmod(submod(word[j], qx, p), invmod_d(v, p, minv), p, minv);
    }
    __syncwarp();

    /* ---- 4. Koetter: candidates start at y^j and stay a basis of
            the module cut out by the constraints seen so far ---- */
    for (int i = lane; i < (DYC + 1) * CANDSZ; i += LANES) cand[i] = 0;
    __syncwarp();
    if (lane <= DYC) {
        CROW(cand, lane, lane)[0] = 1;
        wdeg[lane] = WC * lane;
    }
    __syncwarp();

    for (int pt = 0; pt < NR; pt++) {
        unsigned int x = fxs[pt], y = ftg[pt];
        for (int a = lane; a < ROW0; a += LANES) xpow[a] = powmod_d(x, a, p, minv);
        __syncwarp();
        for (int r = 0; r < MC; r++) {
            for (int sh = 0; sh < MC - r; sh++) {
                /* discrepancies: one candidate per lane group */
                for (int j = 0; j <= DYC; j++) {
                    unsigned int acc = 0;
                    for (int b = sh; b <= DYC; b++) {
                        unsigned int cb = binmod[(long long)b * (BT + 1) + sh];
                        if (!cb) continue;
                        unsigned int yp = powmod_d(y, b - sh, p, minv);
                        unsigned int* row = CROW(cand, j, b);
                        int len = ROWLEN(b);
                        for (int a = r + lane; a < len; a += LANES) {
                            unsigned int c = row[a];
                            if (!c) continue;
                            unsigned int co = mulmod(binmod[(long long)a * (BT + 1) + r],
                                                     cb, p, minv);
                            acc = addmod(acc, mulmod(mulmod(co, c, p, minv),
                                                     mulmod(xpow[a - r], yp, p, minv),
                                                     p, minv), p);
                        }
                    }
                    for (int o = 16; o; o >>= 1) {
                        unsigned int v = __shfl_down_sync(0xffffffffu, acc, o);
                        acc = addmod(acc, v, p);
                    }
                    if (lane == 0) delt[j] = acc;
                }
                __syncwarp();

                /* pivot: least weighted degree among nonzero discrepancies,
                   ties to the lower index -- the mirror's rule */
                int star = -1;
                if (lane == 0) {
                    for (int j = 0; j <= DYC; j++)
                        if (delt[j] && (star < 0 || wdeg[j] < wdeg[star])) star = j;
                    flag[1] = star;
                }
                __syncwarp();
                star = flag[1];
                if (star < 0) continue;

                unsigned int istar = invmod_d(delt[star], p, minv);
                /* every other live candidate loses its multiple of the pivot */
                for (int j = 0; j <= DYC; j++) {
                    if (j == star || !delt[j]) continue;
                    unsigned int c = mulmod(delt[j], istar, p, minv);
                    for (int i = lane; i < CANDSZ; i += LANES) {
                        unsigned int pv = cand[star * CANDSZ + i];
                        if (pv) cand[j * CANDSZ + i] =
                            submod(cand[j * CANDSZ + i], mulmod(c, pv, p, minv), p);
                    }
                }
                __syncwarp();
                /* the pivot is multiplied by (x - x_pt): shift up one and
                   subtract x_pt times itself, per row, high to low */
                for (int b = 0; b <= DYC; b++) {
                    unsigned int* row = CROW(cand, star, b);
                    int len = ROWLEN(b);
                    for (int a = len - 1 - lane; a >= 0; a -= LANES) {
                        unsigned int hi = (a >= 1) ? row[a - 1] : 0u;
                        unsigned int lo = mulmod(row[a], x, p, minv);
                        row[a] = submod(hi, lo, p);
                    }
                    __syncwarp();
                }
                if (lane == 0) {
                    wdeg[star] += 1;
                    if (wdeg[star] > WDB) flag[0] = 1;   /* budget exceeded */
                }
                __syncwarp();
                if (flag[0]) {
                    if (lane == 0) {
                        int slot = atomicAdd(surv_count, 1);
                        if (slot < scap) surv[slot] = tid;
                    }
                    return;
                }
            }
        }
    }

    /* ---- 5. the minimal candidate is the interpolant ---- */
    if (lane == 0) {
        int best = 0;
        for (int j = 1; j <= DYC; j++) if (wdeg[j] < wdeg[best]) best = j;
        flag[1] = best;
    }
    __syncwarp();
    for (int i = lane; i < CANDSZ; i += LANES) qsel[i] = cand[flag[1] * CANDSZ + i];
    __syncwarp();

    /* ---- 6. Roth-Ruckenstein: descend on the roots of Q(0, y),
            replaying from qsel on backtrack (branching is rare, so
            replay costs less than a stack of bivariates) ---- */
    int depth = 0, branch[KR];
    for (int i = 0; i < KR; i++) branch[i] = 0;
    while (depth >= 0) {
        /* rebuild `work` = Q after the alphas alph[0..depth-1] */
        for (int i = lane; i < CANDSZ; i += LANES) work[i] = qsel[i];
        __syncwarp();
        for (int lvl = 0; lvl < depth; lvl++) {
            /* strip the common power of x */
            if (lane == 0) {
                int low = ROW0;
                for (int b = 0; b <= DYC; b++) {
                    unsigned int* row = work + ROWOFF[b];
                    int len = ROWLEN(b), f = len;
                    for (int a = 0; a < len; a++) if (row[a]) { f = a; break; }
                    if (f < low) low = f;
                }
                flag[2] = (low == ROW0) ? 0 : low;
            }
            __syncwarp();
            int low = flag[2];
            if (low > 0) {
                for (int b = 0; b <= DYC; b++) {
                    unsigned int* row = work + ROWOFF[b];
                    int len = ROWLEN(b);
                    for (int a = lane; a < len; a += LANES)
                        row[a] = (a + low < len) ? row[a + low] : 0u;
                    __syncwarp();
                }
            }
            /* work <- work(x, x y + alpha) */
            unsigned int al = alph[lvl];
            for (int j = DYC; j >= 0; j--) {
                for (int a = lane; a < ROW0; a += LANES) xpow[a] = 0;
                __syncwarp();
                for (int b = j; b <= DYC; b++) {
                    unsigned int c = mulmod(binmod[(long long)b * (BT + 1) + j],
                                            powmod_d(al, b - j, p, minv), p, minv);
                    if (!c) continue;
                    unsigned int* row = work + ROWOFF[b];
                    int len = ROWLEN(b);
                    for (int a = lane; a < len; a += LANES)
                        xpow[a] = addmod(xpow[a], mulmod(c, row[a], p, minv), p);
                    __syncwarp();
                }
                unsigned int* dst = work + ROWOFF[j];
                int dlen = ROWLEN(j);
                for (int a = dlen - 1 - lane; a >= 0; a -= LANES)
                    dst[a] = (a >= j) ? xpow[a - j] : 0u;
                __syncwarp();
            }
        }

        /* the branches here: roots of the constant column */
        int nb = 0;
        if (lane == 0) {
            unsigned int col[UMAX], rt[UMAX];
            for (int b = 0; b <= DYC; b++) col[b] = work[ROWOFF[b]];
            for (int b = DYC + 1; b < UMAX; b++) col[b] = 0;
            int dc = u_deg(col, UMAX);
            if (dc < 0) { rt[0] = 0; nb = 1; }
            else nb = u_roots(rt, col, dc, p, minv);
            flag[3] = nb;
            for (int i = 0; i < nb && i < UMAX; i++) rts[i] = rt[i];
        }
        __syncwarp();
        nb = flag[3];

        if (branch[depth] >= nb) {        /* exhausted: back up */
            branch[depth] = 0;
            depth--;
            if (depth >= 0) branch[depth]++;
            continue;
        }
        if (lane == 0) alph[depth] = rts[branch[depth]];
        __syncwarp();

        if (depth == KR - 1) {
            /* a complete g: reassemble f = q_Y + V_Y g and verify */
            unsigned int agree = 0;
            for (int j = lane; j < SC; j += LANES) {
                unsigned int x = dom[j];
                unsigned int num = 0, den = 0;
                int oncore = 0;
                for (int b = 0; b < 2 * LC; b++) if (cxs[b] == x) { oncore = 1; num = cys[b]; }
                unsigned int qx;
                if (oncore) qx = num;
                else {
                    for (int b = 0; b < 2 * LC; b++) {
                        unsigned int u = mulmod(cwt[b],
                                                invmod_d(submod(x, cxs[b], p), p, minv),
                                                p, minv);
                        num = addmod(num, mulmod(u, cys[b], p, minv), p);
                        den = addmod(den, u, p);
                    }
                    qx = mulmod(num, invmod_d(den, p, minv), p, minv);
                }
                unsigned int x2 = mulmod(x, x, p, minv), v = 1;
                for (int c = 0; c < LC; c++) {
                    unsigned int yy = mulmod(dom[ci[c]], dom[ci[c]], p, minv);
                    v = mulmod(v, submod(x2, yy, p), p, minv);
                }
                unsigned int g = 0;
                for (int a = KR - 1; a >= 0; a--) g = addmod(mulmod(g, x, p, minv), alph[a], p);
                unsigned int f = addmod(qx, mulmod(v, g, p, minv), p);
                memb[j] = f;
                agree += (f == word[j]) ? 1u : 0u;
            }
            for (int o = 16; o; o >>= 1) agree += __shfl_down_sync(0xffffffffu, agree, o);
            if (lane == 0) {
                if (agree >= TC) {
                    int slot = atomicAdd(out_count, 1);
                    if (slot < cap)
                        for (int j = 0; j < SC; j++)
                            out_vals[(long long)slot * SC + j] = memb[j];
                }
            }
            __syncwarp();
            branch[depth]++;
        } else {
            depth++;
            branch[depth] = 0;
        }
    }
}
"""


def kernel_layout(s, k, t, warps=4, wdb=None):
    """The shared-memory plan for the (s, k, t) kernel.

    `wdb` is the weighted-degree budget the triangular row allocation
    is cut to. Its default is twice the average candidate degree --
    the total is conserved (the starting basis contributes
    w dy(dy+1)/2 and each constraint exactly one more), so twice the
    mean is generous: at (42, 9, 21) it gives 84 against a measured
    maximum of 74 over 120 cores. Cores that exceed it are not decoded
    on the device; they are emitted as survivors and finished by the
    Rust decoder, so the budget trades device throughput for host
    work, never correctness."""
    l, nr, kr, tr, m, d, dy = residual_cell(s, k, t)
    w = kr - 1
    cons = nr * m * (m + 1) // 2
    if wdb is None:
        total_deg = w * dy * (dy + 1) // 2 + cons
        wdb = -(-2 * total_deg // (dy + 1))
    rowoff, off = [], 0
    for b in range(dy + 2):
        rowoff.append(off)
        if b <= dy:
            off += max(wdb - w * b + 1, 1)
    candsz = rowoff[-1]
    row0 = max(wdb + 1, dy + 2)
    umax = dy + 2
    percore = ((dy + 1) * candsz    # the Koetter candidates
               + 2 * candsz         # the chosen interpolant and the descent's
               + row0               # x powers / row scratch
               + 6 * l              # core points, values, weights
               + 3 * nr             # free points, targets, V_Y
               + l + 2 * (dy + 1) + kr + s + umax + 4)
    return dict(WARPS=warps, SC=s, NF=s // 2, KC=k, TC=t, LC=l, NR=nr, KR=kr,
                TR=tr, MC=m, DYC=dy, WC=w, WDB=wdb, CANDSZ=candsz, ROW0=row0,
                PERCORE=percore, BT=row0,
                ROWOFF=", ".join(str(o) for o in rowoff))


def plan_launch(s, k, t, shared_limit_kb=96, wdb=None):
    """Warps per block that fit the device's shared memory, and the
    per-core cost. Returns (warps, per_core_kb, fits).

    (64, 31, 42) does not fit at any warp count with the default
    budget: dy = 13 and m = 6 make a candidate 2772 u32, so the
    fourteen of them want 175 KB against a 96-164 KB ceiling. That
    cell needs a tighter budget (paying in survivors) or an
    interpolation that does not hold every candidate at once -- it is
    not merely the slower cell, as its core count suggests, but a
    different problem. (64, 31, 43) fits comfortably."""
    cfg = kernel_layout(s, k, t, 1, wdb)
    per_core_kb = cfg["PERCORE"] * 4 / 1024
    warps = 1
    while warps * 2 * per_core_kb <= shared_limit_kb and warps < 8:
        warps *= 2
    return warps, per_core_kb, per_core_kb <= shared_limit_kb


_KERNEL_CACHE = {}


def _kernel(s, k, t, warps=4, wdb=None):
    """Compile (or fetch) the kernel specialised to (s, k, t). The
    cell's constants are substituted into the source, so every loop
    bound is a compile-time constant -- the house pattern of
    decode_gpu.py, for the same reason."""
    import cupy as cp
    from decode_gpu import _MODMATH
    key = (s, k, t, warps, wdb)
    if key not in _KERNEL_CACHE:
        cfg = kernel_layout(s, k, t, warps, wdb)
        src = _MODMATH + (_KERNEL % cfg)
        kern = cp.RawKernel(src, "core_residual")
        shared = cfg["PERCORE"] * warps * 4
        if shared > 48 * 1024:
            kern.max_dynamic_shared_size_bytes = shared
        _KERNEL_CACHE[key] = (kern, cfg, shared)
    return _KERNEL_CACHE[key]


def gpu_list_paired(p, points, k, word, t, cores=None, warps=4, wdb=None,
                    cap=1 << 20, tile=1 << 22, verbose=False):
    """The exact list at (s, k, t) by the GPU core sweep, over `cores`
    (default all). Survivors -- cores the device declined on the
    degree budget -- are finished by the Rust decoder, and every
    member the device emits is re-verified there too, so the result is
    exact whatever the kernel did."""
    import cupy as cp
    import numpy as np
    s = len(points)
    l, nr, kr, tr, m, d, dy = residual_cell(s, k, t)
    kern, cfg, shared = _kernel(s, k, t, warps, wdb)
    total = math.comb(s // 2, l)
    lo, hi = (0, total) if cores is None else (cores.start, cores.stop)

    dom_d = cp.asarray(points, dtype=cp.uint32)
    word_d = cp.asarray(word, dtype=cp.uint32)
    nb = np.zeros((s // 2 + 1) * (l + 1), dtype=np.uint64)
    for a in range(s // 2 + 1):
        for b in range(min(a, l) + 1):
            nb[a * (l + 1) + b] = math.comb(a, b)
    binom_d = cp.asarray(nb)
    bt = cfg["BT"]
    bm = np.zeros((bt + 1) * (bt + 1), dtype=np.uint32)
    for a in range(bt + 1):
        bm[a * (bt + 1)] = 1
        for b in range(1, a + 1):
            bm[a * (bt + 1) + b] = (int(bm[(a - 1) * (bt + 1) + b - 1])
                                    + int(bm[(a - 1) * (bt + 1) + b])) % p
    binmod_d = cp.asarray(bm)
    minv = np.uint64((1 << 64) // int(p))

    out = cp.zeros(cap * s, dtype=cp.uint32)
    ocount = cp.zeros(1, dtype=cp.int32)
    scap = 1 << 16
    surv = cp.zeros(scap, dtype=cp.int64)
    scount = cp.zeros(1, dtype=cp.int32)
    for base in range(lo, hi, tile):
        span = min(tile, hi - base)
        blocks = (span + warps - 1) // warps
        kern((blocks,), (warps * 32,),
             (np.uint32(p), minv, dom_d, word_d, binom_d, binmod_d,
              np.int64(base), np.int64(hi), out, ocount, np.int32(cap),
              surv, scount, np.int32(scap)),
             shared_mem=shared)
    cp.cuda.Stream.null.synchronize()

    n_out, n_surv = int(ocount[0]), int(scount[0])
    if n_out > cap:
        raise RuntimeError(f"member buffer overflowed ({n_out} > {cap}); raise cap")
    if n_surv > scap:
        raise RuntimeError(f"survivor buffer overflowed ({n_surv} > {scap})")
    members = {tuple(int(v) for v in row)
               for row in cp.asnumpy(out[:n_out * s]).reshape(n_out, s)}
    if verbose:
        print(f"  device: {n_out} raw members, {n_surv} survivors "
              f"({100.0 * n_surv / max(hi - lo, 1):.3f}% of cores)")
    # survivors and verification both go through the Rust authority
    for idx in cp.asnumpy(surv[:n_surv]).tolist():
        members.update(
            tuple(int(v) for v in row)
            for row in vanish.list_decode_paired_range(p, points, k, word, t,
                                                       int(idx), int(idx) + 1))
    verified = [f for f in members
                if sum(a == b for a, b in zip(f, word)) >= t]
    return sorted(verified)


def validate(verbose=True):
    """The GPU gate: the kernel equals the mirror and the Rust decoder
    on every gate cell and word. Run on the pod before any s = 64
    number is quoted."""
    ok = True
    cells = [(65537, 16, 7, 10), (65537, 16, 5, 9), (97, 16, 7, 10),
             (65537, 32, 15, 21)]
    for p, s, k, t in cells:
        points, words = _battery(p, s, k, t)
        cfg = kernel_layout(s, k, t)
        if verbose:
            print(f"({s},{k},{t}) p={p}: wdb={cfg['WDB']} candsz={cfg['CANDSZ']} "
                  f"shared/core={cfg['PERCORE'] * 4 / 1024:.1f} KB")
        for name, w, must in words:
            got = gpu_list_paired(p, points, k, w, t, verbose=verbose)
            truth = sorted(tuple(int(v) for v in row)
                           for row in vanish.list_decode_paired(p, points, k, w, t))
            mirror = sorted(list_paired_mirror(p, points, k, w, t))
            missing = [c for c in must if tuple(c) not in set(got)]
            good = got == truth == mirror and not missing
            ok &= good
            if verbose:
                note = f" [{len(missing)} plants MISSING]" if missing else ""
                print(f"  {'PASS' if good else 'FAIL'}  ({s},{k},{t}) p={p} "
                      f"{name}: gpu == mirror == rust ({len(got)}){note}")
    print("VALIDATE:", "PASS" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------------------
# Gates


def _battery(p, s, k, t):
    """Gate words. Random and codeword words are the easy path; the
    plants are the load-bearing ones (rule 9b -- no empty list is
    believed without a plant that must be found), and `two_plant`
    forces a multi-member list, so the branching descent is exercised
    rather than just the single-root path that random words take.

    Returns (points, words) with words as (name, word, must_contain)."""
    points = paired_domain(p, s)
    state = 88172645463325252

    def rnd():
        nonlocal state
        state ^= (state << 13) % (1 << 64)
        state ^= state >> 7
        state ^= (state << 17) % (1 << 64)
        return state % p

    words = [(f"random{i}", [rnd() for _ in range(s)], []) for i in range(2)]
    msg = [rnd() for _ in range(k)]
    cw = [_horner(msg, x, p) for x in points]
    words.append(("codeword", list(cw), [cw]))
    near = list(cw)
    near[3] = (near[3] + 1) % p
    words.append(("near", near, [cw]))
    planted = list(cw)
    for i in range(s - t):
        planted[i] = (planted[i] + 1 + i) % p
    words.append(("planted", planted, [cw]))
    # the structured word of the campaign: top, negated at the real pair
    top = [(pow(x, k, p) + pow(x, s - 1, p)) % p for x in points]
    flip = [(p - v) % p if points[i] in (1, p - 1) else v
            for i, v in enumerate(top)]
    words.append(("flip", flip, []))
    words.append(_two_plant(p, points, k, t, rnd))
    return points, words


def _two_plant(p, points, k, t, rnd):
    """A word carrying two codewords at agreement t, so the list has
    at least two members.

    A and B = A + D differ by the nonzero D of degree k - 1 vanishing
    on the first k - 1 points -- the most two distinct codewords can
    share. The word takes their common value there and then splits the
    remaining points between them, which fits because
    2t - k + 1 <= s at every gate cell."""
    s = len(points)
    shared = list(range(k - 1))
    d_poly = [1]
    for i in shared:
        d_poly = _mul(d_poly, [(-points[i]) % p, 1], p)
    a_msg = [rnd() for _ in range(k)]
    a = [_horner(a_msg, x, p) for x in points]
    b = [(a[i] + _horner(d_poly, points[i], p)) % p for i in range(s)]
    need = t - len(shared)
    rest = [i for i in range(s) if i not in shared]
    assert len(rest) >= 2 * need, "two_plant does not fit this cell"
    w = [rnd() for _ in range(s)]
    for i in shared:
        w[i] = a[i]
    for i in rest[:need]:
        w[i] = a[i]
    for i in rest[need:2 * need]:
        w[i] = b[i]
    return "two_plant", w, [a, b]


def selfcheck(verbose=True):
    """The mirror equals the Rust decoder on every gate cell, word and
    prime. No GPU needed; this is the algorithm gate."""
    ok = True
    # the small cells at both primes, then the battery cell itself at
    # the battery prime -- (32, 15, 21) is where the campaign's
    # equality is asserted, so the mirror is held to it too
    cells = [(p, s, k, t) for p in (65537, 97) for s, k, t in ((16, 7, 10), (16, 5, 9))]
    cells.append((65537, 32, 15, 21))
    for p, s, k, t in cells:
        points, words = _battery(p, s, k, t)
        for name, w, must in words:
            mine = sorted(list_paired_mirror(p, points, k, w, t))
            truth = sorted(tuple(int(v) for v in row)
                           for row in vanish.list_decode_paired(p, points, k, w, t))
            missing = [c for c in must if tuple(c) not in set(mine)]
            good = mine == truth and not missing
            # the re-encoded interpolation must reach the same list;
            # only on the small cells, where it is cheap
            if s == 16:
                recoded = sorted(list_paired_mirror(p, points, k, w, t,
                                                    reencode=True))
                if recoded != mine:
                    good = False
                    print(f"  FAIL  ({s},{k},{t}) p={p} {name}: "
                          f"reencode {len(recoded)} != plain {len(mine)}")
            ok &= good
            if verbose:
                note = f" [{len(missing)} plants MISSING]" if missing else ""
                print(f"  {'PASS' if good else 'FAIL'}  ({s},{k},{t}) p={p} "
                      f"{name}: mirror == rust ({len(mine)} members){note}")
    print("SELFCHECK:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tabulate", action="store_true",
                    help="print the cell shape table")
    ap.add_argument("--selfcheck", action="store_true",
                    help="the mirror gate: mirror == vanish, any machine")
    ap.add_argument("--validate", action="store_true",
                    help="the GPU gate: kernel == mirror == vanish (needs CUDA)")
    ap.add_argument("--layout", action="store_true",
                    help="print the kernel's shared-memory plan per cell")
    a = ap.parse_args()
    if a.tabulate:
        tabulate()
    if a.layout:
        for cell in ((64, 31, 43), (64, 31, 42), (32, 15, 21), (16, 7, 10)):
            cfg = kernel_layout(*cell)
            warps, kb, fits = plan_launch(*cell)
            print(f"{str(cell):>14} wdb={cfg['WDB']:>4} candsz={cfg['CANDSZ']:>5} "
                  f"per-core={kb:>6.1f} KB  "
                  + (f"{warps} warps/block = {warps * kb:>5.1f} KB"
                     if fits else "DOES NOT FIT (see plan_launch)"))
    if a.selfcheck:
        raise SystemExit(0 if selfcheck() else 1)
    if a.validate:
        raise SystemExit(0 if validate() else 1)
    if not (a.tabulate or a.selfcheck or a.validate or a.layout):
        ap.print_help()


if __name__ == "__main__":
    main()
