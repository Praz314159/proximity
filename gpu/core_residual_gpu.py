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

Usage:
  python core_residual_gpu.py --tabulate    # cell shape table
  python core_residual_gpu.py --selfcheck   # the mirror gate
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
    a = ap.parse_args()
    if a.tabulate:
        tabulate()
    if a.selfcheck:
        raise SystemExit(0 if selfcheck() else 1)
    if not (a.tabulate or a.selfcheck):
        ap.print_help()


if __name__ == "__main__":
    main()
