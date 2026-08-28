"""Type stubs for the vanish native module."""

from typing import List, Tuple

import numpy as np
import numpy.typing as npt

def bucket_dist_q1(p: int, s: int, r: int) -> npt.NDArray[np.uint64]:
    """Full exact q=1 bucket distribution; out[lam] counts r-subsets with e_1 = lam."""

def bucket_dist_q2(p: int, s: int, r: int) -> npt.NDArray[np.uint64]:
    """Full exact q=2 joint distribution over (e_1, e_2), shape (p, p)."""

def census_direct(p: int, s: int, cmax: int, wmax: int) -> List[int]:
    """Weight-capped kernel census; out[w] = # nonzero kernel vectors of weight w."""

def census_mitm(p: int, s: int, cmax: int) -> List[int]:
    """Full kernel census by weight (meet-in-the-middle; s <= 32 at cmax = 2)."""

def bucket_e(p: int, s: int, r: int, lam: List[int]) -> int:
    """Exact single bucket at e-values lam (q = len(lam) <= 8, s <= 32); p-independent."""

def buckets_e(p: int, s: int, r: int, q: int, lams: List[List[int]]) -> npt.NDArray[np.uint64]:
    """Exact buckets for many lambdas sharing one table build."""

def rung_lambda_e(p: int, s: int, r: int, q: int) -> List[int]:
    """The common (e_1..e_q) of the Theorem-A rung family."""

def decompose_bucket_q1(p: int, s: int, r: int, lam: int) -> Tuple[int, List[int]]:
    """Anatomy of a q=1 bucket: (total, per-weight class counts); total = DP bucket."""

def m_struct(s: int, r: int, q: int) -> int:
    """Quantized-ladder structural maximum C(s/2^t - [r0!=0], floor(r/2^t))."""

def subgroup(p: int, s: int) -> List[int]:
    """Order-s subgroup of F_p^* as consecutive powers [w^0, ..., w^{s-1}]."""

def is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin for n < 2^64."""

def factor(n: int) -> List[int]:
    """Full prime factorization, sorted with multiplicity."""

def dist_stats_q1(p: int, s: int, r: int) -> Tuple[int, int, int, int, int]:
    """(max, argmax, occupied, total, exact second moment) for the q=1 distribution."""

def sweep_stats_q1(s: int, r: int, primes: List[int]) -> List[Tuple[int, int, int, int, int, int]]:
    """Parallel sweep: rows of (p, max, argmax, occupied, total, second_moment)."""

def certify_q1(p: int, s: int, r: int) -> Tuple[int, int, int]:
    """(tier, m_struct, zero_bucket): 1 = all structural, 2 = zero bucket structural, 3 = inflated."""

def primes_1_mod(s: int, lo: int, hi: int) -> List[int]:
    """Primes p = 1 (mod s) in [lo, hi)."""

def class_size(s: int, r: int, w: int) -> int:
    """Structural class size C(s/2 - w, (r - w)/2); 0 if infeasible."""

def decompose_many(s: int, r: int, primes: List[int]) -> List[Tuple[int, int, List[int]]]:
    """Parallel zero-bucket decompositions: rows of (p, total, per-weight counts)."""

def attack_best(n: int, k: int, list_bits: float) -> Tuple[float, float, int, int, int, float] | None:
    """Best ladder attack: (delta_star, deficit, t, s_g, r, log2_list)."""

def attack_antipodal(n: int, k: int, list_bits: float) -> Tuple[float, float, int, int, int, float] | None:
    """Antipodal (Table-5) baseline, same shape as attack_best."""

def attack_ceiling(n: int, k: int, list_bits: float) -> float:
    """Framework ceiling delta_min - H2(rate)/list_bits."""

def toy_soundness(p: int, s: int, r: int) -> Tuple[int, float, int]:
    """(winning challenges, exact soundness, structural class count)."""

def rung_buckets_many(s: int, r: int, qs: List[int], primes: List[int]) -> List[Tuple[int, List[int]]]:
    """Parallel rung-bucket sweep: rows of (p, [exact bucket per q])."""

def certify_many(s: int, r: int, primes: List[int]) -> List[Tuple[int, int, int, int]]:
    """Parallel certificates: rows of (p, tier, m_struct, zero_bucket)."""

def norms_bad_set(s: int, wmax: int, cmax: int) -> List[Tuple[int, List[int], bool]]:
    """Complete bad set: rows of (p, per-weight counts, census_fallback)."""

def norms_n_max(s: int, wmax: int, cmax: int) -> List[str]:
    """Per-weight maximum cyclotomic norms (decimal strings)."""

def badset_from_gpu_json(
    paths: List[str], s: int, wmax: int, out_prefix: str
) -> Tuple[int, List[int], List[int], int]:
    """Ingest GPU norm-table shards (JSON files or binary-dump prefixes) into a
    bad set. Writes <out_prefix>.{primes,counts,flags}.bin (all counts u64 le)
    and returns (n_rows, mass_by_weight, n_max_by_weight, entries_parsed);
    mass_by_weight must equal C(s/2, w) * 2^w when the shards are complete.
    Crash-safe: the accumulator checkpoints to <out_prefix>.ckpt.* after every
    completed shard, a rerun resumes from it automatically (re-factoring at
    most one shard), and the checkpoint is removed once outputs are written."""

def list_decode(p: int, domain: List[int], k: int, word: List[int], t: int) -> npt.NDArray[np.uint64]:
    """Exact list decode of RS[F_p, domain, k]: every codeword (evaluation
    vector) agreeing with `word` on >= t of the n = len(domain) coordinates.
    `domain` is any distinct-point set (e.g. subgroup(p, s)). Requires t >= k."""

def list_decode_paired(p: int, points: List[int], k: int, word: List[int], t: int) -> npt.NDArray[np.uint64]:
    """Exact list decode past the fiber count on a paired domain
    (points[i + n] = -points[i]): every codeword agreeing on >= t of the
    s = 2n points, via core enumeration + Guruswami-Sudan residual
    decodes. Needs t > n and (t - 2l)^2 > (s - 2l)(k - 2l - 1), l = t - n."""

def list_decode_paired_range(
    p: int, points: List[int], k: int, word: List[int], t: int, lo: int, hi: int
) -> npt.NDArray[np.uint64]:
    """One shard of list_decode_paired: the members through cores lo..hi of
    0..paired_core_count(..). The union over a partition of the full range,
    deduplicated, is the exact list — long sweeps run as resumable chunks."""

def paired_core_count(p: int, points: List[int], k: int, t: int) -> int:
    """The number of cores list_decode_paired enumerates at (k, t) — the
    index space of list_decode_paired_range. Errors when the cell is not
    decodable."""

def list_decode_paired_sampled(
    p: int, points: List[int], k: int, word: List[int], t: int, samples: int, seed: int
) -> npt.NDArray[np.uint64]:
    """Sampled lower bound of list_decode_paired: members found through
    `samples` uniform cores; a subset of the true list, deterministic in seed."""

def list_sizes(p: int, domain: List[int], k: int, words: List[List[int]], t: int) -> List[int]:
    """Exact distinct list sizes for a batch of words in one sweep:
    shared barycentric tables per information set, lex-first dedup,
    rayon inside; GIL released."""

def anneal_pencil(
    p: int, domain: List[int], k: int, t: int, petals: int, steps: int, seed: int
) -> Tuple[List[int], List[List[int]], List[int]]:
    """One code-first optimization run: random pencil seed annealed to maximize
    list size. Returns (center, members, size_trajectory). Loop over `seed` to
    collect a discovery dataset."""

def optimize_pencil(
    p: int, domain: List[int], k: int, t: int, petals: int, max_flips: int, seed: int
) -> Tuple[List[int], List[List[int]], List[int]]:
    """One code-first run to convergence: random pencil seed, greedily hill-climb
    boundary-alignment flips until no flip increases the list (a true local
    maximum). Returns (center, members, monotone size_trajectory)."""

def optimize_word(
    p: int, domain: List[int], k: int, t: int, word: List[int], max_flips: int
) -> Tuple[List[int], npt.NDArray[np.uint64], List[int]]:
    """Greedy list-size climb to convergence FROM `word` (warm start): returns
    (center, members as an (L, n) array, size_trajectory). Deterministic."""

def pencil_seed(p: int, domain: List[int], k: int, petals: int, seed: int) -> List[int]:
    """A random pencil seed word (random (k-1)-core + petal codewords); the
    unbiased code-first start. Deterministic in seed."""

SymRow = Tuple[int, float, int, float, int, List[Tuple[int, int]]]

def decode_profile(
    p: int, domain: List[int], k: int, word: List[int], t: int
) -> Tuple[
    npt.NDArray[np.uint64],
    List[Tuple[int, int]],
    float,
    List[SymRow],
    float,
    int,
]:
    """Decode + full structural profile: (members (L, n), agreement_sizes as
    [(size, count)], size_entropy, per-e_i stats (index, entropy_bits,
    distinct, max_class_fraction, mode_value, distribution), joint_entropy,
    joint_distinct). The canonical frozen-invariant probe."""

def c5_word(p: int, s: int, r: int, lam: List[int]) -> List[int]:
    """The additive frozen-head word sum_i (-1)^i lam_i x^{r-i} on mu_s
    (identifier kept for API stability)."""

def top_word(p: int, s: int, r: int, c: int) -> List[int]:
    """The proven multiplicative extremal word x^{r-1} - (-1)^{r+1} zeta^c
    x^{s-1} (Theorem B_mult): exact list = the Graham-Sloane class of c."""

def word_from_syndrome(p: int, domain: List[int], r: int, b: List[int]) -> List[int]:
    """w = sum_j (-1)^j b_j x^{r-1+j}, pinned to D_S(w) = <b, e(complement)>."""

def gs_class_counts(s: int, t: int) -> npt.NDArray[np.uint64]:
    """Graham-Sloane class counts N_c(s, t) for c = 0..s-1 (DP, exact)."""

def moment_cloud(p: int, domain: List[int], r: int) -> npt.NDArray[np.uint64]:
    """(C(n,r), n-r+1) array: row S (lex over index subsets) = (e_0..e_{n-r})
    of the COMPLEMENT of S — the fixed point cloud of the syndrome layer."""

def cut_counts(p: int, domain: List[int], r: int, bs: List[List[int]]) -> npt.NDArray[np.uint64]:
    """|Z(b)| for many syndromes at once (streaming; convention
    D_S(w) = sum_j b_j e_j(complement))."""

def cut_max_sparse(p: int, domain: List[int], r: int, support: List[int]) -> Tuple[int, List[int]]:
    """Exhaustive sparse-cut max over all words on a 3- or 4-coordinate moment
    support (last coeff normalized to -1). The audited certification kernel."""

def rref_mod(rows: List[List[int]], p: int) -> Tuple[int, List[List[int]], List[int]]:
    """(rank, rref rows, pivot columns) over F_p."""

def nullspace_mod(rows: List[List[int]], p: int) -> List[List[int]]:
    """Right-nullspace basis of the row span over F_p."""

def reduce_mod_span(vecs: List[List[int]], span: List[List[int]], p: int) -> List[List[int]]:
    """Canonical residues of vecs modulo the row span (RREF elimination)."""

def inv_mod(vals: List[int], p: int) -> npt.NDArray[np.uint64]:
    """Batch modular inverses (Montgomery trick; one exponentiation total)."""

def e_syms(p: int, rows: List[List[int]]) -> List[npt.NDArray[np.uint64]]:
    """Elementary-symmetric vectors (e_0..e_m) per row of values."""

def dd_rows(p: int, domain: List[int], subsets: List[List[int]]) -> List[List[int]]:
    """Divided-difference functional rows: D_T(w) = row . w."""

class Cyclo:
    """An element of Z[zeta_s] (negacyclic half-basis coefficients)."""
    def __init__(self, coeffs: list[int]) -> None: ...
    @staticmethod
    def monomial(s: int, exp: int) -> "Cyclo": ...
    @staticmethod
    def one_minus(s: int, exp: int) -> "Cyclo": ...
    @staticmethod
    def prod_one_minus(s: int, exps: list[int]) -> "Cyclo": ...
    @staticmethod
    def e_vector(s: int, exps: list[int], m: int) -> list["Cyclo"]: ...
    def coeffs(self) -> list[int]: ...
    def s(self) -> int: ...
    def add(self, o: "Cyclo") -> "Cyclo": ...
    def sub(self, o: "Cyclo") -> "Cyclo": ...
    def mul(self, o: "Cyclo") -> "Cyclo": ...
    def mul_ntt(self, o: "Cyclo") -> "Cyclo": ...
    def neg(self) -> "Cyclo": ...
    def dilate(self, d: int) -> "Cyclo": ...
    def galois(self, m: int) -> "Cyclo": ...
    def conj(self) -> "Cyclo": ...
    def eval_at(self, x: int, p: int) -> int: ...
    def norm_mod(self, p: int) -> int: ...
    def norm_i128(self) -> int: ...
    def norm_crt(self) -> int:
        """Exact field norm at any height (CRT over norm_mod at 62-bit
        good primes; arbitrary-precision int). The wrapper that
        norm_i128's overflow error points to."""
    def weight(self) -> int: ...
    def sq_sum(self) -> int: ...
    def height(self) -> int: ...
    def is_zero(self) -> bool: ...
    def as_int(self) -> int | None: ...
    def eq_int(self, v: int) -> bool: ...

def fold(half: int, exp: int) -> Tuple[int, int]:
    """The negacyclic fold: zeta^exp = sign * zeta^index on the
    half-basis; returns (index, sign) with sign in {+1, -1}."""

def norms_mod_batch(coeffs: List[List[int]], p: int, s: int) -> List[int]:
    """Batch Cyclo.norm_mod over many coefficient vectors at one good
    prime (subgroup built once; rayon-parallel)."""

def exact_value_census(s: int, r: int, coord: int) -> Tuple[int, int, List[int]]:
    """Exact Z[zeta_s] value census of coordinate coord over all
    r-subsets: (distinct, intrinsic_floor, top5_multiplicities).
    Prime-independent; integer-exact."""

KOALABEAR: int
"""The KoalaBear prime 2^31 - 2^24 + 1 (field::named)."""

BABYBEAR: int
"""The BabyBear prime 2^31 - 2^27 + 1 (field::named)."""

def certify_clean(p: int, s_top: int, wmax_large: int) -> List[Tuple[int, int, List[int]]]:
    """Tower cleanliness certificate (ring::primes): one (s, tier,
    counts) per level from s_top down to 8. tier: 0 Certified, 1
    CertifiedAtUnit (zero class exact; counts = [-2,2] census by
    weight), 2 CertifiedToWeight (counts = [wmax]), 3 Dirty (counts =
    {-1,0,1} census by weight). GIL released."""

def elias_row(s: int, total_len: int, base_q: int, ext_degree: int, target_bits: float) -> Tuple[int, int, float, float, float, bool]:
    """One certified Table-4-style row: (z_star, n, delta_star,
    lg_sound_lo, lg_sound_hi, crossing_pinned) for interleaving width s,
    rate 1/2, base alphabet base_q, |F| = base_q^ext_degree. Present only
    when the wheel is built with the certified feature."""

class Descent:
    """The level-halving operation s -> s/2 at (p, s, k): channel
    splits, channel syndromes, derived words. Build once per cell;
    build per-word views with word()."""
    def __init__(self, p: int, s: int, k: int) -> None: ...
    def k_even(self) -> int: ...
    def k_odd(self) -> int: ...
    def half_points(self) -> list[int]: ...
    def channels(self, word: list[int]) -> tuple[list[int], list[int]]: ...
    def unfold(self, wev: list[int], wod: list[int]) -> list[int]: ...
    def monomial_coeffs(self, word: list[int]) -> list[int]: ...
    def word(self, word: list[int]) -> "WordView": ...

class WordView:
    """Per-word descent data (from Descent.word): cached interpolant
    coefficients and channel words; per-pair and per-core queries."""
    def coeffs(self) -> list[int]: ...
    def channel_words(self) -> tuple[list[int], list[int]]: ...
    def channel_syndromes(self) -> tuple[list[int], list[int], list[int]]: ...
    def effective_syndrome(self, x: int, xp: int) -> list[int]: ...
    def psi_y(self, core: list[int]) -> list[tuple[int, int]]: ...
    def psi_y_stats(self, core: list[int]) -> tuple[int, int, int, int]:
        """(total, distinct, max_fiber, collisions)."""
    def stratum_identity_check(self) -> tuple[int, int]:
        """Both sides of the stratum identity (GIL released)."""
    def member_functional(self, core: list[int], i1: int, i2: int) -> int: ...

def fold_unit(s: int, e: int) -> Cyclo:
    """The fold unit u_e = (1 + zeta^e)/(1 - zeta^e), exact closed form."""

def foldunit_rank_certificate(s: int) -> Tuple[float, float, bool]:
    """(det_lo, det_hi, independent): certified fold-unit independence."""

def foldunit_alpha_certificate(
    level: int,
) -> Tuple[int, List[List[int]], List[int], float, float]:
    """Certified atom-address table at level (power of two, 16..=8192):
    (denom, alpha, torsion2s, residual_bound, height_gap) with
    alpha[j-1] = denom * alpha_j, certified exact (interval residual
    below the height gap + two-camera torsion pin)."""

def valuemap_census(
    p: int, level: int, h1: List[int], h2: List[int],
    size: int, class_: int, point: int,
) -> Tuple[int, int, int, int, int]:
    """(total, distinct, max_fiber, argmax, second_moment) of the MITM census."""

def valuemap_histogram(
    p: int, level: int, h1: List[int], h2: List[int],
    size: int, class_: int, point: int,
) -> npt.NDArray[np.uint64]:
    """Fiber-size histogram: out[k] = number of values with fiber size k."""

def valuemap_distribution(
    p: int, level: int, h1: List[int], h2: List[int],
    size: int, class_: int, point: int,
) -> Tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]:
    """(values, multiplicities), sorted by value."""

def valuemap_sweep(
    level: int, h1: List[int], h2: List[int],
    size: int, class_: int, point: int, primes: List[int],
) -> List[Tuple[int, int, int, int, int, int]]:
    """Parallel prime sweep: rows (p, total, distinct, max, argmax, second_moment)."""

def valuemap_fiber(
    p: int, level: int, h1: List[int], h2: List[int],
    size: int, class_: int, point: int, value: int,
) -> int:
    """Fiber size of one target value."""

def valuemap_fiber_members(
    p: int, level: int, h1: List[int], h2: List[int],
    size: int, class_: int, point: int, value: int, cap: int,
) -> List[List[int]]:
    """Members of one fiber as sorted exponent lists (up to cap)."""

def skeleton_totals(level: int) -> Tuple[int, int, int]:
    """Exact G1 skeleton-DP totals at level 32 or 64:
    (window, budget_pairs, budget_skeletons)."""

def skeleton_census(level: int) -> Tuple[int, int, int, int]:
    """Exact G1-criterion census at level 32 or 64 via the MITM join:
    (m1_pairs, m2_pairs, solvable_pairs, solutions). Level 64 is a
    minutes-scale many-core run (S4: 262s on 252 threads)."""

class VsSpace:
    """The vanishing-syndrome geometry VS(s, k) on mu_s in F_p — the dual
    (quotient) view of RS[F_p, mu_s, k]. Convention authority: domain =
    generator powers; subsets = ascending index tuples, lex-ranked;
    syndrome b_j = (-1)^j c_{k+j} so D_S(w) = <b, e(complement of S)>."""
    def __init__(self, p: int, s: int, k: int) -> None: ...
    @property
    def p(self) -> int: ...
    @property
    def s(self) -> int: ...
    @property
    def k(self) -> int: ...
    @property
    def r(self) -> int: ...
    @property
    def syndrome_dim(self) -> int: ...
    def domain(self) -> list[int]: ...
    def syndrome(self, word: list[int]) -> list[int]: ...
    def word(self, b: list[int]) -> list[int]: ...
    def moment_row(self, subset: list[int]) -> list[int]: ...
    def incident(self, b: list[int], subset: list[int]) -> bool: ...
    def divided_difference(self, word: list[int], subset: list[int]) -> int: ...
    def subset_rank(self, subset: list[int]) -> int: ...
    def subset_unrank(self, rank: int) -> list[int]: ...
    def twist_subset(self, subset: list[int]) -> list[int]: ...
    def invert_subset(self, subset: list[int]) -> list[int]: ...
    def subset_orbit_canon(self, subset: list[int]) -> list[int]: ...
    def core(self, subset: list[int]) -> tuple[list[int], int]: ...
    def strata_counts(self, b: list[int]) -> list[int]: ...
    def top_word(self, c: int) -> list[int]: ...
    def coordinate_word(self, j: int) -> list[int]: ...
    def fold_ladder_word(self) -> list[int]: ...
    def gs_class_counts(self) -> list[int]: ...
    def cut_counts(self, bs: list[list[int]]) -> list[int]: ...
    def list_sizes_cut(self, words: list[list[int]], t: int) -> list[int]:
        """Cut-driven exact list sizes for a batch of words at
        agreement t >= r — the ownership identity as an algorithm;
        GIL released."""
    def cut_max_sparse(self, support: list[int]) -> tuple[int, list[int]]: ...
    def certificate(self) -> dict: ...
