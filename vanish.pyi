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
    """The additive C.5 word sum_i (-1)^i lam_i x^{r-i} on mu_s."""

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
    def cut_max_sparse(self, support: list[int]) -> tuple[int, list[int]]: ...
    def certificate(self) -> dict: ...
