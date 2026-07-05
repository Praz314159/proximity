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

def buckets_e(p: int, s: int, r: int, q: int, lams: List[List[int]]) -> List[int]:
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
