"""Pairwise Feature Computation Module.

Computes fine-grained similarity signals across Name, Address, Numeric, and Channel Provenance.
Optimized for high-speed calculation during candidate evaluation.
"""

from typing import Dict, List, Set, Tuple


def char_ngrams(s: str, n: int) -> Set[str]:
    """Generate set of character n-grams."""
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def dice_sim(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Dice similarity coefficient between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    return 2.0 * intersection / (len(set_a) + len(set_b))


def jaccard_sim(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return len(set_a & set_b) / union


def containment_sim(toks_a: Set[str], toks_b: Set[str]) -> float:
    """Compute asymmetric containment similarity (overlap / min(len_a, len_b))."""
    if not toks_a or not toks_b:
        return 0.0
    intersection = len(toks_a & toks_b)
    return intersection / min(len(toks_a), len(toks_b))


def compute_pairwise_features(
    q_name_norm: str,
    q_name_suf: str,
    q_name_toks: Set[str],
    q_name_c2: Set[str],
    q_name_c3: Set[str],
    q_addr_norm: str,
    q_addr_toks: Set[str],
    q_addr_c2: Set[str],
    q_addr_c3: Set[str],
    q_nums: List[str],
    c_name_norm: str,
    c_name_suf: str,
    c_name_toks: Set[str],
    c_name_c2: Set[str],
    c_name_c3: Set[str],
    c_addr_norm: str,
    c_addr_toks: Set[str],
    c_addr_c2: Set[str],
    c_addr_c3: Set[str],
    c_nums: List[str],
    channel_mask: int,
) -> Dict[str, float]:
    """Compute full suite of similarity features for a (query, candidate) pair."""
    # 1. Name features
    name_eq = 1.0 if (q_name_norm and q_name_norm == c_name_norm) else 0.0
    name_suf_eq = 1.0 if (q_name_suf and q_name_suf == c_name_suf) else 0.0
    name_tok_jac = jaccard_sim(q_name_toks, c_name_toks)
    name_char2 = dice_sim(q_name_c2, c_name_c2)
    name_char3 = dice_sim(q_name_c3, c_name_c3)
    name_contain = containment_sim(q_name_toks, c_name_toks)
    name_len_diff = float(abs(len(q_name_norm) - len(c_name_norm)))
    name_tok_diff = float(abs(len(q_name_toks) - len(c_name_toks)))

    # 2. Address features
    addr_missing = 1.0 if not c_addr_norm else 0.0
    if not c_addr_norm or not q_addr_norm:
        addr_eq = 0.0
        addr_tok_jac = 0.0
        addr_char2 = 0.0
        addr_char3 = 0.0
        addr_contain = 0.0
        addr_len_diff = 0.0
    else:
        addr_eq = 1.0 if (q_addr_norm == c_addr_norm) else 0.0
        addr_tok_jac = jaccard_sim(q_addr_toks, c_addr_toks)
        addr_char2 = dice_sim(q_addr_c2, c_addr_c2)
        addr_char3 = dice_sim(q_addr_c3, c_addr_c3)
        addr_contain = containment_sim(q_addr_toks, c_addr_toks)
        addr_len_diff = float(abs(len(q_addr_norm) - len(c_addr_norm)))

    # 3. Numeric features
    q_num_set = set(q_nums)
    c_num_set = set(c_nums)
    shared_nums = q_num_set & c_num_set
    num_overlap = float(len(shared_nums))
    max_num_len = max(len(q_num_set), len(c_num_set), 1)
    num_overlap_ratio = num_overlap / max_num_len

    first_num_match = (
        1.0 if (q_nums and c_nums and q_nums[0] == c_nums[0]) else 0.0
    )
    last_num_match = (
        1.0 if (q_nums and c_nums and q_nums[-1] == c_nums[-1]) else 0.0
    )

    # 4. Provenance
    ch_count = float(bin(channel_mask).count("1"))

    return {
        "name_eq": name_eq,
        "name_suf_eq": name_suf_eq,
        "name_tok_jac": name_tok_jac,
        "name_char2": name_char2,
        "name_char3": name_char3,
        "name_contain": name_contain,
        "name_len_diff": name_len_diff,
        "name_tok_diff": name_tok_diff,
        "addr_eq": addr_eq,
        "addr_tok_jac": addr_tok_jac,
        "addr_char2": addr_char2,
        "addr_char3": addr_char3,
        "addr_contain": addr_contain,
        "addr_len_diff": addr_len_diff,
        "addr_missing": addr_missing,
        "num_overlap": num_overlap,
        "num_overlap_ratio": num_overlap_ratio,
        "first_num_match": first_num_match,
        "last_num_match": last_num_match,
        "channel_mask": float(channel_mask),
        "ch_count": ch_count,
    }
