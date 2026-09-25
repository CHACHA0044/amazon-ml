"""Vectorized-ish pair-level similarity feature computation (pure Python cores).

Used by the discovery scripts.  Heavy features are computed off precomputed
normalized strings; cheap exact-match flags are computed via polars joins.
"""
import numpy as np


def ngrams(s, n):
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def name_tokens(s):
    return s.split() if s else []


def token_jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


def token_containment(a, b):
    """fraction of a's tokens found in b."""
    if not a:
        return 1.0
    sa = set(a)
    sb = set(b)
    if not sb:
        return 0.0
    return len(sa & sb) / len(sa)


def char_jaccard(a, b, n):
    ga, gb = ngrams(a, n), ngrams(b, n)
    if not ga and not gb:
        return 1.0
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def numeric_overlap(a_tok, b_tok):
    """a_tok, b_tok: space-joined sorted unique numeric-ish tokens."""
    if not a_tok or not b_tok:
        return 0
    sa = set(a_tok.split())
    sb = set(b_tok.split())
    return len(sa & sb)


RARE_VOCAB_SET = None
RARE_DF_CUT = 25


def set_rare_vocab(vocab):
    global RARE_VOCAB_SET
    RARE_VOCAB_SET = set(vocab)


def rare_shared(name1, name2):
    if RARE_VOCAB_SET is None:
        return 0
    t1 = set(name_tokens(name1)) & RARE_VOCAB_SET
    if not t1:
        return 0
    t2 = set(name_tokens(name2)) & RARE_VOCAB_SET
    return len(t1 & t2)


def heavy_features_batch(name_q, name_c, addr_q, addr_c, num_q=None, num_c=None):
    """Compute heavy features for aligned lists. Returns dict of np arrays."""
    n = len(name_q)
    if num_q is None:
        num_q = [""] * n
    if num_c is None:
        num_c = [""] * n
    out = {
        "name_tok_jac": np.zeros(n, dtype=np.float32),
        "name_contain_q": np.zeros(n, dtype=np.float32),
        "name_contain_c": np.zeros(n, dtype=np.float32),
        "addr_tok_jac": np.zeros(n, dtype=np.float32),
        "addr_contain_q": np.zeros(n, dtype=np.float32),
        "addr_contain_c": np.zeros(n, dtype=np.float32),
        "name_char2": np.zeros(n, dtype=np.float32),
        "name_char3": np.zeros(n, dtype=np.float32),
        "addr_char2": np.zeros(n, dtype=np.float32),
        "addr_char3": np.zeros(n, dtype=np.float32),
        "num_overlap": np.zeros(n, dtype=np.int32),
        "rare_shared": np.zeros(n, dtype=np.int32),
    }
    for i in range(n):
        nq = name_tokens(name_q[i])
        nc = name_tokens(name_c[i])
        aq = name_tokens(addr_q[i])
        ac = name_tokens(addr_c[i])
        out["name_tok_jac"][i] = token_jaccard(nq, nc)
        out["name_contain_q"][i] = token_containment(nq, nc)
        out["name_contain_c"][i] = token_containment(nc, nq)
        out["addr_tok_jac"][i] = token_jaccard(aq, ac)
        out["addr_contain_q"][i] = token_containment(aq, ac)
        out["addr_contain_c"][i] = token_containment(ac, aq)
        out["name_char2"][i] = char_jaccard(name_q[i], name_c[i], 2)
        out["name_char3"][i] = char_jaccard(name_q[i], name_c[i], 3)
        out["addr_char2"][i] = char_jaccard(addr_q[i], addr_c[i], 2)
        out["addr_char3"][i] = char_jaccard(addr_q[i], addr_c[i], 3)
        out["num_overlap"][i] = numeric_overlap(num_q[i], num_c[i])
        out["rare_shared"][i] = rare_shared(name_q[i], name_c[i])
    return out


def auc_score(pos, neg):
    """Area under the ROC curve (Mann-Whitney U) for a single feature.
    pos/neg are float arrays. Handles ties via average ranking."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    combined = np.concatenate([pos, neg])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty_like(combined)
    ranks[order] = np.arange(1, len(combined) + 1)
    # average ranks for ties
    sorted_vals = combined[order]
    tie_edges = np.flatnonzero(np.diff(sorted_vals) != 0)
    starts = np.concatenate([[0], tie_edges + 1])
    ends = np.concatenate([tie_edges + 1, [len(combined)]])
    for s, e in zip(starts, ends):
        if e - s > 1:
            ranks[order[s:e]] = (s + 1 + e) / 2.0
    n_pos = len(pos)
    r_sum_pos = ranks[:n_pos].sum()
    u = r_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * len(neg))