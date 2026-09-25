"""Per-S1 Macro-Aware Selection Policies for Entity Resolution (EXP-002).

Implements and evaluates alternative selection strategies to optimize the official
macro-averaged F0.5 score:
  - Policy A: Global confidence threshold
  - Policy B: Tiered confidence thresholds
  - Policy C: Per-S1 ranked top-K with dynamic confidence window
  - Policy D: Anchor-based expansion (Strict singleton protection)
  - Policy E: Mutually exclusive global cluster assignment (Bipartite matching)
"""

from typing import Dict, List, Set, Tuple


def select_policy_a(scored_candidates: List[Tuple[str, float, int]], threshold: float = 0.75) -> Set[str]:
    """Policy A: Flat global threshold."""
    return {cid for cid, conf, _ in scored_candidates if conf >= threshold}


def select_policy_b(
    scored_candidates: List[Tuple[str, float, int]],
    tier1_thresh: float = 0.88,
    tier2_thresh: float = 0.75,
    include_tier3: bool = False,
) -> Set[str]:
    """Policy B: Tiered confidence threshold."""
    selected = set()
    for cid, conf, tier in scored_candidates:
        if tier == 1 and conf >= tier1_thresh:
            selected.add(cid)
        elif tier == 2 and conf >= tier2_thresh:
            selected.add(cid)
        elif tier == 3 and include_tier3 and conf >= 0.65:
            selected.add(cid)
    return selected


def select_policy_c(
    scored_candidates: List[Tuple[str, float, int]],
    anchor_min_conf: float = 0.85,
    expansion_delta: float = 0.15,
    min_cand_conf: float = 0.72,
    max_matches_per_s1: int = 11,
) -> Set[str]:
    """Policy C: Ranked selection with dynamic confidence decay from top anchor."""
    if not scored_candidates:
        return set()

    # Sort descending by confidence
    sorted_cands = sorted(scored_candidates, key=lambda x: x[1], reverse=True)
    top_cid, top_conf, top_tier = sorted_cands[0]

    # If top candidate is not strong enough, reject all (protect singletons!)
    if top_conf < anchor_min_conf:
        return set()

    selected = {top_cid}
    min_allowed_conf = max(min_cand_conf, top_conf - expansion_delta)

    for cid, conf, tier in sorted_cands[1:max_matches_per_s1]:
        if conf >= min_allowed_conf and tier in (1, 2):
            selected.add(cid)

    return selected


def select_policy_d(
    scored_candidates: List[Tuple[str, float, int]],
    tier1_min_conf: float = 0.88,
    tier2_anchor_conf: float = 0.82,
    tier2_expansion_conf: float = 0.75,
    max_matches_per_s1: int = 11,
) -> Set[str]:
    """Policy D: High-confidence anchor with conditional expansion.
    
    Logic:
      - If at least one Tier 1 anchor exists: entity is proven non-singleton,
        accept all Tier 1 + Tier 2 with conf >= tier2_expansion_conf (0.75).
      - If NO Tier 1 anchor exists: require high bar (conf >= tier2_anchor_conf, e.g. 0.82)
        to prevent turning true singletons into false positives.
    """
    if not scored_candidates:
        return set()

    # Partition candidates by tier
    t1_cands = [(cid, conf) for cid, conf, tier in scored_candidates if tier == 1 and conf >= tier1_min_conf]
    t2_cands = [(cid, conf) for cid, conf, tier in scored_candidates if tier == 2]

    selected = set()

    if t1_cands:
        # We have strong anchor(s)
        for cid, _ in t1_cands:
            selected.add(cid)
        # Expand with reliable Tier 2 candidates
        for cid, conf in t2_cands:
            if conf >= tier2_expansion_conf and len(selected) < max_matches_per_s1:
                selected.add(cid)
    else:
        # No Tier 1 anchor - apply conservative singleton shield
        for cid, conf in t2_cands:
            if conf >= tier2_anchor_conf and len(selected) < max_matches_per_s1:
                selected.add(cid)

    return selected
