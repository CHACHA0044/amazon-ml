"""Deterministic Scoring and Tiered Rule Engine for Entity Resolution.

Implements the evidence hierarchy:
  ADDRESS -> NUMERIC -> NAME

Calculates continuous confidence scores and assigns candidate acceptance tiers:
  - TIER 1: Very High Precision Anchor (Confidence >= 0.90)
  - TIER 2: High Precision Match (Confidence >= 0.75)
  - TIER 3: Conditional Match (Confidence >= 0.50)
"""

from typing import Dict, Tuple


def evaluate_pair_rules(feat: Dict[str, float], country: str = "US") -> Tuple[float, int]:
    """Evaluate deterministic matching rules and return (confidence_score, tier).
    
    Tiers:
      1: Very High Precision (Anchor)
      2: High Precision (Corroborated)
      3: Conditional (Requires strong S1 context)
      0: Rejected
    """
    n_eq = feat["name_eq"] > 0.5 or feat["name_suf_eq"] > 0.5
    n_jac = feat["name_tok_jac"]
    n_c2 = feat["name_char2"]
    n_c3 = feat["name_char3"]
    n_cnt = feat["name_contain"]

    a_eq = feat["addr_eq"] > 0.5
    a_jac = feat["addr_tok_jac"]
    a_c2 = feat["addr_char2"]
    a_c3 = feat["addr_char3"]
    a_cnt = feat["addr_contain"]
    a_miss = feat["addr_missing"] > 0.5

    num_ov = feat["num_overlap"] >= 1.0
    first_num = feat["first_num_match"] > 0.5
    last_num = feat["last_num_match"] > 0.5

    # ---------------------------------------------------------
    # TIER 1: VERY HIGH PRECISION ANCHORS (Confidence >= 0.90)
    # ---------------------------------------------------------

    # 1. Exact/Suffix Name with any address/numeric corroboration
    if n_eq and (a_eq or a_jac >= 0.35 or a_c2 >= 0.50 or num_ov or first_num):
        conf = 0.95 + 0.04 * min(1.0, a_jac + (0.05 if num_ov else 0.0))
        return conf, 1

    # 2. Exact Address with reasonable name agreement
    if a_eq and (n_jac >= 0.30 or n_c2 >= 0.40 or n_cnt >= 0.50):
        conf = 0.94 + 0.05 * min(1.0, n_jac)
        return conf, 1

    # 3. Strong Name AND Strong Address
    if (n_jac >= 0.65 or n_c3 >= 0.70) and (a_jac >= 0.45 or a_c2 >= 0.60):
        conf = 0.92 + 0.05 * min(1.0, (n_jac + a_jac) / 2.0)
        return conf, 1

    # 4. Numeric Agreement + Solid Name + Solid Address
    if (num_ov or first_num) and (n_jac >= 0.45 or n_c2 >= 0.55 or n_cnt >= 0.60) and (a_jac >= 0.30 or a_c2 >= 0.45):
        conf = 0.91 + 0.05 * min(1.0, (n_jac + a_jac) / 2.0)
        return conf, 1

    # ---------------------------------------------------------
    # TIER 2: HIGH PRECISION (Confidence >= 0.75)
    # ---------------------------------------------------------

    # 5. Strong Address with Numeric Support (even if name is rewritten)
    if (a_jac >= 0.50 or a_c3 >= 0.65) and (num_ov or first_num) and (n_c2 >= 0.30 or n_cnt >= 0.35):
        conf = 0.85 + 0.05 * a_jac
        return conf, 2

    # 6. Solid Name and Moderate Address
    if (n_jac >= 0.55 or n_c3 >= 0.60) and (a_jac >= 0.35 or a_c2 >= 0.50):
        conf = 0.80 + 0.05 * n_jac
        return conf, 2

    # 7. Numeric agreement with moderate name
    if (num_ov or first_num or last_num) and (n_jac >= 0.60 or (n_c2 >= 0.70 and n_cnt >= 0.60)):
        conf = 0.78 + 0.05 * n_jac
        return conf, 2

    # 8. Address missing on candidate, but exact name match with distinct tokens
    if a_miss and n_eq and feat["name_len_diff"] <= 3:
        conf = 0.76
        return conf, 2

    # ---------------------------------------------------------
    # TIER 3: CONDITIONAL (Confidence >= 0.50)
    # ---------------------------------------------------------

    # 9. Moderate Address and minor name overlap
    if a_jac >= 0.45 and (n_jac >= 0.25 or n_c2 >= 0.35):
        conf = 0.65 + 0.05 * a_jac
        return conf, 3

    # 10. Near exact name with weak address
    if (n_jac >= 0.80 or (n_eq and feat["name_len_diff"] <= 2)) and (a_c2 >= 0.30 or a_cnt >= 0.30):
        conf = 0.60
        return conf, 3

    # Default: rejected
    return 0.0, 0
