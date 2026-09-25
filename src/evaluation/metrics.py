"""Evaluation metrics module for Amazon ML Challenge 2026.

OFFICIAL METRIC (authoritative source: problem_statement.txt):

    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

    "Computed as a macro-average: F_0.5 is calculated per Source 1 entity,
     then averaged across all Source 1 entities in the evaluation set.
     Singletons are included in that average. A Source 1 entity with no true
     matches scores 1.0 when you correctly predict an empty list, and 0.0 when
     you predict any match for it."

Therefore the leaderboard score is the simple mean of the per-entity F_0.5
values (each Source 1 entity contributes equally regardless of match
cardinality).  This is NOT the micro-averaged F_0.5 (global TP/FP/FN), which
earlier drafts of our docs described.  That earlier description contradicted
problem_statement.txt and is now corrected here.

Per-entity F_0.5 conventions (derived from the spec):
    true == empty and pred == empty   -> 1.0 (singleton correctly predicted)
    true == empty and pred != empty   -> 0.0 (false merge on a singleton)
    true != empty and pred == empty   -> 0.0 (recall 0)
    true != empty and pred != empty   -> standard formula with per-entity
                                         precision = tp/(tp+fp) and
                                         recall    = tp/(tp+fn)
"""

from typing import Dict, Iterable, List, Optional, Set, Tuple, Union


def parse_ground_truth_line(line: str) -> Tuple[str, Set[str]]:
    """Parse a single line from train_ground_truth.tsv or submission.tsv.

    Format: source1_entity_id\\tmatched_entity_ids (comma-separated or empty)
    """
    parts = line.strip().split("\t")
    s1_id = parts[0]
    if len(parts) > 1 and parts[1]:
        matched = set(x.strip() for x in parts[1].split(",") if x.strip())
    else:
        matched = set()
    return s1_id, matched


def compute_entity_f05(pred_set: Set[str], true_set: Set[str]) -> float:
    """F_0.5 for a single Source 1 entity (official definition).

    Conventions:
      - both empty: 1.0 (correct singleton prediction)
      - pred not empty, true empty: 0.0 (false merge)
      - pred empty, true not empty: 0.0 (missed matches)
      - otherwise: standard formula.
    """
    if not pred_set and not true_set:
        return 1.0

    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    denom = (0.25 * precision) + recall
    if denom <= 0.0:
        return 0.0
    return (1.25 * precision * recall) / denom


def compute_f05_score(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> Dict[str, Union[float, int]]:
    """OFFICIAL macro-averaged F_0.5: mean of per-Source-1-entity F_0.5.

    Every Source 1 entity contributes equally, singletons included.

    Returns:
        Dict with keys: precision, recall, f05, tp, fp, fn, n_entities,
        plus micro_precision, micro_recall, micro_f05 for comparison.
    """
    entity_scores: List[float] = []
    total_tp = total_fp = total_fn = 0
    n_entities = 0

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        n_entities += 1
        entity_scores.append(compute_entity_f05(pred_set, true_set))

        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    f05_macro = sum(entity_scores) / len(entity_scores) if entity_scores else 0.0

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_denom = (0.25 * micro_precision) + micro_recall
    micro_f05 = (1.25 * micro_precision * micro_recall) / micro_denom if micro_denom > 0 else 0.0

    return {
        "precision": round(micro_precision, 6),  # global (micro) precision
        "recall": round(micro_recall, 6),        # global (micro) recall
        "f05": round(f05_macro, 6),              # OFFICIAL macro-averaged F_0.5
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "n_entities": n_entities,
        "micro_f05": round(micro_f05, 6),
        "f05_stdev": round(_stdev(entity_scores), 6) if entity_scores else 0.0,
    }


def compute_f05_micro(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> Dict[str, Union[float, int]]:
    """Micro-averaged (pair-level) F_0.5 for comparison only.

    This is NOT the official competition metric.  Provided to quantify how the
    two aggregations diverge.
    """
    total_tp = total_fp = total_fn = 0
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        total_tp += len(pred_set & true_set)
        total_fp += len(pred_set - true_set)
        total_fn += len(true_set - pred_set)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    denom = (0.25 * precision) + recall
    f05 = (1.25 * precision * recall) / denom if denom > 0 else 0.0

    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f05": round(f05, 6),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def _stdev(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return var ** 0.5