"""Scoring helpers: exact-match accuracy, macro-F1, confusion matrices, and
set-based precision/recall for the semicolon-joined risk_flags field.
"""

from collections import Counter, defaultdict
from typing import Dict, List


def accuracy(predicted: List[str], expected: List[str]) -> float:
    if not predicted:
        return 0.0
    correct = sum(1 for p, e in zip(predicted, expected) if p == e)
    return correct / len(predicted)


def macro_f1(predicted: List[str], expected: List[str]) -> float:
    labels = sorted(set(expected) | set(predicted))
    if not labels:
        return 0.0
    f1s = []
    for label in labels:
        tp = sum(1 for p, e in zip(predicted, expected) if p == label and e == label)
        fp = sum(1 for p, e in zip(predicted, expected) if p == label and e != label)
        fn = sum(1 for p, e in zip(predicted, expected) if p != label and e == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def confusion_matrix(predicted: List[str], expected: List[str]) -> Dict[str, Counter]:
    matrix: Dict[str, Counter] = defaultdict(Counter)
    for p, e in zip(predicted, expected):
        matrix[e][p] += 1
    return matrix


def set_field_precision_recall(predicted: List[str], expected: List[str]) -> Dict[str, float]:
    """For semicolon-joined fields like risk_flags / supporting_image_ids.
    Treats "none" as an empty set. Micro-averaged over all rows.
    """
    tp = fp = fn = 0
    for p, e in zip(predicted, expected):
        p_set = set(x for x in p.split(";") if x and x != "none")
        e_set = set(x for x in e.split(";") if x and x != "none")
        tp += len(p_set & e_set)
        fp += len(p_set - e_set)
        fn += len(e_set - p_set)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


FIELDS_EXACT = [
    "evidence_standard_met",
    "issue_type",
    "object_part",
    "claim_status",
    "valid_image",
    "severity",
]

FIELDS_MACRO_F1 = ["claim_status", "issue_type", "object_part"]

FIELDS_SET = ["risk_flags", "supporting_image_ids"]


def score(predictions: List[Dict[str, str]], expected_rows: List[Dict[str, str]]) -> Dict:
    report: Dict = {"n": len(predictions), "accuracy": {}, "macro_f1": {}, "set_fields": {}}

    for field in FIELDS_EXACT:
        predicted = [r.get(field, "") for r in predictions]
        expected = [r.get(field, "") for r in expected_rows]
        report["accuracy"][field] = accuracy(predicted, expected)

    for field in FIELDS_MACRO_F1:
        predicted = [r.get(field, "") for r in predictions]
        expected = [r.get(field, "") for r in expected_rows]
        report["macro_f1"][field] = macro_f1(predicted, expected)

    for field in FIELDS_SET:
        predicted = [r.get(field, "") for r in predictions]
        expected = [r.get(field, "") for r in expected_rows]
        report["set_fields"][field] = set_field_precision_recall(predicted, expected)

    report["claim_status_confusion"] = confusion_matrix(
        [r.get("claim_status", "") for r in predictions],
        [r.get("claim_status", "") for r in expected_rows],
    )
    return report
